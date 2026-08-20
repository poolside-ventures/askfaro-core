//! EmbeddingGemma-300M embedder via `ort` + `tokenizers` (RFC §6 default).
//! Opt-in behind the `embeddinggemma` feature — this is the heavy native dep.
//!
//! The pipeline is exactly the Phase-1 spike's, validated to reproduce the
//! Python `onnxruntime` reference at cosine 1.0:
//!  - tokenize with `add_special_tokens = true` (Gemma `<bos>`/`<eos>`);
//!  - read the in-graph-pooled, L2-normalized `sentence_embedding` output
//!    (no pooling/normalization here — the graph owns it);
//!  - apply the retrieval prompt prefixes (distinct for query vs document).

use std::path::Path;
use std::sync::Mutex;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::search::contract::{document_prompt, query_prompt};
use crate::search::embed::EmbedEngine;

/// Errors loading the EmbeddingGemma embedder.
#[derive(Debug, thiserror::Error)]
pub enum GemmaError {
    #[error("failed to load embedding model: {0}")]
    Load(String),
}

/// EmbeddingGemma's context window. The ONNX graph hard-fails past it — the
/// RotaryEmbedding cos/sin cache is sized at export time, so token 2049 is a
/// runtime error, not degraded quality. Callers chunk by OTHER tokenizers'
/// counts (Honcho: 1500 tiktoken tokens, which can be 4x as many SentencePiece
/// tokens), so the tokenizer truncates to the window here — the same behavior
/// as the Python sentence-transformers reference (max_seq_length = 2048).
const MAX_TOKENS: usize = 2048;

/// Session knobs that move the embedder's RESIDENT footprint. Defaults are what
/// the shipping embedder uses; they trade a little throughput for memory, which
/// is the right trade for a model that stays warm for the life of the app on a
/// machine that is also holding a 4-bit brain.
pub struct GemmaOptions {
    /// ONNX Runtime's CPU arena allocator. Off by default: it is a high-water
    /// mark that is never released, and the sequence lengths this embedder sees
    /// vary continuously (every query and document is a different token count),
    /// so the mark keeps climbing.
    pub cpu_arena: bool,
    /// ORT's memory-pattern planner, which pre-allocates one block sized to the
    /// first run's shapes. Off by default for the same reason: it is a win for
    /// fixed shapes and a leak for varying ones.
    pub memory_pattern: bool,
    /// Intra-op threads. `None` leaves ORT's default (one per core).
    pub intra_threads: Option<usize>,
}

impl Default for GemmaOptions {
    fn default() -> Self {
        GemmaOptions {
            cpu_arena: false,
            memory_pattern: false,
            intra_threads: None,
        }
    }
}

/// EmbeddingGemma embedder. Construct once with [`GemmaEmbedder::load_variant`]
/// (loading the model is the expensive step), then embed many texts.
pub struct GemmaEmbedder {
    // ort `Session::run` takes `&mut self`; a Mutex keeps `EmbedEngine` `&self`.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    space: String,
}

impl GemmaEmbedder {
    /// Load a variant from the model cache root — the entry point callers want,
    /// because it is the one that cannot pair the wrong graph with the wrong
    /// directory or the wrong space.
    pub fn load_variant(
        variant: &crate::search::models::GemmaVariant,
        cache_root: &Path,
    ) -> Result<Self, GemmaError> {
        Self::load_graph(variant.spec.dir(cache_root), variant.graph, variant.space)
    }

    /// Load a named graph from the directory, declaring the space it writes.
    ///
    /// The quantized exports keep their own file names (`model_quantized.onnx`
    /// + `model_quantized.onnx_data`), because the graph's external-data record
    /// names its weight file: renaming the pair to `model.onnx` breaks the load.
    /// And a variant that shifts vectors is a different space (the hard rule,
    /// §7), so the caller names both together or neither.
    pub fn load_graph(
        model_dir: impl AsRef<Path>,
        graph: &str,
        space: &str,
    ) -> Result<Self, GemmaError> {
        Self::load_with(model_dir, graph, space, &GemmaOptions::default())
    }

    /// Load with explicit session options. See [`GemmaOptions`] — the defaults
    /// are what the embedder ships with, and they are chosen for resident
    /// footprint, not for throughput.
    pub fn load_with(
        model_dir: impl AsRef<Path>,
        graph: &str,
        space: &str,
        opts: &GemmaOptions,
    ) -> Result<Self, GemmaError> {
        let dir = model_dir.as_ref();
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| GemmaError::Load(e.to_string()))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| GemmaError::Load(e.to_string()))?;
        let mut builder = Session::builder()
            .map_err(|e| GemmaError::Load(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| GemmaError::Load(e.to_string()))?
            .with_memory_pattern(opts.memory_pattern)
            .map_err(|e| GemmaError::Load(e.to_string()))?;
        if let Some(n) = opts.intra_threads {
            builder = builder
                .with_intra_threads(n)
                .map_err(|e| GemmaError::Load(e.to_string()))?;
        }
        // Registering the CPU provider explicitly is the only way to reach
        // `DisableCpuMemArena`. The arena never returns memory: it grows to the
        // high-water mark of every shape the session has ever seen and holds it
        // for the life of the process, which for a warm embedder is forever.
        builder = builder
            .with_execution_providers([ort::ep::CPU::default()
                .with_arena_allocator(opts.cpu_arena)
                .build()])
            .map_err(|e| GemmaError::Load(e.to_string()))?;
        let session = builder
            .commit_from_file(dir.join(graph))
            .map_err(|e| GemmaError::Load(e.to_string()))?;
        Ok(GemmaEmbedder {
            session: Mutex::new(session),
            tokenizer,
            // The model identity is the space name (the hard rule, §7).
            space: space.to_string(),
        })
    }

    /// `None` is the contract [`EmbedEngine`] gives callers, and it carries no
    /// reason. So the reason is printed here instead of being dropped: a host
    /// that logs "embedding failed" and nothing else leaves no way to tell a
    /// text too long for the graph from a corrupt model file from a poisoned
    /// mutex. Same failure mode, and same fix, as the parse path in
    /// `generation::llama_cpp` (5c8c230).
    fn embed_one(&self, prompted: &str) -> Option<Vec<f32>> {
        match self.run_one(prompted) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!(
                    "embeddinggemma: embed failed for {} chars: {e}",
                    prompted.len()
                );
                None
            }
        }
    }

    fn run_one(&self, prompted: &str) -> Result<Vec<f32>, String> {
        let enc = self
            .tokenizer
            .encode(prompted, true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();
        let n = ids.len();
        // The graph hard-fails past its window rather than degrading, and the
        // tokenizer is configured to truncate there, so a sequence that arrives
        // longer than MAX_TOKENS means the truncation did not apply (a tokenizer
        // file carrying its own config, for instance). Say which of the two it
        // is instead of letting ort report an opaque shape error.
        if n > MAX_TOKENS {
            return Err(format!(
                "{n} tokens after truncation, past the {MAX_TOKENS}-token window"
            ));
        }
        let id_tensor =
            Tensor::from_array(([1_usize, n], ids)).map_err(|e| format!("input_ids: {e}"))?;
        let mask_tensor =
            Tensor::from_array(([1_usize, n], mask)).map_err(|e| format!("attention_mask: {e}"))?;
        let mut session = self.session.lock().map_err(|e| format!("session lock: {e}"))?;
        let outputs = session
            .run(ort::inputs![
                "input_ids" => id_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(|e| format!("onnx run ({n} tokens): {e}"))?;
        let (_, data) = outputs
            .get("sentence_embedding")
            .ok_or_else(|| {
                format!(
                    "no sentence_embedding output; graph exposes [{}]",
                    outputs.keys().collect::<Vec<_>>().join(", ")
                )
            })?
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("sentence_embedding extract: {e}"))?;
        if data.is_empty() {
            return Err(format!("empty sentence_embedding for {n} tokens"));
        }
        Ok(data.to_vec())
    }
}

impl EmbedEngine for GemmaEmbedder {
    fn space(&self) -> &str {
        &self.space
    }
    fn embed_documents(&self, texts: &[&str]) -> Vec<Option<Vec<f32>>> {
        texts
            .iter()
            .map(|t| self.embed_one(&document_prompt(t)))
            .collect()
    }
    fn embed_query(&self, text: &str) -> Option<Vec<f32>> {
        self.embed_one(&query_prompt(text))
    }
}
