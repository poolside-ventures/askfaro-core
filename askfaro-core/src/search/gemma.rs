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

/// EmbeddingGemma embedder. Construct once with [`GemmaEmbedder::load`] (loading
/// the model is the expensive step), then embed many texts.
pub struct GemmaEmbedder {
    // ort `Session::run` takes `&mut self`; a Mutex keeps `EmbedEngine` `&self`.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    space: String,
}

impl GemmaEmbedder {
    /// Load from a model directory containing `model.onnx` (+ its `.onnx_data`)
    /// and `tokenizer.json` — i.e. `EMBEDDINGGEMMA_300M_FP32.dir(cache_root)`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, GemmaError> {
        let dir = model_dir.as_ref();
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| GemmaError::Load(e.to_string()))?;
        let session = Session::builder()
            .map_err(|e| GemmaError::Load(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| GemmaError::Load(e.to_string()))?
            .commit_from_file(dir.join("model.onnx"))
            .map_err(|e| GemmaError::Load(e.to_string()))?;
        Ok(GemmaEmbedder {
            session: Mutex::new(session),
            tokenizer,
            // The model identity is the space name (the hard rule, §7).
            space: crate::search::models::EMBEDDINGGEMMA_SPACE.to_string(),
        })
    }

    fn embed_one(&self, prompted: &str) -> Option<Vec<f32>> {
        let enc = self.tokenizer.encode(prompted, true).ok()?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();
        let n = ids.len();
        let id_tensor = Tensor::from_array(([1_usize, n], ids)).ok()?;
        let mask_tensor = Tensor::from_array(([1_usize, n], mask)).ok()?;
        let mut session = self.session.lock().ok()?;
        let outputs = session
            .run(ort::inputs![
                "input_ids" => id_tensor,
                "attention_mask" => mask_tensor,
            ])
            .ok()?;
        let (_, data) = outputs["sentence_embedding"]
            .try_extract_tensor::<f32>()
            .ok()?;
        Some(data.to_vec())
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
