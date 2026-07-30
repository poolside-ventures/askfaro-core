//! In-process llama.cpp generation provider.
//!
//! Sibling of [`apple_fm`](crate::generation::apple_fm), never nested under it:
//! the two have disjoint dependency trees and either can be enabled without the
//! other. This one runs GGUF weights on whatever accelerator the consumer built
//! `llama-cpp-2` for (Metal, CUDA, Vulkan, CPU), which is why the app does not
//! need a separate inference runtime installed alongside it.
//!
//! ## What is ours and what is upstream's
//!
//! Almost none of the hard part is ours. The `llama-cpp-2` bindings expose only
//! the LEGACY `llama_chat_apply_template` (role + content, no tools), but
//! llama.cpp's `common/chat.h` has the real machinery, and `libllama-common.a`
//! is already compiled and linked by `llama-cpp-sys-2`. A thin C++ shim
//! (`chat_shim.cpp`) bridges to it, so we get, from upstream and per model
//! family:
//!
//! - the chat template with tool schemas rendered into it,
//! - tool-call parsing into structured `{name, arguments}`,
//! - the reasoning/content split, with the thinking tags read from the model,
//! - an auto-generated grammar for constrained tool calls.
//!
//! ## Three ways this fails silently, all found the hard way
//!
//! Every one of them produces *plausible text* rather than an error, so they are
//! encoded here rather than left to be rediscovered:
//!
//! 1. `common_chat_parser_params`'s convenience constructor copies only `format`
//!    and `generation_prompt`, leaving the compiled PEG arena empty. Gemma 4
//!    resolves to the `peg-gemma4` format, so an empty arena means every tool
//!    call falls through to raw `content`. The shim wires the arena explicitly.
//! 2. `prompt` and `generation_prompt` are separate fields. Send only the first
//!    and the model emits its own turn opener, which defeats the parser.
//! 3. The end-of-turn token must stay in the text handed to the parser, because
//!    the grammar matches a COMPLETE assistant turn.

// `token_to_str` is deprecated in favour of `token_to_piece`, which requires an
// `encoding_rs::Decoder` the caller must thread through. Not worth an extra
// dependency here: the text is immediately handed to the parser, which wants the
// special tokens rendered exactly as this produces them.
#![allow(deprecated)]

mod ffi;

use std::path::{Path, PathBuf};
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

use crate::generation::{
    Availability, GenError, GenerateRequest, GenerateResponse, GenerationEngine, Timings, ToolCall,
};

/// Hard cap on generated tokens per turn, so a runaway model cannot hang a UI.
/// Reported through [`Timings::truncated`] when hit, because a truncated answer
/// must never be mistaken for a brief one.
const MAX_OUTPUT_TOKENS: usize = 2048;

/// Configuration for [`LlamaCppEngine`].
#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    /// Path to the GGUF weights.
    pub model_path: PathBuf,
    /// Total context window in tokens. A property of the MODEL, so the caller
    /// supplies it from its model profile; this crate does not decide it.
    pub n_ctx: u32,
    /// Layers to offload to the GPU. A large value means "all it will take".
    pub n_gpu_layers: u32,
    /// Whether the model may emit a reasoning channel.
    pub enable_thinking: bool,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            n_ctx: 16384,
            n_gpu_layers: 1000,
            enable_thinking: true,
        }
    }
}

/// On-device generation backed by an in-process llama.cpp.
///
/// Construction is cheap and loads nothing; the weights land on the first
/// [`generate`](GenerationEngine::generate) and then stay resident. That
/// laziness is load-bearing rather than tidy: a cold load measured **12.5s**, so
/// an engine rebuilt per call would be unusable. Keep one alive.
pub struct LlamaCppEngine {
    cfg: LlamaCppConfig,
    loaded: Option<Loaded>,
}

struct Loaded {
    backend: LlamaBackend,
    model: LlamaModel,
    chat: ffi::Chat,
}

impl LlamaCppEngine {
    /// Cheap to construct; loads nothing. Check
    /// [`availability`](GenerationEngine::availability) first.
    pub fn new(cfg: LlamaCppConfig) -> Self {
        Self { cfg, loaded: None }
    }

    /// True once the weights are resident.
    pub fn is_warm(&self) -> bool {
        self.loaded.is_some()
    }

    /// Cheap probe: does the configured GGUF exist? Deliberately does NOT load.
    pub fn availability_for(model_path: &Path) -> Availability {
        if model_path.as_os_str().is_empty() {
            return Availability::NotEnabled("no model path configured".into());
        }
        if !model_path.exists() {
            return Availability::NotEnabled(format!(
                "model not downloaded: {}",
                model_path.display()
            ));
        }
        Availability::Available
    }

    fn ensure_loaded(&mut self) -> Result<u64, GenError> {
        if self.loaded.is_some() {
            return Ok(0);
        }
        let t = Instant::now();
        let backend = LlamaBackend::init().map_err(|e| GenError::Generate(e.to_string()))?;
        let params = LlamaModelParams::default().with_n_gpu_layers(self.cfg.n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, &self.cfg.model_path, &params)
            .map_err(|e| GenError::Generate(format!("load {}: {e}", self.cfg.model_path.display())))?;

        // The template comes from the GGUF itself, so the shim never has to guess
        // which model family it is rendering for.
        let template = model
            .chat_template(None)
            .map_err(|e| GenError::Generate(format!("model has no chat template: {e}")))?
            .to_string()
            .map_err(|e| GenError::Generate(format!("chat template is not utf-8: {e}")))?;
        let chat = ffi::Chat::new(&template)
            .map_err(|e| GenError::Generate(format!("chat shim init: {e}")))?;

        self.loaded = Some(Loaded { backend, model, chat });
        Ok(t.elapsed().as_millis() as u64)
    }
}

impl GenerationEngine for LlamaCppEngine {
    /// Reports only what can be known without a model path in hand. A caller
    /// with a configured engine should prefer [`Self::availability_for`].
    fn availability() -> Availability {
        Availability::Available
    }

    fn generate(&mut self, req: GenerateRequest) -> Result<GenerateResponse, GenError> {
        let load_ms = self.ensure_loaded()?;
        let enable_thinking = self.cfg.enable_thinking;
        let n_ctx = self.cfg.n_ctx;
        let loaded = self.loaded.as_mut().expect("just loaded");

        // --- render the prompt through the model's own template -------------
        let applied = loaded
            .chat
            .apply(&req, enable_thinking)
            .map_err(|e| GenError::Generate(format!("chat template: {e}")))?;

        // prompt + generation_prompt. See failure mode 2 in the module docs.
        let prompt = format!("{}{}", applied.prompt, applied.generation_prompt);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            // n_batch caps one decode() call. It defaults far below a real
            // prompt: five tool schemas render to ~18k chars and llama.cpp
            // asserts `n_tokens_all <= cparams.n_batch`. Size it to the window.
            .with_n_batch(n_ctx);
        let mut ctx = loaded
            .model
            .new_context(&loaded.backend, ctx_params)
            .map_err(|e| GenError::Generate(e.to_string()))?;

        let tokens = loaded
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| GenError::Generate(e.to_string()))?;
        if tokens.len() >= n_ctx as usize {
            return Err(GenError::ContextWindowExceeded);
        }

        let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
        let last = tokens.len() - 1;
        for (i, t) in tokens.iter().enumerate() {
            batch
                .add(*t, i as i32, &[0], i == last)
                .map_err(|e| GenError::Generate(e.to_string()))?;
        }

        let t_prefill = Instant::now();
        ctx.decode(&mut batch)
            .map_err(|e| GenError::Generate(e.to_string()))?;
        let prefill_ms = t_prefill.elapsed().as_millis() as u64;

        // --- decode ---------------------------------------------------------
        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let mut n_cur = tokens.len() as i32;
        let mut raw = String::new();
        let mut output_tokens = 0usize;
        let mut truncated = true;

        let t_decode = Instant::now();
        while output_tokens < MAX_OUTPUT_TOKENS {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            // The end-of-turn token stays in the text: the parser's grammar
            // matches a complete turn. See failure mode 3 in the module docs.
            raw.push_str(
                &loaded
                    .model
                    .token_to_str(token, Special::Tokenize)
                    .map_err(|e| GenError::Generate(e.to_string()))?,
            );
            output_tokens += 1;
            if loaded.model.is_eog_token(token) {
                truncated = false;
                break;
            }
            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| GenError::Generate(e.to_string()))?;
            n_cur += 1;
            ctx.decode(&mut batch)
                .map_err(|e| GenError::Generate(e.to_string()))?;
        }
        let decode_ms = t_decode.elapsed().as_millis() as u64;

        // --- parse ----------------------------------------------------------
        let parsed = loaded
            .chat
            .parse(&raw)
            .map_err(|e| GenError::Generate(format!("chat parse: {e}")))?;

        let tool_calls: Vec<ToolCall> = parsed
            .tool_calls
            .into_iter()
            .map(|c| ToolCall {
                name: c.name,
                // Upstream hands arguments back as a JSON string; a caller
                // wants the object. A non-object here is a malformed call, so
                // it becomes Null rather than being silently dropped.
                arguments: serde_json::from_str(&c.arguments).unwrap_or(serde_json::Value::Null),
            })
            .collect();

        let text = parsed.content.trim().to_string();
        Ok(GenerateResponse {
            abstained: tool_calls.is_empty() && text.is_empty(),
            text,
            tool_calls,
            model_ms: load_ms + prefill_ms + decode_ms,
            reasoning: parsed.reasoning_content,
            timings: Timings {
                load_ms,
                prefill_ms,
                decode_ms,
                prompt_tokens: tokens.len() as u32,
                output_tokens: output_tokens as u32,
                truncated,
            },
        })
    }
}
