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
use llama_cpp_2::context::LlamaContext;
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
    /// Kept beside `loaded` rather than inside it: the shim holds no borrow of
    /// the model (it is built from the template STRING), so it does not
    /// participate in the drop-order constraint above.
    chat: Option<ffi::Chat>,
}

/// The loaded engine.
///
/// **Field order is load-bearing.** Rust drops fields in declaration order, and
/// `ctx` borrows `model`, which borrows `backend`, so they must be declared in
/// that order to be torn down in it. Getting this wrong is not a leak, it is a
/// SIGABRT: ggml-metal asserts at exit that its resource sets were released
/// (`GGML_ASSERT([rsets->data count] == 0)`), which in an app reads as a crash on
/// quit. An earlier version leaked the model to dodge the self-reference and hit
/// exactly that.
struct Loaded {
    /// Lives across turns, and so does its KV cache.
    ///
    /// Rebuilding it per turn is what the first version did, and it cost **16.8s
    /// of prefill on every turn** in the 20-case parity bench: a thread replays
    /// an identical multi-thousand-token prefix each turn and a fresh context
    /// re-reads all of it. Phase 0 measured prefix reuse at 165x, so discarding
    /// the cache is the most expensive mistake available here.
    ///
    /// The `'static` is a lifetime extension over `model` below, sound because
    /// `model` is boxed (a stable address, never moved) and this field is dropped
    /// first.
    ctx: LlamaContext<'static>,
    /// The tokens currently in the KV cache, so the next turn decodes only what
    /// diverges from them.
    cached: Vec<llama_cpp_2::token::LlamaToken>,
    /// Boxed for a stable address that `ctx` can borrow. Dropped after `ctx`.
    #[allow(dead_code)]
    model: Box<LlamaModel>,
    /// Dropped last; llama.cpp requires it to outlive the model.
    #[allow(dead_code)]
    backend: Box<LlamaBackend>,
}

/// SAFETY: the engine owns raw llama.cpp handles (`LlamaContext`, and the
/// sampler it builds per turn), which are not `Sync` and so make the struct
/// non-`Send` by inference. It IS safe to move between threads:
///
///  - llama.cpp contexts are not bound to the thread that created them; the
///    constraint is that a context must not be used CONCURRENTLY.
///  - every method here takes `&mut self`, so Rust already forbids concurrent
///    use through a shared reference, and a host that wants cross-thread access
///    holds it behind a lock (the desktop keeps it in `Arc<Mutex<Option<_>>>`).
///
/// `Sync` is deliberately NOT implemented: a `Mutex<LlamaCppEngine>` is `Sync`
/// on the strength of `Send` alone, so nothing needs it, and claiming it would
/// assert concurrent-use safety that llama.cpp does not offer.
///
/// This became necessary when the context moved into the struct to keep the KV
/// cache across turns; before that the engine held only handles that were
/// already `Send`.
unsafe impl Send for LlamaCppEngine {}

impl LlamaCppEngine {
    /// Cheap to construct; loads nothing. Check
    /// [`availability`](GenerationEngine::availability) first.
    pub fn new(cfg: LlamaCppConfig) -> Self {
        Self { cfg, loaded: None, chat: None }
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
        let backend = Box::new(LlamaBackend::init().map_err(|e| GenError::Generate(e.to_string()))?);
        let params = LlamaModelParams::default().with_n_gpu_layers(self.cfg.n_gpu_layers);
        let model = Box::new(
            LlamaModel::load_from_file(&*backend, &self.cfg.model_path, &params).map_err(|e| {
                GenError::Generate(format!("load {}: {e}", self.cfg.model_path.display()))
            })?,
        );

        // The template comes from the GGUF itself, so the shim never has to guess
        // which model family it is rendering for.
        let template = model
            .chat_template(None)
            .map_err(|e| GenError::Generate(format!("model has no chat template: {e}")))?
            .to_string()
            .map_err(|e| GenError::Generate(format!("chat template is not utf-8: {e}")))?;
        let chat = ffi::Chat::new(&template)
            .map_err(|e| GenError::Generate(format!("chat shim init: {e}")))?;

        // n_batch caps how many tokens one decode() call may carry, and it
        // defaults far below a real prompt: the Gemma 4 template renders the tool
        // schemas into tens of thousands of characters, which asserts out at
        // `n_tokens_all <= cparams.n_batch`. Size it to the window.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(self.cfg.n_ctx))
            .with_n_batch(self.cfg.n_ctx);
        // SAFETY: `model` and `backend` are boxed, so their addresses are stable
        // and outlive `ctx`; `Loaded` declares `ctx` first so it is dropped first.
        let model_ref: &'static LlamaModel = unsafe { &*(&*model as *const LlamaModel) };
        let backend_ref: &'static LlamaBackend = unsafe { &*(&*backend as *const LlamaBackend) };
        let ctx = model_ref
            .new_context(backend_ref, ctx_params)
            .map_err(|e| GenError::Generate(e.to_string()))?;

        self.loaded = Some(Loaded {
            ctx,
            cached: Vec::new(),
            model,
            backend,
        });
        self.chat = Some(chat);
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
        let applied = self
            .chat
            .as_mut()
            .expect("chat is set with loaded")
            .apply(&req, enable_thinking)
            .map_err(|e| GenError::Generate(format!("chat template: {e}")))?;

        // prompt + generation_prompt. See failure mode 2 in the module docs.
        let prompt = format!("{}{}", applied.prompt, applied.generation_prompt);

        let tokens = loaded
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| GenError::Generate(e.to_string()))?;
        if tokens.len() >= n_ctx as usize {
            return Err(GenError::ContextWindowExceeded);
        }

        // --- prefix reuse -----------------------------------------------------
        // A thread replays an identical prefix every turn (system prompt, tool
        // schemas, history), so only the divergent tail needs decoding. Trim the
        // cache at the first differing token and prefill from there.
        //
        // `reuse` is capped one below the common length: llama.cpp needs at least
        // one token to decode in order to produce logits to sample from, so
        // reusing the ENTIRE prompt would leave nothing to run.
        let common = loaded
            .cached
            .iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let reuse = common.min(tokens.len().saturating_sub(1));
        if reuse < loaded.cached.len() {
            loaded
                .ctx
                .kv_cache_seq_rm(0, Some(reuse as u32), None)
                .map_err(|e| GenError::Generate(format!("kv trim: {e}")))?;
        }
        let fresh = &tokens[reuse..];

        let mut batch = LlamaBatch::new(fresh.len().max(512), 1);
        let last = fresh.len() - 1;
        for (i, t) in fresh.iter().enumerate() {
            batch
                .add(*t, (reuse + i) as i32, &[0], i == last)
                .map_err(|e| GenError::Generate(e.to_string()))?;
        }

        let t_prefill = Instant::now();
        loaded.ctx.decode(&mut batch)
            .map_err(|e| GenError::Generate(e.to_string()))?;
        let prefill_ms = t_prefill.elapsed().as_millis() as u64;

        // --- decode ---------------------------------------------------------
        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let mut n_cur = tokens.len() as i32;
        // What the cache holds once this turn's prompt is in. Generated tokens are
        // appended below so the next turn's common-prefix scan sees them too.
        let mut cached = tokens.clone();
        let mut raw = String::new();
        let mut output_tokens = 0usize;
        let mut truncated = true;

        let t_decode = Instant::now();
        while output_tokens < MAX_OUTPUT_TOKENS {
            let token = sampler.sample(&loaded.ctx, batch.n_tokens() - 1);
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
            cached.push(token);
            loaded.ctx
                .decode(&mut batch)
                .map_err(|e| GenError::Generate(e.to_string()))?;
        }
        loaded.cached = cached;
        let decode_ms = t_decode.elapsed().as_millis() as u64;

        // --- parse ----------------------------------------------------------
        let parsed = self
            .chat
            .as_mut()
            .expect("chat is set with loaded")
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
