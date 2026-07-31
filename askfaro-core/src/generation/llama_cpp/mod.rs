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
use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeParams};

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
    /// Optional MTP drafter for speculative decoding.
    ///
    /// Worth **+30% decode** on E4B (29.8 to 38.8 tok/s) for 77 MB, and lossless:
    /// verified tokens are exactly what greedy would have produced, which was
    /// confirmed byte-for-byte across all 20 bench cases. `None` runs the plain
    /// decode loop.
    pub draft_path: Option<PathBuf>,
    /// Draft tokens to propose per step. **3 is a measured optimum, not a
    /// default to tune away from**: acceptance falls as the window grows and
    /// rejected drafts still cost a verify pass, so 5 lands at 33.3 tok/s and 8
    /// at 24.3, the latter BELOW not speculating at all (34.0).
    pub draft_n_max: i32,
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
            draft_path: None,
            draft_n_max: 3,
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
    /// The decoder: either a plain context, or an MTP speculative wrapper that
    /// OWNS the target context (and the drafter's).
    ///
    /// One enum rather than two `Option`s so "both set" and "neither set" are
    /// unrepresentable; the speculative path takes the context by value, so they
    /// genuinely cannot coexist.
    dec: Decoder,
    /// The tokens currently in the KV cache, so the next turn decodes only what
    /// diverges from them.
    cached: Vec<llama_cpp_2::token::LlamaToken>,
    /// Boxed for a stable address the drafter's context borrows. Dropped after
    /// `dec`, which holds that context.
    #[allow(dead_code)]
    draft_model: Option<Box<LlamaModel>>,
    /// Boxed for a stable address that the context can borrow. Dropped after `dec`.
    #[allow(dead_code)]
    model: Box<LlamaModel>,
    /// Dropped last; llama.cpp requires it to outlive the models.
    #[allow(dead_code)]
    backend: Box<LlamaBackend>,
}

/// Plain decode, or MTP speculative decode.
enum Decoder {
    /// Lives across turns, and so does its KV cache.
    ///
    /// Rebuilding it per turn is what the first version did, and it cost **16.8s
    /// of prefill on every turn** in the 20-case parity bench: a thread replays
    /// an identical multi-thousand-token prefix each turn and a fresh context
    /// re-reads all of it. Phase 0 measured prefix reuse at 165x, so discarding
    /// the cache is the most expensive mistake available here.
    ///
    /// The `'static` is a lifetime extension over `model`, sound because `model`
    /// is boxed (a stable address, never moved) and this is dropped first.
    Plain(LlamaContext<'static>),
    /// Owns the target context, so all decode goes through
    /// `target_context_mut()`.
    Mtp(Box<MtpSpeculative<'static>>),
}

impl Decoder {
    fn ctx(&mut self) -> &mut LlamaContext<'static> {
        match self {
            Decoder::Plain(c) => c,
            Decoder::Mtp(s) => s.target_context_mut(),
        }
    }
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

    /// Load the weights now, returning the load time in ms (0 if already warm).
    ///
    /// Generation loads lazily, which is right for a process that may never
    /// generate, but it makes the FIRST turn pay the whole cold load: measured
    /// 12.5s to 26s for E2B and ~60s for E4B's 5.15 GB. A host that knows a turn
    /// is coming (an app that just opened) can pay that in the background
    /// instead of in front of a waiting user.
    ///
    /// Idempotent, so a host may call it on every foreground without checking.
    pub fn warm(&mut self) -> Result<u64, GenError> {
        self.ensure_loaded()
    }

    /// Drop the weights, freeing their memory; the engine stays usable and will
    /// reload lazily on the next generate.
    ///
    /// The counterpart to `warm`. Holding E4B resident costs 5.15 GB, which is a
    /// third of a 16 GB machine, so a host that has been idle for a while wants
    /// that back. Returns whether anything was actually released.
    pub fn unload(&mut self) -> bool {
        let was = self.loaded.is_some();
        // `chat` is derived from the model's template, so it must go too or a
        // reload would keep a template belonging to weights that are gone.
        self.loaded = None;
        self.chat = None;
        was
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

    /// Load the MTP drafter and give it its own context.
    ///
    /// Returns the drafter's context plus the boxed model that context borrows;
    /// the caller must keep the model alive for at least as long as the context,
    /// which `Loaded`'s field order enforces.
    /// The drafter's context must be BOUND to the target's, not built standalone.
    ///
    /// Gemma 4's MTP head is not an independent model that happens to be small;
    /// it reads the target's backbone hidden states, so llama.cpp refuses a
    /// plain context for it: "Gemma4Assistant requires ctx_other to be set".
    /// That is why this takes the target context and uses
    /// `new_context_with_ctx_other`, and why the target must exist first.
    fn load_drafter(
        backend: &LlamaBackend,
        backend_ref: &'static LlamaBackend,
        ctx_params: LlamaContextParams,
        path: &Path,
        target: &LlamaContext<'_>,
    ) -> Result<(LlamaContext<'static>, Box<LlamaModel>), GenError> {
        // The drafter is tiny (77 MB); offload it fully, same as the target.
        let params = LlamaModelParams::default().with_n_gpu_layers(1000);
        let dm = Box::new(
            LlamaModel::load_from_file(backend, path, &params)
                .map_err(|e| GenError::Generate(format!("load drafter {}: {e}", path.display())))?,
        );
        // SAFETY: identical to the target model above -- boxed, so its address is
        // stable, and `Loaded` drops the context before the model.
        let dm_ref: &'static LlamaModel = unsafe { &*(&*dm as *const LlamaModel) };
        let ctx = dm_ref
            .new_context_with_ctx_other(backend_ref, ctx_params, target)
            .map_err(|e| GenError::Generate(e.to_string()))?;
        Ok((ctx, dm))
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
            .new_context(backend_ref, ctx_params.clone())
            .map_err(|e| GenError::Generate(e.to_string()))?;

        // The drafter, when one is configured. Failing to load it is NOT fatal:
        // speculation is a pure speed optimisation and a turn without it is
        // correct, just slower. Degrading to plain decode beats refusing to
        // answer because an 77 MB accessory is missing or mismatched.
        let (dec, draft_model) = match &self.cfg.draft_path {
            Some(p) if p.exists() => {
                match Self::load_drafter(&*backend, backend_ref, ctx_params, p, &ctx) {
                    Ok((spec_ctx, dm)) => match MtpSpeculative::new(
                        ctx,
                        spec_ctx,
                        MtpSpeculativeParams {
                            n_max: self.cfg.draft_n_max,
                            ..Default::default()
                        },
                    ) {
                        Ok(s) => (Decoder::Mtp(Box::new(s)), Some(dm)),
                        Err(e) => {
                            // `MtpSpeculative::new` consumed both contexts, so
                            // there is nothing to fall back WITH; rebuild one.
                            eprintln!("llama_cpp: MTP init failed ({e}); plain decode");
                            let c = model_ref
                                .new_context(backend_ref, LlamaContextParams::default()
                                    .with_n_ctx(std::num::NonZeroU32::new(self.cfg.n_ctx))
                                    .with_n_batch(self.cfg.n_ctx))
                                .map_err(|e| GenError::Generate(e.to_string()))?;
                            (Decoder::Plain(c), None)
                        }
                    },
                    Err(e) => {
                        eprintln!("llama_cpp: drafter load failed ({e}); plain decode");
                        (Decoder::Plain(ctx), None)
                    }
                }
            }
            Some(p) => {
                eprintln!("llama_cpp: drafter not found at {}; plain decode", p.display());
                (Decoder::Plain(ctx), None)
            }
            None => (Decoder::Plain(ctx), None),
        };

        self.loaded = Some(Loaded {
            dec,
            cached: Vec::new(),
            draft_model,
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
                .dec
                .ctx()
                .kv_cache_seq_rm(0, Some(reuse as u32), None)
                .map_err(|e| GenError::Generate(format!("kv trim: {e}")))?;
        }
        let fresh = &tokens[reuse..];

        // The speculative path needs the whole prompt announced up front, before
        // any decode, so the drafter's own context is primed from the same text.
        if let Decoder::Mtp(spec) = &mut loaded.dec {
            spec.begin(&tokens)
                .map_err(|e| GenError::Generate(format!("mtp begin: {e}")))?;
        }

        let mut batch = LlamaBatch::new(fresh.len().max(512), 1);
        let last = fresh.len() - 1;
        for (i, t) in fresh.iter().enumerate() {
            batch
                .add(*t, (reuse + i) as i32, &[0], i == last)
                .map_err(|e| GenError::Generate(e.to_string()))?;
        }

        let t_prefill = Instant::now();
        loaded
            .dec
            .ctx()
            .decode(&mut batch)
            .map_err(|e| GenError::Generate(e.to_string()))?;
        if let Decoder::Mtp(spec) = &mut loaded.dec {
            spec.process(&batch)
                .map_err(|e| GenError::Generate(format!("mtp process: {e}")))?;
        }
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
        let mut draft_proposed = 0usize;
        let mut draft_accepted = 0usize;

        let t_decode = Instant::now();
        // `next` is the token that has been decided but not yet written into the
        // KV cache. Both loops below share this invariant.
        let mut next = sampler.sample(loaded.dec.ctx(), batch.n_tokens() - 1);

        'gen: while output_tokens < MAX_OUTPUT_TOKENS {
            sampler.accept(next);
            // The end-of-turn token stays in the text: the parser's grammar
            // matches a complete turn. See failure mode 3 in the module docs.
            raw.push_str(
                &loaded
                    .model
                    .token_to_str(next, Special::Tokenize)
                    .map_err(|e| GenError::Generate(e.to_string()))?,
            );
            output_tokens += 1;
            if loaded.model.is_eog_token(next) {
                truncated = false;
                break;
            }

            match &mut loaded.dec {
                // --- plain: one token in, one token out ---------------------
                Decoder::Plain(ctx) => {
                    batch.clear();
                    batch
                        .add(next, n_cur, &[0], true)
                        .map_err(|e| GenError::Generate(e.to_string()))?;
                    cached.push(next);
                    n_cur += 1;
                    ctx.decode(&mut batch)
                        .map_err(|e| GenError::Generate(e.to_string()))?;
                    next = sampler.sample(ctx, batch.n_tokens() - 1);
                }
                // --- speculative: propose k, verify all k in ONE pass --------
                //
                // This is why speculation pays at all: verifying k proposals
                // costs about the same as generating one token, because decode
                // is bound by streaming the weights, not by the arithmetic. The
                // sampled result is IDENTICAL to plain greedy decode by
                // construction, since every draft is only kept when it matches
                // what the target model would itself have produced.
                Decoder::Mtp(spec) => {
                    let drafts = spec
                        .draft(n_cur, next, &cached)
                        .map_err(|e| GenError::Generate(format!("mtp draft: {e}")))?;
                    draft_proposed += drafts.len();

                    // `next` at n_cur, then each proposal after it. Every
                    // position needs logits, because each one is a verification
                    // point rather than just context.
                    batch.clear();
                    batch
                        .add(next, n_cur, &[0], true)
                        .map_err(|e| GenError::Generate(e.to_string()))?;
                    for (i, d) in drafts.iter().enumerate() {
                        batch
                            .add(*d, n_cur + 1 + i as i32, &[0], true)
                            .map_err(|e| GenError::Generate(e.to_string()))?;
                    }
                    cached.push(next);
                    let pos_of_next = n_cur;

                    spec.target_context_mut()
                        .decode(&mut batch)
                        .map_err(|e| GenError::Generate(e.to_string()))?;
                    spec.process(&batch)
                        .map_err(|e| GenError::Generate(format!("mtp process: {e}")))?;

                    // Logits at batch index j predict the token AFTER the token
                    // at index j. So index 0 (which holds `next`) is checked
                    // against the first proposal, and so on down the chain.
                    let mut accepted = 0usize;
                    let mut correction = None;
                    for (i, d) in drafts.iter().enumerate() {
                        let t = sampler.sample(spec.target_context_mut(), i as i32);
                        if t == *d {
                            accepted += 1;
                        } else {
                            correction = Some(t);
                            break;
                        }
                    }
                    // All k matched, so index k yields a free extra token: the
                    // verify pass already computed it.
                    let follow = match correction {
                        Some(t) => t,
                        None => sampler.sample(spec.target_context_mut(), drafts.len() as i32),
                    };
                    draft_accepted += accepted;

                    spec.accept(accepted as u16)
                        .map_err(|e| GenError::Generate(format!("mtp accept: {e}")))?;

                    // Emit what was accepted. These are committed tokens, so
                    // they must run the same EOG and cap checks as any other.
                    for d in drafts.iter().take(accepted) {
                        sampler.accept(*d);
                        raw.push_str(
                            &loaded
                                .model
                                .token_to_str(*d, Special::Tokenize)
                                .map_err(|e| GenError::Generate(e.to_string()))?,
                        );
                        output_tokens += 1;
                        cached.push(*d);
                        if loaded.model.is_eog_token(*d) {
                            truncated = false;
                            n_cur = pos_of_next + 1 + accepted as i32;
                            break 'gen;
                        }
                        if output_tokens >= MAX_OUTPUT_TOKENS {
                            n_cur = pos_of_next + 1 + accepted as i32;
                            break 'gen;
                        }
                    }

                    // Rejected proposals are still sitting in the KV cache, so
                    // drop them: keeping them would corrupt every later position.
                    // A no-op when everything was accepted.
                    n_cur = pos_of_next + 1 + accepted as i32;
                    spec.target_context_mut()
                        .kv_cache_seq_rm(0, Some(n_cur as u32), None)
                        .map_err(|e| GenError::Generate(format!("kv rollback: {e}")))?;
                    next = follow;
                }
            }
        }
        loaded.cached = cached;
        let decode_ms = t_decode.elapsed().as_millis() as u64;
        if draft_proposed > 0 {
            eprintln!(
                "llama_cpp: mtp accepted {draft_accepted}/{draft_proposed} drafted tokens ({:.0}%)",
                draft_accepted as f64 / draft_proposed as f64 * 100.0
            );
        }

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
