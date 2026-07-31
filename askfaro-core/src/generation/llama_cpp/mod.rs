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
    Availability, GenError, GenerateRequest, GenerateResponse, GenerationEngine, Msg, Timings,
    ToolCall,
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
    /// How many KV cache slots the context supports.
    ///
    /// Must cover the highest `GenerateRequest::slot` a host will use. A context
    /// built with the default of ONE sequence rejects slot 1 outright, and the
    /// symptom is not a clear error about sequences: llama.cpp reports "failed
    /// to initialize batch" and then `n_tokens == 0`.
    pub n_slots: u32,
    /// Directory to persist slot 0's KV prefix in, or `None` to never persist.
    ///
    /// Prefix reuse already makes the SECOND turn of a session cheap; this makes
    /// the FIRST one cheap too, by writing the computed prefix to disk once and
    /// restoring it on every later load. The measured cost it removes is the
    /// whole cold prefill: ~27s at ~280 tok/s over a ~7,700-token prompt, paid
    /// on the first turn of every launch.
    ///
    /// Costs disk proportional to the cached prefix. The size is logged and
    /// reported through [`PrefixReport::bytes`], because it is the one price of
    /// this that nothing else would surface.
    pub prefix_cache_dir: Option<PathBuf>,
    /// Host-supplied identity folded into the persisted prefix's file name.
    ///
    /// A saved state is valid ONLY for the exact weights it was computed with,
    /// and llama.cpp validates SHAPE, not identity: two GGUFs of the same
    /// architecture at different quantizations deserialize into each other
    /// without complaint and answer plausibly rather than erroring. This crate
    /// keys on what it can see (path, file size, context params); a host that
    /// knows more, the sha256 it verified at download time or a build id, puts
    /// it here and the file name changes with it.
    pub state_key: String,
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
            // Two: the agent loop, plus background one-shots.
            n_slots: 2,
            // Off unless a host names a directory: persisting state is a policy
            // decision (where it lives, when it is cleared) and this crate has
            // no business inventing a cache location.
            prefix_cache_dir: None,
            state_key: String::new(),
        }
    }
}

/// What [`LlamaCppEngine::ensure_prefix`] found or did.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PrefixReport {
    /// Tokens in the persisted prefix.
    pub tokens: u32,
    /// Size of the state file on disk. The whole cost of this feature.
    pub bytes: u64,
    /// True when this call recomputed and rewrote the file.
    pub rebuilt: bool,
    /// Wall clock spent rebuilding; 0 when the existing file was already good.
    pub ms: u64,
    /// Where it lives, so a host can report or delete it.
    pub path: String,
}

/// Where slot 0's prefix stands relative to the file on disk.
///
/// Three states rather than a bool because "restored but not yet checked
/// against a real prompt" is genuinely different from "known good": a restore
/// is only trustworthy once some prompt has been shown to start with it, and
/// that check cannot happen until a turn arrives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixState {
    /// No `prefix_cache_dir`, so none of this runs.
    Disabled,
    /// Enabled, but nothing was restored: no file yet, or it failed to load.
    Missing,
    /// `n` tokens came off disk into sequence 0 and have not been validated.
    Restored(usize),
    /// Built or validated this session; nothing further to do.
    Live,
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
    /// The tokens currently in each KV cache SLOT, so the next turn on that slot
    /// decodes only what diverges from them.
    ///
    /// Per slot, because one cache shared by two workloads is two workloads that
    /// are always cold: the desktop's agent loop (~6,000 tokens) and its
    /// background one-shots (~340) evicted each other on every call. Slots map
    /// to llama.cpp sequence ids.
    cached: std::collections::HashMap<u32, Vec<llama_cpp_2::token::LlamaToken>>,
    /// Whether slot 0's cache came off disk, and whether that has been checked.
    prefix: PrefixState,
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
    /// 12.5s to 26s for E2B and ~37s for E4B's 5.15 GB. A host that knows a turn
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
    ///
    /// Cheap to reverse, which is what makes releasing reasonable: a re-`warm`
    /// measured **~3-5s** against the 37s cold load, because the weights are
    /// mmapped and the OS page cache survives the unload. Do not assume the cold
    /// number applies to a reload.
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
        // `n_ctx` is the TOTAL across sequences: llama.cpp gives each sequence
        // `n_ctx / n_seq_max`. So asking for two slots at face value would have
        // silently HALVED the caller's window, from the profile's 16,384 to
        // 8,192, and the first symptom would be a mid-conversation
        // `NoKvCacheSlot` rather than anything naming the window.
        //
        // `n_ctx` here is a property of the MODEL, so it is honoured per slot and
        // the total is scaled instead. The cost is KV memory, which scales with
        // the number of slots; that is the real price of running two workloads
        // on one engine, and it is worth naming rather than discovering.
        let slots = self.cfg.n_slots.max(1);
        let total_ctx = self.cfg.n_ctx.saturating_mul(slots);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(total_ctx))
            .with_n_batch(self.cfg.n_ctx)
            .with_n_seq_max(slots);
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
                                    .with_n_ctx(std::num::NonZeroU32::new(total_ctx))
                                    .with_n_batch(self.cfg.n_ctx)
                                    .with_n_seq_max(slots))
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

        // A persisted prefix, if one was written by an earlier session. This is
        // the whole point of the feature: a fresh context normally starts empty
        // and the first turn pays the full prefill.
        //
        // Restoring is deliberately best-effort. Every way this can go wrong,
        // absent file, older llama.cpp state format, a shape that does not match
        // this context, comes back as a clean `Err` from llama.cpp (it catches
        // its own deserializer's exceptions and removes the half-written
        // sequence), and the correct response to all of them is the same: carry
        // on cold. A cold start is slow, not wrong.
        let mut dec = dec;
        let mut cached = std::collections::HashMap::new();
        let mut prefix = PrefixState::Disabled;
        if let Some(path) = Self::prefix_path_for(&self.cfg) {
            prefix = PrefixState::Missing;
            if path.exists() {
                let t_restore = Instant::now();
                match dec
                    .ctx()
                    .state_seq_load_file(&path, 0, self.cfg.n_ctx as usize)
                {
                    Ok((toks, bytes)) => {
                        eprintln!(
                            "llama_cpp: restored a {}-token KV prefix from {} ({:.1} MB) in {}ms",
                            toks.len(),
                            path.display(),
                            bytes as f64 / 1e6,
                            t_restore.elapsed().as_millis(),
                        );
                        prefix = PrefixState::Restored(toks.len());
                        cached.insert(0, toks);
                    }
                    Err(e) => eprintln!(
                        "llama_cpp: could not restore the KV prefix from {} ({e}); \
                         this launch pays a cold prefill and will rewrite it",
                        path.display()
                    ),
                }
            }
        }

        self.loaded = Some(Loaded {
            dec,
            cached,
            prefix,
            draft_model,
            model,
            backend,
        });
        self.chat = Some(chat);
        Ok(t.elapsed().as_millis() as u64)
    }

    /// Where this configuration's persisted prefix lives, if enabled.
    fn prefix_path_for(cfg: &LlamaCppConfig) -> Option<PathBuf> {
        let dir = cfg.prefix_cache_dir.as_ref()?;
        Some(dir.join(format!("prefix-{}.kv", Self::prefix_key_for(cfg))))
    }

    /// The invalidation key, as a file name.
    ///
    /// **A stale restore is not an error, it is wrong output.** llama.cpp's
    /// deserializer validates the state's SHAPE (layer count, per-layer k/v
    /// sizes, quantization type) and nothing about which weights produced it, so
    /// a state file from a different fine-tune of the same architecture loads
    /// happily and answers plausible nonsense. Everything the state depends on
    /// therefore goes in the name, and a file whose name does not match is never
    /// opened rather than being opened and checked.
    ///
    /// The prompt itself is deliberately NOT in here, because it cannot be:
    /// tokenizing a prefix needs the model loaded, and the name has to be known
    /// before that. The prompt is checked directly instead: `state_seq_load_file`
    /// hands back the exact tokens the state was saved with, and the first turn
    /// asserts the real prompt starts with them (see `settle_prefix`), which is
    /// strictly stronger than comparing hashes.
    fn prefix_key_for(cfg: &LlamaCppConfig) -> String {
        use sha2::{Digest, Sha256};
        let len = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        let mut h = Sha256::new();
        h.update(b"askfaro-kv-prefix-v1\0");
        h.update(cfg.state_key.as_bytes());
        h.update(b"\0");
        h.update(cfg.model_path.as_os_str().as_encoded_bytes());
        h.update(len(&cfg.model_path).to_le_bytes());
        match &cfg.draft_path {
            // The drafter shares the target's KV memory (`is_mem_shared`, which
            // is why Gemma 4's MTP head needs `ctx_other`), so attaching one
            // changes what is in the cache. Not a cosmetic part of the key.
            Some(p) => {
                h.update(p.as_os_str().as_encoded_bytes());
                h.update(len(p).to_le_bytes());
            }
            None => h.update(b"no-drafter"),
        }
        h.update(cfg.n_ctx.to_le_bytes());
        h.update(cfg.n_slots.to_le_bytes());
        h.update(cfg.n_gpu_layers.to_le_bytes());
        h.update([u8::from(cfg.enable_thinking)]);
        hex::encode(&h.finalize()[..8])
    }

    /// Tokenize what the template renders for `req` plus one probe user turn.
    fn render_tokens(
        &mut self,
        req: &GenerateRequest,
        probe: &str,
    ) -> Result<Vec<llama_cpp_2::token::LlamaToken>, GenError> {
        let mut r = req.clone();
        r.messages.clear();
        r.messages.push(Msg {
            role: "user".into(),
            content: probe.into(),
        });
        let enable_thinking = self.cfg.enable_thinking;
        let applied = self
            .chat
            .as_mut()
            .expect("chat is set with loaded")
            .apply(&r, enable_thinking)
            .map_err(|e| GenError::Generate(format!("chat template: {e}")))?;
        let prompt = format!("{}{}", applied.prompt, applied.generation_prompt);
        self.loaded
            .as_ref()
            .expect("loaded")
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| GenError::Generate(e.to_string()))
    }

    /// The largest token prefix that every conversation using `req`'s system
    /// block and tool set must begin with.
    ///
    /// Found by DIFFING two renders that differ only in the user's first
    /// character, rather than by rendering the system block alone and assuming
    /// it is a prefix of the whole. That assumption is exactly the kind that
    /// holds for one model family and breaks silently for the next: several
    /// chat templates fold the system message INTO the first user turn, so a
    /// system-only render is not a prefix of anything. Two probes make the
    /// boundary an observation instead.
    fn stable_prefix(
        &mut self,
        req: &GenerateRequest,
    ) -> Result<Vec<llama_cpp_2::token::LlamaToken>, GenError> {
        let a = self.render_tokens(req, "a")?;
        let b = self.render_tokens(req, "b")?;
        let n = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
        if n == 0 {
            return Err(GenError::Generate(
                "two renders of the same system block share no tokens; the template is not \
                 deterministic and no prefix can be cached"
                    .into(),
            ));
        }
        Ok(a[..n].to_vec())
    }

    /// Compute, prefill and persist the prefix for `req`, replacing whatever is
    /// on disk.
    ///
    /// `tokens`, when given, is a real prompt this prefix must be a prefix OF.
    /// Checked before anything is written, because a prefix the live prompt does
    /// not start with is worse than none: it would be restored next launch, fail
    /// the match, and force a full re-prefill on top of the restore.
    fn build_prefix(
        &mut self,
        req: &GenerateRequest,
        tokens: Option<&[llama_cpp_2::token::LlamaToken]>,
        path: &Path,
    ) -> Result<PrefixReport, GenError> {
        let t = Instant::now();
        let prefix = self.stable_prefix(req)?;
        if prefix.len() >= self.cfg.n_ctx as usize {
            return Err(GenError::ContextWindowExceeded);
        }
        if let Some(tokens) = tokens {
            if !tokens.starts_with(&prefix) {
                eprintln!(
                    "llama_cpp: the {}-token prefix computed from this request is NOT a prefix of \
                     the {}-token prompt it came from, so nothing was persisted. The template \
                     renders the first user turn differently depending on what follows it.",
                    prefix.len(),
                    tokens.len(),
                );
                // Sequence 0 still holds whatever was restored, and this is the
                // one path that reaches the normal reuse scan with a cache it
                // could not vouch for. Cleared, so that scan finds nothing and
                // prefills from scratch, rather than trimming a restored state
                // back to a common point the sliding-window layers cannot cover.
                let loaded = self.loaded.as_mut().expect("loaded");
                loaded
                    .dec
                    .ctx()
                    .kv_cache_seq_rm(0, None, None)
                    .map_err(|e| GenError::Generate(format!("kv clear: {e}")))?;
                loaded.cached.remove(&0);
                // Live, not Missing: retrying this every turn would recompute a
                // prefix already known not to fit, once per message, forever.
                loaded.prefix = PrefixState::Live;
                return Ok(PrefixReport {
                    path: path.display().to_string(),
                    ..Default::default()
                });
            }
        }

        let loaded = self.loaded.as_mut().expect("loaded");
        // Sequence 0 must contain the prefix and NOTHING else: the state file
        // records every cell in the sequence, and the token list beside it is
        // only metadata. Saving a longer sequence under a shorter token list is
        // how a restore comes back with cells nothing accounts for.
        loaded
            .dec
            .ctx()
            .kv_cache_seq_rm(0, None, None)
            .map_err(|e| GenError::Generate(format!("kv clear: {e}")))?;
        loaded.cached.remove(&0);

        let mut batch = LlamaBatch::new(prefix.len().max(512), 1);
        let last = prefix.len() - 1;
        for (i, tok) in prefix.iter().enumerate() {
            batch
                .add(*tok, i as i32, &[0], i == last)
                .map_err(|e| GenError::Generate(e.to_string()))?;
        }
        // Mirrors a real turn's prefill exactly, drafter bookkeeping included.
        // Gemma 4's MTP head shares the target's KV memory rather than keeping
        // its own, which is what makes persisting the target's sequence enough
        // for speculation to work on the restored turn as well.
        if let Decoder::Mtp(spec) = &mut loaded.dec {
            spec.begin(&prefix)
                .map_err(|e| GenError::Generate(format!("mtp begin: {e}")))?;
        }
        loaded
            .dec
            .ctx()
            .decode(&mut batch)
            .map_err(|e| GenError::Generate(e.to_string()))?;
        if let Decoder::Mtp(spec) = &mut loaded.dec {
            spec.process(&batch)
                .map_err(|e| GenError::Generate(format!("mtp process: {e}")))?;
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| GenError::Generate(format!("prefix cache dir: {e}")))?;
        }
        // Written to a temp name and renamed, so a process killed mid-write
        // leaves the previous good file rather than a truncated one that reads
        // as present.
        let tmp = path.with_extension("kv.tmp");
        let bytes = loaded
            .dec
            .ctx()
            .state_seq_save_file(&tmp, 0, &prefix)
            .map_err(|e| GenError::Generate(format!("kv state save: {e}")))?;
        std::fs::rename(&tmp, path)
            .map_err(|e| GenError::Generate(format!("prefix cache rename: {e}")))?;
        loaded.cached.insert(0, prefix.clone());
        // Trusted only when this was checked against a real prompt.
        //
        // A host-initiated build (`ensure_prefix`, with no prompt to check
        // against) is a GUESS at what the app will send: it assembles the
        // system block and tool set from a second place in the code, and if
        // that drifts by one character the prefix is wrong. Left as
        // `Restored`, so the first real turn validates it exactly as it
        // validates one read off disk, and discards it whole rather than
        // trimming into a sliding window it cannot cover.
        loaded.prefix = match tokens {
            Some(_) => PrefixState::Live,
            None => PrefixState::Restored(prefix.len()),
        };

        // Anything under a different key is dead by construction: only one
        // configuration is live per install, and a model upgrade would otherwise
        // leave its predecessor's state behind forever.
        Self::sweep_stale_prefixes(path);

        let ms = t.elapsed().as_millis() as u64;
        eprintln!(
            "llama_cpp: persisted a {}-token KV prefix to {} ({:.1} MB) in {ms}ms",
            prefix.len(),
            path.display(),
            bytes as f64 / 1e6,
        );
        Ok(PrefixReport {
            tokens: prefix.len() as u32,
            bytes: bytes as u64,
            rebuilt: true,
            ms,
            path: path.display().to_string(),
        })
    }

    /// Delete every `prefix-*.kv` beside `keep` that is not `keep`.
    fn sweep_stale_prefixes(keep: &Path) {
        let Some(dir) = keep.parent() else { return };
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p == keep {
                continue;
            }
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("prefix-") && (name.ends_with(".kv") || name.ends_with(".kv.tmp")) {
                let _ = std::fs::remove_file(&p);
            }
        }
    }

    /// Decide what to do with slot 0's prefix before a turn is prefilled.
    ///
    /// Returns the milliseconds spent, which belong to prefill: a turn that had
    /// to build the prefix did the same work a turn without one would have done,
    /// only in a form the next launch can reuse.
    fn settle_prefix(
        &mut self,
        req: &GenerateRequest,
        tokens: &[llama_cpp_2::token::LlamaToken],
    ) -> Result<u64, GenError> {
        let Some(path) = Self::prefix_path_for(&self.cfg) else {
            return Ok(0);
        };
        let state = self.loaded.as_ref().expect("loaded").prefix;
        match state {
            PrefixState::Disabled | PrefixState::Live => Ok(0),
            PrefixState::Missing => Ok(self.build_prefix(req, Some(tokens), &path)?.ms),
            PrefixState::Restored(n) => {
                let matches = {
                    let loaded = self.loaded.as_ref().expect("loaded");
                    let prior = loaded.cached.get(&0).map_or(&[][..], Vec::as_slice);
                    prior.len() == n && tokens.len() > n && tokens[..n] == prior[..n]
                };
                if matches {
                    self.loaded.as_mut().expect("loaded").prefix = PrefixState::Live;
                    return Ok(0);
                }
                // The prompt moved under the saved state: a changed system
                // block, a changed tool set, a different model profile.
                //
                // Dropped WHOLE and rebuilt, never trimmed back to the common
                // point, which is the one thing the ordinary turn-to-turn path
                // does here and must not do with a restored state. Gemma 4
                // attends over a sliding window, so the SWA half of the cache
                // holds only the positions immediately before where the save
                // ended. Trimming to an earlier point leaves those layers
                // attending over cells that were never restored, and the result
                // of that is plausible text, not an error.
                eprintln!(
                    "llama_cpp: the persisted prefix no longer matches this prompt ({n} tokens \
                     restored); discarding it and rebuilding"
                );
                Ok(self.build_prefix(req, Some(tokens), &path)?.ms)
            }
        }
    }

    /// Make sure the persisted prefix for `req` exists and is current.
    ///
    /// The engine does this by itself on the first turn, which is where it costs
    /// nothing: that turn was going to prefill the whole prompt anyway, so
    /// splitting the prefill in two and saving the first half is free. This is
    /// for a host that would rather pay it somewhere the user is already
    /// waiting, straight after the weights download say, than on whichever
    /// message happens to come first.
    ///
    /// `req` supplies the system block and the tool set; its `messages` are
    /// ignored, because a prefix by definition ends before the conversation.
    pub fn ensure_prefix(&mut self, req: &GenerateRequest) -> Result<PrefixReport, GenError> {
        let Some(path) = Self::prefix_path_for(&self.cfg) else {
            return Err(GenError::Invalid(
                "no prefix_cache_dir configured, so there is nowhere to persist a prefix".into(),
            ));
        };
        self.ensure_loaded()?;
        let prefix = self.stable_prefix(req)?;
        let current = {
            let loaded = self.loaded.as_ref().expect("loaded");
            path.exists()
                && loaded
                    .cached
                    .get(&0)
                    .is_some_and(|c| c.starts_with(&prefix))
        };
        if current {
            // Deliberately does NOT promote to `Live`. Agreeing with the file on
            // disk is not validation when this call is the thing that WROTE that
            // file: a host whose prompt assembly has drifted produces a prefix
            // that matches its own last one perfectly and matches the app's real
            // turn not at all. Whatever state the restore left stands, so the
            // first real prompt is still the arbiter.
            return Ok(PrefixReport {
                tokens: prefix.len() as u32,
                bytes: std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
                rebuilt: false,
                ms: 0,
                path: path.display().to_string(),
            });
        }
        self.build_prefix(req, None, &path)
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

        // --- render the prompt through the model's own template -------------
        let applied = self
            .chat
            .as_mut()
            .expect("chat is set with loaded")
            .apply(&req, enable_thinking)
            .map_err(|e| GenError::Generate(format!("chat template: {e}")))?;

        // prompt + generation_prompt. See failure mode 2 in the module docs.
        let prompt = format!("{}{}", applied.prompt, applied.generation_prompt);

        let tokens = self
            .loaded
            .as_ref()
            .expect("just loaded")
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| GenError::Generate(e.to_string()))?;
        if tokens.len() >= n_ctx as usize {
            return Err(GenError::ContextWindowExceeded);
        }

        // --- persisted prefix -------------------------------------------------
        // Ahead of the reuse scan below, because it decides what the scan will
        // find: it either validates a prefix restored from disk, or builds and
        // saves one out of THIS prompt. Building costs nothing extra, because the
        // tokens it prefills are tokens this turn was about to prefill anyway,
        // just split so the stable half can be written down.
        //
        // Slot 0 only. Other slots carry a host's background one-shots, which
        // are short, varied and have no prefix worth a file.
        let prefix_ms = if req.slot == 0 {
            self.settle_prefix(&req, &tokens)?
        } else {
            0
        };
        let loaded = self.loaded.as_mut().expect("just loaded");

        // --- prefix reuse -----------------------------------------------------
        // A thread replays an identical prefix every turn (system prompt, tool
        // schemas, history), so only the divergent tail needs decoding. Trim the
        // cache at the first differing token and prefill from there.
        //
        // `reuse` is capped one below the common length: llama.cpp needs at least
        // one token to decode in order to produce logits to sample from, so
        // reusing the ENTIRE prompt would leave nothing to run.
        // Slot 0 speculates; any other slot is a plain decode on its own
        // sequence. `MtpSpeculative` binds to sequence 0, and a background
        // one-shot has nothing to gain from a drafter in any case.
        let slot = req.slot;
        let speculative = slot == 0 && matches!(loaded.dec, Decoder::Mtp(_));
        let prior = loaded.cached.entry(slot).or_default();
        let common = prior
            .iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let reuse = common.min(tokens.len().saturating_sub(1));
        let prior_len = prior.len();
        if reuse < prior_len {
            loaded
                .dec
                .ctx()
                .kv_cache_seq_rm(slot as i32, Some(reuse as u32), None)
                .map_err(|e| GenError::Generate(format!("kv trim: {e}")))?;
        }
        let fresh = &tokens[reuse..];

        // The speculative path needs the whole prompt announced up front, before
        // any decode, so the drafter's own context is primed from the same text.
        if speculative {
            if let Decoder::Mtp(spec) = &mut loaded.dec {
                spec.begin(&tokens)
                    .map_err(|e| GenError::Generate(format!("mtp begin: {e}")))?;
            }
        }

        let mut batch = LlamaBatch::new(fresh.len().max(512), 1);
        let last = fresh.len() - 1;
        for (i, t) in fresh.iter().enumerate() {
            batch
                .add(*t, (reuse + i) as i32, &[slot as i32], i == last)
                .map_err(|e| GenError::Generate(e.to_string()))?;
        }

        let t_prefill = Instant::now();
        loaded
            .dec
            .ctx()
            .decode(&mut batch)
            .map_err(|e| GenError::Generate(e.to_string()))?;
        if speculative {
            if let Decoder::Mtp(spec) = &mut loaded.dec {
                spec.process(&batch)
                    .map_err(|e| GenError::Generate(format!("mtp process: {e}")))?;
            }
        }
        // Building the prefix IS prefill, so it is reported as prefill. Hiding it
        // would make the turn that pays for the file look free and every later
        // turn look unchanged, which is the opposite of what the number is for.
        let prefill_ms = t_prefill.elapsed().as_millis() as u64 + prefix_ms;

        // --- decode ---------------------------------------------------------
        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let mut n_cur = tokens.len() as i32;
        // What the cache holds once this turn's prompt is in. Generated tokens are
        // appended below so the next turn's common-prefix scan sees them too.
        let mut cached = tokens.clone();
        let mut raw = String::new();
        let mut output_tokens = 0usize;
        let mut truncated = true;
        // The rendered text of the token that ended the turn, so it can be
        // trimmed off the answer after parsing.
        let mut stop_piece: Option<String> = None;
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
            let piece = loaded
                .model
                .token_to_str(next, Special::Tokenize)
                .map_err(|e| GenError::Generate(e.to_string()))?;
            raw.push_str(&piece);
            output_tokens += 1;
            if loaded.model.is_eog_token(next) {
                truncated = false;
                // Remember how this turn ended, so the marker can come off the
                // answer later. Gemma 4 declares NO template stop strings, so
                // `additional_stops` is empty and the end is a TOKEN; its
                // rendered text is still model-supplied, which is what makes
                // this safe to strip without hardcoding `<turn|>`.
                stop_piece = Some(piece);
                break;
            }

            // Dispatched on `speculative`, not on the decoder type. A non-zero
            // slot must take the plain path EVEN when a drafter is loaded,
            // because MtpSpeculative binds to sequence 0 and would otherwise
            // write another slot's cache.
            match (speculative, &mut loaded.dec) {
                // --- plain: one token in, one token out ---------------------
                (false, dec) => {
                    let ctx = dec.ctx();
                    batch.clear();
                    batch
                        .add(next, n_cur, &[slot as i32], true)
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
                (true, Decoder::Mtp(spec)) => {
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
                        let piece = loaded
                            .model
                            .token_to_str(*d, Special::Tokenize)
                            .map_err(|e| GenError::Generate(e.to_string()))?;
                        raw.push_str(&piece);
                        output_tokens += 1;
                        cached.push(*d);
                        if loaded.model.is_eog_token(*d) {
                            truncated = false;
                            // A turn can end inside an ACCEPTED draft, so the
                            // marker has to be captured here too, not only on
                            // the directly sampled path above.
                            stop_piece = Some(piece);
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
                // `speculative` is only true when the decoder IS Mtp.
                (true, Decoder::Plain(_)) => unreachable!("speculative implies a drafter"),
            }
        }
        loaded.cached.insert(slot, cached);
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

        // Trim the template's own stop markers off the answer.
        //
        // Stopping on end-of-turn TOKENS leaves the marker in the decoded text,
        // and upstream's parser deliberately keeps it in `content` because the
        // grammar matches a complete turn INCLUDING it. So Gemma 4 answers
        // "Today is July 31, 2026.<turn|>" and the marker reaches the user.
        //
        // Trimmed here, AFTER the parse, so the grammar still sees the complete
        // turn it wants; only the user-facing string is cleaned. The markers come
        // from `additional_stops`, i.e. from the model's own template, so this
        // holds for the next model family without a per-family constant. A
        // hardcoded "<turn|>" would have been the same fix in appearance and a
        // maintenance trap in fact.
        //
        // Not covered by the tool-calling bench, which grades tool names and
        // arguments and never asserts on answer TEXT: all 20 cases passed with
        // this broken.
        let mut text = parsed.content.trim();
        // Template-declared stops first, then the end-of-turn token's own text.
        // Gemma 4 declares none of the former, so in practice the latter is what
        // does the work here; both are model-supplied, which is the point.
        for stop in applied
            .additional_stops
            .iter()
            .chain(stop_piece.iter())
        {
            if let Some(stripped) = text.strip_suffix(stop.as_str()) {
                text = stripped.trim_end();
            }
        }
        let text = text.to_string();
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
                draft_proposed: draft_proposed as u32,
                draft_accepted: draft_accepted as u32,
            },
        })
    }
}
