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

use llama_cpp_2::context::params::{KvCacheType as LlamaKvCacheType, LlamaContextParams};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{LlamaStateSeqFlags, SeqState};
use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeParams};

use crate::generation::{
    Availability, CancelHandle, GenError, GenerateRequest, GenerateResponse, GenerationEngine, Msg,
    Timings, ToolCall,
};

/// Hard cap on generated tokens per turn, so a runaway model cannot hang a UI.
/// Reported through [`Timings::truncated`] when hit, because a truncated answer
/// must never be mistaken for a brief one.
const MAX_OUTPUT_TOKENS: usize = 2048;

/// What the reasoning-budget sampler forces into the think channel ahead of
/// the close tag when the budget runs out.
///
/// NOT decoration. llama.cpp's own budget PR measured Qwen 3.5 9B at 89%
/// with a wrap-up message against 79% with a bare forced close, and the s1
/// budget-forcing paper found the same: the model answers from partial
/// reasoning gracefully only when the trace ENDS like a decision instead of
/// ending mid-sentence. First person and present tense because it is spliced
/// into the model's own voice; the leading newline separates it from
/// whatever half-written thought it interrupts.
const REASONING_BUDGET_WRAPUP: &str = "\nOkay, I have to stop thinking now and answer with what I have.\n";

/// How many tokens the persisted prefix stops SHORT of the true stable
/// point, so it ends inside the tool block rather than on the template's
/// post-tools glue. See `stable_prefix` for why this decides whether a
/// registry append can extend the old state or must recompute ~35s of
/// prefill. Sized to comfortably cover the glue (~5-15 tokens on Gemma 4)
/// across template families.
const KV_PREFIX_GLUE_MARGIN: usize = 48;

/// How far before the prompt's end a windowed-SWA checkpoint is taken.
///
/// The next turn's divergence from this turn's cache lands almost entirely in
/// the TAIL: the region where a replayed history renders the last exchange
/// differently than the raw generated text did (measured drift: ~15-20 tokens
/// before the prompt end). A checkpoint taken this far back covers it, so the
/// next turn restores and re-prefills tens of tokens instead of thousands.
/// History further back is a template render of the same messages both times
/// and does not drift.
const SWA_CHECKPOINT_TAIL_MARGIN: usize = 64;

/// Physical micro-batch for the DRAFTER's context. Its compute buffer scales
/// with this (~1 MiB per token of ubatch on E4B), and the drafter needs large
/// batches for nothing: verified by the gated MTP tests and the production
/// bench holding their acceptance rate with this at 128 (517 -> 145 MiB).
const DRAFT_N_UBATCH: u32 = 128;

/// Checkpoints kept per slot: the current turn's and the previous one's.
/// Each costs host RAM proportional to the SWA window's occupied cells (a few
/// tens of MB at a 16K window on E4B), which is exactly the memory this
/// feature exists to NOT spend on the GPU, so the ring stays small.
const SWA_CHECKPOINTS_PER_SLOT: usize = 2;

/// Smallest shared prompt head worth PINNING on a background slot.
///
/// The pinned-head machinery learns the shared instruction head as the
/// smallest common prefix observed between consecutive prompts on the slot.
/// Two unrelated prompts still share the template's opening tokens (a
/// handful), and pinning those would spend a state copy to save nothing
/// while evicting a pin that was earning its keep for the majority prompt
/// family. Below this floor a divergence is treated as "unrelated prompt",
/// not as new evidence about the head.
const PINNED_HEAD_MIN_TOKENS: usize = 64;

/// How long a throttled engine log stays quiet between emissions.
///
/// The rewind guard's per-occurrence lines are diagnostic at one occurrence
/// and noise at backlog-drain volume (dozens of background one-shots in a
/// row each printing "re-prefilling all N tokens"). One line per window,
/// carrying the count of what it suppressed, keeps the signal.
const LOG_THROTTLE_WINDOW: std::time::Duration = std::time::Duration::from_secs(60);

/// A windowed-SWA rollback point: the sequence's PARTIAL (SWA-cache-only)
/// state at a known token boundary, plus the tokens it stands for.
///
/// The base (full-attention) cache needs no snapshot — it never evicts, so
/// rewinding it is an ordinary trim. What is unrecoverable under
/// `swa_full: false` is the sliding window's contents at an earlier position;
/// this is that window, captured while it was live. The token list is the
/// validity proof: a checkpoint may be restored only into a prompt that
/// starts with exactly these tokens, the same continue-only contract the
/// persisted disk prefix uses.
struct SwaCheckpoint {
    tokens: Vec<llama_cpp_2::token::LlamaToken>,
    state: SeqState,
}

/// First occurrence prints verbatim; repeats inside [`LOG_THROTTLE_WINDOW`]
/// are counted instead of printed, and the next occurrence after the window
/// prints again with the suppressed count attached.
struct LogThrottle {
    last: Option<Instant>,
    suppressed: u64,
}

impl LogThrottle {
    const fn new() -> Self {
        Self { last: None, suppressed: 0 }
    }

    /// `Some(n)` means print now, mentioning the `n` occurrences swallowed
    /// since the last print; `None` means stay quiet.
    fn permit(&mut self) -> Option<u64> {
        match self.last {
            Some(t) if t.elapsed() < LOG_THROTTLE_WINDOW => {
                self.suppressed += 1;
                None
            }
            _ => {
                self.last = Some(Instant::now());
                Some(std::mem::take(&mut self.suppressed))
            }
        }
    }
}

/// Renders a [`LogThrottle::permit`] count for appending to a log line.
fn suppressed_note(n: u64) -> String {
    if n == 0 {
        String::new()
    } else {
        format!(" ({n} similar occurrences suppressed in the last minute)")
    }
}

/// The process-wide llama.cpp backend, shared by every engine.
///
/// llama-cpp-2 guards `LlamaBackend::init()` with a process-global flag that
/// only resets when the returned handle drops, so two engines that each owned
/// a backend could never be loaded at the same time: whichever called
/// `ensure_loaded` second failed with `BackendAlreadyInitialized`. That is not
/// a theoretical race — Scope desktop runs a chat engine and a memory
/// extraction engine in one process, and the extraction engine could never
/// load while the chat brain was warm (and vice versa), which silently killed
/// on-device memory derivation.
///
/// One backend initialized once and never freed is the shape llama.cpp itself
/// documents (init once per process). Never freeing it is deliberate and
/// safe: the ggml-metal exit assert that shaped [`Loaded`]'s drop order fires
/// when the device is torn down while contexts still hold resource sets, and
/// a backend that is never torn down cannot trip it. The handful of bytes of
/// ggml global state lives for the process either way.
///
/// Initialization failures are NOT cached: if some other code in the process
/// holds its own `LlamaBackend` (a host that predates this function), init
/// fails now but succeeds after that holder drops, so the next call retries.
fn shared_backend() -> Result<&'static LlamaBackend, GenError> {
    use std::sync::{Mutex, OnceLock};
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let _g = INIT.lock().map_err(|e| GenError::Generate(e.to_string()))?;
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let b = LlamaBackend::init()
        .map_err(|e| GenError::Generate(format!("llama backend init: {e}")))?;
    Ok(BACKEND.get_or_init(|| b))
}

/// What makes two model loads interchangeable.
///
/// Everything here is an input to `llama_model_load_from_file`, so two configs
/// that agree on all of it would produce byte-identical weights in memory and
/// may share one load. Context parameters are deliberately absent: they belong
/// to `llama_context`, which is per-context by construction, and holding them
/// in this key is what would force a second load for a second window.
///
/// The path is canonicalized and its length taken, the same "what this crate
/// can see" identity [`LlamaCppEngine::prefix_key_for`] uses. A file swapped in
/// place at the same length is not detected here either; a host that knows more
/// than the filesystem does keeps its weights at distinct paths.
#[derive(PartialEq, Eq, Hash)]
struct ModelKey {
    path: PathBuf,
    len: u64,
    n_gpu_layers: u32,
    use_mmap: bool,
    cpu_tensor_overrides: Vec<String>,
}

impl ModelKey {
    fn of(cfg: &LlamaCppConfig) -> Self {
        let path =
            std::fs::canonicalize(&cfg.model_path).unwrap_or_else(|_| cfg.model_path.clone());
        let len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        Self {
            path,
            len,
            n_gpu_layers: cfg.n_gpu_layers,
            use_mmap: cfg.use_mmap,
            cpu_tensor_overrides: cfg.cpu_tensor_overrides.clone(),
        }
    }
}

/// How many times weights have actually been read off disk in this process.
static MODEL_LOADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The process-wide weight registry: one loaded [`LlamaModel`] per [`ModelKey`],
/// handed out to every context that wants it.
///
/// **Weights are the expensive resident thing and contexts are not.** E4B is
/// 5.15 GB and 12-40s to load; a context over it is a KV cache sized by its own
/// window. Before this, one engine meant one model meant one window, so a host
/// that wanted a second window either scaled the shared one up (llama.cpp
/// divides a context's KV evenly across its sequences, so every workload gets
/// the largest one's window) or built a second engine — which loaded the same
/// 5.15 GB a second time and could OOM Metal with both warm.
///
/// [`Weak`](std::sync::Weak), not `Arc`: an entry must not keep weights alive
/// after the last engine unloads, or `unload` would stop freeing memory, which
/// is the one thing it exists to do. A dead entry is replaced by the next load.
fn shared_model(
    cfg: &LlamaCppConfig,
    backend: &'static LlamaBackend,
) -> Result<std::sync::Arc<LlamaModel>, GenError> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock, Weak};
    static MODELS: OnceLock<Mutex<HashMap<ModelKey, Weak<LlamaModel>>>> = OnceLock::new();

    let key = ModelKey::of(cfg);
    // Held across the load on purpose: two engines racing for the SAME weights
    // must produce one load, not two, and the second caller wants the first
    // one's result rather than its own 40 seconds. The price is that a load
    // also blocks a concurrent load of DIFFERENT weights, which no host does
    // (a device runs one model family at a time) and which costs waiting,
    // never correctness.
    let mut reg = MODELS
        .get_or_init(Default::default)
        .lock()
        .map_err(|e| GenError::Generate(e.to_string()))?;
    if let Some(m) = reg.get(&key).and_then(Weak::upgrade) {
        return Ok(m);
    }
    let m = Arc::new(load_model(cfg, backend)?);
    MODEL_LOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    reg.insert(key, Arc::downgrade(&m));
    // Entries whose weights are gone: dropped here rather than in `unload`,
    // which has no lock and no key. The map holds one small entry per distinct
    // model a process ever loaded, so this is tidiness, not a leak fix.
    reg.retain(|_, w| w.strong_count() > 0);
    Ok(m)
}

/// Read the weights off disk. Every caller goes through [`shared_model`].
fn load_model(
    cfg: &LlamaCppConfig,
    backend: &'static LlamaBackend,
) -> Result<LlamaModel, GenError> {
    // The override list stores the pattern as a RAW pointer into each
    // CString, so these must outlive `load_from_file` — hence a named
    // binding rather than a temporary inside the loop, which would be freed
    // before the model reads it.
    let patterns: Vec<std::ffi::CString> = cfg
        .cpu_tensor_overrides
        .iter()
        .map(|p| {
            std::ffi::CString::new(p.as_str()).map_err(|_| {
                GenError::Invalid(format!("tensor override has an interior NUL: {p:?}"))
            })
        })
        .collect::<Result<_, _>>()?;
    let base = LlamaModelParams::default()
        .with_n_gpu_layers(cfg.n_gpu_layers)
        .with_use_mmap(cfg.use_mmap);
    // llama.cpp's own load-abort hook: the progress callback is consulted
    // per tensor group, and returning false stops the read. Installed ONLY
    // when a handle is armed, because installing one is not neutral:
    // llama.cpp substitutes its own progress logger when the field is null,
    // and taking that over would change what an uncancellable build prints.
    // Records that the callback is what stopped the read, which is a
    // different question from whether the flag is set. llama.cpp reports an
    // abort and a genuine failure as the same error, so asking the flag
    // would report a corrupt model as a cancel whenever a stop happened to
    // coincide with one, and a host is told to treat `Cancelled` as nothing
    // to report. That turns a broken download into silence.
    let vetoed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let base = match cfg.cancel.clone() {
        Some(c) => {
            let vetoed = vetoed.clone();
            base.with_progress_callback(move |_| {
                if c.is_cancelled() {
                    vetoed.store(true, std::sync::atomic::Ordering::Relaxed);
                    return false;
                }
                true
            })
        }
        None => base,
    };
    let mut params = Box::pin(base);
    for p in &patterns {
        params.as_mut().add_cpu_buft_override(p);
    }
    LlamaModel::load_from_file(backend, &cfg.model_path, &params).map_err(|e| {
        if vetoed.load(std::sync::atomic::Ordering::Relaxed) {
            GenError::Cancelled
        } else {
            GenError::Generate(format!("load {}: {e}", cfg.model_path.display()))
        }
    })
}

/// One context over the shared weights, and the slots it serves.
///
/// **A context's slots are necessarily equal-sized**: llama.cpp divides a
/// context's KV cache evenly across its sequences, so `n_ctx` here is the
/// window EACH of this context's slots gets and the total is scaled by
/// `n_slots`. Two workloads that want different windows therefore need two
/// contexts, which is what this type is for.
///
/// The measured reason to bother: Gemma 4 E4B at a 64K window costs 1,024 MiB
/// of KV per slot against 256 MiB at 16K. A desktop running an agent loop plus
/// short background extraction turns on one 64K two-slot context pays the
/// agent's window twice; giving the background workload its own 16K context
/// gives back 768 MiB at zero decode cost.
#[derive(Debug, Clone)]
pub struct ContextSpec {
    /// Window in tokens for EACH slot of this context.
    pub n_ctx: u32,
    /// How many slots (llama.cpp sequences) this context serves.
    pub n_slots: u32,
    /// Physical micro-batch for this context, or `None` to inherit
    /// [`LlamaCppConfig::n_ubatch`].
    ///
    /// Per-context because the compute buffer is per-context and scales with
    /// it (~1 MiB per token of ubatch on E4B), so a second context inherits a
    /// prefill-throughput setting chosen for the first one and charges a second
    /// full-size buffer for it. A background slot that prefills a few hundred
    /// tokens per job has no use for the big dispatch the agent's window was
    /// sized for.
    pub n_ubatch: Option<u32>,
}

impl Default for ContextSpec {
    fn default() -> Self {
        Self {
            n_ctx: 16384,
            n_slots: 1,
            n_ubatch: None,
        }
    }
}

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
    /// Context window in tokens for each slot of the BASE context. A property
    /// of the MODEL, so the caller supplies it from its model profile; this
    /// crate does not decide it.
    ///
    /// A workload that wants a different window gets its own context; see
    /// [`extra_contexts`](Self::extra_contexts).
    pub n_ctx: u32,
    /// Whether sliding-window-attention layers keep a FULL-size KV cache
    /// (`true`) or only their window (`false`, the default).
    ///
    /// `false` is the memory that pays for everything else: measured on
    /// Gemma 4 E4B at a 16K window, 896 MiB per slot full-size against
    /// 296 MiB windowed, identical prefill throughput. What makes it safe to
    /// DEFAULT to is the checkpoint machinery in `generate_impl`: a windowed
    /// cache cannot rewind (positions behind the window are evicted, llama.cpp
    /// permits the trim anyway, and the output silently degrades — and
    /// ordinary turn 2 already rewinds, by the few-token render drift between
    /// a replayed history and the raw generated tail), so the engine snapshots
    /// the window near each prompt's tail (PARTIAL_ONLY state, host RAM) and
    /// restores the newest covering snapshot when a turn rewinds. Divergence
    /// deeper than every checkpoint degrades to a full re-prefill: slow,
    /// never wrong, never an unload. This is llama-server's own answer
    /// (context checkpoints) built into the engine.
    ///
    /// `true` buys free arbitrary rewinds at the full KV price, and turns the
    /// checkpoint machinery off; the guard stays inert (`pos_min` never moves).
    pub swa_full: bool,
    /// Physical micro-batch: how many tokens one Metal kernel dispatch carries
    /// during prefill. `None` keeps llama.cpp's default (512).
    ///
    /// This is the prefill throughput knob (`n_batch` above it is only the
    /// LOGICAL cap on tokens per `decode()` call). Community numbers on Apple
    /// Silicon put 2048-vs-512 at 2-3x prompt eval; the price is compute
    /// buffer memory, which scales with it. NEVER pass `Some(0)`: llama.cpp
    /// treats 0 as "use n_batch", which with our window-sized n_batch would
    /// size the compute buffer for a 16K-token dispatch.
    pub n_ubatch: Option<u32>,
    /// Layers to offload to the GPU. A large value means "all it will take".
    pub n_gpu_layers: u32,
    /// Tensor-name patterns (llama.cpp regex) to keep in the CPU buffer type
    /// even when the layer they belong to is offloaded to the GPU.
    ///
    /// This exists because on a memory-constrained device the accelerator
    /// budget, not the model file, is the binding constraint — and the two are
    /// not the same memory. Measured on an iPhone 14 Pro (A16) 2026-08-11:
    /// Metal's working set is 4,096 MiB, Gemma 4 E2B q4_0 wires 3,179 MiB of
    /// it, and the process was killed for a SYSTEM-wide `vm-pageshortage`
    /// (free memory 36 MiB, wired 4,040 MiB) while its OWN footprint was a
    /// harmless 875 MiB. Metal buffers are wired and unpageable; CPU-side
    /// tensors stay file-backed mmap pages the kernel can evict.
    ///
    /// The lever that matters for this family is `per_layer_token_embd`:
    /// 1,838 MiB of E2B's 3,179 MiB, 58% of the model, and a LOOKUP TABLE
    /// read one row per token per layer rather than a matrix multiplied by
    /// anything. Keeping it off the accelerator is what the per-layer-embedding
    /// design is for. Q4_0 is also the quantization that makes this free in
    /// both directions: llama.cpp implements `get_rows` for it on CPU and GPU
    /// alike, where the K-quants only support one side.
    ///
    /// Empty (the default) preserves the previous behaviour exactly.
    ///
    /// **Has no effect while [`use_mmap`](Self::use_mmap) is true.** Measured
    /// 2026-08-11: with mmap on, llama.cpp maps the whole file into the
    /// accelerator's buffer and the override changes the reported Metal size by
    /// exactly zero bytes — it only says which backend COMPUTES with the tensor.
    /// llama.cpp warns about the combination ("tensor overrides to CPU are used
    /// with mmap enabled") but does not refuse it, so this is a silent no-op
    /// unless mmap is also turned off.
    pub cpu_tensor_overrides: Vec<String>,
    /// Whether to memory-map the weights. `true` is llama.cpp's default and the
    /// right answer almost everywhere: pages are file-backed, shared, evictable
    /// under pressure, and a second load is nearly instant.
    ///
    /// Turn it OFF only to make [`cpu_tensor_overrides`](Self::cpu_tensor_overrides)
    /// real, and know what that trades: without mmap each tensor is allocated
    /// in its assigned backend, so an overridden tensor genuinely leaves the
    /// accelerator — but it lands in ANONYMOUS host memory, which is charged to
    /// the process and cannot be evicted, only compressed. On a phone that
    /// moves pressure from the system-wide wired total onto the per-process
    /// footprint. Whether that is a win depends on which limit is binding, so
    /// measure both rather than assuming.
    pub use_mmap: bool,
    /// Whether the model may emit a reasoning channel.
    pub enable_thinking: bool,
    /// How many KV cache slots the BASE context supports.
    ///
    /// Must cover the highest `GenerateRequest::slot` a host will use, together
    /// with [`extra_contexts`](Self::extra_contexts). A context built with the
    /// default of ONE sequence rejects slot 1 outright, and the symptom is not
    /// a clear error about sequences: llama.cpp reports "failed to initialize
    /// batch" and then `n_tokens == 0`.
    ///
    /// Every slot of one context has the SAME window ([`n_ctx`](Self::n_ctx)),
    /// because llama.cpp divides a context's KV evenly across its sequences.
    /// Slots that want different windows go in different contexts.
    pub n_slots: u32,
    /// Further contexts over the same loaded weights, each with its own window.
    ///
    /// Empty is the default and the previous behaviour exactly: one context,
    /// [`n_slots`](Self::n_slots) slots of [`n_ctx`](Self::n_ctx) tokens each.
    ///
    /// Slots are numbered consecutively across contexts: the base context owns
    /// slots `0..n_slots`, the first extra the `n_slots` that follow, and so on.
    /// So a desktop that keeps its agent on slot 0 and background work on slot 1
    /// asks for `n_slots: 1` plus one extra context of `n_slots: 1`, and neither
    /// side changes which slot number it sends.
    ///
    /// The weights are loaded ONCE regardless (see [`shared_model`]); what each
    /// entry costs is its own KV cache and compute buffer. That is the whole
    /// point: KV is what scales with the window, and a background workload that
    /// never fills 64K should not be charged for one.
    ///
    /// Slot 0 keeps everything that is written against it: the persisted KV
    /// prefix and the MTP drafter both live on the base context.
    pub extra_contexts: Vec<ContextSpec>,
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
    /// Element type of the K cache. [`KvCacheType::F16`] is llama.cpp's default.
    ///
    /// The KV cache is the one buffer that grows with the conversation rather
    /// than with the weights, so quantizing it is the only lever that gets
    /// cheaper the longer a session runs. `Q8_0` halves it.
    ///
    /// **This is not a free setting and the generic advice does not transfer.**
    /// Whether it is safe is a property of the MODEL: published KL-divergence
    /// comparisons put Gemma an order of magnitude worse than Qwen under q8_0
    /// KV, and a model whose attention is already mostly windowed has little
    /// KV left to save. Measure the model in hand rather than adopting a
    /// number from another family.
    pub cache_type_k: KvCacheType,
    /// Element type of the V cache. See [`cache_type_k`](Self::cache_type_k).
    ///
    /// **A quantized V requires flash attention.** llama.cpp refuses the
    /// context outright ("V cache quantization requires flash_attn",
    /// `llama_init_from_model`) rather than degrading, so a host that turns
    /// flash attention off and quantizes V gets a load failure, not slow
    /// answers. Flash attention is AUTO by default here and resolves to
    /// enabled on Metal.
    ///
    /// Both types additionally require the block size to divide the head
    /// dimension, checked per layer at context creation.
    pub cache_type_v: KvCacheType,
    /// The host's stop signal for an in-flight call, or `None` to make every
    /// call run to completion.
    ///
    /// On the config rather than on the engine because of WHEN the host needs
    /// it: the first call is also the one that pays the cold load (measured up
    /// to 40.4s for E4B's 5.15 GB), and a host that builds the engine lazily
    /// inside its own lock has no chance to ask an engine for a handle before
    /// that. Building the handle first and passing it in covers the load, the
    /// prefill and the decode of the very first turn.
    ///
    /// Arming this is what turns cancellation ON. With `None` the engine takes
    /// exactly the code path it took before cancellation existed, prefill
    /// batching and model-load callbacks included, so a host that does not want
    /// a stop button pays nothing for one.
    pub cancel: Option<CancelHandle>,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            draft_path: None,
            draft_n_max: 3,
            n_ctx: 16384,
            swa_full: false,
            n_ubatch: None,
            n_gpu_layers: 1000,
            cpu_tensor_overrides: Vec::new(),
            use_mmap: true,
            enable_thinking: true,
            // Two: the agent loop, plus background one-shots.
            n_slots: 2,
            extra_contexts: Vec::new(),
            // Off unless a host names a directory: persisting state is a policy
            // decision (where it lives, when it is cleared) and this crate has
            // no business inventing a cache location.
            prefix_cache_dir: None,
            state_key: String::new(),
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            cancel: None,
        }
    }
}

/// Element type for one half of the KV cache.
///
/// Its own enum rather than the binding's `llama_cpp_2::context::params::
/// KvCacheType`, so a host can name a cache type without depending on the
/// binding crate at the same version this one resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCacheType {
    /// llama.cpp's default, and the only one that needs no flash attention.
    #[default]
    F16,
    F32,
    BF16,
    /// Half the bytes of `F16`. The usual first stop, and the one with
    /// published quality data.
    Q8_0,
    Q5_1,
    Q5_0,
    Q4_1,
    Q4_0,
}

impl From<KvCacheType> for LlamaKvCacheType {
    fn from(t: KvCacheType) -> Self {
        match t {
            KvCacheType::F16 => Self::F16,
            KvCacheType::F32 => Self::F32,
            KvCacheType::BF16 => Self::BF16,
            KvCacheType::Q8_0 => Self::Q8_0,
            KvCacheType::Q5_1 => Self::Q5_1,
            KvCacheType::Q5_0 => Self::Q5_0,
            KvCacheType::Q4_1 => Self::Q4_1,
            KvCacheType::Q4_0 => Self::Q4_0,
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
    /// What the loaded template says it supports. Read once at load, because it
    /// is a property of the template and re-asking per turn would be pure cost.
    /// `None` until the weights land, and empty when upstream does not report.
    caps: Option<std::collections::BTreeMap<String, bool>>,
    /// Capability complaints already printed, so each is said once per engine
    /// rather than once per turn.
    warned: std::collections::HashSet<String>,
    /// Per-slot LEARNED length of the shared prompt head, in tokens: the
    /// smallest common prefix observed between consecutive prompts on the
    /// slot (ignoring sub-[`PINNED_HEAD_MIN_TOKENS`] overlaps as unrelated).
    /// Lives on the engine, not on [`Loaded`]: the length survives an
    /// unload/reload so the first prefill after a warm-up can re-pin
    /// immediately instead of re-learning from a fresh divergence. Only ever
    /// shrinks, which is safe: a pin shorter than the true head restores
    /// less but is never wrong.
    head_len: std::collections::HashMap<u32, usize>,
    /// Throttles for the rewind guard's per-occurrence lines, which flood at
    /// backlog-drain volume (one line per background job).
    log_full_reprefill: LogThrottle,
    log_pinned_restore: LogThrottle,
}

/// Where one host slot lives: which context, which sequence inside it, and how
/// big a window it therefore has.
///
/// Copy, and computed once per turn, because the borrow discipline below
/// depends on it: the context is reached as `loaded.decs[at.ctx]`, a borrow of
/// ONE field, so the checkpoint rings and the token records in the sibling
/// fields stay reachable at the same time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlotAddr {
    /// The host's slot number, which keys everything the engine records.
    slot: u32,
    /// Index into [`Loaded::decs`].
    ctx: usize,
    /// The llama.cpp sequence id INSIDE that context. Equal to `slot` only for
    /// the base context.
    seq: i32,
    /// Tokens this slot may hold.
    n_ctx: u32,
}

/// The loaded engine.
///
/// **Field order is load-bearing.** Rust drops fields in declaration order, and
/// every context borrows `model`, so they must be declared (and torn down)
/// first. Getting this wrong is not a leak, it is a SIGABRT: ggml-metal asserts
/// at exit that its resource sets were released
/// (`GGML_ASSERT([rsets->data count] == 0)`), which in an app reads as a crash
/// on quit. An earlier version leaked the model to dodge the self-reference and
/// hit exactly that.
///
/// The backend is NOT a field: it is the process-wide [`shared_backend`],
/// which outlives every `Loaded` by construction.
struct Loaded {
    /// One decoder per context, base first. Each is either a plain context or
    /// an MTP speculative wrapper that OWNS the target context (and the
    /// drafter's).
    ///
    /// One enum rather than two `Option`s so "both set" and "neither set" are
    /// unrepresentable; the speculative path takes the context by value, so they
    /// genuinely cannot coexist.
    ///
    /// Several entries because a context's slots are all the same size and two
    /// workloads may not want the same window; see [`ContextSpec`].
    decs: Vec<Decoder>,
    /// Host slot -> where it lives, indexed by slot number.
    slots: Vec<SlotAddr>,
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
    /// Per-slot windowed-SWA rollback points, newest first. Empty rings when
    /// `swa_full` is true (nothing to roll back — the full cache rewinds for
    /// free). Dropped with the engine, which is correct: a checkpoint is only
    /// valid against the llama.cpp state that produced it.
    checkpoints: std::collections::HashMap<u32, std::collections::VecDeque<SwaCheckpoint>>,
    /// One PINNED checkpoint per non-zero slot: the slot's prompts' shared
    /// instruction head, never evicted by the per-turn ring.
    ///
    /// The ring hugs each prompt's TAIL, which serves a conversation (the
    /// next prompt extends this one) and is useless for a background slot's
    /// one-shots: every job replaces the whole prompt, so no tail checkpoint
    /// is ever a prefix of the next job and each job re-prefilled everything
    /// (measured: dozens of 500-3,200-token full re-prefills in one backlog
    /// drain, all diverging at the ~390-token shared head). What one-shots DO
    /// share is that head, so the slot keeps exactly one checkpoint there,
    /// the analogue of slot 0's persisted disk prefix, in host RAM because
    /// background prompts are assembled by the host per call and have no
    /// stable identity worth a file. Slot 0 never pins; the disk prefix
    /// already covers it.
    pinned_heads: std::collections::HashMap<u32, SwaCheckpoint>,
    /// Boxed for a stable address the drafter's context borrows. Dropped after
    /// `decs`, which holds that context. Only the base context drafts.
    #[allow(dead_code)]
    draft_model: Option<Box<LlamaModel>>,
    /// The shared weights, from [`shared_model`]. Heap-allocated, so its address
    /// is stable for the `'static` reborrow every context takes; dropped after
    /// `decs`, and the weights themselves only when this is the last handle.
    model: std::sync::Arc<LlamaModel>,
}

impl Loaded {
    /// Where `slot` lives.
    ///
    /// A slot past the last one declared addresses the last context, where
    /// llama.cpp rejects it exactly as it always has ("failed to initialize
    /// batch", then `n_tokens == 0`): the sequence id is out of that context's
    /// range by construction, since the declared slots cover every context up
    /// to its end. Unchanged behaviour on purpose — an over-range slot has been
    /// a decode-level failure the engine recovers from by unloading, and
    /// `tests/llama_two_engines.rs` pins that recovery.
    fn addr(&self, slot: u32) -> SlotAddr {
        match self.slots.get(slot as usize) {
            Some(&a) => a,
            None => {
                let last = *self.slots.last().expect("at least one slot");
                SlotAddr {
                    slot,
                    ctx: last.ctx,
                    seq: (slot - last.slot) as i32 + last.seq,
                    n_ctx: last.n_ctx,
                }
            }
        }
    }
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
        Self {
            cfg,
            loaded: None,
            chat: None,
            caps: None,
            warned: Default::default(),
            head_len: Default::default(),
            log_full_reprefill: LogThrottle::new(),
            log_pinned_restore: LogThrottle::new(),
        }
    }

    /// True once the weights are resident.
    pub fn is_warm(&self) -> bool {
        self.loaded.is_some()
    }

    /// The slot's recorded prefix length, and the highest position actually
    /// present in its KV cache. `None` before the weights load.
    ///
    /// These two must agree after every turn: the recorded prefix is what the
    /// NEXT turn's common-prefix scan trusts, and when the cache holds more than
    /// the record says, that turn appends on top of cells nobody accounts for
    /// (same sequence, same positions, both visible to attention) and answers
    /// from a corrupted context. Nothing about that failure is observable from
    /// the outside, which is why the invariant is exposed rather than inferred:
    /// see `tests/llama_kv_invariant.rs`.
    ///
    /// A cache holding N tokens reports a max position of N-1, so the two agree
    /// when `prefix_len == pos_max + 1`.
    pub fn kv_prefix_state(&mut self, slot: u32) -> Option<(usize, i32)> {
        let loaded = self.loaded.as_mut()?;
        let at = loaded.addr(slot);
        let prefix_len = loaded.cached.get(&slot).map_or(0, Vec::len);
        let pos_max = loaded.decs[at.ctx].ctx().kv_cache_seq_pos_max(at.seq);
        Some((prefix_len, pos_max))
    }

    /// The window each slot actually has, read from llama.cpp rather than from
    /// the configuration.
    ///
    /// Empty before the weights load. This is the observable half of
    /// [`LlamaCppConfig::extra_contexts`]: a host that asked for a 64K agent
    /// slot and a 16K background slot can check that is what it got, and so can
    /// a diagnostics view, without inferring it from a config it does not hold.
    ///
    /// Reported per SLOT, not per context, because a context's `n_ctx` is the
    /// total across its sequences and a slot only ever gets its share.
    pub fn slot_windows(&mut self) -> std::collections::BTreeMap<u32, u32> {
        let Some(loaded) = self.loaded.as_mut() else {
            return Default::default();
        };
        let addrs = loaded.slots.clone();
        addrs
            .iter()
            .map(|a| {
                // llama.cpp reports a context's TOTAL, and the binding does not
                // expose `n_seq_max`, so the share is taken over the slots this
                // engine mapped into that context — which is the same number it
                // built the context with.
                let seqs = addrs.iter().filter(|b| b.ctx == a.ctx).count().max(1) as u32;
                (a.slot, loaded.decs[a.ctx].ctx().n_ctx() / seqs)
            })
            .collect()
    }

    /// How many times GGUF weights have been read off disk in this process.
    ///
    /// The point of the shared registry is that this does NOT go up when a
    /// second context, or a second engine, wants weights that are already
    /// resident: a load is 5.15 GB and 12-40s for E4B, and a second copy of it
    /// is what could OOM Metal with both warm. Exposed so that can be asserted
    /// rather than believed — process RSS cannot say it, because the weights
    /// are mmapped (`model buffer size` prints 0.00 MiB and RSS for one
    /// identical configuration has been observed at both 6.44 and 1.66 GiB
    /// across two runs).
    pub fn model_loads() -> u64 {
        MODEL_LOADS.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// What the loaded template declares it supports, as the jinja layer names
    /// it: `supports_tools`, `supports_tool_calls`, `supports_object_arguments`,
    /// `supports_system_role`, and so on.
    ///
    /// Empty before the weights load, and empty if upstream does not report
    /// them. The engine already warns on a mismatch by itself (see
    /// `warn_unsupported`); this is for a host that wants to check BEFORE
    /// building a transcript it cannot send, or to show it in a diagnostics
    /// view.
    pub fn template_caps(&self) -> std::collections::BTreeMap<String, bool> {
        self.caps.clone().unwrap_or_default()
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
    ///
    /// Cancellable when [`LlamaCppConfig::cancel`] is armed, which is the point:
    /// this is the call that pays the 40.4s.
    pub fn warm(&mut self) -> Result<u64, GenError> {
        self.clear_cancel();
        self.ensure_loaded()
    }

    /// Drop a stop signal left over from an earlier call, so a cancel only ever
    /// applies to the call it arrives during. See [`CancelHandle`] for the race
    /// this trades away.
    fn clear_cancel(&self) {
        if let Some(c) = &self.cfg.cancel {
            c.clear();
        }
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
    ///
    /// Frees this engine's contexts (their KV caches and compute buffers) always,
    /// and the WEIGHTS only when no other engine still holds them: they are
    /// shared through [`shared_model`], and the last handle out turns off the
    /// lights. `true` reports that this engine had something to release, not
    /// that 5 GB came back.
    pub fn unload(&mut self) -> bool {
        let was = self.loaded.is_some();
        // `chat` is derived from the model's template, so it must go too or a
        // reload would keep a template belonging to weights that are gone.
        self.loaded = None;
        self.chat = None;
        // Derived from the template, so it goes with it. Keeping stale caps
        // across a reload would describe weights that are gone.
        self.caps = None;
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
        backend: &'static LlamaBackend,
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
        // The drafter's SCRATCH buffer, not the target's: compute-buffer size
        // scales linearly with n_ubatch (measured 522/261/145 MiB at
        // 512/256/128), and inheriting the target's 512 gave a 77 MB drafter a
        // 517 MiB scratch allocation. The drafter's own work is drafting ≤
        // draft_n_max tokens per step; the only large batches it sees are the
        // target's prefill chunks passing through `spec.process`, which a
        // smaller ubatch merely splits — the drafter's forward pass is so
        // cheap that the extra dispatches are noise. n_batch stays the
        // target's, so no batch is ever rejected as too large.
        let ctx_params = ctx_params.with_n_ubatch(DRAFT_N_UBATCH);
        // SAFETY: identical to the target model above -- boxed, so its address is
        // stable, and `Loaded` drops the context before the model.
        let dm_ref: &'static LlamaModel = unsafe { &*(&*dm as *const LlamaModel) };
        let ctx = dm_ref
            .new_context_with_ctx_other(backend, ctx_params, target)
            .map_err(|e| GenError::Generate(e.to_string()))?;
        Ok((ctx, dm))
    }

    /// The layout this configuration asks for: the base context first, then
    /// whatever extras were declared.
    ///
    /// One place, because the slot map, the context build and
    /// [`slot_windows`](Self::slot_windows) must agree about which slot lives
    /// where, and a host reading a window that the engine does not actually
    /// serve would be worse than no accessor at all.
    fn context_specs(&self) -> Vec<ContextSpec> {
        let base = ContextSpec {
            n_ctx: self.cfg.n_ctx,
            n_slots: self.cfg.n_slots.max(1),
            n_ubatch: self.cfg.n_ubatch,
        };
        std::iter::once(base)
            .chain(self.cfg.extra_contexts.iter().cloned().map(|s| ContextSpec {
                n_slots: s.n_slots.max(1),
                n_ubatch: s.n_ubatch.or(self.cfg.n_ubatch),
                ..s
            }))
            .collect()
    }

    /// The slot map for a layout: slot numbers run consecutively across
    /// contexts, so a host that adds a context does not renumber the slots it
    /// was already sending.
    fn slot_map(specs: &[ContextSpec]) -> Vec<SlotAddr> {
        let mut slots = Vec::new();
        for (ctx, spec) in specs.iter().enumerate() {
            for seq in 0..spec.n_slots {
                slots.push(SlotAddr {
                    slot: slots.len() as u32,
                    ctx,
                    seq: seq as i32,
                    n_ctx: spec.n_ctx,
                });
            }
        }
        slots
    }

    /// The context parameters, in ONE place.
    ///
    /// There are three call sites (the base context, each extra one, and the
    /// rebuild after an MTP init failure) and the drafter derives its own from
    /// whatever the base produced. When they were written out twice they had
    /// already drifted once; a KV element type that differs between the two
    /// would be worse than the `n_ubatch` drift was, because the persisted
    /// prefix is keyed on it and the mismatch shows up as a state file that
    /// silently will not restore.
    ///
    /// `n_ctx` is the TOTAL across sequences: llama.cpp gives each sequence
    /// `n_ctx / n_seq_max`. So asking for two slots at face value would
    /// silently HALVE the caller's window, from the profile's 16,384 to 8,192,
    /// and the first symptom would be a mid-conversation `NoKvCacheSlot` rather
    /// than anything naming the window. [`ContextSpec::n_ctx`] is a property of
    /// the MODEL and of the workload, so it is honoured per slot and the total
    /// is scaled here instead.
    ///
    /// `n_batch` caps how many tokens one `decode()` call may carry, and it
    /// defaults far below a real prompt: the Gemma 4 template renders the tool
    /// schemas into tens of thousands of characters, which asserts out at
    /// `n_tokens_all <= cparams.n_batch`. Sized to this context's window, which
    /// is also the longest prompt it will accept.
    fn ctx_params(&self, spec: &ContextSpec) -> LlamaContextParams {
        let slots = spec.n_slots.max(1);
        let mut p = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(spec.n_ctx.saturating_mul(slots)))
            .with_n_batch(spec.n_ctx)
            .with_n_seq_max(slots)
            .with_swa_full(self.cfg.swa_full)
            .with_type_k(self.cfg.cache_type_k.into())
            .with_type_v(self.cfg.cache_type_v.into());
        if let Some(ub) = spec.n_ubatch.filter(|&ub| ub > 0) {
            p = p.with_n_ubatch(ub);
        }
        p
    }

    fn ensure_loaded(&mut self) -> Result<u64, GenError> {
        if self.loaded.is_some() {
            return Ok(0);
        }
        let t = Instant::now();
        let backend = shared_backend()?;
        // One load per (weights, model params) in the process, however many
        // engines and contexts want them. See `shared_model`.
        let model = shared_model(&self.cfg, backend)?;

        // The template comes from the GGUF itself, so the shim never has to guess
        // which model family it is rendering for.
        let template = model
            .chat_template(None)
            .map_err(|e| GenError::Generate(format!("model has no chat template: {e}")))?
            .to_string()
            .map_err(|e| GenError::Generate(format!("chat template is not utf-8: {e}")))?;
        let mut chat = ffi::Chat::new(&template)
            .map_err(|e| GenError::Generate(format!("chat shim init: {e}")))?;

        // The base context, then one per extra spec. Each has its OWN window and
        // its own KV cache; they share only the weights, which is where the
        // memory is.
        let specs = self.context_specs();
        let ctx_params = self.ctx_params(&specs[0]);
        // SAFETY: `model` is an `Arc`, so the weights sit at a stable heap
        // address for as long as any handle lives, and `Loaded` holds one and
        // declares `decs` first so every context is dropped before it. The
        // backend is genuinely `'static` (shared_backend), no extension needed.
        let model_ref: &'static LlamaModel = unsafe { &*(&*model as *const LlamaModel) };
        let ctx = model_ref
            .new_context(backend, ctx_params.clone())
            .map_err(|e| GenError::Generate(e.to_string()))?;

        // The drafter, when one is configured. Failing to load it is NOT fatal:
        // speculation is a pure speed optimisation and a turn without it is
        // correct, just slower. Degrading to plain decode beats refusing to
        // answer because an 77 MB accessory is missing or mismatched.
        let (dec, draft_model) = match &self.cfg.draft_path {
            Some(p) if p.exists() => {
                match Self::load_drafter(backend, ctx_params, p, &ctx) {
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
                            let p = self.ctx_params(&specs[0]);
                            let c = model_ref
                                .new_context(backend, p)
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
        // The extras. Built after the base one because the drafter binds to the
        // base context and must find it already there; plain contexts, because
        // the MTP head shares the target's KV memory and there is one of those.
        let mut decs = vec![dec];
        for spec in &specs[1..] {
            let p = self.ctx_params(spec);
            let c = model_ref
                .new_context(backend, p)
                .map_err(|e| GenError::Generate(e.to_string()))?;
            decs.push(Decoder::Plain(c));
        }

        let mut cached = std::collections::HashMap::new();
        let mut prefix = PrefixState::Disabled;
        if let Some(path) = Self::prefix_path_for(&self.cfg) {
            prefix = PrefixState::Missing;
            if path.exists() {
                let t_restore = Instant::now();
                match decs[0]
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
            decs,
            slots: Self::slot_map(&specs),
            cached,
            prefix,
            checkpoints: Default::default(),
            pinned_heads: Default::default(),
            draft_model,
            model,
        });
        // Read once, here, while the template is fresh. `warn_unsupported`
        // consults it per turn and must never pay FFI for the privilege.
        self.caps = Some(chat.caps());
        self.chat = Some(chat);
        Ok(t.elapsed().as_millis() as u64)
    }

    /// Where this configuration's persisted prefix lives, if enabled.
    fn prefix_path_for(cfg: &LlamaCppConfig) -> Option<PathBuf> {
        let dir = cfg.prefix_cache_dir.as_ref()?;
        Some(dir.join(format!("prefix-{}.kv", Self::prefix_key_for(cfg))))
    }

    /// Public form of [`Self::prefix_path_for`], for hosts that install a
    /// PREBUILT prefix artifact: a state file computed elsewhere (a release
    /// pipeline) for these exact weights and this exact prompt must land
    /// under the engine's own key name to be found by the launch restore.
    /// The name is machine-specific (the key hashes local paths), which is
    /// why the artifact cannot ship pre-named and the host has to ask.
    ///
    /// Adopting a foreign file is safe by the same contract as any restore:
    /// `state_seq_load_file` hands back the token list the state was saved
    /// with, and the first real prompt must start with it or the whole thing
    /// is discarded and rebuilt locally.
    pub fn prefix_artifact_path(cfg: &LlamaCppConfig) -> Option<PathBuf> {
        Self::prefix_path_for(cfg)
    }

    /// The state's SHAPE, as a short id for whoever publishes and whoever
    /// downloads a prebuilt artifact.
    ///
    /// [`Self::prefix_key_for`] already answers "may this file be opened", but
    /// only on the machine that holds the file: it hashes local paths, so it
    /// cannot name an artifact in a bucket. A published artifact therefore
    /// needs its own name for the same question, and the name has to be
    /// derivable on both sides from the same fields.
    ///
    /// **This exists because the URL could not express a shape change and a
    /// shape change is silent.** The published key was
    /// `{model-id}-{weights-sha}/v1/{prompt-fingerprint}`, which covers the
    /// WEIGHTS and the PROMPT and nothing else. When the desktop moved its
    /// window from 16,384 to 65,536 on 2026-08-20, the publisher kept building
    /// at the old default and the URL stayed byte-identical, so a fresh install
    /// would fetch a 16k-shaped state, rename it into its own 64k key, and hand
    /// llama.cpp a cache built under a different layout. That is the same class
    /// of mismatch the weights half of the key exists to prevent, and the token
    /// check at restore cannot see it either: the tokens are right, the values
    /// are for a different cache.
    ///
    /// The weights and the paths are deliberately NOT in here. The weights are
    /// the other half of the published key already, and the paths are the part
    /// that cannot travel. Everything else that [`Self::prefix_key_for`] hashes
    /// is, so adding a field there and forgetting it here is the one drift this
    /// is meant to make impossible — keep the two lists together.
    pub fn prefix_shape_id(cfg: &LlamaCppConfig) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"askfaro-kv-prefix-shape-v1\0");
        h.update(cfg.n_ctx.to_le_bytes());
        h.update(cfg.n_slots.to_le_bytes());
        h.update(cfg.n_gpu_layers.to_le_bytes());
        for p in &cfg.cpu_tensor_overrides {
            h.update(p.as_bytes());
            h.update(b"\0");
        }
        h.update([u8::from(cfg.use_mmap)]);
        h.update([u8::from(cfg.enable_thinking)]);
        h.update([u8::from(cfg.swa_full)]);
        h.update([cfg.cache_type_k as u8, cfg.cache_type_v as u8]);
        hex::encode(&h.finalize()[..6])
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
        // The BASE context's shape, and deliberately not the extra contexts':
        // the prefix is slot 0's, it is saved out of and restored into the base
        // context, and an extra context is a separate `llama_context` that
        // cannot change a cell of it. Hashing the whole layout would throw away
        // the agent's ~27s prefill every time a host retuned a background
        // window — an invalidation that buys nothing.
        h.update(cfg.n_ctx.to_le_bytes());
        h.update(cfg.n_slots.to_le_bytes());
        h.update(cfg.n_gpu_layers.to_le_bytes());
        // Which tensors sit in which buffer type changes where the computation
        // happens, so a state computed under one split must not be offered to
        // another. Same reasoning as n_gpu_layers directly above.
        for p in &cfg.cpu_tensor_overrides {
            h.update(p.as_bytes());
            h.update(b"\0");
        }
        h.update([u8::from(cfg.use_mmap)]);
        h.update([u8::from(cfg.enable_thinking)]);
        // The SWA cache's SIZE is part of the persisted state's layout, so a
        // state saved windowed must never be offered to a full-size context or
        // vice versa; llama.cpp validates shape and would likely reject the
        // mismatch, but "likely rejected" is not the contract this key exists
        // to provide.
        h.update([u8::from(cfg.swa_full)]);
        // The cache's ELEMENT TYPE is the other half of its layout. A state
        // written with an f16 cache and restored into a q8_0 one is the same
        // class of hazard as the quantization mismatch `state_key` exists for:
        // the byte count differs, so this one would at least be caught, but
        // being caught means a silent full re-prefill on every launch rather
        // than an error anyone sees.
        h.update([cfg.cache_type_k as u8, cfg.cache_type_v as u8]);
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
        r.messages.push(Msg::user(probe));
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
        // Held back from the true common point by a margin, so the PERSISTED
        // prefix ends inside the stable tool block instead of on the
        // template's post-tools glue (the user-turn opener). The glue sits
        // exactly where the next appended tool begins, so a prefix saved
        // through it can never be a strict prefix of the prompt after a
        // registry APPEND, and build_prefix's restore-and-extend could never
        // fire; measured, that turned an ~800-token update into a ~35s full
        // recompute. The margin costs one prefill of these tokens per restore
        // (~0.2s) and buys extendability for the commonest prompt change
        // there is.
        let n = n.saturating_sub(KV_PREFIX_GLUE_MARGIN).max(1);
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
        let cancel = self.cfg.cancel.clone();
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
                loaded.decs[0]
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
        loaded.decs[0]
            .ctx()
            .kv_cache_seq_rm(0, None, None)
            .map_err(|e| GenError::Generate(format!("kv clear: {e}")))?;
        loaded.cached.remove(&0);

        // Restore-and-extend: when the SUPERSEDED state file holds a strict
        // prefix of the new prefix, restore it and prefill only the tail.
        // This is the common shape of a prompt update: tools are APPENDED to
        // the registry, the head (system text, existing tools) is untouched,
        // and recomputing it from scratch costs ~35s of prefill at this
        // model's measured ~290 tok/s ceiling for ~800 tokens of actual
        // change. Head-of-prompt edits fail the starts_with check and pay the
        // full rebuild they genuinely need. The restore lands in a sequence
        // just cleared above, and extending a restored windowed-SWA state
        // forward is an ordinary contiguous continuation; what a windowed
        // state cannot do is REWIND, which this never does.
        let mut start = 0usize;
        if path.exists() {
            match loaded.decs[0].ctx().state_seq_load_file(&path, 0, self.cfg.n_ctx as usize) {
                Ok((toks, _)) if !toks.is_empty() && prefix.len() > toks.len()
                    && prefix[..toks.len()] == toks[..] =>
                {
                    eprintln!(
                        "llama_cpp: the superseded prefix ({} tokens) is a prefix of the new \
                         one ({}); extending it instead of recomputing",
                        toks.len(),
                        prefix.len(),
                    );
                    start = toks.len();
                }
                Ok(_) => {
                    // Restored but not a strict prefix: drop it whole and
                    // rebuild from scratch. Never trim a windowed state.
                    loaded.decs[0]
                        .ctx()
                        .kv_cache_seq_rm(0, None, None)
                        .map_err(|e| GenError::Generate(format!("kv clear: {e}")))?;
                }
                Err(_) => {}
            }
        }

        // Mirrors a real turn's prefill exactly, drafter bookkeeping included.
        // Gemma 4's MTP head shares the target's KV memory rather than keeping
        // its own, which is what makes persisting the target's sequence enough
        // for speculation to work on the restored turn as well.
        // The window this batch may later be reused for, in tokens, and 0 when
        // there is no drafter. Doubles as the `feed the drafter` flag.
        let drafting = if matches!(loaded.decs[0], Decoder::Mtp(_)) {
            1 + self.cfg.draft_n_max.max(0) as usize
        } else {
            0
        };
        if let Decoder::Mtp(spec) = &mut loaded.decs[0] {
            spec.begin(&prefix)
                .map_err(|e| GenError::Generate(format!("mtp begin: {e}")))?;
        }
        let at = loaded.addr(0);
        Self::prefill_span(loaded, at, &prefix, start, prefix.len(), drafting, cancel.as_ref())?;
        // A cancel that lands during the LAST chunk still reaches here, and the
        // state file is the one artifact of a stopped call that would outlive
        // the process: written now, every later launch would restore it as if
        // it had been asked for. The tokens themselves are recorded, because
        // they are genuinely in the cache.
        if cancel.as_ref().is_some_and(CancelHandle::is_cancelled) {
            loaded.cached.insert(0, prefix.clone());
            return Err(GenError::Cancelled);
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| GenError::Generate(format!("prefix cache dir: {e}")))?;
        }
        // Written to a temp name and renamed, so a process killed mid-write
        // leaves the previous good file rather than a truncated one that reads
        // as present.
        let tmp = path.with_extension("kv.tmp");
        let bytes = loaded.decs[0]
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

    /// Read `tokens[start..end]` into `seq`, returning the batch that carried
    /// the last of them so the caller can sample from its logits.
    ///
    /// **This is the only interruption point that can reach inside prefill.**
    /// A per-token check in the decode loop cannot: prefill is ONE
    /// `llama_decode` over the whole prompt, and llama.cpp splits it into
    /// physical micro-batches internally, so the Rust side does not get a turn
    /// until the first token is sampled. Measured on E2B at `n_ubatch` 128,
    /// 6,297 prompt tokens: 13.4s of it, and E4B's ~7,700-token prompts are the
    /// ~27s the persisted prefix exists to remove.
    ///
    /// The upstream answer to that would be ggml's abort callback, and it is not
    /// available here: llama-cpp-2 exposes neither the raw `llama_context` nor
    /// `llama_context_params`, and on Metal it would be inert anyway. The
    /// Metal backend does not publish `ggml_backend_set_abort_callback` from
    /// its registry, so `llama_context::set_abort_callback` installs the hook
    /// on the CPU backend only, and ggml-metal's own abort check sits inside
    /// its Metal-capture debug branch.
    ///
    /// So the split is made HERE instead, at the same boundary llama.cpp would
    /// have split on, which leaves each dispatch exactly the shape it had. A
    /// prefill split at a token boundary is arithmetically identical to an
    /// unsplit one under causal attention, which the checkpoint machinery in
    /// `generate_impl` already relies on.
    ///
    /// Unarmed, `chunk` is the whole span: one batch, one decode, one
    /// allocation, which is what every caller got before cancellation existed.
    fn prefill_span(
        loaded: &mut Loaded,
        at: SlotAddr,
        tokens: &[llama_cpp_2::token::LlamaToken],
        start: usize,
        end: usize,
        draft_slots: usize,
        cancel: Option<&CancelHandle>,
    ) -> Result<LlamaBatch<'static>, GenError> {
        let span = &tokens[start..end];
        let chunk = match cancel {
            None => span.len(),
            Some(_) => loaded.decs[at.ctx].ctx().n_ubatch().max(1) as usize,
        };
        // `draft_slots` is in the capacity because `generate_impl` goes on to
        // reuse this batch for its speculative steps. Unarmed that was free,
        // since the capacity came from the prompt and dwarfed a draft step. Once
        // `chunk` is `n_ubatch`, a good interruption granularity would otherwise
        // silently become a batch too small for the window that follows it.
        let mut batch = LlamaBatch::new(chunk.min(span.len()).max(512).max(draft_slots), 1);
        let mut done = 0usize;
        while done < span.len() {
            if cancel.is_some_and(CancelHandle::is_cancelled) {
                // What the sequence actually holds, recorded before returning.
                // Everything up to here was decoded contiguously from position
                // 0, so this is a truthful continue-only cache the next turn can
                // extend, the same contract the persisted prefix and the SWA
                // checkpoints are validated against. Trimming BACK to where the
                // turn started would be the wrong repair: a windowed SWA cache
                // cannot rewind.
                loaded.cached.insert(at.slot, tokens[..start + done].to_vec());
                return Err(GenError::Cancelled);
            }
            let upto = (done + chunk).min(span.len());
            batch.clear();
            // Logits on the span's final token only. An intermediate chunk is
            // pure context, and asking for its logits would pay a 262K-row
            // output projection for a row nothing reads.
            for (i, t) in span[done..upto].iter().enumerate() {
                batch
                    .add(
                        *t,
                        (start + done + i) as i32,
                        &[at.seq],
                        done + i == span.len() - 1,
                    )
                    .map_err(|e| GenError::Generate(e.to_string()))?;
            }
            loaded.decs[at.ctx]
                .ctx()
                .decode(&mut batch)
                .map_err(|e| GenError::Generate(e.to_string()))?;
            if draft_slots > 0 {
                if let Decoder::Mtp(spec) = &mut loaded.decs[at.ctx] {
                    spec.process(&batch)
                        .map_err(|e| GenError::Generate(format!("mtp process: {e}")))?;
                }
            }
            done = upto;
        }
        Ok(batch)
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

    /// Say, once, when a request asks the template for something it cannot do.
    ///
    /// **Every failure this guards against is fluent, not fatal.** The template
    /// renders whatever it understands and drops the rest, so the model gets a
    /// transcript missing the part the host cared about and answers from what
    /// survived. Scope shipped exactly that for months: tool calls written into
    /// message text, never rendered as tool calls, and a model that imitated the
    /// prose it was shown and invented the data. Nothing errored, and the only
    /// evidence was in the answers.
    ///
    /// Warnings rather than errors on purpose. A template that reports
    /// `supports_tool_calls = false` may still produce something usable, and a
    /// host mid-conversation is not helped by a hard failure it cannot act on.
    /// The point is that the next person who hits this reads one line at startup
    /// instead of rediscovering it from a benchmark.
    ///
    /// Deduplicated per engine: this runs on every turn, and a warning printed
    /// once per turn is a warning nobody reads.
    fn warn_unsupported(&mut self, req: &GenerateRequest) {
        let caps = match &self.caps {
            Some(c) => c,
            None => return,
        };
        // A missing key means this upstream does not report that capability.
        // Absence is not denial, so an unknown capability never warns.
        let supported = |k: &str| caps.get(k).copied().unwrap_or(true);

        let mut say: Vec<String> = Vec::new();
        if !req.tools.is_empty() && !supported("supports_tools") {
            say.push(format!(
                "{} tool schema(s) were sent, but this template declares supports_tools=false: \
                 they will not appear in the prompt and the model cannot call them",
                req.tools.len()
            ));
        }
        if req.messages.iter().any(|m| !m.tool_calls.is_empty())
            && !supported("supports_tool_calls")
        {
            say.push(
                "the transcript replays assistant tool CALLS, but this template declares \
                 supports_tool_calls=false: they will be dropped from the prompt, so the model \
                 sees a conversation in which it never called anything"
                    .into(),
            );
        }
        // TWO checks, not four, and deliberately. The first draft also warned on
        // `supports_parallel_tool_calls` and on `supports_object_arguments`, and
        // the second one fired on Gemma 4 the very first time it ran, on the
        // path the bench scores 100%. It was wrong: upstream calls
        // `workaround::func_args_not_string` when a template supports object
        // arguments and converts our JSON string itself (`chat.cpp`, around the
        // `original_caps().supports_object_arguments` branch). The parallel-calls
        // case was never verified either way.
        //
        // A warning that fires on a working path is worse than no warning: it
        // teaches the reader that these lines are noise, which is precisely how
        // the failure this whole mechanism exists to catch stayed invisible.
        // Only warn about what has been checked. Adding a third means verifying
        // it against upstream first, the same way these two were.

        for s in say {
            // `warned` is the dedup set, so each distinct complaint is said once
            // per engine rather than once per turn.
            if self.warned.insert(s.clone()) {
                eprintln!("llama_cpp: {s}");
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
        self.clear_cancel();
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

/// Sample at `idx`, holding the token to `grmr` when a grammar is active.
///
/// Upstream's rejection-sampling pattern (common/sampling.cpp with
/// `grammar_first == false`): sample from the plain chain, test THAT ONE
/// token against the grammar with a 1-element apply, and only when it is
/// rejected pay the full-vocab grammar mask and take the best surviving
/// token. Under greedy decode the argmax is grammar-legal on almost every
/// step, so the constrained path costs one cheap check per token instead of
/// a 262K-piece grammar evaluation per token.
///
/// The grammar advances ONLY here, exactly once per committed token.
/// `LlamaSampler::sample` auto-accepts into the CHAIN it is called on, but
/// `grmr` lives outside the chain, so the double-accept corruption the
/// decode loop's invariant documents cannot recur through this path.
fn sample_with_grammar(
    ctx: &mut LlamaContext<'static>,
    chain: &mut LlamaSampler,
    grmr: Option<&mut LlamaSampler>,
    idx: i32,
) -> llama_cpp_2::token::LlamaToken {
    let token = chain.sample(ctx, idx);
    let Some(grmr) = grmr else {
        return token;
    };
    // The cheap test: does the grammar admit this exact token? Logit 0.0 in,
    // -INF out means rejected.
    let mut probe = llama_cpp_2::token::data_array::LlamaTokenDataArray::from_iter(
        [llama_cpp_2::token::data::LlamaTokenData::new(token, 0.0, 0.0)],
        false,
    );
    probe.apply_sampler(grmr);
    let admitted = probe
        .data
        .first()
        .is_some_and(|d| d.logit() > f32::NEG_INFINITY);
    if admitted {
        grmr.accept(token);
        return token;
    }
    // Rejection: the model's argmax is not grammar-legal (prose where JSON
    // must continue, or end-of-turn before the value is complete). Pay the
    // full mask once and take the best surviving token. The grammar always
    // admits SOMETHING from a live state: a complete grammar admits the EOG
    // token, an incomplete one admits at least one continuation.
    let mut cands = ctx.token_data_array_ith(idx);
    cands.apply_sampler(grmr);
    let token = cands.sample_token_greedy();
    grmr.accept(token);
    token
}

impl GenerationEngine for LlamaCppEngine {
    /// Reports only what can be known without a model path in hand. A caller
    /// with a configured engine should prefer [`Self::availability_for`].
    fn availability() -> Availability {
        Availability::Available
    }

    fn generate(&mut self, req: GenerateRequest) -> Result<GenerateResponse, GenError> {
        self.clear_cancel();
        let out = self.generate_impl(req);
        match &out {
            // Fires before any decode touches the KV cache, and hosts retry it
            // in a trim-and-resend loop; the engine state is intact, so paying
            // a reload on every iteration of that loop would be pure waste.
            Err(GenError::ContextWindowExceeded) => {}
            // Request rejection, not inference failure: every `Invalid` in
            // `generate_impl` is returned before any decode touches the KV
            // cache (validation runs ahead of the weight load, and the sampler
            // is built ahead of the reuse trim), so the warm weights are still
            // good. Unloading here would make one malformed request cost the
            // next good one a multi-second reload.
            Err(GenError::Invalid(_)) => {}
            // The host stopped this turn, which is not a fault to recover from.
            // Every cancel point leaves the slot's KV cache and its token
            // bookkeeping agreeing with each other, which is the contract that
            // makes the engine reusable, so unloading here would throw away
            // multi-gigabyte warm weights, and the reload, to fix nothing.
            Err(GenError::Cancelled) => {}
            // Any other failure may have left llama.cpp's state inconsistent
            // with the slot bookkeeping, and the engine cannot tell from here.
            // The desktop's memory deriver found the concrete case: a Metal
            // OOM aborts the decode, llama.cpp trims the sequence, and every
            // later call on that engine fails position validation ("failed to
            // initialize batch"), forever. llama.cpp's own log says it:
            // "recreate the backend to recover" — the erroring object is the
            // per-context ggml scheduler, so dropping the context IS the
            // recovery, and the next generate reloads clean (~3-5s warm).
            Err(_) => {
                self.unload();
            }
            Ok(_) => {}
        }
        out
    }
}

impl LlamaCppEngine {
    fn generate_impl(&mut self, req: GenerateRequest) -> Result<GenerateResponse, GenError> {
        // Request validation FIRST, ahead of the weight load: a malformed
        // request must cost an error, not a multi-gigabyte model load. The
        // wrapper in `generate` relies on this ordering when it declines to
        // unload on `Invalid` — every `Invalid` below is returned before any
        // decode has touched the KV cache.
        if req.json_schema.is_some() && !req.tools.is_empty() {
            return Err(GenError::Invalid(
                "json_schema and tools are mutually exclusive: upstream's response-format \
                 grammar takes precedence over the tool-call rules (Gemma 4's PEG returns \
                 before building them), so the tools would render into the prompt but be \
                 unsampleable — dropped silently, which this engine refuses to do"
                    .into(),
            ));
        }
        // Upstream requires the schema to be a JSON OBJECT and silently ignores
        // any other shape (`has_response_format = inputs.json_schema.is_object()`
        // in every family handler), which for a constrained request means the
        // one failure this feature exists to prevent: a reply that was asked to
        // conform and simply is not. Rejected loudly here instead.
        if req.json_schema.as_ref().is_some_and(|s| !s.is_object()) {
            return Err(GenError::Invalid(
                "json_schema must be a JSON Schema OBJECT; upstream ignores any other shape \
                 and the reply would be silently unconstrained"
                    .into(),
            ));
        }

        let load_ms = self.ensure_loaded()?;
        // Both read up front so the code below stays readable once `loaded`
        // holds a mutable borrow of `self`.
        let cancel = self.cfg.cancel.clone();
        let draft_n_max = self.cfg.draft_n_max.max(0) as usize;
        let enable_thinking = req.enable_thinking.unwrap_or(self.cfg.enable_thinking);
        // Where this request's slot lives, and how big a window it therefore
        // has. Read once, before `loaded` is borrowed, and Copy, so the borrow
        // splitting below stays field-level.
        let at = self.loaded.as_ref().expect("just loaded").addr(req.slot);
        let n_ctx = at.n_ctx;

        // --- render the prompt through the model's own template -------------
        let applied = self
            .chat
            .as_mut()
            .expect("chat is set with loaded")
            .apply(&req, enable_thinking)
            .map_err(|e| GenError::Generate(format!("chat template: {e}")))?;

        // Does this template do what the request is asking of it? Warned once per
        // engine, before the prompt is tokenized, because the failure is silent
        // in both directions: a template that ignores `tool_calls` renders a
        // transcript with the tool use missing, and the model answers fluently
        // from what is left. That cost a full multi-step agent loop for months
        // and was found by benchmark rather than by anything saying so.
        self.warn_unsupported(&req);

        // A json_schema request is honored by the FAMILY handler in upstream's
        // chat layer (the same `inputs.json_schema` llama-server fills from an
        // OpenAI `response_format`), which renders it into `applied.grammar` —
        // for Gemma 4, "optional thought channel, then a ```json fenced body
        // constrained to the schema", with the fences consumed by the parser.
        // Checked here, not assumed: a family whose handler does not implement
        // response_format returns an empty (or lazy, trigger-gated) grammar,
        // and applying nothing would return an ordinary unconstrained reply to
        // a caller that asked for a guarantee — the silent failure this whole
        // engine module is written against. Still ahead of prefill, so the
        // engine state is untouched on rejection.
        let grammar: Option<&str> = match &req.json_schema {
            None => None,
            Some(_) => {
                if applied.grammar.is_empty() || applied.grammar_lazy {
                    return Err(GenError::Invalid(format!(
                        "json_schema was requested but this model's chat format produced {} \
                         response-format grammar, so the constraint cannot be enforced",
                        if applied.grammar.is_empty() { "no" } else { "only a lazy" }
                    )));
                }
                Some(applied.grammar.as_str())
            }
        };

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

        // --- reasoning budget -------------------------------------------------
        // Upstream's reasoning-budget sampler (common/reasoning-budget.cpp, the
        // machinery behind llama-server's `--reasoning-budget`), built per
        // request. It watches the committed token stream for the template's own
        // think-open tag, counts the thinking tokens, and at the budget forces
        // a wrap-up message plus the think-CLOSE tag token-by-token; the answer
        // that follows is generated normally and never truncated. This is the
        // "shape the thinking, never truncate the answer" knob: MAX_OUTPUT_TOKENS
        // above is a runaway backstop, not a latency control.
        //
        // Grammar-constrained turns are excluded on purpose: their operating
        // point is thinking-off extraction, and upstream itself sequences the
        // budget sampler against LAZY grammars only, which this engine does not
        // run (its json_schema grammar is eager). Tags come from the template
        // via the chat shim, so a model family without a declared think channel
        // degrades to "no budget" rather than to guessed tags.
        let mut rbudget: Option<ffi::ReasoningBudget> = None;
        if let Some(budget) = req.reasoning_budget.filter(|b| *b >= 0) {
            if grammar.is_none()
                && enable_thinking
                && !applied.thinking_start.is_empty()
                && !applied.thinking_end.is_empty()
            {
                let model = &self.loaded.as_ref().expect("just loaded").model;
                let tok = |s: &str| -> Result<Vec<i32>, GenError> {
                    Ok(model
                        .str_to_token(s, AddBos::Never)
                        .map_err(|e| GenError::Generate(format!("tokenize think tag: {e}")))?
                        .into_iter()
                        .map(|t| t.0)
                        .collect())
                };
                let start = tok(&applied.thinking_start)?;
                let end = tok(&applied.thinking_end)?;
                let forced = tok(&format!("{REASONING_BUDGET_WRAPUP}{}", applied.thinking_end))?;
                match ffi::ReasoningBudget::new(&start, &end, forced, budget) {
                    Some(mut rb) => {
                        // Mirror upstream's prefill step: a continuation render
                        // can OPEN the think block inside the generation prompt,
                        // and the matcher must see it or the budget never arms.
                        // The leading-space skip is upstream's too: some
                        // tokenizers prepend a space token the text never had.
                        for (i, t) in model
                            .str_to_token(&applied.generation_prompt, AddBos::Never)
                            .map_err(|e| GenError::Generate(e.to_string()))?
                            .into_iter()
                            .enumerate()
                        {
                            if i == 0 {
                                let piece = model
                                    .token_to_str(t, Special::Tokenize)
                                    .unwrap_or_default();
                                let gp_ws =
                                    applied.generation_prompt.starts_with(char::is_whitespace);
                                if piece.starts_with(char::is_whitespace) && !gp_ws {
                                    continue;
                                }
                            }
                            rb.accept(t.0);
                        }
                        rbudget = Some(rb);
                    }
                    // A budget that cannot build must not fail the turn: the
                    // reply without it is correct, just longer.
                    None => eprintln!(
                        "llama_cpp: reasoning budget requested but the sampler failed to build; \
                         thinking is unbounded this turn"
                    ),
                }
            }
        }
        // The drafter must never straddle the budget boundary: it proposes
        // without seeing the budget, so a window that crossed it would commit
        // thinking tokens the mask had already cut off. One extra step of
        // margin over the window size covers `next` itself.
        let draft_window = self.cfg.draft_n_max.max(0) + 2;

        let loaded = self.loaded.as_mut().expect("just loaded");

        // The chain is plain greedy even on constrained turns; the grammar
        // lives OUTSIDE it and is enforced by `sample_with_grammar`'s
        // rejection-sampling, which is upstream's own pattern
        // (common/sampling.cpp, `grammar_first == false`). An in-chain
        // grammar sampler evaluates the grammar against all 262K vocab
        // pieces on EVERY token (the cost tracked upstream as llama.cpp
        // issue #4218); the rejection form tests only the one sampled token
        // and pays the full mask solely when the argmax is not
        // grammar-legal, which under greedy decode is the rare step, not
        // the common one. Enforcement is unchanged: a non-conforming token
        // still cannot COMMIT, it just gets rejected after the fact instead
        // of masked before.
        //
        // Built BEFORE anything below mutates this slot's cache, so a
        // failure here leaves the engine consistent (the `Invalid` arm of
        // `generate`'s wrapper counts on that). The grammar sampler holds
        // the C-side vocab pointer, not a Rust borrow, so `loaded` stays
        // freely mutable.
        // The close-tag bias, the median half of the thinking-length pair (the
        // reasoning budget above is the tail half; the field docs on
        // `GenerateRequest` carry the measurements). Under greedy decode a
        // positive bias means "close the think channel wherever its logit gets
        // within `bias` of the argmax", which ends easy derivations early and
        // does nothing mid-derivation. It rides the sampler CHAIN, so it also
        // steers the speculative verify pass; the drafter proposes without it
        // and simply loses the draft at the flip point (measured: acceptance
        // held at 70% with budget + bias active).
        //
        // Same guards as the budget, same reasons: no thinking channel means
        // nothing to bias, and a grammar-constrained turn is the extraction
        // operating point this feature must not touch. Biasing only the FIRST
        // token of the close tag is deliberate: it is the decision token, and
        // once it commits, the rest of the tag follows from the model itself.
        let close_bias = req
            .reasoning_close_bias
            .filter(|b| b.is_finite() && *b != 0.0 && enable_thinking && grammar.is_none())
            .and_then(|bias| {
                let close = loaded
                    .model
                    .str_to_token(&applied.thinking_end, AddBos::Never)
                    .ok()?;
                Some((close.first().copied()?, bias))
            });
        let mut sampler = match close_bias {
            Some((first, bias)) => LlamaSampler::chain_simple([
                LlamaSampler::logit_bias(
                    loaded.model.n_vocab(),
                    &[llama_cpp_2::token::logit_bias::LlamaLogitBias::new(first, bias)],
                ),
                LlamaSampler::greedy(),
            ]),
            None => LlamaSampler::chain_simple([LlamaSampler::greedy()]),
        };
        let mut grmr = match grammar {
            Some(g) => Some(LlamaSampler::grammar(&loaded.model, g, "root").map_err(|e| {
                GenError::Invalid(format!("json_schema produced an unusable grammar: {e}"))
            })?),
            None => None,
        };

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
        //
        // A grammar-constrained turn never speculates, even on slot 0. The
        // grammar sampler is STATEFUL, and the speculative path's contract
        // with it is untested: the verify loop's early exits leave sampled-
        // but-uncommitted tokens accepted into the grammar, and the drafter
        // proposes without seeing the grammar at all, so acceptance on
        // constrained output is unmeasured. Constrained turns are background
        // extraction (short JSON on a non-latency-critical path), so the
        // plain loop's correctness is worth more than an unproven speedup.
        // Skipping a turn is safe for the drafter: `spec.begin` re-announces
        // the whole prompt on every speculative turn.
        let slot = req.slot;
        let speculative =
            slot == 0 && matches!(loaded.decs[at.ctx], Decoder::Mtp(_)) && grammar.is_none();
        // In tokens, because prefill_span sizes its batch to hold this: the
        // batch it returns is the one the speculative steps below reuse.
        let draft_slots = if speculative { 1 + draft_n_max } else { 0 };
        let prior = loaded.cached.entry(slot).or_default();
        let common = prior
            .iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let mut reuse = common.min(tokens.len().saturating_sub(1));
        let prior_len = prior.len();
        if reuse < prior_len {
            // A real divergence between two prompts on a background slot is
            // evidence about the slot's SHARED HEAD: everything before the
            // divergence point was rendered identically by two independent
            // jobs. The smallest such prefix observed is where the pinned
            // checkpoint belongs; smaller can only under-restore, never
            // corrupt. Overlaps below the floor are two unrelated prompts
            // agreeing on template boilerplate, not head evidence; counting
            // them would shrink the pin to nothing.
            if slot != 0 && !self.cfg.swa_full && common >= PINNED_HEAD_MIN_TOKENS {
                self.head_len
                    .entry(slot)
                    .and_modify(|h| *h = (*h).min(common))
                    .or_insert(common);
            }
            // A windowed SWA cache (`swa_full: false`) cannot REWIND: positions
            // behind the window were evicted as decode advanced, so restarting
            // from an earlier point leaves the SWA layers attending over holes.
            // llama.cpp does not stop this — `seq_rm` succeeds and the output
            // is silently degraded, not an error (found the hard way: a
            // fallback written against seq_rm FAILING never fired, because it
            // does not fail). The guard is llama-server's, minus the n_swa the
            // safe bindings do not expose: `pos_min` is the oldest position
            // still present, so `pos_min <= 0` means nothing was evicted and
            // any trim point is sound; anything else forces a full re-prefill
            // of this slot. Conservative on purpose — with a windowed cache
            // even a 10-token rewind needs 10 EVICTED positions back, so
            // mid-history divergence genuinely costs the whole prompt (this is
            // exactly why upstream built context checkpoints; those are the
            // refinement if divergence turns out to be common). Appended turns
            // — the overwhelmingly common case — never trim and never pay
            // this. With `swa_full: true`, pos_min stays 0 and nothing
            // changes.
            let evicted = loaded.decs[at.ctx].ctx().kv_cache_seq_pos_min(at.seq) > 0;
            if evicted {
                // The way back: a checkpoint at or before the divergence whose
                // tokens this prompt still starts with. Newest first, because
                // the newest covering checkpoint minimizes the re-prefill.
                let ring_hit = loaded.checkpoints.get(&slot).and_then(|ring| {
                    ring.iter()
                        .find(|c| c.tokens.len() <= reuse && tokens.starts_with(&c.tokens))
                });
                // The ring first (its checkpoints sit deeper when they cover
                // at all), then the slot's pinned head. Same validity
                // contract for both: the prompt must start with exactly the
                // checkpoint's tokens, AND the divergence must lie at or past
                // its boundary, because the trim below keeps the base cache's cells
                // up to `n`, which only match this prompt within the common
                // prefix. For a background slot's one-shots the ring never
                // covers (each job replaces the whole prompt) and the pinned
                // head is the difference between re-prefilling the job's
                // unique content and re-prefilling everything.
                let pinned = ring_hit.is_none();
                let ckpt = ring_hit.or_else(|| {
                    loaded
                        .pinned_heads
                        .get(&slot)
                        .filter(|c| c.tokens.len() <= reuse && tokens.starts_with(&c.tokens))
                });
                // Rewind to the checkpoint boundary — plain and valid for
                // the full-size base cache; whatever it leaves in the SWA
                // cache does not matter, because the restore clears the
                // sequence there before inserting (state_read_meta begins
                // with seq_rm on the destination).
                //
                // A PARTIAL rewind only means anything for an attention KV
                // cache. On a hybrid recurrent architecture there is no
                // per-position history to truncate and llama.cpp refuses:
                // `llama_memory_recurrent::seq_rm` returns false for any
                // range that is not the whole sequence, surfacing here as
                // "Couldn't remove partial sequence". Measured on LFM2.5-2.6B
                // (short-convolution blocks + Gated Delta Net state): 12 of 30
                // bench cases failed the whole turn on this.
                //
                // A refusal is not a reason to lose the turn. This branch
                // already treats "re-prefill everything" as its floor, so a
                // refused rewind degrades to exactly that: slow, never wrong.
                let rewound = match ckpt.map(|c| c.tokens.len()) {
                    Some(n) => match loaded.decs[at.ctx].ctx().kv_cache_seq_rm(
                        at.seq,
                        Some(n as u32),
                        None,
                    ) {
                        Ok(_) => true,
                        Err(e) => {
                            if let Some(sup) = self.log_full_reprefill.permit() {
                                eprintln!(
                                    "llama_cpp: slot {slot} cannot rewind to a checkpoint on this \
                                     architecture ({e}); re-prefilling the whole prompt (slow, \
                                     not wrong){}",
                                    suppressed_note(sup),
                                );
                            }
                            false
                        }
                    },
                    None => false,
                };
                if let Some(c) = ckpt.filter(|_| rewound) {
                    let n = c.tokens.len();
                    loaded.decs[at.ctx]
                        .ctx()
                        .state_seq_set(&c.state, at.seq)
                        .map_err(|e| GenError::Generate(format!("checkpoint restore: {e}")))?;
                    if pinned {
                        // Fires once per background job during a backlog
                        // drain, so it is throttled; the ring line below is
                        // per user-visible turn and stays verbatim.
                        if let Some(sup) = self.log_pinned_restore.permit() {
                            eprintln!(
                                "llama_cpp: slot {slot} diverges at token {reuse}, behind the \
                                 SWA window; restored the {n}-token pinned head, re-prefilling \
                                 {} instead of {} tokens{}",
                                tokens.len() - n,
                                tokens.len(),
                                suppressed_note(sup),
                            );
                        }
                    } else {
                        eprintln!(
                            "llama_cpp: slot {slot} diverges at token {reuse}, behind the SWA \
                             window; restored the {n}-token checkpoint, re-prefilling {} instead \
                             of {} tokens",
                            tokens.len() - n,
                            tokens.len(),
                        );
                    }
                    reuse = n;
                } else {
                    // Slot 0 has one more way back before paying the whole
                    // prompt: the persisted prefix file. At a conversation
                    // boundary the new prompt keeps the system+tools prefix
                    // and replaces everything after it, so the divergence
                    // lands just past the prefix: behind the SWA window, past
                    // every checkpoint (those hug the PREVIOUS conversation's
                    // tail and are not prefixes of the new prompt), and
                    // squarely covered by the file. The multistep bench paid
                    // ~32s of full prefill at every case boundary for exactly
                    // this gap.
                    //
                    // Restored WHOLE into the sequence and never trimmed to an
                    // earlier point, same as the launch-time restore, and
                    // validated the way `settle_prefix` validates: this prompt
                    // must strictly extend the file's own token list. Any
                    // failure or mismatch drops the sequence whole and falls
                    // back to the full re-prefill this branch was already
                    // resigned to.
                    let mut restored = 0usize;
                    if slot == 0 {
                        if let Some(path) = Self::prefix_path_for(&self.cfg) {
                            if path.exists() {
                                loaded.decs[at.ctx]
                                    .ctx()
                                    .kv_cache_seq_rm(0, None, None)
                                    .map_err(|e| GenError::Generate(format!("kv clear: {e}")))?;
                                let t_restore = Instant::now();
                                match loaded.decs[at.ctx]
                                    .ctx()
                                    .state_seq_load_file(&path, 0, self.cfg.n_ctx as usize)
                                {
                                    Ok((toks, _)) => {
                                        let n = toks.len();
                                        if n > 0 && tokens.len() > n && tokens[..n] == toks[..] {
                                            eprintln!(
                                                "llama_cpp: slot 0 diverges at token {reuse}, \
                                                 behind the SWA window with no covering \
                                                 checkpoint; restored the {n}-token persisted \
                                                 prefix from {} in {}ms, re-prefilling {} \
                                                 instead of {} tokens",
                                                path.display(),
                                                t_restore.elapsed().as_millis(),
                                                tokens.len() - n,
                                                tokens.len(),
                                            );
                                            restored = n;
                                        } else {
                                            loaded.decs[at.ctx]
                                                .ctx()
                                                .kv_cache_seq_rm(0, None, None)
                                                .map_err(|e| {
                                                    GenError::Generate(format!("kv clear: {e}"))
                                                })?;
                                            eprintln!(
                                                "llama_cpp: the persisted prefix ({n} tokens on \
                                                 disk) is not a prefix of this prompt; discarded \
                                                 it whole",
                                            );
                                        }
                                    }
                                    Err(e) => eprintln!(
                                        "llama_cpp: could not restore the persisted prefix from \
                                         {} ({e}); continuing without it",
                                        path.display(),
                                    ),
                                }
                            }
                        }
                    }
                    if restored == 0 {
                        // Throttled: at backlog-drain volume this fired once
                        // per job and flooded the log. The first occurrence
                        // prints verbatim (it is the diagnostic one); repeats
                        // collapse into a per-minute count.
                        if let Some(sup) = self.log_full_reprefill.permit() {
                            eprintln!(
                                "llama_cpp: slot {slot} diverges at token {reuse} but the SWA \
                                 window has moved past it and no checkpoint covers it; \
                                 re-prefilling all {} tokens (slow, not wrong){}",
                                tokens.len(),
                                suppressed_note(sup),
                            );
                        }
                    }
                    reuse = restored;
                }
            }
            if reuse == 0 {
                loaded.decs[at.ctx]
                    .ctx()
                    .kv_cache_seq_rm(at.seq, None, None)
                    .map_err(|e| GenError::Generate(format!("kv clear: {e}")))?;
            } else if reuse < prior_len && !evicted {
                // Same refusal as the checkpoint rewind above, reached by the
                // ordinary divergence path: a recurrent cache cannot be
                // trimmed to a position. Drop the sequence whole and re-prefill
                // rather than losing the turn.
                if loaded.decs[at.ctx]
                    .ctx()
                    .kv_cache_seq_rm(at.seq, Some(reuse as u32), None)
                    .is_err()
                {
                    loaded.decs[at.ctx]
                        .ctx()
                        .kv_cache_seq_rm(at.seq, None, None)
                        .map_err(|e| GenError::Generate(format!("kv clear: {e}")))?;
                    reuse = 0;
                }
            }
        }
        // The speculative path needs the whole prompt announced up front, before
        // any decode, so the drafter's own context is primed from the same text.
        if speculative {
            if let Decoder::Mtp(spec) = &mut loaded.decs[at.ctx] {
                spec.begin(&tokens)
                    .map_err(|e| GenError::Generate(format!("mtp begin: {e}")))?;
            }
        }

        let t_prefill = Instant::now();

        // --- windowed-SWA checkpoints -----------------------------------------
        // Up to two capture points, each taken DURING this turn's prefill by
        // splitting it there, so every snapshot sees a window that is complete
        // by construction (decode reached it contiguously). Cost: one
        // PARTIAL_ONLY state copy to host RAM per point.
        //
        //  - The RING point sits near the prompt's TAIL, because that is where
        //    a conversation's next turn diverges: the replayed history
        //    re-renders the last exchange a few tokens differently than the
        //    raw generated text (measured ~15-20 tokens of drift), while
        //    everything earlier is the same template render both times. It is
        //    what turns the rewind guard's full re-prefill into a tail
        //    re-prefill.
        //  - The PINNED point sits at a background slot's learned shared
        //    HEAD, because that is where the next one-shot diverges: each job
        //    replaces everything after the common instruction block, so no
        //    tail checkpoint ever covers it. (Re)captured only while absent
        //    or stale; a matching pin costs nothing per turn.
        //
        // The second entry of a same-position tie is dropped; positions never
        // coincide persistently (the head is fixed, prompt lengths vary), so
        // a skipped capture just retries next turn.
        let mut capture: Vec<(usize, bool)> = Vec::new(); // (position, is_pinned)
        if !self.cfg.swa_full {
            if tokens.len() > SWA_CHECKPOINT_TAIL_MARGIN {
                let p = reuse.max(tokens.len() - SWA_CHECKPOINT_TAIL_MARGIN);
                let dup = loaded
                    .checkpoints
                    .get(&slot)
                    .is_some_and(|r| r.iter().any(|c| c.tokens.len() == p && tokens.starts_with(&c.tokens)));
                if !dup && p < tokens.len() {
                    capture.push((p, false));
                }
            }
            if slot != 0 {
                if let Some(&h) = self.head_len.get(&slot) {
                    let stale = loaded
                        .pinned_heads
                        .get(&slot)
                        .is_none_or(|c| c.tokens.len() != h || !tokens.starts_with(&c.tokens));
                    // `reuse < h`: a state at `h` can only be captured while
                    // decode passes through it; a turn that restored AT the
                    // head has nothing to recapture, and a turn reusing past
                    // it will be covered by a later divergent turn's prefill.
                    if stale && reuse < h && h < tokens.len() {
                        capture.push((h, true));
                    }
                }
            }
        }
        capture.sort_by_key(|&(p, _)| p);
        capture.dedup_by_key(|e| e.0);

        let mut prefilled = reuse;
        for &(cap, pin) in &capture {
            // Decode up to the capture point first (no-op when the reuse
            // boundary already IS the capture point, i.e. an appended turn
            // whose fresh tail is shorter than the margin).
            if cap > prefilled {
                Self::prefill_span(loaded, at, &tokens, prefilled, cap, draft_slots, cancel.as_ref())?;
                prefilled = cap;
            }
            // Best-effort: a turn without a checkpoint is a turn that may pay
            // a full re-prefill later, not a wrong turn.
            match loaded.decs[at.ctx]
                .ctx()
                .state_seq_get(at.seq, LlamaStateSeqFlags::PARTIAL_ONLY)
            {
                Ok(state) => {
                    let ckpt = SwaCheckpoint {
                        tokens: tokens[..cap].to_vec(),
                        state,
                    };
                    if pin {
                        eprintln!(
                            "llama_cpp: pinned the {cap}-token shared head on slot {slot}"
                        );
                        loaded.pinned_heads.insert(slot, ckpt);
                    } else {
                        let ring = loaded.checkpoints.entry(slot).or_default();
                        ring.push_front(ckpt);
                        ring.truncate(SWA_CHECKPOINTS_PER_SLOT);
                    }
                }
                Err(e) => eprintln!("llama_cpp: checkpoint capture failed on slot {slot} ({e})"),
            }
        }

        let mut batch = Self::prefill_span(
            loaded,
            at,
            &tokens,
            prefilled,
            tokens.len(),
            draft_slots,
            cancel.as_ref(),
        )?;
        // Building the prefix IS prefill, so it is reported as prefill. Hiding it
        // would make the turn that pays for the file look free and every later
        // turn look unchanged, which is the opposite of what the number is for.
        let prefill_ms = t_prefill.elapsed().as_millis() as u64 + prefix_ms;

        // --- decode ---------------------------------------------------------
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
        // While the budget sampler is FORCING, the next token is dictated, not
        // sampled: upstream expresses that by masking every other logit and
        // this loop, being greedy, takes the dictated token directly. The
        // logits at the position still exist (the decode ran), so behavior is
        // identical to upstream's mask + argmax.
        let forced_next = |rb: &Option<ffi::ReasoningBudget>| -> Option<llama_cpp_2::token::LlamaToken> {
            rb.as_ref()
                .filter(|rb| rb.forcing())
                .and_then(|rb| rb.next_forced())
                .map(llama_cpp_2::token::LlamaToken)
        };
        // `next` is the token that has been decided but not yet written into the
        // KV cache. Both loops below share this invariant.
        let mut next = match forced_next(&rbudget) {
            Some(t) => t,
            None => sample_with_grammar(
                loaded.decs[at.ctx].ctx(),
                &mut sampler,
                grmr.as_mut(),
                batch.n_tokens() - 1,
            ),
        };

        // NO explicit `sampler.accept` anywhere in this loop, and that is a
        // correctness invariant, not an omission: `llama_sampler_sample`
        // already accepts the token it returns into the chain (upstream
        // llama-sampler.cpp, end of `llama_sampler_sample`), so every token
        // this loop commits has been accepted exactly once at the moment it
        // was sampled. An explicit accept on top is a DOUBLE accept — a no-op
        // for stateless greedy, which is how it sat here harmlessly, but state
        // corruption for the grammar sampler: the second accept of a
        // value-completing token walks the grammar past its end, llama-cpp-2's
        // `accept` swallows the C++ exception that reports it (`let _ =
        // try_accept(...)`), and the NEXT sample aborts the process on
        // `GGML_ASSERT(!stacks.empty())`. Found by the json_schema test the
        // first time it ran.
        let mut cancelled = false;
        'gen: while output_tokens < MAX_OUTPUT_TOKENS {
            // One check per token, which bounds a stop to a single decode step
            // (a speculative window at most). Here rather than anywhere else in
            // the loop because of the invariant this point has and no other
            // does: `next` is sampled but not yet in the cache, and the previous
            // iteration has already rolled back its rejected drafts, so `cached`
            // and the KV agree exactly. Nothing needs repairing, and the
            // write-back below is the same line a completed turn runs.
            if cancel.as_ref().is_some_and(CancelHandle::is_cancelled) {
                cancelled = true;
                break 'gen;
            }
            // The end-of-turn token stays in the text: the parser's grammar
            // matches a complete turn. See failure mode 3 in the module docs.
            let piece = loaded
                .model
                .token_to_str(next, Special::Tokenize)
                .map_err(|e| GenError::Generate(e.to_string()))?;
            raw.push_str(&piece);
            output_tokens += 1;
            // Every committed token feeds the budget's state machine, exactly
            // once, in emission order: upstream's `common_sampler_accept`
            // contract. The drafts committed in the speculative arm below do
            // their own accepts; `next` is accepted only here.
            if let Some(rb) = rbudget.as_mut() {
                rb.accept(next.0);
            }
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
            //
            // A speculative turn degrades to single-token steps while the
            // budget sampler is forcing (the drafter cannot propose a dictated
            // sequence) and just before exhaustion (a window must not straddle
            // the boundary, or it would commit thinking tokens the mask had
            // already cut off). The degradation is a handful of tokens per
            // turn; the drafter is kept in sync through them below, so
            // speculation resumes for the ANSWER: the part still worth
            // accelerating.
            let spec_now = speculative
                && rbudget
                    .as_ref()
                    .map_or(true, |rb| !rb.forcing() && !rb.near_exhaustion(draft_window));
            match (spec_now, &mut loaded.decs[at.ctx]) {
                // --- plain: one token in, one token out ---------------------
                (false, dec) => {
                    let ctx = dec.ctx();
                    batch.clear();
                    batch
                        .add(next, n_cur, &[at.seq], true)
                        .map_err(|e| GenError::Generate(e.to_string()))?;
                    cached.push(next);
                    n_cur += 1;
                    ctx.decode(&mut batch)
                        .map_err(|e| GenError::Generate(e.to_string()))?;
                    // A speculative TURN passing through here (budget forcing
                    // or near exhaustion) must keep feeding the drafter: MTP
                    // reads the target's hidden states, and a gap would
                    // degrade every window after forcing ends.
                    if speculative {
                        if let Decoder::Mtp(spec) = dec {
                            spec.process(&batch)
                                .map_err(|e| GenError::Generate(format!("mtp process: {e}")))?;
                        }
                    }
                    next = match forced_next(&rbudget) {
                        Some(t) => t,
                        None => sample_with_grammar(
                            dec.ctx(),
                            &mut sampler,
                            grmr.as_mut(),
                            batch.n_tokens() - 1,
                        ),
                    };
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

                    // Emit what was accepted. These are committed tokens, so
                    // they must run the same EOG and cap checks as any other.
                    // A committed draft needs no sampler.accept either: it is
                    // only committed when the verify sample returned the same
                    // token, and THAT sample already accepted it (see the
                    // loop-level invariant above).
                    //
                    // `committed` can stop short of `accepted`: the budget
                    // sampler transitions per COMMITTED token, and once it
                    // starts forcing, the drafts after that point are thinking
                    // tokens upstream's mask would have cut off: they are
                    // treated exactly like rejected proposals. The pre-window
                    // `near_exhaustion` guard makes this rare (a mid-window
                    // activation with a tiny budget), not impossible.
                    let mut committed = 0usize;
                    let mut budget_cut = false;
                    // A turn that ends inside an accepted draft leaves this
                    // loop by flag, not by `break 'gen`, and that is the whole
                    // point: leaving 'gen from in here jumped over the KV
                    // rollback below, so the proposals this window did NOT
                    // commit stayed in the cache at positions `cached` (written
                    // back at the end of the turn) does not know about. The next
                    // turn then read that record, found its prompt extends it,
                    // took the `reuse == prior_len` fast path that trims
                    // nothing, and decoded on top of the stale cells. Both cells
                    // carry the same sequence and position, so attention sees
                    // both: a corrupted answer, not a slow one.
                    let mut end_turn = false;
                    for d in drafts.iter().take(accepted) {
                        let piece = loaded
                            .model
                            .token_to_str(*d, Special::Tokenize)
                            .map_err(|e| GenError::Generate(e.to_string()))?;
                        raw.push_str(&piece);
                        output_tokens += 1;
                        cached.push(*d);
                        committed += 1;
                        if let Some(rb) = rbudget.as_mut() {
                            rb.accept(d.0);
                        }
                        if loaded.model.is_eog_token(*d) {
                            truncated = false;
                            // A turn can end inside an ACCEPTED draft, so the
                            // marker has to be captured here too, not only on
                            // the directly sampled path above.
                            stop_piece = Some(piece);
                            end_turn = true;
                            break;
                        }
                        if output_tokens >= MAX_OUTPUT_TOKENS {
                            end_turn = true;
                            break;
                        }
                        if rbudget.as_ref().is_some_and(|rb| rb.forcing()) {
                            budget_cut = true;
                            break;
                        }
                    }

                    // Rejected proposals are still sitting in the KV cache, so
                    // drop them: keeping them would corrupt every later position.
                    // A no-op when everything was accepted. A budget cut rolls
                    // back the same way: the surviving drafts were never
                    // committed, so the cache must not carry them. Every exit
                    // from the commit loop above reaches this, including the two
                    // that end the turn, which is what keeps the loop-top
                    // invariant ("`cached` and the KV agree exactly") true for
                    // the last window as well as every earlier one.
                    n_cur = pos_of_next + 1 + committed as i32;
                    spec.target_context_mut()
                        .kv_cache_seq_rm(0, Some(n_cur as u32), None)
                        .map_err(|e| GenError::Generate(format!("kv rollback: {e}")))?;
                    // Told the COMMITTED count, after the commit loop, so a
                    // budget cut reports the same boundary the KV rollback
                    // just enforced.
                    spec.accept(committed as u16)
                        .map_err(|e| GenError::Generate(format!("mtp accept: {e}")))?;
                    if end_turn {
                        break 'gen;
                    }
                    next = if budget_cut {
                        // The forced close begins at the cut, dictated rather
                        // than sampled; `follow` belonged to the discarded
                        // continuation.
                        forced_next(&rbudget).unwrap_or(follow)
                    } else {
                        follow
                    };
                }
                // `speculative` is only true when the decoder IS Mtp.
                (true, Decoder::Plain(_)) => unreachable!("speculative implies a drafter"),
            }
        }
        loaded.cached.insert(slot, cached);
        // After the write-back, so the tokens this turn did decode stay
        // reusable: an identical retry reuses everything up to its divergence
        // exactly as it would after a completed turn. Before the parse, because
        // a half-written turn is not a turn and must never reach a caller as
        // one.
        if cancelled {
            return Err(GenError::Cancelled);
        }
        let decode_ms = t_decode.elapsed().as_millis() as u64;
        if draft_proposed > 0 {
            eprintln!(
                "llama_cpp: mtp accepted {draft_accepted}/{draft_proposed} drafted tokens ({:.0}%)",
                draft_accepted as f64 / draft_proposed as f64 * 100.0
            );
        }

        // --- parse ----------------------------------------------------------
        // The end-of-turn marker comes OFF before the parse, not after it.
        //
        // Failure mode 3 in the module docs says the marker must stay in, and
        // that was read off Gemma 4, where it happens to be harmless: the
        // gemma4 PEG rule ends at `zero_or_more(message) + scan_to_toolcall +
        // tool_call` with NO `p.end()`, so an unconsumed `<turn|>` never fails
        // the match, it just gets swallowed into `content` and stripped below.
        // Tolerated, not required — which is not the same claim.
        //
        // The LFM2 / LFM2.5 rule DOES end with `p.end()`, so the mandatory
        // `<|im_end|>` fails the strict parse on every well-formed turn, and
        // the salvage retry recovers content but not tool calls. Measured on
        // LFM2.5-2.6B q4_0: 28 of 30 bench cases died as
        // `GenError::Generate("chat parse: ... at pos N: <|im_end|>")`.
        //
        // Both stop kinds are model-supplied (template-declared stops, then
        // the rendered text of the token that ended the turn), so nothing here
        // hardcodes a family's marker.
        let mut parse_input = raw.as_str();
        for stop in applied.additional_stops.iter().chain(stop_piece.iter()) {
            if let Some(stripped) = parse_input.strip_suffix(stop.as_str()) {
                parse_input = stripped;
            }
        }
        let parsed = self
            .chat
            .as_mut()
            .expect("chat is set with loaded")
            .parse(parse_input)
            .map_err(|e| GenError::Generate(format!("chat parse: {e}")))?;

        if parsed.salvaged {
            eprintln!(
                "llama_cpp: chat parse salvaged from partial ({} tool call(s), {} content chars); \
                 the tail of this generation did not parse",
                parsed.tool_calls.len(),
                parsed.content.len()
            );
        }

        // THE SILENT DROP, which is worse than the throw the salvage above
        // catches. The PEG parser runs LENIENT, so a match that consumes the
        // reasoning block and then fails to make sense of the tool call does
        // not raise: it returns a message with no calls and no content, and the
        // turn evaporates. No error, no warning, nothing for the user.
        //
        // Reproduced on 2026-08-13 by bench case `note_long_multiline` (E4B,
        // shipping config, greedy so deterministic): the model reasoned its way
        // to the right tool and the right arguments across 237 output tokens,
        // then produced `toolName: null`, `args: null`, empty content, and no
        // exception. The body it was asked to write contains the template's own
        // string delimiter, which is the second mechanism described in
        // llama.cpp #25986.
        //
        // Reasoning is present and output tokens were spent, so "the model said
        // nothing" is not a possible reading. Fail loudly instead: an error the
        // agent loop can see and retry beats a blank turn the user cannot
        // distinguish from a refusal.
        if parsed.tool_calls.is_empty() && parsed.content.trim().is_empty() && !raw.trim().is_empty()
        {
            return Err(GenError::Generate(format!(
                "chat parse produced neither a tool call nor content from {} raw chars \
                 ({} chars of reasoning); the generation did not match the template's \
                 grammar and was silently dropped",
                raw.len(),
                parsed.reasoning_content.len()
            )));
        }

        let tool_calls: Vec<ToolCall> = parsed
            .tool_calls
            .into_iter()
            .map(|c| ToolCall {
                name: c.name,
                // Upstream hands arguments back as a JSON string; a caller
                // wants the object. A non-object here is a malformed call, so
                // it becomes Null rather than being silently dropped.
                arguments: serde_json::from_str(&c.arguments).unwrap_or(serde_json::Value::Null),
                // Carried so the host can put it on the matching tool result.
                // Dropping it here would make `Msg::tool_call_id` unfillable
                // from a real turn, which is the only way it is ever filled.
                id: c.id,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every field the persisted state depends on, at a non-default value, so a
    /// test can flip exactly one and see the key move.
    fn cfg() -> LlamaCppConfig {
        LlamaCppConfig {
            model_path: PathBuf::from("/models/a.gguf"),
            draft_path: Some(PathBuf::from("/models/draft.gguf")),
            n_ctx: 16384,
            n_slots: 2,
            n_gpu_layers: 1000,
            enable_thinking: true,
            // Non-default (default is false), like every other field here.
            swa_full: true,
            prefix_cache_dir: Some(PathBuf::from("/cache")),
            state_key: "model-sha-aaa".into(),
            // Non-default (default is F16), like every other field here.
            cache_type_k: KvCacheType::Q8_0,
            cache_type_v: KvCacheType::Q8_0,
            ..Default::default()
        }
    }

    /// The key is what stops a state file being opened by a configuration it was
    /// not computed for, and a wrong answer here is not an error: llama.cpp
    /// validates the SHAPE of a saved state and nothing about which weights
    /// produced it, so a mismatched file deserializes happily and answers
    /// plausible nonsense. Each of these fields changes the tokens or the cache
    /// layout, so each MUST change the file name.
    #[test]
    fn every_input_that_changes_the_state_changes_the_key() {
        let base = LlamaCppEngine::prefix_key_for(&cfg());

        let mut c = cfg();
        c.state_key = "model-sha-bbb".into();
        assert_ne!(base, LlamaCppEngine::prefix_key_for(&c), "state_key");

        let mut c = cfg();
        c.model_path = PathBuf::from("/models/b.gguf");
        assert_ne!(base, LlamaCppEngine::prefix_key_for(&c), "model_path");

        let mut c = cfg();
        c.n_ctx = 8192;
        assert_ne!(base, LlamaCppEngine::prefix_key_for(&c), "n_ctx");

        let mut c = cfg();
        c.n_slots = 1;
        assert_ne!(base, LlamaCppEngine::prefix_key_for(&c), "n_slots");

        let mut c = cfg();
        c.n_gpu_layers = 0;
        assert_ne!(base, LlamaCppEngine::prefix_key_for(&c), "n_gpu_layers");

        let mut c = cfg();
        c.enable_thinking = false;
        assert_ne!(base, LlamaCppEngine::prefix_key_for(&c), "enable_thinking");

        // The SWA cache's size is part of the saved state's layout.
        let mut c = cfg();
        c.swa_full = false;
        assert_ne!(base, LlamaCppEngine::prefix_key_for(&c), "swa_full");

        // The drafter shares the target's KV memory, so attaching or removing
        // one changes what is in the cache.
        let mut c = cfg();
        c.draft_path = None;
        assert_ne!(base, LlamaCppEngine::prefix_key_for(&c), "drafter removed");

        let mut c = cfg();
        c.draft_path = Some(PathBuf::from("/models/other-draft.gguf"));
        assert_ne!(base, LlamaCppEngine::prefix_key_for(&c), "drafter swapped");

        // The cache's element type sizes every entry in it, so it is part of
        // the saved layout exactly as `swa_full` is. K and V are separately
        // settable and must be separately keyed.
        let mut c = cfg();
        c.cache_type_k = KvCacheType::F16;
        assert_ne!(base, LlamaCppEngine::prefix_key_for(&c), "cache_type_k");

        let mut c = cfg();
        c.cache_type_v = KvCacheType::F16;
        assert_ne!(base, LlamaCppEngine::prefix_key_for(&c), "cache_type_v");
    }

    /// The mirror of the above, and the half that makes the cache worth having:
    /// an unchanged configuration must resolve to the SAME file every launch, or
    /// nothing is ever reused and every start rebuilds.
    #[test]
    fn an_unchanged_configuration_keeps_its_key() {
        assert_eq!(
            LlamaCppEngine::prefix_key_for(&cfg()),
            LlamaCppEngine::prefix_key_for(&cfg())
        );
    }

    /// Where the file lives is policy the host sets; moving the directory must
    /// not rename the file inside it, or a host that relocates its cache silently
    /// orphans every prefix it has ever built.
    #[test]
    fn the_directory_is_not_part_of_the_key() {
        let mut c = cfg();
        c.prefix_cache_dir = Some(PathBuf::from("/somewhere/else"));
        assert_eq!(
            LlamaCppEngine::prefix_key_for(&cfg()),
            LlamaCppEngine::prefix_key_for(&c)
        );
    }

    /// A background context is not part of slot 0's state. Keying on it would
    /// mean that retuning the background window discards the agent's persisted
    /// prefix and pays a ~27s cold prefill on the next launch, for a file that
    /// is still perfectly valid: the prefix is saved out of and restored into
    /// the base context, and an extra context is a separate `llama_context`.
    #[test]
    fn an_extra_context_does_not_move_the_prefix_key() {
        let mut c = cfg();
        c.extra_contexts = vec![ContextSpec {
            n_ctx: 4096,
            n_slots: 1,
            n_ubatch: Some(128),
        }];
        assert_eq!(
            LlamaCppEngine::prefix_key_for(&cfg()),
            LlamaCppEngine::prefix_key_for(&c)
        );
    }

    /// The published artifact's shape id moves when the WINDOW moves.
    ///
    /// This is the regression that armed a live hazard on 2026-08-20: the
    /// desktop went from 16,384 to 65,536 while the publish path said `v1`
    /// either way, so a 16k-shaped state was addressable at the URL a 64k
    /// install fetches. The URL now carries this id on both sides, and the
    /// property that makes it worth carrying is exactly this one.
    #[test]
    fn the_shape_id_moves_with_the_window() {
        let mut c = cfg();
        c.n_ctx = 65536;
        assert_ne!(
            LlamaCppEngine::prefix_shape_id(&cfg()),
            LlamaCppEngine::prefix_shape_id(&c)
        );
    }

    /// ...and does NOT move for a background context, for the same reason
    /// [`an_extra_context_does_not_move_the_prefix_key`] gives. The two hashes
    /// read the same fields and must agree about which ones; a published
    /// artifact that went stale whenever a host retuned a background window
    /// would send every install back to a cold build for nothing.
    #[test]
    fn an_extra_context_does_not_move_the_shape_id() {
        let mut c = cfg();
        c.extra_contexts = vec![ContextSpec {
            n_ctx: 4096,
            n_slots: 1,
            n_ubatch: Some(128),
        }];
        assert_eq!(
            LlamaCppEngine::prefix_shape_id(&cfg()),
            LlamaCppEngine::prefix_shape_id(&c)
        );
    }

    /// Slot numbers run consecutively ACROSS contexts, so a host that adds a
    /// second context does not renumber the slots it was already sending: the
    /// desktop keeps its agent on slot 0 and its background work on slot 1
    /// either way, and only the window changes.
    #[test]
    fn slots_number_consecutively_across_contexts() {
        let mut c = cfg();
        c.n_ctx = 65536;
        c.n_slots = 1;
        c.extra_contexts = vec![ContextSpec {
            n_ctx: 16384,
            n_slots: 2,
            n_ubatch: None,
        }];
        let e = LlamaCppEngine::new(c);
        let specs = e.context_specs();
        let map = LlamaCppEngine::slot_map(&specs);
        assert_eq!(
            map.iter()
                .map(|a| (a.slot, a.ctx, a.seq, a.n_ctx))
                .collect::<Vec<_>>(),
            vec![(0, 0, 0, 65536), (1, 1, 0, 16384), (2, 1, 1, 16384)],
        );
    }

    /// One context's slots are all the same size — llama.cpp divides its KV
    /// evenly across sequences — so the TOTAL is what gets scaled, and a host
    /// that asks for two 16K slots must not silently get two 8K ones.
    #[test]
    fn a_contexts_window_is_per_slot_and_the_total_is_scaled() {
        let mut c = cfg();
        c.n_ctx = 16384;
        c.n_slots = 2;
        c.extra_contexts = vec![ContextSpec {
            n_ctx: 4096,
            n_slots: 1,
            n_ubatch: None,
        }];
        let e = LlamaCppEngine::new(c);
        let specs = e.context_specs();
        let p: Vec<_> = specs
            .iter()
            .map(|s| e.ctx_params(s))
            .map(|p| (p.n_ctx().map(|n| n.get()), p.n_batch()))
            .collect();
        assert_eq!(p, vec![(Some(32768), 16384), (Some(4096), 4096)]);
    }

    #[test]
    fn no_cache_directory_means_no_path() {
        let mut c = cfg();
        c.prefix_cache_dir = None;
        assert!(LlamaCppEngine::prefix_path_for(&c).is_none());
        let p = LlamaCppEngine::prefix_path_for(&cfg()).expect("configured");
        assert_eq!(p.parent().unwrap(), Path::new("/cache"));
        assert!(p.file_name().unwrap().to_string_lossy().starts_with("prefix-"));
        assert_eq!(p.extension().unwrap(), "kv");
    }

    /// Upstream treats a non-object `json_schema` as "no response format" and
    /// renders an ordinary unconstrained turn (`has_response_format =
    /// inputs.json_schema.is_object()` in every family handler). For a caller
    /// that asked for a guarantee, silence is the failure — the engine must
    /// reject the request instead, before the weights load.
    #[test]
    fn a_non_object_schema_is_rejected_not_silently_ignored() {
        use crate::generation::GenerationEngine;
        let mut e = LlamaCppEngine::new(LlamaCppConfig {
            model_path: PathBuf::from("/nonexistent/never-loaded.gguf"),
            ..Default::default()
        });
        let err = e
            .generate(GenerateRequest {
                system: "s".into(),
                messages: vec![Msg::user("hi")],
                json_schema: Some(serde_json::json!(["not", "an", "object"])),
                ..Default::default()
            })
            .expect_err("a non-object schema must be rejected");
        assert!(
            matches!(err, GenError::Invalid(_)),
            "expected Invalid (validation before load), got: {err}"
        );
    }

    /// A request that is malformed on its face must be rejected before the
    /// weights load: `Invalid`, not a load error from the bogus path — and per
    /// the wrapper's contract, without unloading anything.
    #[test]
    fn json_schema_with_tools_is_rejected_before_any_load() {
        use crate::generation::{GenerationEngine, ToolSchema};
        let mut e = LlamaCppEngine::new(LlamaCppConfig {
            model_path: PathBuf::from("/nonexistent/never-loaded.gguf"),
            ..Default::default()
        });
        let err = e
            .generate(GenerateRequest {
                system: "s".into(),
                messages: vec![Msg::user("hi")],
                tools: vec![ToolSchema {
                    name: "t".into(),
                    description: "d".into(),
                    parameters: serde_json::json!({"type": "object"}),
                }],
                json_schema: Some(serde_json::json!({"type": "object"})),
                ..Default::default()
            })
            .expect_err("tools + json_schema must be rejected");
        assert!(
            matches!(err, GenError::Invalid(_)),
            "expected Invalid (validation before load), got: {err}"
        );
    }

    /// `sweep_stale_prefixes` deletes superseded state files, and it runs in a
    /// directory that also holds multi-gigabyte weights. It must remove only what
    /// it wrote.
    #[test]
    fn the_sweep_removes_old_prefixes_and_nothing_else() {
        let dir = std::env::temp_dir().join("faro-prefix-sweep-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let keep = dir.join("prefix-keepme.kv");
        for name in [
            "prefix-keepme.kv",
            "prefix-old.kv",
            "prefix-older.kv.tmp",
            "weights.gguf",
            "prefix-inputs.json",
            "notes.txt",
        ] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }

        LlamaCppEngine::sweep_stale_prefixes(&keep);

        assert!(keep.exists(), "the live prefix must survive");
        assert!(!dir.join("prefix-old.kv").exists(), "a superseded prefix must go");
        assert!(!dir.join("prefix-older.kv.tmp").exists(), "a half-written prefix must go");
        // The weights are the expensive thing in this directory; a sweep that
        // took them would cost a multi-gigabyte re-download.
        assert!(dir.join("weights.gguf").exists(), "weights must never be touched");
        // The host keeps its own records here too.
        assert!(dir.join("prefix-inputs.json").exists(), "host sidecars must survive");
        assert!(dir.join("notes.txt").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
