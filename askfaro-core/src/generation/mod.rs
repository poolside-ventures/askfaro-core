//! # askfaro-core::generation
//!
//! On-device text generation + tool-calling, provider-agnostic. This crate owns
//! the [`GenerationEngine`] contract and the OpenAI-shaped request/response types
//! that consuming apps (the on-device agent) speak; concrete providers are opt-in
//! behind cargo features.
//!
//! The default build is model-free — just serde + thiserror, no platform deps —
//! so it cross-compiles unchanged and a host can depend on the types without
//! pulling a model runtime. Concrete engines are opt-in behind features; today
//! that means `llama-cpp`.
//!
//! An Apple Foundation Models provider lived here until 2026-08-01. Its 4,096-
//! token window could not hold a real tool registry, and its Swift bridge
//! flattened every message to `"role: content"`, so it could not replay a tool
//! call even in principle. It was removed rather than kept as a second engine
//! that no consumer could actually run.
//!
//! Tool *selection* is deliberately out of scope: the caller passes only the
//! already-chosen tool subset in [`GenerateRequest::tools`]. The companion
//! `progressive` module produces that subset from a catalog.
//!
//! ```
//! use askfaro_core::generation::{GenerateRequest, Msg, ToolSchema};
//! use serde_json::json;
//!
//! let req = GenerateRequest {
//!     system: "You are a helpful assistant.".into(),
//!     messages: vec![Msg::user("Mark task t_8f3a done")],
//!     tools: vec![ToolSchema {
//!         name: "scope_task".into(),
//!         description: "Update a task".into(),
//!         parameters: json!({"type": "object", "properties": {"task_id": {"type": "string"}}}),
//!     }],
//!     ..Default::default()
//! };
//! assert_eq!(req.tools.len(), 1);
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// In-process llama.cpp. Not OS-gated: the same code runs on every platform and
/// the accelerator is chosen by the consumer's `llama-cpp-2` features. That is
/// the whole reason it was preferred over an Apple-only runtime, and why it is
/// the one that survived.
#[cfg(feature = "llama-cpp")]
pub mod llama_cpp;

/// Weight specs for the local providers. Gated on `model` (the shared
/// provisioning types) rather than on a specific engine, so a host can offer the
/// download before deciding to compile a runtime in.
#[cfg(feature = "model")]
pub mod models;

#[cfg(feature = "llama-cpp")]
pub use llama_cpp::{ContextSpec, KvCacheType, LlamaCppConfig, LlamaCppEngine, PrefixReport};

/// One conversation turn. `role` is the OpenAI role (`"system"`, `"user"`,
/// `"assistant"`, `"tool"`); the engine maps it to the provider's transcript.
///
/// **A tool call is structured, not prose.** For a year this struct was `role`
/// plus `content`, which meant a host replaying an agent loop had nowhere to put
/// "the assistant called `scopy_task` with these arguments" except inside the
/// text. Scope's desktop wrote `[called scopy_task({...})]` and
/// `[scopy_task result: {...}]` as ordinary message text, and the model, shown
/// bracket prose where its template has real `<|tool_call>` and
/// `<|tool_response>` tokens, wrote bracket prose back: it invented
/// `[tool_response]` blocks full of fabricated records and echoed our own
/// `[called ...]` syntax to the user as their answer. Measured at 0% correct
/// continuation calls across the whole multi-step bench.
///
/// The fields below are what a chat template's tool branches read. They are
/// optional so a plain user/assistant transcript is unchanged, but a host
/// replaying tool use MUST fill them, or the template's tool arms never fire and
/// the failure is silent and fluent.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Msg {
    pub role: String,
    pub content: String,
    /// On an `assistant` turn: the calls it made. Rendered by the template as
    /// real tool-call markup rather than described in `content`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// On a `tool` turn: which tool produced this result.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_name: String,
    /// On a `tool` turn: the id of the call it answers, matching
    /// [`ToolCall::id`]. Templates that pair calls with results by id need it;
    /// ones that pair positionally ignore it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_call_id: String,
}

impl Msg {
    /// A plain user turn.
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into(), ..Default::default() }
    }

    /// A plain assistant turn (text only, no calls).
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into(), ..Default::default() }
    }

    /// An assistant turn that CALLED tools. `content` is whatever text came with
    /// them, usually empty.
    pub fn assistant_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            tool_calls,
            ..Default::default()
        }
    }

    /// A tool result. `content` is the tool's output as the model should read it;
    /// pass the payload itself, not a description of it.
    pub fn tool_result(
        tool_name: impl Into<String>,
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            tool_name: tool_name.into(),
            tool_call_id: tool_call_id.into(),
            ..Default::default()
        }
    }
}

/// An OpenAI function-tool definition. `parameters` is a JSON Schema object — the
/// provider renders it into the model's own chat template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// JSON Schema for the function arguments (`{"type": "object", ...}`).
    pub parameters: Value,
}

/// A generation request. `tools` is the already-selected subset — this crate does
/// not choose tools.
// PartialEq only: `reasoning_close_bias` is an f32, and floats have no total
// equality. Nothing keyed a request by Eq (it was derived, not used).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// System / instructions prompt.
    pub system: String,
    /// Conversation so far, oldest first.
    pub messages: Vec<Msg>,
    /// The tool subset the model may call this turn (may be empty).
    pub tools: Vec<ToolSchema>,
    /// Which KV cache slot this request belongs to. Defaults to 0.
    ///
    /// A host that runs more than one KIND of request against one engine needs
    /// more than one cache. The desktop runs an agent loop with a ~6,000-token
    /// prompt alongside background one-shots of ~340 tokens (reply-intent,
    /// follow-up assessment), and with a single cache each evicts the other:
    /// measured 19,281ms of prefill on a turn that should have reused almost all
    /// of it, and 3,441ms for the 344-token one-shot that displaced it. Two
    /// workloads, one cache, both always cold.
    ///
    /// Slots map to llama.cpp sequences, so their caches are independent and
    /// each keeps its own prefix. Note slot 0 is the only one that speculates:
    /// `MtpSpeculative` binds to sequence 0, and a background one-shot has
    /// nothing to gain from a drafter anyway.
    #[serde(default)]
    pub slot: u32,
    /// Override the engine's `enable_thinking` for THIS request; `None` uses
    /// the engine's configuration.
    ///
    /// This is what lets one engine serve two workloads instead of two
    /// engines holding the same weights: the desktop's chat runs with
    /// thinking on while its memory extraction runs with thinking off (the
    /// proven extraction operating point), and thinking is a property of the
    /// rendered prompt, not of the loaded weights.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// Cap the model's REASONING channel at this many tokens for THIS request;
    /// `None` (or any negative value) leaves thinking unbounded.
    ///
    /// This is upstream llama.cpp's reasoning-budget sampler (the mechanism
    /// behind llama-server's `--reasoning-budget`), not an output cap: when the
    /// budget runs out mid-thought, a short wrap-up message plus the model's
    /// own think-close tag are forced token-by-token, and the ANSWER that
    /// follows is never truncated. `Some(0)` closes the think channel the
    /// moment it opens: per-turn "don't think" semantics WITHOUT changing the
    /// rendered prompt, so unlike flipping [`Self::enable_thinking`] it leaves
    /// the persisted KV prefix and the slot's cache untouched.
    ///
    /// Ignored on engines without a thinking-tag template, and on
    /// grammar-constrained (`json_schema`) turns, whose operating point is
    /// thinking-off already.
    #[serde(default)]
    pub reasoning_budget: Option<i32>,
    /// Additive logit bias on the FIRST token of the model's think-close tag,
    /// applied for the whole turn; `None` applies no bias.
    ///
    /// The complement to [`Self::reasoning_budget`]: the budget bounds the
    /// TAIL deterministically, this shifts the MEDIAN by letting easy
    /// derivations close early (under greedy decode, the close wins wherever
    /// its logit comes within the bias of the argmax). Measured together on
    /// the 26-case Scope bench at budget 128 / bias +4.0: 100% fully correct,
    /// p50 3,468ms to 2,826ms, median thinking halved. On its own a bias
    /// cannot bound anything (p95 was unmoved in every solo run), so ship it
    /// WITH a budget, not instead of one.
    ///
    /// Same guards as the budget: ignored when thinking is off, when the
    /// template declares no think tags, and on grammar-constrained
    /// (`json_schema`) turns.
    #[serde(default)]
    pub reasoning_close_bias: Option<f32>,
    /// Constrain the reply to JSON matching this JSON Schema (an OBJECT, per
    /// upstream), enforced at the SAMPLER: tokens outside the grammar are
    /// masked, so a non-conforming reply is unrepresentable rather than merely
    /// discouraged.
    ///
    /// This exists because asking is not enforcing. The desktop's memory shim
    /// honored OpenAI `response_format` by pasting the schema into the system
    /// prompt, and some Honcho deriver units still answered in prose — and with
    /// greedy decode the argmax continuation is prose EVERY time, so retries
    /// changed nothing and the unit burned in the queue forever. A hosted
    /// backend's `response_format` works because the server constrains decoding;
    /// for a local engine, this field is that server-side enforcement.
    ///
    /// The whole mechanism is upstream's: the schema rides
    /// `common_chat_templates_inputs.json_schema` — the same field llama-server
    /// fills from an OpenAI `response_format` — and the model FAMILY's chat
    /// handler shapes the grammar. Gemma 4's permits an optional thinking
    /// channel before the constrained body and strips its ```json fences at
    /// parse, so `enable_thinking` composes freely with this. Mutually
    /// exclusive with `tools`: upstream's response-format grammar takes
    /// precedence over the call rules, which would leave the sent tools
    /// rendered but unsampleable — rejected as [`GenError::Invalid`] rather
    /// than dropped silently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<Value>,
}

/// A single tool invocation the model emitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    /// Decoded arguments object.
    pub arguments: Value,
    /// The template's own id for this call, when it emitted one.
    ///
    /// Carried so a host can put it back on the matching [`Msg::tool_result`]:
    /// templates that pair a result to its call by id have nothing to pair with
    /// otherwise, and a turn with two calls in flight is exactly where that
    /// stops being cosmetic. Empty for families that pair positionally.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub id: String,
}

/// The model's response. A turn is either tool calls (`tool_calls` non-empty),
/// a text reply (`text` non-empty), or an abstention (`abstained` true, both
/// empty) — the model declined to act and had nothing to say.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    /// True when the model produced neither a tool call nor any text.
    pub abstained: bool,
    /// Wall-clock inference time in milliseconds.
    pub model_ms: u64,

    /// The model's reasoning, separated from `text`.
    ///
    /// Reasoning models emit a distinct channel before the answer, and on Gemma 4
    /// roughly 96% of decode goes here. Folding it into `text` is not cosmetic:
    /// it is how a turn comes back looking empty. Empty for engines whose model
    /// does not reason, or when reasoning was disabled.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning: String,

    /// Where the time went. `model_ms` alone cannot tell a caller *why* a turn
    /// was slow, and every on-device latency decision so far has rested on this
    /// split: fewer tools fixes prefill, fewer output tokens fixes decode,
    /// keep-alive fixes load. Engines fill what they can measure and leave the
    /// rest zero.
    #[serde(default, skip_serializing_if = "Timings::is_empty")]
    pub timings: Timings,
}

/// Per-turn cost attribution. All durations in milliseconds; zero means "not
/// measured by this engine", not "instant".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timings {
    /// Weight load. Non-zero only when the engine had to (re)load the model,
    /// which is why it is reported apart from prefill rather than folded in.
    pub load_ms: u64,
    /// Reading the prompt. Scales with how much you send, so this is the number
    /// tool selection moves.
    pub prefill_ms: u64,
    /// Writing the answer. Scales with how much the model says, so this is the
    /// number a reasoning budget moves.
    pub decode_ms: u64,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    /// True when generation stopped at a cap rather than at end-of-turn, so a
    /// truncated answer is never mistaken for a brief one.
    pub truncated: bool,
    /// Draft tokens proposed by a speculative drafter, and how many the target
    /// model accepted. Both 0 when speculation is off.
    ///
    /// Reported rather than merely logged because acceptance is the number that
    /// says whether speculation is EARNING its place: it decides the speedup,
    /// and it is also the only way to tell "the drafter is working" from "the
    /// drafter silently fell back to plain decode", which is the failure mode
    /// this whole path specialises in. A host that surfaces timings should
    /// surface these in the same place, or the question can only be answered
    /// from a terminal.
    #[serde(default, skip_serializing_if = "crate::generation::is_zero_u32")]
    pub draft_proposed: u32,
    #[serde(default, skip_serializing_if = "crate::generation::is_zero_u32")]
    pub draft_accepted: u32,
}

/// Serde helper: omit speculative counters when speculation is off.
pub(crate) fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

impl Timings {
    /// True when nothing was measured, so the field can be omitted from JSON.
    pub fn is_empty(&self) -> bool {
        *self == Timings::default()
    }
}

/// Whether an engine can run on this device *right now*, cheaply (no model load).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    /// Ready to generate.
    Available,
    /// This build/OS can never run the engine (e.g. wrong OS version, missing
    /// framework). The string is a human-readable reason.
    Unsupported(String),
    /// Supported in principle but not currently usable — the user can fix it
    /// (e.g. Apple Intelligence disabled, model still downloading).
    NotEnabled(String),
}

/// Errors surfaced by an engine. Provider/internal details are flattened to
/// strings so the public API never leaks the underlying runtime's types.
#[derive(Debug, thiserror::Error)]
pub enum GenError {
    /// The engine is not available on this device (see [`Availability`]).
    #[error("generation engine unavailable: {0}")]
    Unavailable(String),
    /// The prompt + tool schemas overflowed the model's context window. Callers
    /// should shrink the tool subset (see the `progressive` module) and retry.
    #[error("context window exceeded")]
    ContextWindowExceeded,
    /// The request was malformed (e.g. a tool's `parameters` was not a JSON
    /// Schema object).
    #[error("invalid request: {0}")]
    Invalid(String),
    /// Inference failed.
    #[error("generation failed: {0}")]
    Generate(String),
    /// The host stopped this call through its [`CancelHandle`].
    ///
    /// A variant of its own rather than a `Generate` string, because "the user
    /// pressed stop" and "inference broke" want opposite responses: nothing to
    /// report, nothing to retry, and no recovery to perform. An engine that
    /// returns this leaves itself usable, weights and KV cache included, so the
    /// next call runs normally.
    #[error("generation cancelled")]
    Cancelled,
}

/// A host's stop signal for work that is already running.
///
/// Cloneable and shared: the host keeps one clone OUTSIDE whatever lock the
/// engine lives behind and gives another to the engine. That is the only shape
/// that works, because an engine held in a `Mutex` cannot be asked to stop
/// THROUGH that mutex: the call to be stopped is holding it. Scope desktop
/// measured turns at p50 16.5s, p90 31.8s and max 65.4s, which is how long
/// quitting waited for the lock it needed to drop the engine.
///
/// The engine clears the flag at the start of every call that can honour it, so
/// a cancel applies to the call it arrives during and never latches into the
/// next one.
///
/// Read that literally: **one `cancel` stops one call.** In the host shape
/// above, an engine behind a lock, the window where a stop is dropped is not an
/// instant but the whole time another caller sits queued on that lock. The
/// in-flight call aborts, releases the lock, and the queued call clears the flag
/// and runs to completion, because nothing here can tell "raised while I was
/// queued" from "left over from a call that already ended". A host that wants a
/// backlog stopped has to stop enqueueing or keep raising it; a host that wants
/// one turn stopped, which is what a stop button is, gets exactly that. Making
/// the engine able to tell the difference means a generation counter or an
/// explicit re-arm, and neither is worth inventing before something needs it.
///
/// A default handle is not armed on anything. It becomes live only where an
/// engine takes one (for llama.cpp, `LlamaCppConfig::cancel`).
#[derive(Debug, Clone, Default)]
pub struct CancelHandle(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl CancelHandle {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask whatever this handle is armed on to stop. Callable from any thread,
    /// any number of times.
    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether a stop has been asked for and not yet consumed.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Withdraw a stop. An engine calls this itself at the start of every call
    /// it can honour, so a host normally never needs to; it is public for the
    /// one case that is not covered, a handle carried from one engine to the
    /// next after a cancel was raised and never consumed.
    ///
    /// `Relaxed` throughout this type: the flag publishes no data, so there is
    /// nothing for an acquire/release pair to order. The only guarantee needed
    /// is that a store becomes visible to the polling thread, which every
    /// ordering gives.
    pub fn clear(&self) {
        self.0.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// A provider-agnostic on-device generation engine.
///
/// Construct an engine cheaply, then keep it alive and call [`generate`] per turn.
/// The model is loaded lazily on the first [`generate`] (and kept warm across
/// calls, like the `stt` module's `SttEngine`) — never on construction.
///
/// [`generate`]: GenerationEngine::generate
pub trait GenerationEngine {
    /// Cheap capability probe — must NOT load the model. Callers check this
    /// before constructing/using the engine.
    fn availability() -> Availability
    where
        Self: Sized;

    /// Run one generation turn. Loads the model on first call if needed.
    fn generate(&mut self, req: GenerateRequest) -> Result<GenerateResponse, GenError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A clone must see the original's stop, or the whole design fails: the
    /// host keeps one clone outside the engine's lock and the engine holds the
    /// other, and they are the same flag or they are useless.
    #[test]
    fn a_cloned_handle_shares_one_flag_across_threads() {
        let host = CancelHandle::new();
        let engine_side = host.clone();
        assert!(!engine_side.is_cancelled());

        let h = std::thread::spawn(move || host.cancel());
        h.join().unwrap();
        assert!(engine_side.is_cancelled());

        // What the engine does at the start of every call it can honour, so a
        // stop never latches into the following one.
        engine_side.clear();
        assert!(!engine_side.is_cancelled());
    }

    #[test]
    fn request_roundtrips_through_json() {
        let req = GenerateRequest {
            system: "sys".into(),
            messages: vec![Msg::user("hi")],
            tools: vec![ToolSchema {
                name: "t".into(),
                description: "d".into(),
                parameters: json!({"type": "object"}),
            }],
            ..Default::default()
        };
        let s = serde_json::to_string(&req).unwrap();
        let back: GenerateRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(req, back);
    }

    /// `json_schema` must roundtrip when set and vanish from the wire when not:
    /// requests are exchanged with hosts that predate the field, and an
    /// ever-present `"json_schema": null` would be a schema change for them.
    #[test]
    fn json_schema_roundtrips_and_is_absent_when_unset() {
        let req = GenerateRequest {
            system: "sys".into(),
            messages: vec![Msg::user("hi")],
            json_schema: Some(json!({"type": "object", "required": ["name"]})),
            ..Default::default()
        };
        let s = serde_json::to_string(&req).unwrap();
        let back: GenerateRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(req, back);

        let plain = serde_json::to_string(&GenerateRequest::default()).unwrap();
        assert!(!plain.contains("json_schema"), "unset field must not serialize: {plain}");
    }

    #[test]
    fn response_default_is_an_abstention_shape() {
        let r = GenerateResponse::default();
        assert!(r.text.is_empty());
        assert!(r.tool_calls.is_empty());
        assert!(!r.abstained); // default false; engines set it explicitly
    }

    #[test]
    fn tool_call_arguments_decode_to_value() {
        let tc = ToolCall {
            name: "scope_task".into(),
            arguments: json!({"task_id": "t_8f3a", "status": "completed"}),
            id: "call_0".into(),
        };
        assert_eq!(tc.arguments["task_id"], "t_8f3a");
    }

    /// The transcript a replayed agent loop produces must carry the CALL, not a
    /// sentence about it. This is the shape that was wrong for a year: tool use
    /// went into `content` as prose, the template's tool branches never fired,
    /// and the model imitated the prose instead of reading the result.
    #[test]
    fn a_replayed_tool_turn_carries_structure_not_prose() {
        let call = ToolCall {
            name: "scope_task".into(),
            arguments: json!({"task_id": "t_11"}),
            id: "call_0".into(),
        };
        let assistant = Msg::assistant_calls("", vec![call.clone()]);
        assert_eq!(assistant.role, "assistant");
        assert_eq!(assistant.tool_calls, vec![call]);
        // The call must NOT have been described in the text.
        assert!(assistant.content.is_empty());

        let result = Msg::tool_result("scope_task", "call_0", r#"{"ok":true}"#);
        assert_eq!(result.role, "tool");
        assert_eq!(result.tool_name, "scope_task");
        assert_eq!(result.tool_call_id, "call_0");
        // The payload itself, so the model reads data rather than a report of it.
        assert_eq!(result.content, r#"{"ok":true}"#);
    }

    /// A plain conversation must serialize exactly as before, or every existing
    /// persisted prefix is invalidated by a field nothing uses.
    #[test]
    fn a_toolless_message_serializes_unchanged() {
        let json = serde_json::to_string(&Msg::user("hi")).unwrap();
        assert_eq!(json, r#"{"role":"user","content":"hi"}"#);
    }
}
