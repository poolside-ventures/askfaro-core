//! # askfaro-core::generation
//!
//! On-device text generation + tool-calling, provider-agnostic. This crate owns
//! the [`GenerationEngine`] contract and the OpenAI-shaped request/response types
//! that consuming apps (the on-device agent) speak; concrete providers are opt-in
//! behind cargo features.
//!
//! The default build is model-free — just serde + thiserror, no platform deps —
//! so it cross-compiles unchanged and a host can depend on the types without
//! pulling a model runtime. The Apple Foundation Models provider lives behind the
//! `apple-fm` feature and compiles only under
//! `cfg(all(target_os = "macos", feature = "apple-fm"))` (see [`apple_fm`]).
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
//!     messages: vec![Msg { role: "user".into(), content: "Mark task t_8f3a done".into() }],
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

#[cfg(all(target_os = "macos", feature = "apple-fm"))]
pub mod apple_fm;

#[cfg(all(target_os = "macos", feature = "apple-fm"))]
pub use apple_fm::AppleFmEngine;

/// In-process llama.cpp. Not OS-gated, unlike `apple_fm`: the same code runs on
/// every platform and the accelerator is chosen by the consumer's `llama-cpp-2`
/// features. That is the whole reason it was preferred over an Apple-only
/// runtime.
#[cfg(feature = "llama-cpp")]
pub mod llama_cpp;

/// Weight specs for the local providers. Gated on `model` (the shared
/// provisioning types) rather than on a specific engine, so a host can offer the
/// download before deciding to compile a runtime in.
#[cfg(feature = "model")]
pub mod models;

#[cfg(feature = "llama-cpp")]
pub use llama_cpp::{LlamaCppConfig, LlamaCppEngine, PrefixReport};

/// One conversation turn. `role` is the OpenAI role (`"system"`, `"user"`,
/// `"assistant"`, `"tool"`); the engine maps it to the provider's transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Msg {
    pub role: String,
    pub content: String,
}

/// An OpenAI function-tool definition. `parameters` is a JSON Schema object — the
/// provider builds its own per-call schema from it (Apple FM uses a
/// `DynamicGenerationSchema`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// JSON Schema for the function arguments (`{"type": "object", ...}`).
    pub parameters: Value,
}

/// A generation request. `tools` is the already-selected subset — this crate does
/// not choose tools.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
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
}

/// A single tool invocation the model emitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    /// Decoded arguments object.
    pub arguments: Value,
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

    #[test]
    fn request_roundtrips_through_json() {
        let req = GenerateRequest {
            system: "sys".into(),
            messages: vec![Msg {
                role: "user".into(),
                content: "hi".into(),
            }],
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
        };
        assert_eq!(tc.arguments["task_id"], "t_8f3a");
    }
}
