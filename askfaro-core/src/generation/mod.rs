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
//! };
//! assert_eq!(req.tools.len(), 1);
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(all(target_os = "macos", feature = "apple-fm"))]
pub mod apple_fm;

#[cfg(all(target_os = "macos", feature = "apple-fm"))]
pub use apple_fm::AppleFmEngine;

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
