//! Apple Foundation Models provider (macOS/iOS 26+).
//!
//! Compiled only under `cfg(all(target_os = "macos", feature = "apple-fm"))`. The
//! heavy lifting lives in a vendored Swift bridge (`swift/Sources/AppleFM`) that
//! this module calls over a `swift-rs` C-ABI seam: a JSON request goes in, a JSON
//! response comes back. The Swift side builds a per-call `DynamicGenerationSchema`
//! from each tool's JSON Schema, drives a `LanguageModelSession`, and reads tool
//! calls back out of the transcript — the technique validated in the F-7 spike.
//!
//! The system language model is a process-resident singleton on Apple platforms,
//! so it stays warm across calls once first used. [`AppleFmEngine`] therefore does
//! no work on construction; the model is touched lazily on the first
//! [`generate`](GenerationEngine::generate).

use serde::Deserialize;
use swift_rs::{swift, SRString};

use crate::generation::{Availability, GenError, GenerateRequest, GenerateResponse, GenerationEngine, ToolCall};

swift!(fn afm_availability() -> SRString);
swift!(fn afm_generate(request_json: &SRString) -> SRString);

/// On-device generation backed by Apple Foundation Models.
///
/// Keep one instance alive and call [`generate`](GenerationEngine::generate)
/// repeatedly; the underlying model stays warm.
pub struct AppleFmEngine {
    /// Whether a generation has run yet (the model was touched). Purely
    /// informational — the model singleton lives in the Swift runtime.
    warmed: bool,
}

impl AppleFmEngine {
    /// Cheap to construct; loads nothing. Check
    /// [`availability`](GenerationEngine::availability) first.
    pub fn new() -> Self {
        AppleFmEngine { warmed: false }
    }

    /// True once a generation has run (and the model is resident).
    pub fn is_warm(&self) -> bool {
        self.warmed
    }
}

impl Default for AppleFmEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// What the Swift bridge returns: either an `error` marker or a full response.
#[derive(Deserialize)]
struct RawResponse {
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    text: String,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
    #[serde(default)]
    abstained: bool,
    #[serde(default)]
    model_ms: u64,
}

/// Sentinel the Swift side emits for `LanguageModelSession`'s
/// `exceededContextWindowSize`, mapped to a typed error so callers can shrink the
/// tool subset and retry.
const ERR_CONTEXT_WINDOW: &str = "context_window_exceeded";

fn parse_availability(raw: &str) -> Availability {
    match raw.split_once(':') {
        Some(("unsupported", reason)) => Availability::Unsupported(reason.to_string()),
        Some(("notenabled", reason)) => Availability::NotEnabled(reason.to_string()),
        _ if raw == "available" => Availability::Available,
        // Any unexpected shape is treated as unsupported rather than crashing.
        _ => Availability::Unsupported(format!("unrecognized availability: {raw}")),
    }
}

impl GenerationEngine for AppleFmEngine {
    fn availability() -> Availability {
        let raw = unsafe { afm_availability() };
        parse_availability(raw.as_str())
    }

    fn generate(&mut self, req: GenerateRequest) -> Result<GenerateResponse, GenError> {
        match Self::availability() {
            Availability::Available => {}
            Availability::NotEnabled(r) | Availability::Unsupported(r) => {
                return Err(GenError::Unavailable(r))
            }
        }

        let request_json =
            serde_json::to_string(&req).map_err(|e| GenError::Invalid(e.to_string()))?;
        let input: SRString = request_json.as_str().into();
        let raw = unsafe { afm_generate(&input) };
        self.warmed = true;

        let parsed: RawResponse = serde_json::from_str(raw.as_str())
            .map_err(|e| GenError::Generate(format!("malformed bridge response: {e}")))?;

        if let Some(err) = parsed.error {
            return if err == ERR_CONTEXT_WINDOW {
                Err(GenError::ContextWindowExceeded)
            } else {
                Err(GenError::Generate(err))
            };
        }

        Ok(GenerateResponse {
            text: parsed.text,
            tool_calls: parsed.tool_calls,
            abstained: parsed.abstained,
            model_ms: parsed.model_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_parsing() {
        assert_eq!(parse_availability("available"), Availability::Available);
        assert_eq!(
            parse_availability("notenabled:apple intelligence is off"),
            Availability::NotEnabled("apple intelligence is off".into())
        );
        assert_eq!(
            parse_availability("unsupported:device not eligible"),
            Availability::Unsupported("device not eligible".into())
        );
    }

    /// Smoke test: requires macOS 26 with Apple Intelligence enabled. Skips
    /// cleanly (does not fail) when the model isn't available on the box, so CI
    /// on an un-provisioned runner stays green; on a provisioned machine it
    /// asserts a tool call comes back and no context-window overflow occurs.
    #[test]
    fn smoke_two_tool_subset_returns_tool_call() {
        use crate::generation::{Msg, ToolSchema};
        use serde_json::json;

        match AppleFmEngine::availability() {
            Availability::Available => {}
            other => {
                eprintln!("skipping Apple FM smoke test: {other:?}");
                return;
            }
        }

        let tools = vec![
            ToolSchema {
                name: "scope_task".into(),
                description: "Update an existing task: complete it, change priority, reschedule, or delete it.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "The task id, e.g. t_8f3a"},
                        "status": {"type": "string", "enum": ["completed", "in_progress", "cancelled"]}
                    },
                    "required": ["task_id"]
                }),
            },
            ToolSchema {
                name: "scope_contact".into(),
                description: "Create or update a CRM contact record.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"]
                }),
            },
        ];

        let mut engine = AppleFmEngine::new();
        let resp = engine
            .generate(GenerateRequest {
                system: "You manage tasks and a CRM. Call exactly one tool when the user asks for an action a tool handles. Use the exact ids the user gives.".into(),
                messages: vec![Msg {
                    role: "user".into(),
                    content: "Mark task t_8f3a as completed.".into(),
                }],
                tools,
            })
            .expect("generation should not error with a 2-tool subset");

        assert!(
            !resp.tool_calls.is_empty(),
            "expected a tool call, got text: {:?}",
            resp.text
        );
        assert_eq!(resp.tool_calls[0].name, "scope_task");
        assert!(engine.is_warm());
    }
}
