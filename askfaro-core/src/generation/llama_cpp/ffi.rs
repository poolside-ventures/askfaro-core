//! Safe Rust over `chat_shim.cpp`.
//!
//! The shim speaks JSON in both directions so no C++ type ever crosses the
//! boundary. It owns the templates handle and the last `common_chat_params`,
//! because `common_chat_parse` needs the format and the compiled PEG arena from
//! the apply step and serializing those would be pointless work.

use std::ffi::{c_char, CStr, CString};

use serde::Deserialize;

use crate::generation::GenerateRequest;

#[repr(C)]
struct ScopeChatCtx {
    _private: [u8; 0],
}

#[repr(C)]
struct ScopeRbudget {
    _private: [u8; 0],
}

extern "C" {
    fn scope_chat_init(template_str: *const c_char) -> *mut ScopeChatCtx;
    fn scope_chat_apply(
        ctx: *mut ScopeChatCtx,
        messages_json: *const c_char,
        tools_json: *const c_char,
        json_schema: *const c_char,
        enable_thinking: bool,
    ) -> *mut c_char;
    fn scope_chat_parse(ctx: *mut ScopeChatCtx, text: *const c_char) -> *mut c_char;
    fn scope_chat_caps(ctx: *mut ScopeChatCtx) -> *mut c_char;
    fn scope_chat_free(p: *mut c_char);
    fn scope_chat_ctx_free(ctx: *mut ScopeChatCtx);
    fn scope_rbudget_init(
        start: *const i32,
        n_start: usize,
        end: *const i32,
        n_end: usize,
        forced: *const i32,
        n_forced: usize,
        budget: i32,
    ) -> *mut ScopeRbudget;
    fn scope_rbudget_accept(s: *mut ScopeRbudget, token: i32);
    fn scope_rbudget_state(s: *const ScopeRbudget) -> i32;
    fn scope_rbudget_free(s: *mut ScopeRbudget);
}

/// `common_reasoning_budget_state`, mirrored by value. The C side is the
/// authority; `scope_rbudget_state` documents the mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RbudgetState {
    Idle,
    Counting,
    Forcing,
    WaitingUtf8,
    Done,
}

impl RbudgetState {
    fn from_c(v: i32) -> Self {
        match v {
            0 => Self::Idle,
            1 => Self::Counting,
            2 => Self::Forcing,
            3 => Self::WaitingUtf8,
            _ => Self::Done,
        }
    }
}

/// Upstream's reasoning-budget sampler (the one behind llama-server's
/// `--reasoning-budget`), owned from Rust.
///
/// The STATE MACHINE is entirely upstream's; this wrapper adds only the two
/// pieces of bookkeeping upstream keeps private and the decode loop needs:
///
///  - `force_idx`: which forced token comes next. Upstream's sampler expresses
///    it by masking logits, but this engine's greedy loop can take the token
///    directly and skip a 262K-candidate apply; the caller built
///    `forced` itself, so mirroring the index is bookkeeping, not a port. It
///    advances exactly when upstream's `force_pos` does (on accept while
///    FORCING).
///  - `remaining`: the countdown, mirrored so the speculative path can refuse
///    to start a draft window that would cross the budget boundary. It moves
///    exactly when upstream's does (on accept while COUNTING, except the
///    accept that matches the natural end).
pub struct ReasoningBudget {
    s: *mut ScopeRbudget,
    forced: Vec<i32>,
    force_idx: usize,
    remaining: i32,
    budget: i32,
}

// Only ever touched behind `&mut self` from the owning engine, same as `Chat`.
unsafe impl Send for ReasoningBudget {}

impl Drop for ReasoningBudget {
    fn drop(&mut self) {
        unsafe { scope_rbudget_free(self.s) };
    }
}

impl ReasoningBudget {
    pub fn new(start: &[i32], end: &[i32], forced: Vec<i32>, budget: i32) -> Option<Self> {
        if start.is_empty() || end.is_empty() {
            return None;
        }
        let s = unsafe {
            scope_rbudget_init(
                start.as_ptr(),
                start.len(),
                end.as_ptr(),
                end.len(),
                forced.as_ptr(),
                forced.len(),
                budget,
            )
        };
        if s.is_null() {
            return None;
        }
        Some(Self { s, forced, force_idx: 0, remaining: budget, budget })
    }

    pub fn state(&self) -> RbudgetState {
        RbudgetState::from_c(unsafe { scope_rbudget_state(self.s) })
    }

    /// Feed one COMMITTED token through the state machine. Call exactly once
    /// per token the decode loop emits, in emission order: the same contract
    /// as upstream's `common_sampler_accept`.
    pub fn accept(&mut self, token: i32) {
        let pre = self.state();
        unsafe { scope_rbudget_accept(self.s, token) };
        match pre {
            // Upstream advances force_pos on every accept while forcing.
            RbudgetState::Forcing => self.force_idx += 1,
            // Upstream decrements on every counting accept EXCEPT the one that
            // completes the natural end sequence (that transitions to DONE
            // before the decrement).
            RbudgetState::Counting => {
                if self.state() != RbudgetState::Done {
                    self.remaining -= 1;
                }
            }
            // Activation (and re-activation on a second think block) resets
            // the countdown, exactly as upstream resets `remaining`.
            RbudgetState::Idle | RbudgetState::Done => {
                if self.state() == RbudgetState::Counting {
                    self.remaining = self.budget;
                }
            }
            RbudgetState::WaitingUtf8 => {}
        }
    }

    /// True while the sampler is force-feeding the close sequence: the next
    /// token is [`Self::next_forced`], not a model sample.
    pub fn forcing(&self) -> bool {
        self.state() == RbudgetState::Forcing
    }

    /// The token upstream's mask would force next. `None` never happens while
    /// [`Self::forcing`] holds (upstream leaves FORCING when the sequence is
    /// exhausted), but the caller treats it as "sample normally" for safety.
    pub fn next_forced(&self) -> Option<i32> {
        self.forced.get(self.force_idx).copied()
    }

    /// Would a speculative window of `window` tokens risk crossing the budget
    /// boundary? The drafter proposes without seeing the budget, so a window
    /// that straddles exhaustion would commit thinking tokens upstream's mask
    /// had already cut off; the caller degrades those steps to plain decode.
    pub fn near_exhaustion(&self, window: i32) -> bool {
        self.state() == RbudgetState::Counting && self.remaining <= window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives the UPSTREAM sampler (no model needed: tokens are just ids) and
    /// checks the mirrored bookkeeping stays in lockstep with its states.
    #[test]
    fn budget_counts_down_and_forces_the_close_sequence() {
        // start = [10, 11], end = [20, 21], forced = [30, 20, 21], budget = 3.
        let mut rb = ReasoningBudget::new(&[10, 11], &[20, 21], vec![30, 20, 21], 3)
            .expect("sampler builds");
        assert_eq!(rb.state(), RbudgetState::Idle);

        // Not the start sequence: stays idle.
        rb.accept(99);
        assert_eq!(rb.state(), RbudgetState::Idle);

        // Start sequence activates counting with the full budget.
        rb.accept(10);
        rb.accept(11);
        assert_eq!(rb.state(), RbudgetState::Counting);
        assert!(!rb.near_exhaustion(2));
        assert!(rb.near_exhaustion(3));

        // Three thinking tokens exhaust the budget; FORCING begins.
        rb.accept(100);
        rb.accept(101);
        assert_eq!(rb.state(), RbudgetState::Counting);
        rb.accept(102);
        assert_eq!(rb.state(), RbudgetState::Forcing);

        // The forced sequence comes back token by token, then DONE.
        assert_eq!(rb.next_forced(), Some(30));
        rb.accept(30);
        assert_eq!(rb.next_forced(), Some(20));
        rb.accept(20);
        assert_eq!(rb.next_forced(), Some(21));
        rb.accept(21);
        assert_eq!(rb.state(), RbudgetState::Done);
        assert!(!rb.forcing());
    }

    /// A natural close inside the budget must deactivate without forcing:
    /// the model that finishes thinking on its own is the common case.
    #[test]
    fn natural_end_inside_budget_never_forces() {
        let mut rb =
            ReasoningBudget::new(&[10], &[20, 21], vec![30, 20, 21], 100).expect("sampler builds");
        rb.accept(10);
        assert_eq!(rb.state(), RbudgetState::Counting);
        rb.accept(50);
        rb.accept(20);
        rb.accept(21);
        assert_eq!(rb.state(), RbudgetState::Done);
    }

    /// Budget zero forces the close the moment the think block opens: the
    /// per-turn "no thinking" operating point that does NOT change the
    /// rendered prompt (unlike flipping `enable_thinking`, which invalidates
    /// the persisted KV prefix).
    #[test]
    fn budget_zero_forces_immediately_on_activation() {
        let mut rb =
            ReasoningBudget::new(&[10], &[20], vec![20], 0).expect("sampler builds");
        rb.accept(10);
        assert_eq!(rb.state(), RbudgetState::Forcing);
        assert_eq!(rb.next_forced(), Some(20));
        rb.accept(20);
        assert_eq!(rb.state(), RbudgetState::Done);
    }
}

/// What the model's template produced for this turn.
#[derive(Debug, Deserialize)]
pub struct Applied {
    pub prompt: String,
    /// Separate from `prompt` on purpose; both are needed. See the module docs
    /// on `llama_cpp` for what happens when only the first is sent.
    #[serde(default)]
    pub generation_prompt: String,
    /// Upstream's grammar for this turn, from the model family's own chat
    /// handler. Two producers, one field:
    ///
    /// - a `json_schema` request compiles into it (response-format grammar,
    ///   eager) — the engine APPLIES it, that is the whole constrained-output
    ///   feature;
    /// - tool schemas alone also generate one (call grammar, usually lazy) —
    ///   deliberately NOT applied: the first measurement showed Gemma emitting
    ///   a valid call unconstrained, so constraining would be paying for a
    ///   problem we have not observed.
    #[serde(default)]
    pub grammar: String,
    /// True when `grammar` is trigger-gated (tool-call style) rather than
    /// meant to constrain from the first sampled token. A response-format
    /// grammar is eager; the engine refuses to treat a lazy grammar as one.
    #[serde(default)]
    pub grammar_lazy: bool,
    /// Whether the model's own template declares a reasoning channel.
    #[allow(dead_code)]
    #[serde(default)]
    pub supports_thinking: bool,
    /// The template's thinking tags (Gemma 4: `<|channel>thought` /
    /// `<channel|>`), from upstream's per-family chat params. Empty when the
    /// template declares none, which is also what makes a reasoning budget on
    /// such a model a clean no-op rather than a guess at tags.
    #[serde(default)]
    pub thinking_start: String,
    #[serde(default)]
    pub thinking_end: String,
    /// Stop strings the TEMPLATE declares, e.g. Gemma 4's `<turn|>`.
    ///
    /// Stopping on end-of-turn TOKENS is not enough: the marker is still sitting
    /// at the end of the decoded text, and upstream's parser leaves it in
    /// `content` because the grammar wants a complete turn including it. These
    /// are what the text has to be trimmed with, and they come from the model
    /// rather than from a constant we would have to maintain per family.
    #[serde(default)]
    pub additional_stops: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ParsedToolCall {
    pub name: String,
    /// JSON text, as upstream hands it over.
    pub arguments: String,
    /// The template's id for this call, when it emits one. Needed to pair the
    /// result back to the call on the next turn.
    #[serde(default)]
    pub id: String,
}

#[derive(Debug, Deserialize)]
pub struct Parsed {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub reasoning_content: String,
    #[serde(default)]
    pub tool_calls: Vec<ParsedToolCall>,
    /// True when the strict parse failed and the shim recovered this message
    /// from upstream's partial-parse path instead. The turn is usable; the
    /// tail of the generation was not parseable. See `scope_chat_parse`.
    #[serde(default)]
    pub salvaged: bool,
}

/// Errors are returned by the shim as `{"error": "..."}` rather than thrown, so
/// a template or parse failure never unwinds across the FFI boundary.
#[derive(Debug, Deserialize)]
struct ShimError {
    error: String,
}

pub struct Chat {
    ctx: *mut ScopeChatCtx,
}

// The handle is only ever touched behind `&mut self` from the owning engine.
unsafe impl Send for Chat {}

impl Drop for Chat {
    fn drop(&mut self) {
        unsafe { scope_chat_ctx_free(self.ctx) };
    }
}

fn take_string(p: *mut c_char) -> Result<String, String> {
    if p.is_null() {
        return Err("shim returned null".into());
    }
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { scope_chat_free(p) };
    Ok(s)
}

fn decode<T: for<'de> Deserialize<'de>>(json: &str) -> Result<T, String> {
    if let Ok(e) = serde_json::from_str::<ShimError>(json) {
        return Err(e.error);
    }
    serde_json::from_str(json).map_err(|e| e.to_string())
}

impl Chat {
    pub fn new(template: &str) -> Result<Self, String> {
        let c = CString::new(template).map_err(|e| e.to_string())?;
        let ctx = unsafe { scope_chat_init(c.as_ptr()) };
        if ctx.is_null() {
            return Err("scope_chat_init returned null".into());
        }
        Ok(Self { ctx })
    }

    pub fn apply(&mut self, req: &GenerateRequest, enable_thinking: bool) -> Result<Applied, String> {
        let mut messages = Vec::with_capacity(req.messages.len() + 1);
        if !req.system.is_empty() {
            messages.push(serde_json::json!({"role": "system", "content": req.system}));
        }
        for m in &req.messages {
            // The tool fields go across whenever they are set. Sending only role
            // and content is what left the template's tool branches dark, so a
            // replayed agent loop reached the model as prose about tool calls
            // rather than as tool calls.
            let mut j = serde_json::json!({"role": m.role, "content": m.content});
            let o = j.as_object_mut().expect("just built an object");
            if !m.tool_calls.is_empty() {
                o.insert(
                    "tool_calls".into(),
                    serde_json::Value::Array(
                        m.tool_calls
                            .iter()
                            .map(|c| {
                                serde_json::json!({
                                    "name": c.name,
                                    "arguments": c.arguments,
                                    "id": c.id,
                                })
                            })
                            .collect(),
                    ),
                );
            }
            if !m.tool_name.is_empty() {
                o.insert("tool_name".into(), m.tool_name.clone().into());
            }
            if !m.tool_call_id.is_empty() {
                o.insert("tool_call_id".into(), m.tool_call_id.clone().into());
            }
            messages.push(j);
        }
        let tools: Vec<_> = req
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            })
            .collect();

        let m = CString::new(serde_json::Value::Array(messages).to_string()).map_err(|e| e.to_string())?;
        let t = CString::new(serde_json::Value::Array(tools).to_string()).map_err(|e| e.to_string())?;
        // Empty string means "no response format"; the shim maps it to an
        // unset `inputs.json_schema`.
        let s = CString::new(
            req.json_schema.as_ref().map(|v| v.to_string()).unwrap_or_default(),
        )
        .map_err(|e| e.to_string())?;
        let raw =
            unsafe { scope_chat_apply(self.ctx, m.as_ptr(), t.as_ptr(), s.as_ptr(), enable_thinking) };
        decode(&take_string(raw)?)
    }

    /// What the template supports, keyed as the jinja layer names it
    /// (`supports_tools`, `supports_tool_calls`, `supports_object_arguments`,
    /// `supports_system_role`, ...).
    ///
    /// Cheap and constant for a given template, so a caller reads it once at
    /// load. An empty map means the shim could not report them, which is not an
    /// error: an older upstream may not expose caps at all, and refusing to
    /// generate over a missing diagnostic would be worse than the diagnostic.
    pub fn caps(&mut self) -> std::collections::BTreeMap<String, bool> {
        let raw = unsafe { scope_chat_caps(self.ctx) };
        match take_string(raw) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
            Err(_) => Default::default(),
        }
    }

    pub fn parse(&mut self, text: &str) -> Result<Parsed, String> {
        let t = CString::new(text).map_err(|e| e.to_string())?;
        let raw = unsafe { scope_chat_parse(self.ctx, t.as_ptr()) };
        decode(&take_string(raw)?)
    }
}
