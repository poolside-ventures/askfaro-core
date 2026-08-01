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

extern "C" {
    fn scope_chat_init(template_str: *const c_char) -> *mut ScopeChatCtx;
    fn scope_chat_apply(
        ctx: *mut ScopeChatCtx,
        messages_json: *const c_char,
        tools_json: *const c_char,
        enable_thinking: bool,
    ) -> *mut c_char;
    fn scope_chat_parse(ctx: *mut ScopeChatCtx, text: *const c_char) -> *mut c_char;
    fn scope_chat_caps(ctx: *mut ScopeChatCtx) -> *mut c_char;
    fn scope_chat_free(p: *mut c_char);
    fn scope_chat_ctx_free(ctx: *mut ScopeChatCtx);
}

/// What the model's template produced for this turn.
#[derive(Debug, Deserialize)]
pub struct Applied {
    pub prompt: String,
    /// Separate from `prompt` on purpose; both are needed. See the module docs
    /// on `llama_cpp` for what happens when only the first is sent.
    #[serde(default)]
    pub generation_prompt: String,
    /// Auto-generated from the tool schemas. Carried but not applied yet: the
    /// first measurement showed Gemma emitting a valid call unconstrained, so
    /// constraining decoding would be paying for a problem we have not observed.
    /// Kept because it is free here and is the fix if that ever changes.
    #[allow(dead_code)]
    #[serde(default)]
    pub grammar: String,
    /// Whether the model's own template declares a reasoning channel.
    #[allow(dead_code)]
    #[serde(default)]
    pub supports_thinking: bool,
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
        let raw = unsafe { scope_chat_apply(self.ctx, m.as_ptr(), t.as_ptr(), enable_thinking) };
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
