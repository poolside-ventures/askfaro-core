//! Grammar-constrained JSON output — the fix for the deriver units that burned
//! in Scope's memory queue.
//!
//! The failure this guards against: the desktop's memory shim honored OpenAI
//! `response_format` by pasting the schema into the system prompt, and some
//! Honcho deriver units still answered in prose. With greedy decode the argmax
//! continuation is prose EVERY time, so retries were identical and the unit
//! never completed. `GenerateRequest::json_schema` moves the enforcement to the
//! sampler: the schema compiles to GBNF and non-conforming tokens are masked,
//! so prose is unrepresentable.
//!
//! Ignored by default (needs real GGUF weights). Run with:
//!
//!   FARO_TEST_GGUF=/path/to/model.gguf \
//!   cargo test -p askfaro-core --features llama-cpp --test llama_json_schema -- --ignored --nocapture

#![cfg(feature = "llama-cpp")]

use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
};

fn engine() -> LlamaCppEngine {
    let path = std::env::var("FARO_TEST_GGUF").expect("set FARO_TEST_GGUF to a .gguf file");
    LlamaCppEngine::new(LlamaCppConfig {
        model_path: path.into(),
        n_ctx: 2048,
        n_slots: 1,
        // The extraction operating point: the grammar masks the thinking tags
        // from token 0 anyway, and the prompt should agree with what sampling
        // permits.
        enable_thinking: false,
        ..Default::default()
    })
}

/// An open-ended prompt that no instruction tuning maps to JSON — the same
/// shape as the deriver units that failed: unconstrained, the model answers in
/// prose. The grammar must make the conforming reply the ONLY reply.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn a_prose_prompt_is_forced_into_the_schema() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "topics": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["summary", "topics"],
        "additionalProperties": false
    });

    let mut e = engine();
    let resp = e
        .generate(GenerateRequest {
            system: "You are a helpful assistant.".into(),
            // Deliberately NOT a request for JSON: the point under test is that
            // the sampler, not the prompt, is what holds the contract.
            messages: vec![Msg::user("Tell me a little about the city of Paris.")],
            json_schema: Some(schema),
            enable_thinking: Some(false),
            ..Default::default()
        })
        .expect("constrained generate");

    println!(
        "constrained reply ({} tok, truncated={}, decode {}ms = {:.1}ms/tok): {}",
        resp.timings.output_tokens,
        resp.timings.truncated,
        resp.timings.decode_ms,
        resp.timings.decode_ms as f64 / resp.timings.output_tokens.max(1) as f64,
        resp.text
    );
    assert!(!resp.timings.truncated, "the grammar must reach a complete value within the cap");

    let v: serde_json::Value = serde_json::from_str(resp.text.trim())
        .expect("a grammar-constrained reply must parse as JSON");
    assert!(v.get("summary").and_then(|s| s.as_str()).is_some_and(|s| !s.is_empty()));
    assert!(v.get("topics").is_some_and(|t| t.is_array()));
}

/// A nullable field (`"type": ["string", "null"]`) through the response-format
/// grammar. This is the exact verdict shape Scope's follow-up hook sends, and
/// union types are the schema feature most likely to be unsupported by a
/// grammar converter — if upstream's PEG rejects it, the caller must find out
/// from this test, not from every verdict silently erroring in the field.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn a_nullable_union_field_is_accepted_and_enforced() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "needs_followup": {"type": "boolean"},
            "wait_until": {"type": ["string", "null"]}
        },
        "required": ["needs_followup"]
    });

    let mut e = engine();
    let resp = e
        .generate(GenerateRequest {
            system: "You assess whether a sent email still needs a follow-up.".into(),
            messages: vec![Msg::user(
                "I emailed a vendor three days ago asking for a quote and heard nothing back.",
            )],
            json_schema: Some(schema),
            enable_thinking: Some(false),
            ..Default::default()
        })
        .expect("union-typed schema must compile and generate");

    let v: serde_json::Value = serde_json::from_str(resp.text.trim())
        .expect("reply must be valid JSON");
    println!("verdict: {v}");
    assert!(v.get("needs_followup").is_some_and(|b| b.is_boolean()));
    if let Some(w) = v.get("wait_until") {
        assert!(w.is_string() || w.is_null(), "wait_until must be string or null: {w}");
    }
}

/// The same engine must serve constrained and unconstrained turns back to back:
/// the sampler (and its grammar) is per turn, not per engine, and a leaked
/// grammar would strangle every later chat reply into JSON.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn the_grammar_does_not_leak_into_the_next_turn() {
    let mut e = engine();

    let constrained = e
        .generate(GenerateRequest {
            system: "You are a helpful assistant.".into(),
            messages: vec![Msg::user("Name any color.")],
            json_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {"color": {"type": "string"}},
                "required": ["color"]
            })),
            enable_thinking: Some(false),
            ..Default::default()
        })
        .expect("constrained turn");
    serde_json::from_str::<serde_json::Value>(constrained.text.trim())
        .expect("constrained turn must be JSON");

    let free = e
        .generate(GenerateRequest {
            system: "Answer in one short plain-text sentence.".into(),
            messages: vec![Msg::user("Say hello.")],
            enable_thinking: Some(false),
            ..Default::default()
        })
        .expect("unconstrained turn after a constrained one");
    println!("free reply: {}", free.text);
    assert!(!free.text.trim().is_empty(), "the follow-up turn must still answer");
}
