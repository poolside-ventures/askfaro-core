//! End-to-end smoke test for the `llama-cpp` generation provider.
//!
//! Not a unit test on purpose: it needs multi-gigabyte GGUF weights, so it must
//! never run in CI by default.
//!
//! ```sh
//! cargo run --release --features llama-cpp,metal --example llama_generate -- /path/to/model.gguf
//! ```
//!
//! Passing means: a tool call comes back parsed with correct arguments, and the
//! model's reasoning is separated from its answer. Those two are the whole
//! reason this provider exists, and both fail as plausible text rather than as
//! errors, so they are asserted rather than eyeballed.

use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg, ToolSchema,
};
use serde_json::json;

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: llama_generate <path-to.gguf>");
        std::process::exit(2);
    });

    let cfg = LlamaCppConfig {
        model_path: model_path.into(),
        ..Default::default()
    };
    println!("availability: {:?}", LlamaCppEngine::availability_for(&cfg.model_path));

    let mut engine = LlamaCppEngine::new(cfg);

    // The app's real shape: one filtering tool whose two arguments must both be
    // right, so naming the tool alone does not count as success.
    let req = GenerateRequest {
        system: "You are Scopy, a fast on-device assistant. Call a tool when one applies.".into(),
        messages: vec![Msg {
            role: "user".into(),
            content: "Show me all my high priority tasks that are still in progress.".into(),
        }],
        tools: vec![ToolSchema {
            name: "scopy_task_list".into(),
            description: "List tasks in the workspace, filtered by status, priority or search text."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Filter by task status, e.g. in_progress, completed."},
                    "priority": {"type": "string", "description": "Filter by priority, e.g. high, medium, low."}
                }
            }),
        }],
        ..Default::default()
    };

    let res = engine.generate(req).expect("generate failed");

    println!("\n--- response ---");
    println!("text            {:?}", res.text);
    println!("reasoning       {} chars", res.reasoning.len());
    println!("abstained       {}", res.abstained);
    println!("tool_calls      {}", res.tool_calls.len());
    for c in &res.tool_calls {
        println!("   {} {}", c.name, c.arguments);
    }
    println!(
        "timings         load {}ms / prefill {}ms / decode {}ms, {} prompt tok, {} out tok, truncated={}",
        res.timings.load_ms,
        res.timings.prefill_ms,
        res.timings.decode_ms,
        res.timings.prompt_tokens,
        res.timings.output_tokens,
        res.timings.truncated
    );

    let call = res.tool_calls.first().expect("no tool call was parsed");
    assert_eq!(call.name, "scopy_task_list", "wrong tool");
    assert_eq!(call.arguments["priority"], "high", "priority argument wrong");
    assert_eq!(call.arguments["status"], "in_progress", "status argument wrong");
    assert!(!res.reasoning.is_empty(), "reasoning was not separated out");
    assert!(res.timings.decode_ms > 0, "timings not populated");
    println!("\nOK: tool call parsed with correct arguments, reasoning separated.");
}
