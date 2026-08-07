//! Does the KV prefix survive a real CONVERSATION, or only fresh questions?
//!
//! Every prefix number we have was measured on independent turns: ask, answer,
//! throw the transcript away, ask again. The product does the opposite. It
//! replays the whole thread on every turn, and inside one turn it replays the
//! thread plus an assistant turn and a tool result per step. So the prompt grows
//! monotonically and the claim under test is that each turn EXTENDS the cached
//! tokens instead of colliding with them.
//!
//! ```sh
//! cargo run --release --features llama-cpp --example conversation_prefix -- <model.gguf>
//! ```
//!
//! Exits non-zero on failure, like `slot_isolation`. It asserts rather than
//! prints because the failure mode here is not an error: a prefix that stops
//! matching mid-conversation re-prefills silently and the only symptom is
//! seconds. That is precisely what nobody notices in a log.
//!
//! Three things get separated on purpose, because they fail independently:
//!
//!   1. **Restore.** Turn 1 of a fresh process, prefix off disk. Covered by
//!      `llama_bench` too, repeated here so a failure localises.
//!   2. **Replayed history.** Turns 2..N of one conversation, each carrying every
//!      earlier turn. This is what a thread does.
//!   3. **Tool results.** The same, with `role: "tool"` messages in the
//!      transcript. Gemma's template renders a tool turn differently from a user
//!      turn, and "differently" is where a prefix assumption goes wrong.
//!
//! A cheaper version of this test would keep one long conversation and check the
//! total. It would pass with (3) broken, because the expensive turns would hide
//! inside an average. Each turn is therefore checked on its own.

use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg, ToolCall, ToolSchema,
};
use serde_json::json;

/// Big enough that a re-prefill is unmistakable, and shaped like the real one:
/// the app's system block is ~1.9k tokens of instructions ahead of ~7.7k of tool
/// schemas.
fn system_block() -> String {
    let mut s = String::from(
        "You are Scopy, a fast on-device assistant. Answer briefly.\n\nBackground:\n",
    );
    for i in 0..300 {
        s.push_str(&format!(
            "Note {i}: record {i} sits in bucket {}, weight {}, owner user_{}.\n",
            i % 7,
            i * 3 % 11,
            i % 13
        ));
    }
    s
}

fn tools() -> Vec<ToolSchema> {
    vec![
        ToolSchema {
            name: "lookup_record".into(),
            description: "Look up a record by id.".into(),
            parameters: json!({
                "type": "object",
                "properties": {"id": {"type": "string", "description": "record id"}},
                "required": ["id"]
            }),
        },
        ToolSchema {
            name: "list_bucket".into(),
            description: "List the records in a bucket.".into(),
            parameters: json!({
                "type": "object",
                "properties": {"bucket": {"type": "integer", "description": "bucket number"}},
                "required": ["bucket"]
            }),
        },
    ]
}

struct Turn {
    prefill_ms: u64,
    prompt_tokens: u32,
}

fn run(engine: &mut LlamaCppEngine, system: &str, messages: &[Msg]) -> Turn {
    let out = engine
        .generate(GenerateRequest {
            system: system.into(),
            messages: messages.to_vec(),
            tools: tools(),
            slot: 0,
            ..Default::default()
        })
        .expect("generate");
    Turn {
        prefill_ms: out.timings.prefill_ms,
        prompt_tokens: out.timings.prompt_tokens,
    }
}

fn main() {
    let Some(model) = std::env::args().nth(1) else {
        eprintln!("usage: conversation_prefix <model.gguf>");
        std::process::exit(2);
    };

    // Its own directory, not the bench's: this test deliberately builds a prefix
    // from scratch and must not race another harness's file or inherit its
    // state. Cleared up front so a rerun measures a restore rather than whatever
    // the last run happened to leave.
    let prefix_dir = std::env::temp_dir().join("faro-conversation-prefix");
    let _ = std::fs::remove_dir_all(&prefix_dir);

    let cfg = LlamaCppConfig {
        model_path: model.into(),
        prefix_cache_dir: Some(prefix_dir.clone()),
        state_key: "conversation_prefix".into(),
        ..Default::default()
    };
    let mut engine = LlamaCppEngine::new(cfg.clone());

    let system = system_block();
    let prefix_req = GenerateRequest {
        system: system.clone(),
        tools: tools(),
        ..Default::default()
    };
    match engine.ensure_prefix(&prefix_req) {
        Ok(r) => println!("prefix built : {} tokens, {:.1} MB", r.tokens, r.bytes as f64 / 1e6),
        Err(e) => {
            eprintln!("FAILED: could not build the persisted prefix: {e}");
            std::process::exit(1);
        }
    }

    // A NEW engine over the same config, not an unload/warm on this one. The
    // claim is that the state file works for a process that never computed it,
    // and `unload` leaves `cached` behind in the struct, so reusing the engine
    // would test a weaker thing than a relaunch does.
    drop(engine);
    let mut engine = LlamaCppEngine::new(cfg);
    println!("engine       : rebuilt from config; the KV below comes off disk");

    let mut failures: Vec<String> = Vec::new();

    // --- 1. restore ------------------------------------------------------
    let mut messages = vec![Msg::user("Which bucket is record 12 in?")];
    let t1 = run(&mut engine, &system, &messages);
    println!(
        "turn 1       : {:>6}ms prefill / {} prompt tok   (restored prefix)",
        t1.prefill_ms, t1.prompt_tokens
    );
    const BUDGET_MS: u64 = 1_500;
    if t1.prefill_ms > BUDGET_MS {
        failures.push(format!(
            "turn 1 prefill {}ms exceeds {BUDGET_MS}ms: the persisted prefix was not restored \
             into the fresh engine, or did not match the prompt.",
            t1.prefill_ms
        ));
    }

    // --- 2. replayed history ---------------------------------------------
    //
    // Each turn appends the previous answer and a new question, so turn N's
    // prompt is turn N-1's plus a tail. Every one is checked on its own: an
    // average would swallow a single expensive turn, which is the shape this
    // failure actually has.
    let follow_ups = [
        ("Bucket 5.", "And what weight does record 12 have?"),
        ("Weight 3.", "Who owns record 12?"),
        ("user_12.", "List two other records in the same bucket."),
    ];
    for (i, (answer, question)) in follow_ups.iter().enumerate() {
        messages.push(Msg::assistant(*answer));
        messages.push(Msg::user(*question));
        let t = run(&mut engine, &system, &messages);
        println!(
            "turn {}       : {:>6}ms prefill / {} prompt tok   (replayed history)",
            i + 2,
            t.prefill_ms,
            t.prompt_tokens
        );
        if t.prefill_ms > BUDGET_MS {
            failures.push(format!(
                "turn {} prefill {}ms exceeds {BUDGET_MS}ms with {} prompt tokens: replaying the \
                 thread is not extending the cached prompt, so each turn re-reads all of it.",
                i + 2,
                t.prefill_ms,
                t.prompt_tokens
            ));
        }
    }

    // --- 3. tool results in the transcript -------------------------------
    //
    // The distinct risk. A `role: "tool"` message goes through a different arm
    // of the chat template than a user turn, and if that rendering perturbs
    // anything ahead of it the prefix stops matching. Mapped exactly as
    // `local-provider.ts` maps it, envelope included, so this measures the
    // transcript the product actually sends.
    let tool_rounds = [
        ("lookup_record", json!({"id": "12"}), json!({"id": "12", "bucket": 5, "weight": 3})),
        ("list_bucket", json!({"bucket": 5}), json!({"ids": ["5", "19", "33"]})),
        ("lookup_record", json!({"id": "19"}), json!({"id": "19", "bucket": 5, "weight": 8})),
    ];
    for (i, (name, args, result)) in tool_rounds.iter().enumerate() {
        messages.push(Msg::assistant_calls(
            "",
            vec![ToolCall {
                name: (*name).into(),
                arguments: args.clone(),
                id: format!("call_{i}"),
            }],
        ));
        messages.push(Msg::tool_result(*name, format!("call_{i}"), result.to_string()));
        let t = run(&mut engine, &system, &messages);
        println!(
            "tool turn {}  : {:>6}ms prefill / {} prompt tok   (tool result in transcript)",
            i + 1,
            t.prefill_ms,
            t.prompt_tokens
        );
        if t.prefill_ms > BUDGET_MS {
            failures.push(format!(
                "tool turn {} prefill {}ms exceeds {BUDGET_MS}ms with {} prompt tokens: a \
                 transcript containing role=\"tool\" messages is invalidating the prefix. The \
                 template renders a tool turn in a way that changes tokens ahead of it.",
                i + 1,
                t.prefill_ms,
                t.prompt_tokens
            ));
        }
    }

    if failures.is_empty() {
        println!(
            "\npassed: the prefix survived {} turns of replayed history and {} tool results, \
             every one under {BUDGET_MS}ms",
            follow_ups.len() + 1,
            tool_rounds.len()
        );
    } else {
        eprintln!("\nFAILED:");
        for f in &failures {
            eprintln!("  - {f}");
        }
        std::process::exit(1);
    }
}
