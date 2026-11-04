//! Does a request get exactly the add-on it names, and nothing it does not?
//!
//! Runs the same requests on the plain model and under one add-on, switching
//! back and forth, on one slot and across two slots of one context:
//!
//!  - the plain answer after a switch must equal the first plain answer
//!    (nothing of the add-on leaked into the slot's cache or the context);
//!  - the add-on's answer must differ from the plain one (it was applied) and
//!    be stable across switches;
//!  - a slot whose add-on did not change stays warm when another slot of the
//!    same context switched the context's add-on in between.
//!
//! Prints each answer as one JSON line, for comparing with llama-server run
//! with and without `--lora` on the same request.
//!
//! ```sh
//! cargo run --release --features metal --example adapter_switch -- <model.gguf> <adapter.gguf> [text prompt]
//! ```
use askfaro_core::generation::{
    AdapterSpec, GenError, GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
    ToolSchema,
};
use serde_json::json;

const SYSTEM: &str = "You are Scopy, a fast on-device assistant. Use a tool when one fits.";
const USER: &str = "what's on my task list?";

fn tools() -> Vec<ToolSchema> {
    vec![
        ToolSchema {
            name: "scopy_task_list".into(),
            description: "List the user's open tasks.".into(),
            parameters: json!({"type": "object", "properties": {}}),
        },
        ToolSchema {
            name: "scopy_task".into(),
            description: "Create, update or complete a task.".into(),
            parameters: json!({"type": "object", "properties": {
                "action": {"type": "string", "enum": ["create", "update", "complete"]},
                "task_id": {"type": "string"},
                "title": {"type": "string"}}}),
        },
    ]
}

fn ask(engine: &mut LlamaCppEngine, slot: u32, adapter: Option<&str>) -> (String, u64, u32) {
    let out = engine
        .generate(GenerateRequest {
            system: SYSTEM.into(),
            messages: vec![Msg::user(USER)],
            tools: tools(),
            slot,
            enable_thinking: Some(false),
            adapter: adapter.map(str::to_string),
            ..Default::default()
        })
        .expect("generate");
    let calls: Vec<_> = out
        .tool_calls
        .iter()
        .map(|c| json!({"name": c.name, "arguments": c.arguments}))
        .collect();
    let answer = json!({"text": out.text, "tool_calls": calls}).to_string();
    println!(
        "{}",
        json!({"slot": slot, "adapter": adapter, "answer": answer, "prefill_ms": out.timings.prefill_ms})
    );
    (answer, out.timings.prefill_ms, out.timings.prompt_tokens)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(model), Some(adapter)) = (args.next(), args.next()) else {
        eprintln!("usage: adapter_switch <model.gguf> <adapter.gguf> [text prompt]");
        std::process::exit(2);
    };
    let mut engine = LlamaCppEngine::new(LlamaCppConfig {
        model_path: model.into(),
        n_ctx: 4096,
        adapters: vec![AdapterSpec {
            name: "tools".into(),
            path: adapter.into(),
        }],
        ..Default::default()
    });
    let mut failed = false;
    let mut check = |ok: bool, what: &str| {
        println!("{} {what}", if ok { "PASS" } else { "FAIL" });
        failed |= !ok;
    };

    let (plain1, _, _) = ask(&mut engine, 0, None);
    let (tools1, _, _) = ask(&mut engine, 0, Some("tools"));
    let (plain2, _, _) = ask(&mut engine, 0, None);
    let (tools2, _, _) = ask(&mut engine, 0, Some("tools"));
    check(
        plain1 == plain2,
        "the plain answer is unchanged after the slot ran under the add-on",
    );
    check(
        tools1 == tools2,
        "the add-on's answer is the same each time",
    );
    check(
        plain1 != tools1,
        "the add-on changes the answer (it was applied)",
    );

    // Slot 0 back on the plain model and warm; slot 1 then switches the shared
    // context to the add-on; slot 0 must still answer plainly, from its cache.
    let (_, _, tokens) = ask(&mut engine, 0, None);
    let (tools3, _, _) = ask(&mut engine, 1, Some("tools"));
    let (plain3, warm_ms, _) = ask(&mut engine, 0, None);
    check(
        tools3 == tools1,
        "the add-on answers the same on another slot",
    );
    check(
        plain3 == plain1,
        "slot 0 answers plainly after slot 1 switched the context",
    );
    println!("slot 0 prefill after slot 1's switch: {warm_ms}ms for a {tokens}-token prompt");

    let refused = engine.generate(GenerateRequest {
        messages: vec![Msg::user(USER)],
        adapter: Some("nope".into()),
        ..Default::default()
    });
    check(
        matches!(refused, Err(GenError::Invalid(_))),
        "an unknown add-on is refused",
    );

    // A free-text reply, plain and under the add-on, no tools: a longer
    // answer shows a difference in how the add-on is applied that one short
    // tool call can hide. Printed for comparing with llama-server.
    if let Some(text) = args.next() {
        for adapter in [None, Some("tools")] {
            let out = engine
                .generate(GenerateRequest {
                    messages: vec![Msg::user(&text)],
                    slot: 1,
                    enable_thinking: Some(false),
                    adapter: adapter.map(str::to_string),
                    ..Default::default()
                })
                .expect("generate");
            println!("{}", json!({"adapter": adapter, "text": out.text}));
        }
    }

    // Dropped before exit, as an app quitting drops it: an add-on's tensors
    // sit in a Metal buffer, and ggml-metal aborts at exit over any buffer
    // still alive, so this also proves they go with the weights.
    drop(engine);
    std::process::exit(i32::from(failed));
}
