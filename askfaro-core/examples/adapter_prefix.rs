//! Does the persisted prefix work when slot 0 runs under an add-on?
//!
//! A KV state belongs to the weights that computed it, add-on included. With
//! the prefix built on the plain model and the agent turn decoded under an
//! add-on, every turn cleared the slot and read the whole prompt again
//! (desktop, 2026-11-10: 26.8s on 7,873 tokens with the plain prefix restored
//! 32ms earlier). `LlamaCppConfig::prefix_adapter` builds, saves and restores
//! the prefix under the add-on instead. This checks, on real weights:
//!
//!  - a turn under the add-on on a fresh engine reuses the whole restored
//!    prefix, and answers what a cold engine with no prefix answers;
//!  - `ensure_prefix` right after a turn finds the prefix current and leaves
//!    the conversation in the slot, so the next turn extends it;
//!  - a prefix request under any other add-on is refused.
//!
//! `<prompt.json>` is `{system, tools: [{name, description, parameters}]}`,
//! the file `build_prefix` takes.
//!
//! ```sh
//! cargo run --release --features metal --example adapter_prefix -- \
//!   <model.gguf> <adapter.gguf> <drafter.gguf> <prompt.json>
//! ```
use askfaro_core::generation::{
    AdapterSpec, GenError, GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
    ToolSchema,
};

const USER_1: &str = "what's on my task list?";
const USER_2: &str = "and which of those are due this week?";

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(model), Some(adapter), Some(drafter), Some(prompt_path)) =
        (args.next(), args.next(), args.next(), args.next())
    else {
        eprintln!("usage: adapter_prefix <model.gguf> <adapter.gguf> <drafter.gguf> <prompt.json>");
        std::process::exit(2);
    };
    let prompt: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&prompt_path).expect("read prompt.json"))
            .expect("parse prompt.json");
    let system = prompt["system"].as_str().expect("string `system`").to_string();
    let tools: Vec<ToolSchema> = prompt["tools"]
        .as_array()
        .expect("array `tools`")
        .iter()
        .map(|t| ToolSchema {
            name: t["name"].as_str().unwrap_or_default().to_string(),
            description: t["description"].as_str().unwrap_or_default().to_string(),
            parameters: t["parameters"].clone(),
        })
        .collect();

    let dir = std::env::temp_dir().join("faro-adapter-prefix");
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = |prefix: bool| LlamaCppConfig {
        model_path: model.clone().into(),
        draft_path: Some(drafter.clone().into()),
        n_ctx: 16384,
        n_slots: 1,
        enable_thinking: false,
        prefix_cache_dir: prefix.then(|| dir.clone()),
        state_key: "adapter-prefix-example".into(),
        adapters: vec![AdapterSpec { name: "tools".into(), path: adapter.clone().into() }],
        prefix_adapter: prefix.then(|| "tools".to_string()),
        ..Default::default()
    };
    let request = |messages: Vec<Msg>, adapter: Option<&str>| GenerateRequest {
        system: system.clone(),
        tools: tools.clone(),
        messages,
        adapter: adapter.map(str::to_string),
        ..Default::default()
    };
    let answer = |out: &askfaro_core::generation::GenerateResponse| {
        let calls: Vec<_> = out
            .tool_calls
            .iter()
            .map(|c| serde_json::json!({"name": c.name, "arguments": c.arguments}))
            .collect();
        serde_json::json!({"text": out.text, "tool_calls": calls}).to_string()
    };
    let mut failed = false;
    let mut check = |ok: bool, what: String| {
        println!("{} {what}", if ok { "PASS" } else { "FAIL" });
        failed |= !ok;
    };

    // What a cold engine with no prefix answers: the reference.
    let mut cold = LlamaCppEngine::new(cfg(false));
    let cold_out = cold
        .generate(request(vec![Msg::user(USER_1)], Some("tools")))
        .expect("cold turn");
    println!(
        "cold, no prefix: {} prompt tokens, {} cached, prefill {}ms",
        cold_out.timings.prompt_tokens, cold_out.timings.cached_tokens, cold_out.timings.prefill_ms
    );
    drop(cold);

    // Build the prefix where a host does, before any turn.
    let mut builder = LlamaCppEngine::new(cfg(true));
    let refused = builder.ensure_prefix(&request(vec![], None));
    check(
        matches!(refused, Err(GenError::Invalid(_))),
        "a plain-model prefix request is refused when the prefix is the add-on's".into(),
    );
    let built = builder.ensure_prefix(&request(vec![], Some("tools"))).expect("build the prefix");
    check(built.rebuilt && built.tokens > 0, format!("built a {}-token prefix under the add-on in {}ms", built.tokens, built.ms));
    drop(builder);

    // A fresh engine, as after a relaunch or an idle release: restore, then a turn.
    let mut engine = LlamaCppEngine::new(cfg(true));
    let turn1 = engine
        .generate(request(vec![Msg::user(USER_1)], Some("tools")))
        .expect("first turn");
    check(
        turn1.timings.cached_tokens >= built.tokens,
        format!(
            "the first turn reused {} of the {}-token prefix ({} prompt tokens, prefill {}ms against {}ms cold)",
            turn1.timings.cached_tokens,
            built.tokens,
            turn1.timings.prompt_tokens,
            turn1.timings.prefill_ms,
            cold_out.timings.prefill_ms,
        ),
    );
    check(
        answer(&turn1) == answer(&cold_out),
        "the answer on the restored prefix equals the cold engine's".into(),
    );

    // The host's after-turn check: current, and the conversation stays.
    let after = engine.ensure_prefix(&request(vec![], Some("tools"))).expect("after-turn check");
    check(!after.rebuilt, "the after-turn check finds the prefix current and rebuilds nothing".into());
    let turn2 = engine
        .generate(request(
            vec![Msg::user(USER_1), Msg::assistant(&turn1.text), Msg::user(USER_2)],
            Some("tools"),
        ))
        .expect("second turn");
    check(
        turn2.timings.cached_tokens >= built.tokens,
        format!(
            "the second turn reused {} tokens ({} prompt tokens, prefill {}ms)",
            turn2.timings.cached_tokens, turn2.timings.prompt_tokens, turn2.timings.prefill_ms
        ),
    );

    // Dropped before exit: ggml-metal aborts over a buffer still alive.
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
    std::process::exit(i32::from(failed));
}
