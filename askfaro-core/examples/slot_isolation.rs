//! Do two workloads on one engine keep their own KV caches?
//!
//! This is a regression test for the most expensive on-device bug found so far,
//! and one that no benchmark could see because each workload looked fine alone.
//!
//! The desktop runs an agent loop with a ~6,000-token prompt, and background
//! one-shots of ~340 tokens (reply-intent, follow-up assessment) on the SAME
//! engine. With one cache, each call evicted the other's prefix: measured
//! 19,281ms of prefill on a turn that should have reused nearly all of it, and
//! 3,441ms for the tiny one-shot that had just displaced it. Two workloads, one
//! cache, both permanently cold, and every individual number looked plausible.
//!
//! Slots map to llama.cpp sequences. The check: interleave a big prompt on slot
//! 0 with a small one on slot 1 and assert the big one stays WARM across the
//! interruption. Sharing a slot fails this immediately.
//!
//! ```sh
//! cargo run --release --features llama-cpp --example slot_isolation -- <model.gguf>
//! ```
use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg, ToolSchema,
};
use serde_json::json;

/// Big enough that a re-prefill is unmistakable in the timing.
fn big_system() -> String {
    let mut s = String::from("You are a careful assistant. Background material follows.\n\n");
    for i in 0..400 {
        s.push_str(&format!(
            "Fact {i}: item {i} belongs to group {}, with weight {} and a short note.\n",
            i % 7,
            i * 3 % 11
        ));
    }
    s
}

fn tool() -> ToolSchema {
    ToolSchema {
        name: "lookup".into(),
        description: "Look something up.".into(),
        parameters: json!({"type":"object","properties":{"q":{"type":"string"}}}),
    }
}

fn ask(engine: &mut LlamaCppEngine, system: &str, user: &str, slot: u32) -> (u64, u32) {
    let out = engine
        .generate(GenerateRequest {
            system: system.into(),
            messages: vec![Msg::user(user)],
            tools: vec![tool()],
            slot,
            ..Default::default()
        })
        .expect("generate");
    (out.timings.prefill_ms, out.timings.prompt_tokens)
}

fn main() {
    let Some(model) = std::env::args().nth(1) else {
        eprintln!("usage: slot_isolation <model.gguf>");
        std::process::exit(2);
    };
    let mut engine = LlamaCppEngine::new(LlamaCppConfig {
        model_path: model.into(),
        ..Default::default()
    });

    let big = big_system();
    let small = "You answer in one word.";

    // Slot 0 pays the cold prefill once.
    let (cold, big_tokens) = ask(&mut engine, &big, "Name group 3.", 0);
    println!("slot 0 cold      : {cold:>6}ms / {big_tokens} tok");

    // Slot 0 again, undisturbed: this is the warm baseline.
    let (warm, _) = ask(&mut engine, &big, "Name group 4.", 0);
    println!("slot 0 warm      : {warm:>6}ms");

    // A background one-shot on slot 1, exactly what the desktop does.
    let (one_shot, small_tokens) = ask(&mut engine, small, "Say hi.", 1);
    println!("slot 1 one-shot  : {one_shot:>6}ms / {small_tokens} tok");

    // Slot 0 AFTER the interruption. Shared cache means this repeats the cold
    // number; isolated slots mean it stays at the warm one.
    let (after, _) = ask(&mut engine, &big, "Name group 5.", 0);
    println!("slot 0 after     : {after:>6}ms");

    // Generous multiple of the warm baseline: this is looking for a return to
    // COLD (tens of times warm), not for a few milliseconds of jitter.
    let budget = (warm * 5).max(1_500);
    if after > budget {
        eprintln!(
            "\nFAILED: slot 0 went cold after a slot 1 request ({after}ms against a \
             {warm}ms warm baseline). The two workloads are sharing one KV cache, so each \
             call evicts the other's prefix."
        );
        std::process::exit(1);
    }
    println!("\npassed: slot 0 stayed warm ({after}ms vs {warm}ms baseline) across a slot 1 call");
}
