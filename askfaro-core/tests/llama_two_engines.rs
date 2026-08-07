//! Two `LlamaCppEngine`s warm in ONE process — the desktop's real shape: the
//! chat brain and the memory extraction engine coexist, and each can unload
//! and re-warm while the other stays resident.
//!
//! Regression test for `BackendAlreadyInitialized`: llama-cpp-2 guards
//! `LlamaBackend::init()` with a process-global flag that only resets when the
//! handle drops, so per-engine backends meant whichever engine loaded second
//! could not load at all. On Scope desktop that silently killed on-device
//! memory derivation whenever the chat brain was warm. The engines now share
//! one process-wide backend (`shared_backend`).
//!
//! Ignored by default (needs real GGUF weights). Run with:
//!
//!   FARO_TEST_GGUF=/path/to/model.gguf \
//!   cargo test -p askfaro-core --features llama-cpp --test llama_two_engines -- --ignored --nocapture

#![cfg(feature = "llama-cpp")]

use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
};

fn cfg(path: &str) -> LlamaCppConfig {
    LlamaCppConfig {
        model_path: path.into(),
        // Small window and one slot: this test proves coexistence, not
        // capacity, and two engines are resident at once.
        n_ctx: 512,
        n_slots: 1,
        enable_thinking: false,
        ..Default::default()
    }
}

fn one_turn(engine: &mut LlamaCppEngine, who: &str) {
    let resp = engine
        .generate(GenerateRequest {
            system: "Answer with a single short word.".into(),
            messages: vec![Msg {
                role: "user".into(),
                content: "Say OK.".into(),
                ..Default::default()
            }],
            tools: vec![],
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("{who}: generate failed: {e}"));
    println!("{who}: {:?} ({} tok)", resp.text, resp.timings.output_tokens);
    assert!(!resp.text.trim().is_empty(), "{who}: empty generation");
}

#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn two_engines_coexist_in_one_process() {
    let path = std::env::var("FARO_TEST_GGUF").expect("set FARO_TEST_GGUF to a .gguf file");

    // Engine A = the "chat brain": loads first, stays warm.
    let mut a = LlamaCppEngine::new(cfg(&path));
    a.warm().expect("first engine must load");

    // Engine B = the "extraction engine": before the shared backend this
    // failed here with BackendAlreadyInitialized.
    let mut b = LlamaCppEngine::new(cfg(&path));
    b.warm()
        .expect("second engine must load while the first is warm (BackendAlreadyInitialized regression)");

    // Both must actually work, not merely load.
    one_turn(&mut a, "engine A");
    one_turn(&mut b, "engine B");

    // The desktop's release choreography: extraction unloads when the user
    // returns, then reloads at the next idle window — all while the brain
    // stays warm. The reload must not re-trip the init guard.
    assert!(b.unload(), "engine B had weights to release");
    b.warm().expect("engine B must re-warm after unload while A is still warm");
    one_turn(&mut b, "engine B (re-warmed)");
}
