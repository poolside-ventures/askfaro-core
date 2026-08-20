//! ONE load of the weights, SEVERAL contexts, each with its own window.
//!
//! llama.cpp divides a context's KV cache evenly across its sequences, so every
//! slot of one context has the same window and a host that wants a big one and
//! a small one used to have to buy two big ones. On Gemma 4 E4B that is 1,024
//! MiB of KV per slot at 64K against 256 MiB at 16K: a desktop running an agent
//! loop plus short background extraction turns paid the agent's window twice.
//!
//! The obvious workaround — a second `LlamaCppEngine` — is the one that was
//! already tried and rejected: it loads the same 5.15 GB a second time and
//! could OOM Metal whenever both were warm. So the weights are shared through a
//! process-wide registry and contexts are built over them independently, which
//! is what this test pins:
//!
//!  1. two contexts of DIFFERENT `n_ctx` live at once,
//!  2. every slot generates on its own context,
//!  3. the windows are real (a prompt that fits the big slot is refused by the
//!     small one),
//!  4. a second engine reuses the resident weights,
//!  5. and through all of it the weights are read off disk exactly ONCE.
//!
//! One test function on purpose: `model_loads()` is a process-wide counter, and
//! a sibling test loading weights in another thread would make the count a
//! race. This file is its own test binary, so the count here is absolute.
//!
//! RSS cannot answer (5): the weights are mmapped, `model buffer size` prints
//! 0.00 MiB, and the same configuration has been observed at both 6.44 and 1.66
//! GiB across two runs. For the KV numbers, read the loader's own
//! `llama_kv_cache: size = ... MiB` lines — `examples/kv_footprint.rs` prints
//! the shapes side by side.
//!
//! Ignored by default (needs real GGUF weights). Run with:
//!
//!   FARO_TEST_GGUF=/path/to/model.gguf \
//!   cargo test -p askfaro-core --features llama-cpp --test llama_shared_model -- --ignored --nocapture

#![cfg(feature = "llama-cpp")]

use askfaro_core::generation::{
    ContextSpec, GenError, GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
};

/// The agent's window and the background one, both small enough to build in a
/// test and far enough apart to tell apart.
const BIG: u32 = 2048;
const SMALL: u32 = 512;

fn say_ok(slot: u32) -> GenerateRequest {
    GenerateRequest {
        system: "Answer with a single short word.".into(),
        messages: vec![Msg {
            role: "user".into(),
            content: "Say OK.".into(),
            ..Default::default()
        }],
        tools: vec![],
        slot,
        ..Default::default()
    }
}

fn one_turn(engine: &mut LlamaCppEngine, slot: u32, who: &str) {
    let resp = engine
        .generate(say_ok(slot))
        .unwrap_or_else(|e| panic!("{who}: generate failed: {e}"));
    println!("{who}: {:?} ({} tok)", resp.text, resp.timings.output_tokens);
    assert!(!resp.text.trim().is_empty(), "{who}: empty generation");
}

#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn two_windows_over_one_set_of_weights() {
    let path = std::env::var("FARO_TEST_GGUF").expect("set FARO_TEST_GGUF to a .gguf file");

    assert_eq!(
        LlamaCppEngine::model_loads(),
        0,
        "this test owns its process and must start with nothing resident"
    );

    // The desktop's real shape, shrunk: the agent keeps slot 0 on the big
    // window, background work keeps slot 1 and gets a small one. Neither side
    // changes which slot number it sends.
    let mut e = LlamaCppEngine::new(LlamaCppConfig {
        model_path: path.clone().into(),
        n_ctx: BIG,
        n_slots: 1,
        extra_contexts: vec![ContextSpec {
            n_ctx: SMALL,
            n_slots: 1,
            n_ubatch: None,
        }],
        enable_thinking: false,
        ..Default::default()
    });
    e.warm().expect("engine must load");

    assert_eq!(
        LlamaCppEngine::model_loads(),
        1,
        "two contexts must come from ONE read of the weights"
    );

    // Read back from llama.cpp, not from the config: the whole feature is that
    // these two numbers differ.
    let windows = e.slot_windows();
    assert_eq!(
        windows,
        [(0, BIG), (1, SMALL)].into_iter().collect(),
        "each slot must have its own window"
    );

    // Both contexts must actually decode, not merely exist.
    one_turn(&mut e, 0, "slot 0 (big)");
    one_turn(&mut e, 1, "slot 1 (small)");

    // And the small window must BE small. A prompt that fits slot 0 and not
    // slot 1 proves the KV cache was sized per context rather than per engine —
    // which is the memory this exists to give back.
    let long = "banana ".repeat(SMALL as usize);
    let mut over = say_ok(1);
    over.messages[0].content = long.clone();
    assert!(
        matches!(
            e.generate(over).expect_err("must not fit the small window"),
            GenError::ContextWindowExceeded
        ),
        "slot 1 must refuse a prompt past ITS window"
    );
    let mut fits = say_ok(0);
    fits.messages[0].content = long;
    let resp = e
        .generate(fits)
        .expect("the same prompt must fit slot 0's larger window");
    assert!(!resp.text.trim().is_empty(), "slot 0: empty generation");

    // A second engine over the same weights: this is what used to load 5.15 GB
    // twice. A different context layout on purpose — the registry keys on the
    // MODEL parameters, and context size is not one of them.
    let mut b = LlamaCppEngine::new(LlamaCppConfig {
        model_path: path.into(),
        n_ctx: SMALL,
        n_slots: 1,
        enable_thinking: false,
        ..Default::default()
    });
    b.warm().expect("second engine must load while the first is warm");
    assert_eq!(
        LlamaCppEngine::model_loads(),
        1,
        "a second engine must borrow the resident weights, not load them again"
    );
    one_turn(&mut b, 0, "engine B");

    // Still fine after all of that: the first engine's contexts are untouched
    // by anything the second one did.
    one_turn(&mut e, 0, "slot 0 (big, after B)");
    one_turn(&mut e, 1, "slot 1 (small, after B)");
}
