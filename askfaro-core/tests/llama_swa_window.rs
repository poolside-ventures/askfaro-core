//! Windowed SWA cache (`swa_full: false`, OPT-IN): the 600 MiB-per-slot
//! memory lever, and the semantics that keep it from being the default.
//!
//! A windowed cache cannot REWIND. Once decode advances past the window,
//! positions behind it are evicted, and llama.cpp does not stop a trim that
//! rewinds past them — `seq_rm` succeeds and the output is silently degraded.
//! The engine's guard (`pos_min > 0` at a trim) turns that into a full
//! re-prefill: slow, never wrong, never an unload. The first version of this
//! test expected the guard to fire only on DEEP divergence; running it showed
//! it fires on ordinary turn 2 as well, because a replayed history renders a
//! few tokens differently than the raw generated tail. That finding is why
//! `swa_full` defaults to true and why flipping it waits for checkpoint
//! support (upstream's own answer to the same problem).
//!
//! Ignored by default (needs real GGUF weights). Run with:
//!
//!   FARO_TEST_GGUF=/path/to/model.gguf \
//!   FARO_TEST_DRAFT_GGUF=/path/to/mtp-drafter.gguf \  # optional
//!   cargo test -p askfaro-core --features llama-cpp --test llama_swa_window -- --ignored --nocapture

#![cfg(feature = "llama-cpp")]

use askfaro_core::generation::{
    GenerateRequest, GenerateResponse, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
};

fn engine() -> LlamaCppEngine {
    let path = std::env::var("FARO_TEST_GGUF").expect("set FARO_TEST_GGUF to a .gguf file");
    LlamaCppEngine::new(LlamaCppConfig {
        model_path: path.into(),
        draft_path: std::env::var("FARO_TEST_DRAFT_GGUF").ok().map(Into::into),
        n_ctx: 4096,
        n_slots: 1,
        enable_thinking: false,
        // The opt-in under test.
        swa_full: false,
        ..Default::default()
    })
}

fn filler(tag: &str, n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        s.push_str(&format!(
            "{tag} note {i}: the review moved to Thursday, the vendor is quiet, \
             and the migration is still blocked on the sandbox environment. "
        ));
    }
    s
}

fn ask(e: &mut LlamaCppEngine, history: Vec<Msg>) -> GenerateResponse {
    e.generate(GenerateRequest {
        system: "Answer with a single short word.".into(),
        messages: history,
        enable_thinking: Some(false),
        ..Default::default()
    })
    .expect("generate")
}

/// Histories LONGER than the window: every later turn that diverges at all
/// (and replayed turns do, by a few render tokens) must degrade to a full
/// re-prefill — correct output, warm engine, no error. This is the guard
/// working, and also the measured reason `swa_full: false` is not the
/// default.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn beyond_the_window_divergence_costs_a_full_prefill_never_an_error() {
    let mut e = engine();
    let long_a = filler("alpha", 60);

    let t1 = vec![Msg::user(format!("{long_a}\n\nSay OK."))];
    let r1 = ask(&mut e, t1.clone());
    assert!(!r1.text.trim().is_empty());

    // Turn 2: replayed history. Under a windowed cache the render drift at
    // the tail forces a full re-prefill (asserted as: it completes and the
    // answer is sane; cheapness is exactly what windowed SWA cannot offer
    // here, and pretending otherwise is how silent quality bugs ship).
    let mut t2 = t1.clone();
    t2.push(Msg::assistant(r1.text.clone()));
    t2.push(Msg::user("Say OK again."));
    let r2 = ask(&mut e, t2);
    assert!(!r2.text.trim().is_empty(), "post-eviction divergence must not fail the turn");

    // Turn 3: total divergence, far behind the window. Full prompt, full
    // re-prefill, engine stays warm.
    let long_b = filler("omega", 60);
    let r3 = ask(&mut e, vec![Msg::user(format!("{long_b}\n\nSay OK."))]);
    assert!(!r3.text.trim().is_empty());
    assert!(
        r3.timings.prefill_ms > r1.timings.prefill_ms / 2,
        "deep divergence must pay a full re-prefill, not reuse a rewound SWA cache \
         (turn1 {}ms, turn3 {}ms)",
        r1.timings.prefill_ms,
        r3.timings.prefill_ms,
    );
    assert!(e.is_warm(), "the guard must never unload the engine");
}

/// Histories SHORTER than the window: nothing has been evicted (`pos_min`
/// still 0), so the windowed cache behaves exactly like the full one —
/// appended turns reuse the prefix and stay cheap. This is the case that
/// makes `swa_full: false` viable for short-prompt workloads even before
/// checkpoints exist.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn inside_the_window_reuse_still_works() {
    let mut e = engine();
    let short = filler("brief", 8); // well under any plausible window

    let t1 = vec![Msg::user(format!("{short}\n\nSay OK."))];
    let r1 = ask(&mut e, t1.clone());
    assert!(!r1.text.trim().is_empty());

    let mut t2 = t1.clone();
    t2.push(Msg::assistant(r1.text.clone()));
    t2.push(Msg::user("Say OK again."));
    let r2 = ask(&mut e, t2);
    assert!(!r2.text.trim().is_empty());
    assert!(
        r2.timings.prefill_ms < r1.timings.prefill_ms,
        "inside the window an appended turn must reuse the prefix \
         (turn1 {}ms, turn2 {}ms)",
        r1.timings.prefill_ms,
        r2.timings.prefill_ms,
    );
}
