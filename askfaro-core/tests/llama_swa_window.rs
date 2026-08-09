//! Windowed SWA cache (`swa_full: false`, now the DEFAULT): the 600 MiB-per-
//! slot memory win, held together by checkpoints.
//!
//! A windowed cache cannot REWIND: positions behind the window are evicted,
//! llama.cpp permits the trim anyway, and the output silently degrades. An
//! earlier version of this suite proved that ordinary turn 2 already rewinds
//! (replayed history renders a few tokens differently than the raw generated
//! tail), which made windowed mode unshippable on its own: every turn paid a
//! full re-prefill. The engine now does what llama-server does — snapshots
//! the window near each prompt's tail and restores the newest covering
//! snapshot on a rewind — so turn 2 is cheap again, and only divergence
//! deeper than every checkpoint degrades to the full re-prefill.
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

/// Histories LONGER than the window. Turn 2's replay drift rewinds; the
/// checkpoint taken near turn 1's tail must absorb it (cheap turn). Total
/// divergence has no covering checkpoint and must degrade to a full
/// re-prefill on a warm engine — and the checkpoint ring must re-arm after.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn checkpoints_absorb_replay_drift_and_deep_divergence_degrades_cleanly() {
    let mut e = engine();
    let long_a = filler("alpha", 60);

    let t1 = vec![Msg::user(format!("{long_a}\n\nSay OK."))];
    let r1 = ask(&mut e, t1.clone());
    assert!(!r1.text.trim().is_empty());

    // Turn 2: replayed history. The tail render drifts (a rewind), and the
    // checkpoint near turn 1's tail must make it CHEAP — this exact turn
    // paying a full re-prefill is what kept windowed SWA unshippable before
    // checkpoints existed.
    let mut t2 = t1.clone();
    t2.push(Msg::assistant(r1.text.clone()));
    t2.push(Msg::user("Say OK again."));
    let r2 = ask(&mut e, t2);
    assert!(!r2.text.trim().is_empty());
    assert!(
        r2.timings.prefill_ms < r1.timings.prefill_ms / 2,
        "the checkpoint must absorb turn 2's replay drift \
         (turn1 {}ms, turn2 {}ms)",
        r1.timings.prefill_ms,
        r2.timings.prefill_ms,
    );

    // Turn 3: total divergence — no checkpoint can cover a different
    // conversation. Full prompt, full re-prefill, engine stays warm.
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
    assert!(e.is_warm(), "the fallback must never unload the engine");

    // Turn 4: replay of the NEW conversation. The full re-prefill above must
    // have re-armed the ring, so this drift-rewind is cheap again.
    let r4 = ask(
        &mut e,
        vec![
            Msg::user(format!("{long_b}\n\nSay OK.")),
            Msg::assistant(r3.text.clone()),
            Msg::user("Once more."),
        ],
    );
    assert!(!r4.text.trim().is_empty());
    assert!(
        r4.timings.prefill_ms < r3.timings.prefill_ms / 2,
        "checkpoints must re-arm after a full re-prefill \
         (turn3 {}ms, turn4 {}ms)",
        r3.timings.prefill_ms,
        r4.timings.prefill_ms,
    );
}

/// Background one-shots on a non-zero slot: every job REPLACES the whole
/// prompt after a shared instruction head, so no tail checkpoint ever covers
/// the next job and, before the pinned head existed, every job re-prefilled
/// everything (the observed backlog-drain pathology: dozens of
/// "re-prefilling all N tokens" lines, all diverging at the shared head).
///
/// The engine learns the head from the first observed divergence and pins a
/// checkpoint there; from the third job on, only the job's unique content is
/// prefilled.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn pinned_head_absorbs_oneshot_divergence_on_background_slot() {
    let path = std::env::var("FARO_TEST_GGUF").expect("set FARO_TEST_GGUF to a .gguf file");
    let mut e = LlamaCppEngine::new(LlamaCppConfig {
        model_path: path.into(),
        draft_path: std::env::var("FARO_TEST_DRAFT_GGUF").ok().map(Into::into),
        n_ctx: 4096,
        n_slots: 2,
        enable_thinking: false,
        swa_full: false,
        ..Default::default()
    });
    // A shared head long enough to slide the SWA window (so divergence past
    // it is unrecoverable without the pin), and unique tails LONGER than the
    // ring's tail margin: each job's ring checkpoint then sits inside the
    // previous job's unique content, where it can never cover the next job,
    // which is exactly the real workload's shape (100-2,800-token unique
    // email bodies). A short tail would let the ring checkpoint land inside
    // the shared head and mask the pinned path entirely.
    let head = filler("shared-instructions", 60);
    let job = |unique_tag: &str| GenerateRequest {
        system: head.clone(),
        messages: vec![Msg::user(format!("{}\n\nSay OK.", filler(unique_tag, 8)))],
        enable_thinking: Some(false),
        slot: 1,
        ..Default::default()
    };

    // Job 1: cold, full prefill; nothing known about the head yet.
    let r1 = e.generate(job("first-invoice")).expect("job 1");
    assert!(!r1.text.trim().is_empty());

    // Job 2: diverges at the head with no covering checkpoint, so it pays the
    // full re-prefill, and is the divergence the head is LEARNED from.
    let r2 = e.generate(job("second-review")).expect("job 2");
    assert!(!r2.text.trim().is_empty());

    // Job 3 on: the pinned head must absorb the divergence.
    let r3 = e.generate(job("third-migration")).expect("job 3");
    assert!(!r3.text.trim().is_empty());
    assert!(
        r3.timings.prefill_ms < r2.timings.prefill_ms / 2,
        "the pinned head must absorb one-shot divergence from job 3 on \
         (job2 {}ms, job3 {}ms)",
        r2.timings.prefill_ms,
        r3.timings.prefill_ms,
    );

    // And it keeps holding, job after job (the backlog-drain shape).
    let r4 = e.generate(job("fourth-offsite")).expect("job 4");
    assert!(!r4.text.trim().is_empty());
    assert!(
        r4.timings.prefill_ms < r2.timings.prefill_ms / 2,
        "the pinned head must keep absorbing (job2 {}ms, job4 {}ms)",
        r2.timings.prefill_ms,
        r4.timings.prefill_ms,
    );

    // Slot 0's machinery must be untouched by slot 1's pin.
    let r5 = ask(&mut e, vec![Msg::user("Say OK.".to_string())]);
    assert!(!r5.text.trim().is_empty());
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
