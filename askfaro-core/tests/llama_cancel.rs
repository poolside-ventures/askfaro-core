//! Cooperative cancellation: a host can stop an in-flight call instead of
//! waiting it out, and the engine is still good afterwards.
//!
//! The measured shape this exists for: Scope desktop holds the engine behind a
//! `Mutex`, and quitting has to take that mutex to drop it. Turns measured at
//! p50 16.5s, p90 31.8s, max 65.4s, so Cmd+Q froze the app for as long as a
//! turn had left. The three places that time is spent are the three places a
//! stop has to be honoured, and each one is a separate mechanism:
//!
//!  - the model load, through llama.cpp's progress callback,
//!  - prefill, by splitting a prompt into micro-batch chunks (a single
//!    `llama_decode` over 7,600 tokens is uninterruptible from Rust),
//!  - decode, by checking once per token.
//!
//! Ignored by default (needs real GGUF weights). Run with:
//!
//!   FARO_TEST_GGUF=/path/to/model.gguf \
//!   cargo test -p askfaro-core --features llama-cpp --test llama_cancel -- --ignored --nocapture

#![cfg(feature = "llama-cpp")]

use std::time::{Duration, Instant};

use askfaro_core::generation::{
    CancelHandle, GenError, GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
};

fn engine(cancel: &CancelHandle, n_ctx: u32) -> LlamaCppEngine {
    let path = std::env::var("FARO_TEST_GGUF").expect("set FARO_TEST_GGUF to a .gguf file");
    LlamaCppEngine::new(LlamaCppConfig {
        model_path: path.into(),
        n_ctx,
        n_slots: 1,
        enable_thinking: false,
        // Small enough that one chunk is a fraction of a second, which is the
        // bound this test is asserting on.
        n_ubatch: Some(128),
        cancel: Some(cancel.clone()),
        ..Default::default()
    })
}

/// Raise the flag from ANOTHER thread while the engine is busy, which is the
/// only way a host can use this: the thread inside `generate` is the one
/// holding the host's lock.
fn cancel_after(cancel: &CancelHandle, delay: Duration) -> std::thread::JoinHandle<()> {
    let c = cancel.clone();
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        c.cancel();
    })
}

fn long_answer(prompt: &str) -> GenerateRequest {
    GenerateRequest {
        system: "You are verbose. Answer at length and never stop early.".into(),
        messages: vec![Msg::user(prompt)],
        enable_thinking: Some(false),
        ..Default::default()
    }
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

/// The decode-loop point, plus the reusability contract that makes the whole
/// feature worth having: a stopped turn must not cost the next one.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn a_stopped_decode_returns_cancelled_and_the_engine_still_generates() {
    let cancel = CancelHandle::new();
    let mut e = engine(&cancel, 4096);
    // Out of the way, so what this measures is decode and not the load.
    e.warm().expect("warm");

    let t = cancel_after(&cancel, Duration::from_millis(1500));
    let started = Instant::now();
    let err = e
        .generate(long_answer("Count from 1 to 400, one number per line."))
        .expect_err("a cancelled turn must not come back as a reply");
    let elapsed = started.elapsed();
    t.join().unwrap();

    assert!(
        matches!(err, GenError::Cancelled),
        "a stop must be distinguishable from a failure, got: {err}"
    );
    // The cancel lands at 1.5s; anything past that is the abort latency plus
    // the prompt's own prefill, and a per-token check bounds it to one step.
    assert!(
        elapsed < Duration::from_secs(5),
        "cancelling took {elapsed:?}, which is not promptly"
    );
    println!("decode cancel: {elapsed:?} total, {:?} after the stop", elapsed - Duration::from_millis(1500));

    // The point of cooperative cancellation over dropping the engine: the
    // weights are still warm and the KV cache still agrees with the slot
    // bookkeeping, so the next turn is an ordinary turn.
    assert!(e.is_warm(), "a cancelled turn must not unload the weights");
    let ok = e
        .generate(GenerateRequest {
            system: "Answer with a single short word.".into(),
            messages: vec![Msg::user("Say OK.")],
            enable_thinking: Some(false),
            ..Default::default()
        })
        .expect("the engine must be reusable after a cancelled turn");
    assert!(!ok.text.trim().is_empty(), "empty generation after a cancel");
    println!("next turn: {:?} ({} tok)", ok.text, ok.timings.output_tokens);
}

/// The prefill point. A per-token check cannot reach this: the whole prompt
/// goes into ONE `llama_decode`, so without chunking the flag is not read again
/// until the first token is sampled, tens of seconds later.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn a_stopped_prefill_returns_cancelled_before_the_prompt_is_read() {
    let cancel = CancelHandle::new();
    let mut e = engine(&cancel, 8192);
    e.warm().expect("warm");

    // Thousands of tokens, so prefill alone is many seconds of work inside one
    // decode call.
    let prompt = format!("{}\n\nSummarise the above.", filler("alpha", 220));

    // What this prompt costs when nobody stops it, measured rather than
    // assumed. Without it the assertion below is a wall-clock guess that a
    // faster machine turns into a test which passes for the wrong reason: the
    // stop landing in the DECODE loop after prefill had already finished. A
    // different system block keeps the two turns from sharing any prefix, so
    // this one is not paying for the other's cache.
    let cold = e
        .generate(GenerateRequest {
            system: "Reply briefly.".into(),
            messages: vec![Msg::user(prompt.clone())],
            enable_thinking: Some(false),
            ..Default::default()
        })
        .expect("uncancelled reference turn");
    let reference = Duration::from_millis(cold.timings.prefill_ms);
    println!(
        "reference: {} prompt tokens, {reference:?} of prefill",
        cold.timings.prompt_tokens
    );
    assert!(
        reference > Duration::from_secs(3),
        "this prompt prefills in {reference:?}, too fast to prove anything about \
         interrupting prefill; lengthen the filler"
    );

    let stop_at = Duration::from_millis(800);
    let t = cancel_after(&cancel, stop_at);
    let started = Instant::now();
    let err = e
        .generate(long_answer(&prompt))
        .expect_err("a cancelled prefill must not come back as a reply");
    let elapsed = started.elapsed();
    t.join().unwrap();

    assert!(
        matches!(err, GenError::Cancelled),
        "a stopped prefill must report a stop, got: {err}"
    );
    // Stopped well inside the prefill it was asked to stop, which is the claim.
    // A stop honoured only at the decode loop would land past `reference`.
    assert!(
        elapsed < stop_at + reference / 4,
        "prefill ran {elapsed:?} after a stop at {stop_at:?}, against {reference:?} \
         of prefill for this prompt; the stop was not honoured inside prefill"
    );
    println!("prefill cancel: {elapsed:?} total, {:?} after the stop", elapsed - stop_at);

    assert!(e.is_warm(), "a cancelled prefill must not unload the weights");
    let ok = e
        .generate(GenerateRequest {
            system: "Answer with a single short word.".into(),
            messages: vec![Msg::user("Say OK.")],
            enable_thinking: Some(false),
            ..Default::default()
        })
        .expect("the engine must be reusable after a cancelled prefill");
    assert!(!ok.text.trim().is_empty(), "empty generation after a cancel");
}

/// The model-load point, and the reset rule. A cold load is the longest single
/// thing this engine does (measured up to 40.4s for E4B's 5.15 GB), and it
/// happens before any handle could be taken off an engine, which is why the
/// handle is passed IN.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn a_stopped_load_returns_cancelled_and_the_next_warm_succeeds() {
    let cancel = CancelHandle::new();
    let mut e = engine(&cancel, 4096);

    let t = cancel_after(&cancel, Duration::from_millis(150));
    let err = e.warm().expect_err("a cancelled load must not report success");
    t.join().unwrap();

    assert!(
        matches!(err, GenError::Cancelled),
        "a stopped load must report a stop, got: {err}"
    );
    assert!(!e.is_warm(), "an aborted load must leave nothing resident");

    // The reset rule: the flag belongs to the call it arrived during, so the
    // next call starts clean. A latched flag would kill every load from here on.
    e.warm().expect("the next warm must load, not inherit the stop");
    assert!(e.is_warm());
}

/// Nothing set the flag, so nothing changes: the engine must generate exactly
/// as it does without a handle. The armed path takes different code (chunked
/// prefill, a progress callback on the load), so "armed but never used" is its
/// own case.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF"]
fn an_armed_handle_that_is_never_raised_changes_nothing() {
    let cancel = CancelHandle::new();
    let mut e = engine(&cancel, 4096);
    let r = e
        .generate(GenerateRequest {
            system: "Answer with a single short word.".into(),
            messages: vec![Msg::user("Say OK.")],
            enable_thinking: Some(false),
            ..Default::default()
        })
        .expect("an armed but unraised handle must not affect a turn");
    assert!(!r.text.trim().is_empty());
    assert!(!cancel.is_cancelled());
    println!(
        "armed, unraised: {:?} (load {}ms, prefill {}ms, decode {}ms)",
        r.text, r.timings.load_ms, r.timings.prefill_ms, r.timings.decode_ms
    );
}
