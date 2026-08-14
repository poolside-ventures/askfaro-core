//! The invariant the whole prefix-reuse scheme rests on: after a turn, the KV
//! cache holds exactly the tokens the engine recorded for that slot.
//!
//! Break it and nothing throws. The next turn's common-prefix scan reads the
//! record, finds its prompt extends it, takes the fast path that trims nothing,
//! and decodes on top of cells nobody accounted for. Two cells then carry the
//! same sequence and the same position, attention sees both, and the answer is
//! fluent and wrong. This is the same class of failure as
//! `examples/prefix_invalidation.rs`, one layer down.
//!
//! The speculative path is where it actually broke: the commit loop for accepted
//! drafts used to `break 'gen` on the end-of-turn token and on the output cap,
//! which jumped over the KV rollback of the proposals that same window did not
//! commit. A turn ending inside an accepted draft is the common case, not a
//! corner: end-of-turn is the single most predictable token there is, so the
//! drafter proposes it and the target accepts it.
//!
//! Ignored by default (needs real GGUF weights). Run with:
//!
//!   FARO_TEST_GGUF=/path/to/model.gguf \
//!   FARO_TEST_DRAFT_GGUF=/path/to/mtp-drafter.gguf \
//!   cargo test -p askfaro-core --features llama-cpp --test llama_kv_invariant -- --ignored --nocapture

#![cfg(feature = "llama-cpp")]

use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
};

fn engine() -> LlamaCppEngine {
    let path = std::env::var("FARO_TEST_GGUF").expect("set FARO_TEST_GGUF to a .gguf file");
    LlamaCppEngine::new(LlamaCppConfig {
        model_path: path.into(),
        // The drafter is the point of this test, not an optional extra: without
        // it there is no speculative window and nothing to roll back.
        draft_path: Some(
            std::env::var("FARO_TEST_DRAFT_GGUF")
                .expect("set FARO_TEST_DRAFT_GGUF to the MTP drafter .gguf")
                .into(),
        ),
        n_ctx: 4096,
        n_slots: 1,
        enable_thinking: false,
        ..Default::default()
    })
}

fn ask(e: &mut LlamaCppEngine, messages: Vec<Msg>) -> String {
    e.generate(GenerateRequest {
        system: "Answer in one short sentence.".into(),
        messages,
        enable_thinking: Some(false),
        ..Default::default()
    })
    .expect("generate")
    .text
}

fn assert_cache_matches_record(e: &mut LlamaCppEngine, when: &str) {
    let (prefix_len, pos_max) = e.kv_prefix_state(0).expect("engine is warm");
    assert_eq!(
        prefix_len as i32,
        pos_max + 1,
        "{when}: the slot records {prefix_len} tokens but its KV cache holds {} \
         (positions 0..={pos_max}). The next turn trusts the record, so the \
         extra cells are read as context nobody wrote.",
        pos_max + 1,
    );
}

/// Turn 1 alone is enough to catch the regression: the mismatch exists the
/// moment the turn returns, before any second prompt can hide or amplify it.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF and FARO_TEST_DRAFT_GGUF"]
fn a_turn_leaves_the_cache_matching_its_record() {
    let mut e = engine();
    // Short answers end on the end-of-turn token quickly, which is exactly the
    // window this is about.
    let a = ask(&mut e, vec![Msg::user("Say the word orange and nothing else.")]);
    println!("turn 1: {a:?}");
    assert_cache_matches_record(&mut e, "after turn 1");

    // And it must still hold once a second turn has appended to the first, the
    // path that reuses the record rather than rebuilding it.
    let b = ask(
        &mut e,
        vec![
            Msg::user("Say the word orange and nothing else."),
            Msg::assistant(&a),
            Msg::user("Now say the word blue and nothing else."),
        ],
    );
    println!("turn 2: {b:?}");
    assert_cache_matches_record(&mut e, "after turn 2");

    // The behavioural half: a corrupted context does not fail, it drifts. The
    // second answer has to be about the second question.
    assert!(
        b.to_lowercase().contains("blue"),
        "turn 2 answered {b:?}, which does not contain the word it was asked for; \
         a stale cell in the cache is the first thing to suspect",
    );
}

/// Several short turns in a row, because the rollback has to hold on EVERY
/// window, and a single turn only exercises the last one.
#[test]
#[ignore = "requires real GGUF weights on FARO_TEST_GGUF and FARO_TEST_DRAFT_GGUF"]
fn the_record_holds_across_a_conversation() {
    let mut e = engine();
    let mut history = Vec::new();
    for (i, q) in [
        "What is two plus two? Answer with the number only.",
        "And plus three?",
        "And plus four?",
    ]
    .iter()
    .enumerate()
    {
        history.push(Msg::user(*q));
        let a = ask(&mut e, history.clone());
        println!("turn {}: {a:?}", i + 1);
        history.push(Msg::assistant(&a));
        assert_cache_matches_record(&mut e, &format!("after turn {}", i + 1));
    }
}
