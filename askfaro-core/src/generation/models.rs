//! Weight specs for the local generation providers.
//!
//! Same shape as `stt::models` and `search::models`, so a host provisions all
//! on-device models through one pipeline (streaming download, sha256 verify,
//! progress events) and stores them under one cache root the user can inspect.

use crate::model::{ModelFile, ModelSpec};

/// Gemma 4 E4B, instruction-tuned, QAT q4_0. **The shipping brain.**
///
/// Chosen over E2B on quality: E4B is materially better at memory processing,
/// which is the work that matters. E2B's spec was REMOVED 2026-08-08 — its
/// error rate made it unusable in practice (the 20-case bench that once
/// scored it 94% later showed 5 deterministic failures E4B does not have),
/// and nothing shipped it. The numbers below that cite E2B are historical
/// baselines, not an available alternative.
///
/// It is NOT chosen on raw decode, where it runs at roughly half E2B's rate. It
/// wins the thing a user actually waits for, because it says the same thing in
/// 41% fewer tokens: measured p50 4,484ms against E2B's 4,735ms, and p95 7,786ms
/// against 9,357ms. **That win depends on the MTP drafter below**; without
/// speculation E4B is a latency regression (p50 5,802ms), so the two ship
/// together or not at all.
pub const GEMMA4_E4B_IT_QAT_Q4_0: ModelSpec = ModelSpec {
    id: "gemma-4-e4b-it-qat-q4_0",
    display_name: "Gemma 4 E4B (instruction-tuned, QAT q4_0)",
    files: &[ModelFile {
        name: "gemma-4-E4B_q4_0-it.gguf",
        url: "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf",
        sha256: "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee",
        size: 5_154_941_280,
    }],
};

/// The Multi-Token Prediction drafter for E4B, for speculative decoding.
///
/// 77 MB against E4B's 5.15 GB, and worth **+30% decode** (29.8 to 38.8 tok/s).
/// Speculation is lossless by construction and was verified so: all 20 bench
/// cases came back byte-identical with the drafter on and off.
///
/// **q4_0 is deliberate, and the intuition for picking it is the opposite of the
/// obvious one.** Acceptance sits at 70-72% regardless of drafter precision, so
/// it is bounded by the drafter's architecture and the task rather than its
/// numerics. A *better* drafter therefore buys nothing while costing more per
/// step: bf16 (172 MB) measured 32.7 tok/s, BELOW the q8_0 baseline. The lever
/// is drafter COST, not drafter quality.
///
/// Pair only with the matching E4B target. Some community conversions declare a
/// `gemma4_mtp` architecture that llama.cpp does not know and fail to load; this
/// one is the q4_0 conversion of Google's own unquantized assistant.
pub const GEMMA4_E4B_MTP_DRAFTER_Q4_0: ModelSpec = ModelSpec {
    id: "gemma-4-e4b-it-qat-assistant-q4_0",
    display_name: "Gemma 4 E4B MTP drafter (QAT q4_0)",
    files: &[ModelFile {
        name: "gemma-4-E4B-it-qat-assistant-q4_0.gguf",
        url: "https://huggingface.co/cascade-tech/gemma-4-E4B-it-qat-q4_0-unquantized-assistant-gguf/resolve/main/gemma-4-E4B-it-qat-assistant-q4_0.gguf",
        sha256: "866800a8c56e77cc788340e47d79c03e4368451467d757985666dcdbf8c06e90",
        size: 76_978_624,
    }],
};
