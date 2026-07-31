//! Weight specs for the local generation providers.
//!
//! Same shape as `stt::models` and `search::models`, so a host provisions all
//! on-device models through one pipeline (streaming download, sha256 verify,
//! progress events) and stores them under one cache root the user can inspect.

use crate::model::{ModelFile, ModelSpec};

/// Gemma 4 E2B, instruction-tuned, **quantization-aware-trained q4_0**.
///
/// This is Google's own QAT GGUF, chosen deliberately over the many community
/// requantizations: it is the same quantization every measurement in
/// `scope/docs/on-device-runtime-in-process.md` was taken on, so the baseline
/// (100% tool selection, 89% fully-correct, ~65 tok/s decode on an M1 Pro)
/// transfers rather than needing to be re-established.
///
/// It is byte-comparable to what Ollama serves as `gemma4:e2b-it-qat`: 3,349,516,256
/// here against Ollama's 3,349,514,112, a 2,144-byte difference that is Ollama's
/// GGUF metadata rewriting, not different weights. Swapping to a different quant
/// (Q4_K_M, IQ4_XS, an UD variant) is a **measurable change, not a free one**:
/// re-run `desktop/spikes/f7-tool-calling/bench.mjs` before believing the old
/// numbers still hold.
///
/// One file, unlike the ONNX specs: a GGUF carries weights, tokenizer and chat
/// template together, which is also why the engine can read the chat template
/// straight out of the model rather than being told which family it is.
pub const GEMMA4_E2B_IT_QAT_Q4_0: ModelSpec = ModelSpec {
    id: "gemma-4-e2b-it-qat-q4_0",
    display_name: "Gemma 4 E2B (instruction-tuned, QAT q4_0)",
    files: &[ModelFile {
        name: "gemma-4-E2B_q4_0-it.gguf",
        url: "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf",
        sha256: "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634",
        size: 3_349_516_256,
    }],
};

/// Gemma 4 E4B, instruction-tuned, QAT q4_0. **The shipping brain.**
///
/// Chosen over E2B on quality: E4B is materially better at memory processing,
/// which is the work that matters and which the 20-case tool bench is far too
/// saturated to show (both score 94% fully correct there).
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
