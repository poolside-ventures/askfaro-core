//! Weight specs for the local generation providers.
//!
//! Same shape as `stt::models` and `search::models`, so a host provisions all
//! on-device models through one pipeline (streaming download, sha256 verify,
//! progress events) and stores them under one cache root the user can inspect.

use crate::model::{ModelFile, ModelSpec};

/// Gemma 4 E2B, instruction-tuned, **quantization-aware-trained q4_0**.
/// **The mobile brain, and only that.**
///
/// This spec was removed on 2026-08-09 and is restored deliberately, for a
/// workload the disqualification never covered. It was cut because the 20-case
/// bench found 5 deterministic TOOL-CALLING failures (one wrong tool, four
/// argument misses) that E4B does not have — a verdict about an agent loop
/// choosing and parameterising tools. Mobile runs no agent loop: it runs
/// one-shot drafting and two grammar-constrained style derivations, where the
/// sampler makes a malformed answer unrepresentable rather than merely
/// unlikely. On that path E2B scored 26/26 (drafting, both overlay depths and
/// the profile compile) and beat E4B on decisiveness, 30/30 against 24/30; it
/// lost on engaging with specifics, 60% against 90%. See
/// `scope/docs/on-device-capability-status.md`.
///
/// **Do not promote this to the desktop brain on the strength of that.** The
/// tool-calling failures are still real and still unfixed; nothing here
/// re-litigates them.
///
/// What forces the choice is memory. E4B is 5.15 GB of weights before KV cache
/// and Metal compute buffers, which no shipping iPhone can hold in a
/// foregrounded app. E2B at 3.35 GB is the only member of the family that is
/// even a candidate, and whether it fits is itself measured, not assumed.
///
/// Google's own QAT GGUF, chosen over the community requantizations so the
/// existing measurements transfer rather than needing to be re-established.
/// Byte-comparable to what Ollama serves as `gemma4:e2b-it-qat`: 3,349,516,256
/// here against Ollama's 3,349,514,112, a 2,144-byte difference that is Ollama's
/// GGUF metadata rewriting, not different weights. Swapping quant (Q4_K_M,
/// IQ4_XS, a UD variant) is a **measurable change, not a free one**.
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

/// Gemma 4 E4B, instruction-tuned, QAT q4_0. **The shipping DESKTOP brain.**
///
/// Chosen over E2B on quality: E4B is materially better at memory processing,
/// which is the work that matters, and it does not have E2B's deterministic
/// tool-calling failures. That verdict stands for anything running an agent
/// loop. It does not reach mobile, which cannot hold these weights at all —
/// see [`GEMMA4_E2B_IT_QAT_Q4_0`] above for the workload that survives on E2B
/// and the reason it is a different question.
///
/// It is NOT chosen on raw decode, where it runs at roughly half E2B's rate. It
/// wins the thing a user actually waits for, because it says the same thing in
/// 41% fewer tokens: measured p50 4,484ms against E2B's 4,735ms, and p95 7,786ms
/// against 9,357ms. **That win depends on the MTP drafter below**; without
/// speculation E4B is a latency regression (p50 5,802ms), so the two ship
/// together or not at all.
///
/// # Why this URL is ours and not Google's
///
/// The file we serve is Google's file with **two embedding tables requantised
/// from Q6_K to Q4_0** — `token_embd.weight` and `per_layer_token_embd.weight`
/// — and nothing else touched. 5,154,941,280 bytes becomes 4,255,263,488, and
/// the Metal allocation falls 858 MiB, measured, which is the whole point:
/// `per_layer_token_embd` alone is 693 MiB of resident weights that a token
/// never streams, because it is a row gather rather than a matmul.
///
/// **Quality does not move.** 30-case F-7, 64k window, MTP drafter attached, on
/// an M1 Pro: 26 of 27 tool cases, 100% tool selection, 100% abstain, 0 errors
/// — the same score the unmodified weights get in the same harness on the same
/// day. The single miss is not the same case in both (`note_long_multiline`
/// here, `email_draft` on the unmodified run), which is what one case at n=27
/// looks like and is why the score is the claim rather than the case.
///
/// **It is a transform, not a fork, and that is enforced rather than promised.**
/// CI downloads Google's file, checks Google's sha256, runs
/// `scripts/requant_gguf.py`, and asserts the result hashes to the constant
/// below; a transform that ever produced different bytes fails the build
/// instead of publishing something nobody benched. All 664 untouched tensors
/// were verified byte-identical to Google's afterwards.
///
/// **No public GGUF is this file**, and that was established by reading tensor
/// tables over HTTP range reads rather than by trusting names. Every E4B QAT
/// GGUF on the Hub was enumerated and the plausible ones byte-compared against
/// Google's. unsloth's `UD-Q4_K_XL` and its re-uploads have the right TYPE
/// table but share zero of 342 body tensors with Google, because `q4_0 -> q4_0`
/// is not idempotent; they also drop `per_layer_model_proj` to Q4_0. The only
/// near-match, `FadedRedStar/...-heretic-Q4_0_PURE`, carries Google's bytes for
/// 647 of 663 non-embedding tensors — the 16 that differ are
/// `blk.20..35.attn_output`, a deliberate refusal-ablation edit, which is worse
/// than a requantisation, not better. Ours keeps `per_layer_model_proj` at F16
/// and its histogram is exactly 344 Q4_0 + 321 F32 + 1 F16.
///
/// **Licence.** Gemma 4 is Apache-2.0 (the Gemma Terms of Use appendix does not
/// list it), which permits distributing a modified copy. The obligations are
/// met in the file itself: Google's `general.license`, `general.license.link`
/// and `general.base_model.*` keys are copied verbatim, and a
/// `general.description` naming the modification and its provenance is
/// appended.
pub const GEMMA4_E4B_IT_QAT_Q4_0: ModelSpec = ModelSpec {
    id: "gemma-4-e4b-it-qat-q4_0",
    display_name: "Gemma 4 E4B (instruction-tuned, QAT q4_0, embeddings q4_0)",
    files: &[ModelFile {
        // Content-addressed on the sha below, so republishing different bytes
        // can never be served from a cache of the old ones.
        name: "gemma-4-E4B_q4_0-it-embd-q4_0.gguf",
        url: "https://files.scopy.app/ondevice/weights/gemma-4-e4b-it-qat-q4_0-embd-q4_0/7a06b5147b6e/gemma-4-E4B_q4_0-it-embd-q4_0.gguf",
        sha256: "7a06b5147b6e0e6f64734cc7f2050224041a59edea98c9d70860dfa83ccd2724",
        size: 4_255_263_488,
    }],
};

/// Google's unmodified E4B QAT q4_0, the INPUT to the requantisation above.
///
/// Not a model the app provisions: nothing loads these weights, and the desktop
/// brain is [`GEMMA4_E4B_IT_QAT_Q4_0`]. It is here so the publisher and this
/// crate name the same source file, sha and size in one place — the release
/// workflow reads it to fetch and verify Google's original before transforming
/// it, and a reader comparing the two specs can see exactly what changed.
pub const GEMMA4_E4B_IT_QAT_Q4_0_UPSTREAM: ModelSpec = ModelSpec {
    id: "gemma-4-e4b-it-qat-q4_0-upstream",
    display_name: "Gemma 4 E4B (instruction-tuned, QAT q4_0, Google original)",
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
