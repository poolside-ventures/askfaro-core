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
