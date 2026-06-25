//! Model provisioning for search — the EmbeddingGemma [`ModelSpec`], using the
//! shared [`crate::model`] download/verify subsystem (the same one
//! `core-stt` uses). Network-free: the crate owns the spec + sha256
//! verification; the host downloads.
//!
//! **The hard rule (RFC §7):** `model identity == space name`. This spec's `id`
//! is the canonical space name for the EmbeddingGemma vectors; any change that
//! shifts vectors (model, quantized variant, dimensionality) is a NEW space +
//! backfill, never an in-place re-embed.

pub use crate::model::{is_present, missing, verify, ModelFile, ModelSpec};

/// The embedding **space** name for EmbeddingGemma vectors (the hard rule, §7).
///
/// Distinct from the model directory id below because a space name is also a SQL
/// column (`embedding_<space>`), so it must be a valid identifier
/// (`[a-z][a-z0-9_]*` — no hyphens). Changing the model/variant/dims means a new
/// space constant + backfill, never an in-place re-embed.
pub const EMBEDDINGGEMMA_SPACE: &str = "embeddinggemma_300m_fp32";

/// EmbeddingGemma-300M, fp32 ONNX (`onnx-community/embeddinggemma-300m-ONNX`).
///
/// fp32 is the parity-exact reference (Phase 1 spike: cosine 1.0 vs the Python
/// `onnxruntime` pipeline). The shipping shard will likely swap to a quantized
/// variant (q4 ≈ 200 MB) once its recall drop is measured — which, per the hard
/// rule, is a different space id.
///
/// External-data ONNX: two files (`model.onnx` graph + `model.onnx_data`
/// weights) plus the tokenizer. The host downloads all three; the crate verifies
/// all three.
pub const EMBEDDINGGEMMA_300M_FP32: ModelSpec = ModelSpec {
    id: "embeddinggemma-300m-fp32",
    display_name: "EmbeddingGemma 300M (multilingual, fp32)",
    files: &[
        ModelFile {
            name: "model.onnx",
            url: "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model.onnx",
            sha256: "ea91fd315a7c152d427d231746f0f811a1ac93beaba656abfdf2b24e091265e4",
            size: 479_932,
        },
        ModelFile {
            name: "model.onnx_data",
            url: "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model.onnx_data",
            sha256: "ef835ae565d8695236652475903078e8ed794c7c35faf1164d78ec3238e8a88d",
            size: 1_234_521_088,
        },
        ModelFile {
            name: "tokenizer.json",
            url: "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer.json",
            sha256: "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47",
            size: 20_323_312,
        },
    ],
};
