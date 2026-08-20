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
///
/// The device runs the fp16 export and still writes and queries THIS space. That
/// is the one case where the rule is satisfied by measurement instead of by the
/// file name: fp16 does not shift the vectors, it rounds them (see
/// [`EMBEDDINGGEMMA_FP16`] for the numbers). Do not read it as licence to point
/// a quantized variant at this space — q8 and q4 both move results, measured,
/// and both need their own space.
pub const EMBEDDINGGEMMA_SPACE: &str = "embeddinggemma_300m_fp32";

/// A weight variant of EmbeddingGemma: which files to fetch, which graph inside
/// that directory to open, and which space the vectors belong to.
///
/// The three travel together because getting them apart is the failure the hard
/// rule exists to prevent. The graph name is not derivable from the directory —
/// the quantized exports keep their upstream names, because an ONNX graph's
/// external-data record names its own weight file and renaming the pair breaks
/// the load.
pub struct GemmaVariant {
    pub spec: &'static ModelSpec,
    pub graph: &'static str,
    pub space: &'static str,
}

/// **fp16 — what the device ships.** Same weights as fp32 at half the bytes:
/// 609 MiB on disk against 1,198, and 338 MiB resident against 572. It is not a
/// quantization in the sense the hard rule is about, and it is the only variant
/// measured to leave retrieval untouched, so it stays in the fp32 space rather
/// than forcing a backfill.
///
/// The 338 is only reachable with [`GemmaOptions`]' defaults. ONNX Runtime has
/// no fp16 CPU kernels, so it casts to fp32; under ITS defaults it caches those
/// casts forever and this variant measures 640 MiB, worse than the fp32 it
/// replaces. The knobs are worth 302 MiB here and nothing at all on fp32.
///
/// Measured 2026-08-20 on an M1 Pro, against a real 6,696-row device shard whose
/// document vectors are the server's fp32 ones, 300 known-item queries:
///  - cosine(fp32 query vector, fp16 query vector): mean 0.99999992, min
///    0.99999962 — fp32 rounding noise, not a shift in geometry;
///  - top-10 overlap against the identical fp32 document vectors: **1.000**, so
///    not one result moved in or out of a single result set;
///  - known-item MRR 0.4935 for both, to four decimals.
///
/// **The one thing to know before bumping `ort`.** Google's card says
/// EmbeddingGemma activations do not support fp16, and the ONNX card repeats it
/// ("use fp32, q8 or q4"). The warning is real and its mechanism is real: the
/// embedding output is scaled by sqrt(hidden_size) and fp32 hidden states have
/// been measured around 264,000, well past fp16's 65,504 ceiling. It does not
/// materialize here because ONNX Runtime's CPU execution provider has no fp16
/// kernels — it inserts Casts and computes in fp32, which is exactly why the
/// vectors come out identical and why this variant is SLOWER, not faster (query
/// p50 122ms against fp32's 48ms).
///
/// So the safety of this choice rests on an implementation detail of the CPU EP,
/// not on the model supporting fp16. A runtime that genuinely computes in fp16 —
/// CoreML, WebGPU, an NPU, or a future ORT with native ARM fp16 kernels — would
/// produce garbage, silently. `tests/gemma_fp16_parity.rs` is the tripwire: run
/// it against both graphs on any `ort` bump or execution-provider change.
///
/// The three genuinely quantized variants are recorded below for the next person
/// who asks, because "did anyone try q8" deserves a number rather than a guess.
/// Same corpus, same 300 queries. Cosine to the fp32 query vector / top-10
/// overlap / resident, against fp16's 1.000 / 1.000 / 338 MiB:
///  - `model_quantized` (int8), 315 MiB on disk: 0.9943 / 0.966 / **219 to 1,024
///    MiB across runs**. It does no int8 arithmetic at all — the graph is 175
///    `DequantizeLinear` feeding fp32 `MatMul`, with one scalar scale for the
///    whole 201M-parameter embedding table — so it expands at inference. Four
///    times smaller to download, worse RAM than the fp32 it would replace.
///  - `model_q4`, 207 MiB: 0.9703 / 0.906 / 228 MiB, and the only variant that
///    is also FASTER than fp32 (27ms against 48ms). It costs one result in ten,
///    plus a new space and a backfill of every row the server has indexed.
///  - `model_no_gather_q4`, 205 MiB: 0.9584 / 0.866, and roughly 4x slower. It
///    replaces the embedding `Gather` with a one-hot matrix times the full
///    table, for runtimes without `GatherBlockQuantized`; ORT has had that since
///    1.20. Never ship it here.
///  - `model_q4f16`, 186 MiB: **segfaults** on load under `ort` 2.0.0-rc.12 on
///    Apple Silicon (SIGSEGV, not an error return). Not a candidate.
pub const EMBEDDINGGEMMA_FP16: GemmaVariant = GemmaVariant {
    spec: &EMBEDDINGGEMMA_300M_FP16,
    graph: "model_fp16.onnx",
    space: EMBEDDINGGEMMA_SPACE,
};

/// fp32 — the parity-exact reference the space is named for, and what the server
/// still embeds documents with. Kept as a variant so a parity check can load
/// both without hand-assembling the triple.
pub const EMBEDDINGGEMMA_FP32: GemmaVariant = GemmaVariant {
    spec: &EMBEDDINGGEMMA_300M_FP32,
    graph: "model.onnx",
    space: EMBEDDINGGEMMA_SPACE,
};

/// EmbeddingGemma-300M, fp16 ONNX (`onnx-community/embeddinggemma-300m-ONNX`).
/// See [`EMBEDDINGGEMMA_FP16`] for why this one ships.
pub const EMBEDDINGGEMMA_300M_FP16: ModelSpec = ModelSpec {
    id: "embeddinggemma-300m-fp16",
    display_name: "EmbeddingGemma 300M (multilingual, fp16)",
    files: &[
        ModelFile {
            name: "model_fp16.onnx",
            url: "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_fp16.onnx",
            sha256: "dcfaf21ff7cae91af9295366ac0d7352efcadeaf7deefb98f82d5056502d0bf2",
            size: 655_263,
        },
        ModelFile {
            name: "model_fp16.onnx_data",
            url: "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_fp16.onnx_data",
            sha256: "1cd839755aa8e24d5af7f16ef275b12d717a4401bb009099b8c17e4156d3d5d5",
            size: 617_434_112,
        },
        ModelFile {
            name: "tokenizer.json",
            url: "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer.json",
            sha256: "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47",
            size: 20_323_312,
        },
    ],
};

/// EmbeddingGemma-300M, fp32 ONNX (`onnx-community/embeddinggemma-300m-ONNX`).
///
/// fp32 is the parity-exact reference (Phase 1 spike: cosine 1.0 vs the Python
/// `onnxruntime` pipeline), and still what the SERVER embeds documents with.
/// The device moved to [`EMBEDDINGGEMMA_FP16`]; the space is unchanged because
/// the two produce the same vectors to seven decimals.
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
