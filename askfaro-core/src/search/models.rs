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

/// The embedding **space** for the QAT 4-bit embedder — a different space from
/// [`EMBEDDINGGEMMA_SPACE`] because it is a different set of weights, not a
/// different precision of the same ones. Adopting it means a backfill.
pub const EMBEDDINGGEMMA_QAT_Q4_SPACE: &str = "embeddinggemma_300m_qat_q4";

/// **QAT 4-bit — what the device ships.** 199 MiB of weights against fp16's
/// 589, 250 MiB resident against 363, and roughly a third of the query latency.
///
/// The reason this is not the `model_q4.onnx` everyone else uses: every
/// published 4-bit ONNX of EmbeddingGemma is quantized from the ORDINARY
/// checkpoint, and Google also ships a checkpoint whose weights were TRAINED to
/// sit on the 4-bit grid. Nobody had exported that one, so we do
/// (`scripts/build_embeddinggemma_qat_q4.py` in the Scope repo, reproduced and
/// sha-gated by CI before it publishes).
///
/// Measured 2026-08-20 on an M1 Pro against a real 6,696-row device shard,
/// 1,500 documents re-embedded per arm, 200 known-item queries, both sides of
/// each arm using the same weights:
///
/// | variant | disk | resident | query p50 | known-item MRR |
/// |---|---|---|---|---|
/// | fp32, the reference | 1,178 MiB | 572 MiB | 30.8 ms | 0.6671 |
/// | post-training q4 (onnx-community) | 188 MiB | 228 MiB | 23.8 ms | 0.6540 |
/// | QAT fp32 | 1,177 MiB | 578 MiB | 25.8 ms | 0.6589 |
/// | **QAT q4 — this** | **199 MiB** | **250 MiB** | **18.4 ms** | **0.6647** |
///
/// Post-training q4 costs 2.0% of MRR; this costs 0.4%, which at 200 queries is
/// indistinguishable from zero, and it is AHEAD on the deeper cuts (R@5 0.820
/// against fp32's 0.815, R@10 0.890 against 0.875). The control that makes the
/// claim believable is the third row: QAT q4 scores ABOVE its own fp32, which is
/// the signature of weights that were trained for the grid — quantizing them
/// costs nothing. Same quantizer, same settings, both checkpoints; the only
/// variable is which weights went in.
///
/// Read it as "indistinguishable from fp32", not "better than": one corpus, 200
/// queries, one retrieval task.
pub const EMBEDDINGGEMMA_QAT_Q4: GemmaVariant = GemmaVariant {
    spec: &EMBEDDINGGEMMA_300M_QAT_Q4,
    graph: "model_q4.onnx",
    space: EMBEDDINGGEMMA_QAT_Q4_SPACE,
};

/// EmbeddingGemma-300M, QAT 4-bit ONNX — **ours**, served by us, because no
/// public export of the QAT checkpoint exists. See [`EMBEDDINGGEMMA_QAT_Q4`].
///
/// Built from `unsloth/embeddinggemma-300m-qat-q4_0-unquantized` rather than
/// Google's own repo, which is manually gated and answers 401 to CI; the
/// unsloth copy is byte-identical and its revision is pinned by the publish
/// job. Quantized at 4 bits, block 32, symmetric, with the embedding `Gather`
/// included — the same grid the QAT training targeted.
///
/// **Neither file reproduces the way the E4B weights do, and the shas below say
/// which build they came from.** The 254 KB graph is not byte-stable at all: the
/// exporter emits parallel branches in a varying order, so two builds on ONE
/// machine are isomorphic rather than equal. The 208 MB blob is byte-stable per
/// machine but NOT across architectures — the quantizer's float rounding differs,
/// so an arm64 Mac and an x86_64 runner disagree. The first publish attempt
/// caught exactly that, which is the gate working.
///
/// So: the graph is checked into the Scope repo and uploaded verbatim, and the
/// blob is the one `embedder-weights.yml` builds on ubuntu-latest, whose sha is
/// what this pins. Pairing them is sound because the graph is PORTABLE —
/// initializers are named by the sha256 of their contents and sorted by it, so
/// the blob's layout is a function of the model, and one build's graph against
/// another build's blob was verified to produce bitwise identical vectors.
///
/// The gate that actually protects the model is therefore behavioural, not a
/// hash: before uploading anything the job embeds fixed token sequences with the
/// pair and compares against reference vectors recorded from the build every
/// number above was measured on. A model that had drifted would fail there.
///
/// The tokenizer is unchanged and still comes from onnx-community: it is the
/// same file, byte for byte, as the one the fp16 build used, and token ids were
/// checked to match across scripts before any of the numbers above were taken.
pub const EMBEDDINGGEMMA_300M_QAT_Q4: ModelSpec = ModelSpec {
    id: "embeddinggemma-300m-qat-q4",
    display_name: "EmbeddingGemma 300M (multilingual, QAT 4-bit)",
    files: &[
        ModelFile {
            name: "model_q4.onnx",
            url: "https://files.scopy.app/ondevice/weights/embeddinggemma-300m-qat-q4/a94b15f2ffce/model_q4.onnx",
            sha256: "f352a9797f521cffb18d1bfd9369d6d5a09bfc8844d76b5ed8db51150b7281e9",
            size: 254_180,
        },
        ModelFile {
            name: "model_q4.onnx.data",
            url: "https://files.scopy.app/ondevice/weights/embeddinggemma-300m-qat-q4/a94b15f2ffce/model_q4.onnx.data",
            sha256: "a94b15f2ffce5a2dd9066e6fb1e9c309d46c46e02fc3a279e1d162d3ab79e0f6",
            size: 208_456_704,
        },
        ModelFile {
            name: "tokenizer.json",
            url: "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer.json",
            sha256: "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47",
            size: 20_323_312,
        },
    ],
    // Nothing to name: this lands in its own directory (`embeddinggemma-300m-qat-q4`),
    // so the fp16 build it replaces goes away with its directory rather than
    // leaving a file behind inside this one.
    supersedes: &[],
};

/// The QAT checkpoint the artifact above is built FROM. Not a model anything
/// provisions — the publish job reads it to fetch and verify the input before
/// transforming it, so the crate and the workflow name one source rather than
/// two. Same role as `GEMMA4_E4B_IT_QAT_Q4_0_UPSTREAM`.
pub const EMBEDDINGGEMMA_QAT_UPSTREAM_REPO: &str = "unsloth/embeddinggemma-300m-qat-q4_0-unquantized";
/// Pinned revision of [`EMBEDDINGGEMMA_QAT_UPSTREAM_REPO`], so a re-run of the
/// build cannot silently pick up different weights.
pub const EMBEDDINGGEMMA_QAT_UPSTREAM_REV: &str = "a2f8a2faf81988899f996a2fb3a1abe91486403a";
/// sha256 of that checkpoint's `model.safetensors`, checked before the build.
pub const EMBEDDINGGEMMA_QAT_UPSTREAM_SHA256: &str =
    "92b0b41d51116cd40db3d136f90f6176271f267a5c3d82c99a5f19a8ad39005e";

/// fp16 — the previous on-device embedder, kept as the reference half of the
/// fp16 tripwire and as the definition of the space the SERVER still writes.
/// Same weights as fp32 at half the bytes:
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

/// fp32 — the reference half of the fp16 tripwire, and a mirror of what the
/// SERVER embeds documents with. No app provisions it: every on-device consumer
/// downloads [`EMBEDDINGGEMMA_FP16`]. It is here so `gemma_fp16_parity` can name
/// the thing it compares against, and so the file the server runs is written
/// down in the same place as the file the device runs.
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
    supersedes: &[],
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
/// The parity-exact reference (Phase 1 spike: cosine 1.0 vs the Python
/// `onnxruntime` pipeline), and what the SERVER embeds documents with. **Not a
/// model any device provisions** — see [`EMBEDDINGGEMMA_FP32`] for why it is
/// still declared here.
///
/// External-data ONNX: two files (`model.onnx` graph + `model.onnx_data`
/// weights) plus the tokenizer.
pub const EMBEDDINGGEMMA_300M_FP32: ModelSpec = ModelSpec {
    id: "embeddinggemma-300m-fp32",
    display_name: "EmbeddingGemma 300M (multilingual, fp32)",
    supersedes: &[],
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
