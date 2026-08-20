//! Model provisioning for speech — the canonical *spec* for the on-device speech
//! model. The download/verify mechanism now lives in the shared
//! [`crate::model`] crate (so STT and search share it); this module
//! re-exports it and owns only the Parakeet [`ModelSpec`] constant.
//!
//! Network-free by design: the crate owns the spec and verification; the **host**
//! performs the actual download with its platform-native transport.
//!
//! Typical host flow: [`missing`] → download each file → [`verify`] → then
//! [`crate::stt::SttEngine::load`] on [`ModelSpec::dir`].

pub use crate::model::{
    is_present, missing, sha256_file, verify, ModelFile, ModelSpec,
};

/// **The default since 2026-08-20:** NVIDIA Parakeet TDT 0.6B v3, int4 encoder,
/// multilingual. Smaller AND more accurate than the int8 build it replaces.
///
/// The encoder's linear and pointwise-Conv weights are `MatMulNBits` at 4 bits,
/// block 64, asymmetric (depthwise Conv stays float); the decoder/joiner graph
/// and the vocab are byte-identical to [`PARAKEET_TDT_V3_INT8`]'s, same sha256.
/// 390 MiB against 640.
///
/// Measured 2026-08-20 on an M1 Pro, 324 clips / 79 minutes, WER after
/// lowercasing and stripping punctuation — int8 (what we shipped) → int4:
///
/// | set                          | int8   | int4   |
/// |------------------------------|--------|--------|
/// | LibriSpeech clean, n=120     |  1.75% |  1.50% |
/// | LibriSpeech 76-102s, n=12    |  2.66% |  1.73% |
/// | FLEURS nl, n=60              | 11.58% | 10.26% |
/// | FLEURS fr, n=60              |  7.28% |  5.84% |
/// | FLEURS de, n=60              |  8.22% |  5.84% |
/// | everything over 60s, n=24    | 20.88% |  7.59% |
///
/// It wins every bucket, and the last row is the reason this is a correctness
/// fix rather than a size cut. The int8 export carries ONE dynamic activation
/// scale per tensor, and past roughly 30 seconds in a single pass that is not
/// enough: on 82-89s French clips it code-switched into English mid-sentence and
/// truncated hard — 73 characters returned against a 1,223-character reference
/// in the worst case, 86.6% WER. int4 transcribed the same clips in full French
/// at 19.1%. Nobody would have found this from utterance-level benchmarks; every
/// published number for either export is short-form English.
///
/// The cost is speed: RTFx 9.5 against 11.3, because `MatMulNBits` dequantizes
/// into float compute. Still an order of magnitude faster than real time, which
/// is what dictation needs.
pub const PARAKEET_TDT_V3_INT4: ModelSpec = ModelSpec {
    id: "parakeet-tdt-0.6b-v3-int4",
    display_name: "Parakeet TDT 0.6B v3 (multilingual)",
    files: &[
        ModelFile {
            name: "encoder-model.int4.onnx",
            url: "https://huggingface.co/efederici/parakeet-tdt-0.6b-v3-onnx-int4/resolve/main/encoder-model.int4.onnx",
            sha256: "df4f1e5ff7a3af4e9d4b7078055164b11005e5d8a4c100e67f583c23975f7a31",
            size: 390_929_172,
        },
        ModelFile {
            name: "decoder_joint-model.int8.onnx",
            url: "https://huggingface.co/efederici/parakeet-tdt-0.6b-v3-onnx-int4/resolve/main/decoder_joint-model.int8.onnx",
            sha256: "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
            size: 18_202_004,
        },
        ModelFile {
            name: "vocab.txt",
            url: "https://huggingface.co/efederici/parakeet-tdt-0.6b-v3-onnx-int4/resolve/main/vocab.txt",
            sha256: "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
            size: 93_939,
        },
    ],
};

/// The int8 build this replaced. `parakeet-rs` resolves the encoder by trying
/// known names and then globbing `encoder*.onnx`, so the two must never share a
/// directory — they do not, because the directory is the spec id.
pub const PARAKEET_TDT_V3_INT8: ModelSpec = ModelSpec {
    id: "parakeet-tdt-0.6b-v3-int8",
    display_name: "Parakeet TDT 0.6B v3 (multilingual)",
    files: &[
        ModelFile {
            name: "encoder-model.int8.onnx",
            url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.int8.onnx",
            sha256: "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09",
            size: 652_183_999,
        },
        ModelFile {
            name: "decoder_joint-model.int8.onnx",
            url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.int8.onnx",
            sha256: "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
            size: 18_202_004,
        },
        ModelFile {
            name: "vocab.txt",
            url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt",
            sha256: "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
            size: 93_939,
        },
    ],
};
