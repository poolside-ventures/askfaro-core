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

/// The validated default: NVIDIA Parakeet TDT 0.6B v3, int8, multilingual.
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
