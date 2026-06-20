//! Model provisioning — the canonical *spec* for the on-device speech model
//! (files, source URLs, sizes, sha256) plus presence/verification helpers.
//!
//! Network-free by design: the crate owns the spec and verification; the **host**
//! performs the actual download with its platform-native transport (so it can do
//! background download, Wi-Fi-only, and progress UI). That keeps this crate
//! dependency-light and cross-compiling unchanged to every mobile target.
//!
//! Typical host flow: [`missing`] → download each file → [`verify`] → then
//! [`crate::SttEngine::load`] on [`ModelSpec::dir`].

use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// One file that makes up a model.
pub struct ModelFile {
    /// File name on disk (and within the model directory).
    pub name: &'static str,
    /// Where the host can fetch it.
    pub url: &'static str,
    /// Expected lowercase-hex sha256 of the file.
    pub sha256: &'static str,
    /// Expected size in bytes (a cheap pre-hash sanity check).
    pub size: u64,
}

/// A complete on-device model: a set of files under a per-model directory.
pub struct ModelSpec {
    /// Stable id, also used as the on-disk directory name.
    pub id: &'static str,
    /// Human label for UI.
    pub display_name: &'static str,
    /// The files that make up the model.
    pub files: &'static [ModelFile],
}

impl ModelSpec {
    /// Total download size in bytes.
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
    /// The directory this model lives in, under a host-provided cache root.
    pub fn dir(&self, cache_root: &Path) -> PathBuf {
        cache_root.join(self.id)
    }
}

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

/// True if every file is present at the right size (cheap; no hashing).
pub fn is_present(spec: &ModelSpec, cache_root: &Path) -> bool {
    let dir = spec.dir(cache_root);
    spec.files.iter().all(|f| {
        std::fs::metadata(dir.join(f.name))
            .map(|m| m.len() == f.size)
            .unwrap_or(false)
    })
}

/// The files that are absent or the wrong size, and so need downloading.
pub fn missing<'a>(spec: &'a ModelSpec, cache_root: &Path) -> Vec<&'a ModelFile> {
    let dir = spec.dir(cache_root);
    spec.files
        .iter()
        .filter(|f| {
            std::fs::metadata(dir.join(f.name))
                .map(|m| m.len() != f.size)
                .unwrap_or(true)
        })
        .collect()
}

/// Lowercase-hex sha256 of a file, computed in a streaming fashion.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Verify every file's sha256. `Err` names the first file that is missing or whose
/// hash does not match — the host should re-download it.
pub fn verify(spec: &ModelSpec, cache_root: &Path) -> Result<(), String> {
    let dir = spec.dir(cache_root);
    for f in spec.files {
        let path = dir.join(f.name);
        let got = sha256_file(&path).map_err(|e| format!("{}: {e}", f.name))?;
        if got != f.sha256 {
            return Err(format!("{}: sha256 mismatch", f.name));
        }
    }
    Ok(())
}
