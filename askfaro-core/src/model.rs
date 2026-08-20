//! Shared on-device model provisioning — the *mechanism* (files, sizes, sha256,
//! presence/verification) that both `core-stt` and `core-search` build on. Each
//! domain crate owns its own [`ModelSpec`] constants; this crate owns the type
//! and the verify subsystem.
//!
//! Network-free by design: the crate owns the spec and verification; the **host**
//! performs the actual download with its platform-native transport (so it can do
//! background download, Wi-Fi-only, and progress UI). That keeps this crate
//! dependency-light and cross-compiling unchanged to every mobile target.
//!
//! Typical host flow: [`missing`] → download each file → [`verify`] → then load
//! the model from [`ModelSpec::dir`].

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
    /// File names this spec REPLACES inside its own directory, named one by one.
    ///
    /// A model directory is keyed on [`ModelSpec::id`], so a spec that keeps its
    /// id and changes a file name leaves the old file behind forever: nothing in
    /// the provisioning path prunes, [`is_present`] only asks whether the CURRENT
    /// files exist, and the host downloads the new one alongside the old. That is
    /// how the 2026-08-20 requantised E4B would have left an upgrading machine
    /// holding 9.4 GB in a directory meant for 4.26.
    ///
    /// **Why a named list rather than "delete whatever is not in `files`".** A
    /// model directory is not exclusively ours. Scope's persisted KV prefix lives
    /// in the weights directory (`brain_prefix_dir()` IS `GEN_SPEC.dir()`), so
    /// that directory currently also holds `prefix-<hash>.kv` and
    /// `prefix-inputs.json`. A delete-by-default rule needs an allowlist of every
    /// foreign file that has ever been put there, and it silently destroys the
    /// next one somebody adds without knowing to update the list. The failure
    /// mode is data loss with no error. Naming what we replace can only ever
    /// remove something a human decided to remove, and it leaves the migration
    /// history readable in the spec itself.
    ///
    /// Empty for a spec that has never replaced a file, which is most of them.
    pub supersedes: &'static [&'static str],
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

/// True if every file is present at the right size (cheap; no hashing).
pub fn is_present(spec: &ModelSpec, cache_root: &Path) -> bool {
    let dir = spec.dir(cache_root);
    spec.files.iter().all(|f| {
        std::fs::metadata(dir.join(f.name))
            .map(|m| m.len() == f.size)
            .unwrap_or(false)
    })
}

/// Remove the files this spec supersedes, and report what went.
///
/// Call it only AFTER the new files verify: the whole point is that losing the
/// old copy is safe once the replacement is known good, and unsafe before. It
/// is deliberately conservative and silent about everything it was not told to
/// remove.
///
/// Refuses to remove a name that appears in `spec.files`, so a spec that lists
/// a current file as superseded (a copy-paste away) deletes the model it just
/// downloaded rather than the one it replaced. Only ever touches the spec's own
/// directory, and a name containing a path separator is ignored rather than
/// followed.
pub fn drop_superseded(spec: &ModelSpec, cache_root: &Path) -> Vec<PathBuf> {
    let dir = spec.dir(cache_root);
    let mut removed = Vec::new();
    for name in spec.supersedes {
        if name.is_empty() || name.contains('/') || name.contains('\\') {
            continue;
        }
        if spec.files.iter().any(|f| f.name == *name) {
            continue;
        }
        let p = dir.join(name);
        if std::fs::metadata(&p).map(|m| m.is_file()).unwrap_or(false)
            && std::fs::remove_file(&p).is_ok()
        {
            removed.push(p);
        }
    }
    removed
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
