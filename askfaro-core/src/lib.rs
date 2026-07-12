//! # askfaro-core
//!
//! The single on-device SDK core for Faro. One crate, one import; every
//! capability is a cargo feature, so consumers add one dependency and turn on
//! exactly what they need instead of composing a fistful of per-capability
//! crates:
//!
//! ```toml
//! askfaro-core = { git = "https://github.com/poolside-ventures/askfaro-core", \
//!                  default-features = false, features = ["stt", "apple-fm", "progressive"] }
//! ```
//!
//! ## Capabilities (each a feature → a module)
//!
//! - [`model`] — shared model provisioning (sha256 spec + verify; host downloads).
//!   Internal; enabled automatically by `stt` and `search`.
//! - [`search`] — embedded hybrid retrieval (FTS5 lexical + bag-of-words semantic,
//!   RRF). The `embeddinggemma` feature adds the on-device vector embedder.
//! - [`generation`] — provider-agnostic text generation + tool-calling. The
//!   `apple-fm` feature adds the Apple Foundation Models provider (macOS/iOS 26+).
//! - [`progressive`] — catalog selector: relevance ranking ([`search`]) plus
//!   progressive, budget-bounded expansion of a pcx manifest. No LLM call.
//! - [`free_tools`] — deterministic, zero-cost tools that run in-process and
//!   identically on server and device (calc, units, datetime, encoding, phone,
//!   random, timer, timezone, astronomy). The on-device half of the skill layer:
//!   a device app runs these locally instead of round-tripping to the cloud.
//!
//! Heavy/platform dependencies are pulled only by the feature that needs them, so
//! the default build is light and cross-compiles unchanged. The design follows
//! the on-device contract: the crate owns the spec + logic; the host owns the
//! transport (network, model download), and engines stay warm across calls.

#[cfg(feature = "model")]
pub mod model;

#[cfg(feature = "search")]
pub mod search;

#[cfg(feature = "generation")]
pub mod generation;

#[cfg(feature = "progressive")]
pub mod progressive;

#[cfg(feature = "stt")]
pub mod stt;

// The free-tool slice is vendored verbatim from the open-source `faro-core-free`
// crate by `scripts/vendor_free_tools.sh` (core-free stays the single source of
// truth; core-sync CI fails on drift). It carries its own envelope/error types,
// so it lives under one private `free` module and is exposed through the tidy
// `free_tools` surface below. The vendored slice carries the full server surface
// (envelope wrappers, error helpers); the on-device SDK re-exports only the
// execution subset, so allow the rest to sit unused rather than patch generated code.
#[cfg(feature = "free-tools")]
#[allow(dead_code, unused_imports)]
mod free;

/// On-device execution of the free / local tools: deterministic, zero-cost
/// capabilities (calc, units, datetime, encoding, phone, random, timer, timezone,
/// astronomy) that run in-process with no network, credentials, or billing — the
/// same code the server SDK runs, so a local run and a cloud run are byte-for-byte
/// identical. Host apps dispatch a capability locally when it is available on the
/// device and fall back to the cloud otherwise.
#[cfg(feature = "free-tools")]
pub mod free_tools {
    /// The canonical output envelope every free tool returns (Spec 1).
    pub use crate::free::envelope::SkillResult;
    /// Names of the free tools available in this build.
    pub use crate::free::free_tools::available;
    /// Execute a free tool by name with JSON `params`, returning a [`SkillResult`].
    pub use crate::free::free_tools::execute;
    /// Binding-friendly entry point: JSON params string in, envelope JSON out, no
    /// Rust types crossing the FFI boundary. This is what host bridges call.
    pub use crate::free::execute_free_tool_json;
}
