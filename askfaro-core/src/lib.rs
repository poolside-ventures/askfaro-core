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
