//! # askfaro-core-stt
//!
//! On-device speech-to-text for the Faro on-device SDK. A thin, host-facing wrapper
//! over a Parakeet TDT model running on ONNX Runtime (via `parakeet-rs`) — the
//! validated cross-platform path (macOS/Windows/Linux/iOS/Android).
//!
//! Opt-in by design: depend on this crate only when voice input is enabled. The
//! server/Python build never includes it, so the wheel stays small.
//!
//! ```no_run
//! use askfaro_core_stt::SttEngine;
//! let mut engine = SttEngine::load("/path/to/parakeet-model")?;
//! let transcript = engine.transcribe(pcm_f32, 16_000, 1)?;
//! println!("{}", transcript.text);
//! # Ok::<(), askfaro_core_stt::SttError>(())
//! ```

pub mod models;

use std::path::Path;

use parakeet_rs::{ParakeetTDT, Transcriber};

/// Errors surfaced by the engine. Provider/internal details are flattened to a
/// string so the public API never leaks the underlying runtime's types.
#[derive(Debug, thiserror::Error)]
pub enum SttError {
    /// The model directory could not be loaded (missing/invalid encoder, decoder, or vocab).
    #[error("failed to load speech model: {0}")]
    Load(String),
    /// Transcription failed at inference time.
    #[error("transcription failed: {0}")]
    Transcribe(String),
}

/// A transcription result. Kept deliberately small for now; word/segment
/// timestamps and detected language are additive later.
#[derive(Debug, Clone)]
pub struct Transcript {
    /// The transcribed text, with the model's punctuation and casing.
    pub text: String,
}

/// A loaded on-device speech-to-text engine.
///
/// Construct once with [`SttEngine::load`] (the model load is the expensive step),
/// then call [`SttEngine::transcribe`] per utterance.
pub struct SttEngine {
    model: ParakeetTDT,
}

impl SttEngine {
    /// Load a Parakeet TDT model from a directory containing the ONNX encoder,
    /// decoder, and `vocab.txt`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, SttError> {
        let model = ParakeetTDT::from_pretrained(model_dir.as_ref(), None)
            .map_err(|e| SttError::Load(e.to_string()))?;
        Ok(Self { model })
    }

    /// Transcribe PCM audio. The model expects 16 kHz; multi-channel input is
    /// downmixed by the underlying runtime.
    pub fn transcribe(
        &mut self,
        pcm: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Transcript, SttError> {
        let result = self
            .model
            .transcribe_samples(pcm, sample_rate, channels, None)
            .map_err(|e| SttError::Transcribe(e.to_string()))?;
        Ok(Transcript { text: result.text })
    }
}
