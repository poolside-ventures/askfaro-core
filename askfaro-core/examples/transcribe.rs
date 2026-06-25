//! Minimal end-to-end check: load a model, transcribe a 16 kHz wav, print RTF.
//!
//!   cargo run -p askfaro-core --example transcribe --features stt --release -- <model_dir> <wav>

use std::time::Instant;

use askfaro_core::stt::SttEngine;

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/parakeet-rs-test/tdt".to_string());
    let wav = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/parakeet-rs-test/clips/en.wav".to_string());

    let mut reader = hound::WavReader::open(&wav).expect("open wav");
    let spec = reader.spec();
    let pcm: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
    };
    let dur = pcm.len() as f32 / spec.sample_rate as f32 / spec.channels as f32;

    let t = Instant::now();
    let mut engine = SttEngine::load(&model_dir).expect("load model");
    println!("model load: {:.2}s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    let transcript = engine
        .transcribe(pcm, spec.sample_rate, spec.channels)
        .expect("transcribe");
    let wall = t.elapsed().as_secs_f32();
    println!(
        "[{:.1}s audio in {:.2}s -> RTFx {:.1}] {}",
        dur,
        wall,
        dur / wall,
        transcript.text
    );
}
