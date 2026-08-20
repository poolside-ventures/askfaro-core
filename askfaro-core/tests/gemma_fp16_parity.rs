//! The tripwire for the fp16 embedder.
//!
//! The device ships `model_fp16.onnx` into the fp32 embedding space, which is
//! only sound because ONNX Runtime's CPU provider has no fp16 kernels: it casts
//! to fp32 and computes there, so the vectors match. Google's own card says
//! EmbeddingGemma activations do NOT support fp16 (the output is scaled by
//! sqrt(hidden_size); fp32 hidden states reach ~264,000 against fp16's 65,504
//! ceiling), and a backend that genuinely computed in fp16 would produce
//! garbage rather than an error.
//!
//! So this asserts the thing that would be false if that ever changed. Run it on
//! any `ort` bump, any execution-provider change, and any new target:
//!
//! Both variants resolve their own directory under the model cache root, so the
//! test names the same constants the app does rather than two hand-written paths
//! that can drift from them:
//!
//! ```text
//! EMB_GEMMA_CACHE_ROOT="$HOME/Library/Application Support/com.getscopy.desktop/models" \
//! cargo test -p askfaro-core --features embeddinggemma --test gemma_fp16_parity -- --ignored --nocapture
//! ```
//!
//! The fp32 half is not a model any device provisions; fetch it from the URLs in
//! `EMBEDDINGGEMMA_300M_FP32` when you need to run this.

#![cfg(feature = "embeddinggemma")]

use askfaro_core::search::gemma::GemmaEmbedder;
use askfaro_core::search::models::{EMBEDDINGGEMMA_FP16, EMBEDDINGGEMMA_FP32};
use askfaro_core::search::EmbedEngine;

/// Measured floor, not an aspiration: over 300 real queries the worst pair
/// agreed at 0.99999962. Anything below this is a different model, not rounding.
const MIN_COSINE: f32 = 0.9999;

/// Mixed scripts and lengths, because the fp16 range problem would show up first
/// wherever activations are largest.
const TEXTS: &[&str] = &[
    "invoice from last quarter",
    "what did Anna say about the pricing deck",
    "Maschinelles Lernen und künstliche Intelligenz",
    "四半期レビューは木曜日です",
    "the thread where we agreed the launch date, and the follow-up nobody answered",
];

#[test]
#[ignore = "requires both EmbeddingGemma variants under EMB_GEMMA_CACHE_ROOT"]
fn fp16_matches_fp32() {
    let root = std::env::var("EMB_GEMMA_CACHE_ROOT").expect("set EMB_GEMMA_CACHE_ROOT");
    let root = std::path::Path::new(&root);

    let fp32 = GemmaEmbedder::load_variant(&EMBEDDINGGEMMA_FP32, root)
        .expect("load fp32 EmbeddingGemma");
    let fp16 = GemmaEmbedder::load_variant(&EMBEDDINGGEMMA_FP16, root)
        .expect("load fp16 EmbeddingGemma");

    let mut worst = 1.0f32;
    for text in TEXTS {
        for (label, a, b) in [
            ("query", fp32.embed_query(text), fp16.embed_query(text)),
            (
                "document",
                fp32.embed_documents(&[text]).into_iter().next().flatten(),
                fp16.embed_documents(&[text]).into_iter().next().flatten(),
            ),
        ] {
            let a = a.unwrap_or_else(|| panic!("fp32 {label} embed returned None for {text:?}"));
            let b = b.unwrap_or_else(|| panic!("fp16 {label} embed returned None for {text:?}"));
            assert_eq!(a.len(), b.len(), "{label} dimensionality differs for {text:?}");
            assert!(
                b.iter().all(|v| v.is_finite()),
                "fp16 {label} produced a non-finite value for {text:?} — the range \
                 problem the model card warns about has materialized"
            );
            let c = cosine(&a, &b);
            println!("{c:.8}  {label}: {text}");
            worst = worst.min(c);
            assert!(
                c >= MIN_COSINE,
                "fp16 {label} vector for {text:?} agrees with fp32 at only {c}, below {MIN_COSINE}. \
                 The CPU provider is no longer upcasting; fp16 is not safe on this backend."
            );
        }
    }
    println!("worst cosine across {} texts: {worst:.8}", TEXTS.len());
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt())
}
