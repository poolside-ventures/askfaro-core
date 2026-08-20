//! Regression: embedding must succeed (return Some, 768 dims) for inputs at
//! and beyond the model's 2048-token context window, because callers chunk by
//! OTHER tokenizers' counts (Honcho counts with tiktoken at 1500 tokens, which
//! can exceed 2048 Gemma SentencePiece tokens) and a None here becomes an
//! empty embedding upstream.
//!
//!   EMB_GEMMA_DIR=... cargo test -p askfaro-core --features embeddinggemma \
//!     --test gemma_long_input -- --ignored --nocapture

#![cfg(feature = "embeddinggemma")]

use askfaro_core::search::gemma::GemmaEmbedder;
use askfaro_core::search::EmbedEngine;

#[test]
#[ignore = "requires the EmbeddingGemma model on EMB_GEMMA_DIR"]
fn long_inputs_embed() {
    let dir = std::env::var("EMB_GEMMA_DIR").expect("set EMB_GEMMA_DIR to the model dir");
    // The graph file is named per variant, so the shipping fp16 export can be
    // put through the same test as the fp32 reference rather than only the
    // reference being covered.
    let graph = std::env::var("EMB_GEMMA_GRAPH").unwrap_or_else(|_| "model.onnx".to_string());
    let embedder = GemmaEmbedder::load_graph(&dir, &graph, "embeddinggemma_300m_fp32")
        .expect("load EmbeddingGemma");

    // Short sanity input.
    let short = embedder.embed_documents(&["hello world"]);
    println!("short: {:?}", short[0].as_ref().map(|v| v.len()));
    assert_eq!(short[0].as_ref().map(|v| v.len()), Some(768), "short input must embed");

    // Lengths straddling the 2048-token window. Each word tokenizes to >=1
    // token, so 3000 distinct words is safely past the window.
    for words in [500usize, 1500, 2000, 2500, 3000, 6000] {
        let text: String = (0..words).map(|i| format!("word{i} ")).collect();
        let out = embedder.embed_documents(&[text.as_str()]);
        let dims = out[0].as_ref().map(|v| v.len());
        let finite = out[0]
            .as_ref()
            .map(|v| v.iter().all(|x| x.is_finite()))
            .unwrap_or(false);
        println!("words={words} -> dims={dims:?} finite={finite}");
        assert_eq!(dims, Some(768), "input of {words} words must embed, got None");
        assert!(finite, "input of {words} words produced non-finite values");
    }
}
