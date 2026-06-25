//! End-to-end: the full engine with the REAL EmbeddingGemma embedder — model
//! load → index a mixed-language corpus → hybrid search → cross-lingual hit.
//!
//! Ignored by default (needs the ~1.2 GB fp32 model). Run with the model dir
//! (containing model.onnx, model.onnx_data, tokenizer.json) on EMB_GEMMA_DIR:
//!
//!   EMB_GEMMA_DIR=/tmp/embgemma-model \
//!   cargo test -p askfaro-core --features embeddinggemma -- --ignored --nocapture

#![cfg(feature = "embeddinggemma")]

use askfaro_core::search::gemma::GemmaEmbedder;
use askfaro_core::search::sqlite::SqliteBackend;
use askfaro_core::search::{EmbedEngine, IndexDoc, SearchIndex, SearchParams};

#[test]
#[ignore = "requires the EmbeddingGemma model on EMB_GEMMA_DIR"]
fn full_engine_cross_lingual() {
    let dir = std::env::var("EMB_GEMMA_DIR").expect("set EMB_GEMMA_DIR to the model dir");
    let embedder = GemmaEmbedder::load(&dir).expect("load EmbeddingGemma");

    let space = embedder.space().to_string();
    let backend = SqliteBackend::open_in_memory(&[space.as_str()]).unwrap();
    let index = SearchIndex::new(backend, embedder);

    index
        .upsert_many(&[
            IndexDoc::leaf("note", "n_en", "Machine learning", "neural networks and model training"),
            IndexDoc::leaf("note", "n_de", "Maschinelles Lernen", "Künstliche Intelligenz und neuronale Netze"),
            IndexDoc::leaf("note", "n_cook", "Bread recipe", "flour water yeast salt baking oven"),
            IndexDoc::leaf("task", "t_pay", "Stripe payments", "charge card customer checkout"),
        ])
        .unwrap();

    // English query, no shared tokens with the German doc — only the semantic
    // (embedding) half can bridge the languages.
    let hits = index
        .search("artificial intelligence", &SearchParams { k: 3, ..Default::default() })
        .unwrap();

    let ids: Vec<&str> = hits.iter().map(|h| h.object_id.as_str()).collect();
    println!("cross-lingual hits: {:?}", ids);
    assert!(
        ids.contains(&"n_de"),
        "expected the German AI note to be retrieved cross-lingually; got {ids:?}"
    );
}
