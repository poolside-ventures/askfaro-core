//! Reproduce an embedding failure against a real model directory.
//!
//! `EmbedEngine::embed_documents` answers `None` for a text it could not embed,
//! which is all a host can see; this prints the reason the engine now logs
//! alongside it, for texts of the sizes that actually fail in the field (a
//! 3,521-char document was failing on every retry, forever, in the desktop
//! memory sidecar).
//!
//! ```text
//! cargo run --release --features embeddinggemma --example embed_probe -- \
//!   "$HOME/Library/Application Support/com.getscopy.desktop/models"
//! ```
//!
//! An optional second argument is a file whose contents are embedded verbatim,
//! for reproducing one specific document.

#[cfg(feature = "embeddinggemma")]
fn main() {
    use askfaro_core::search::gemma::GemmaEmbedder;

    let mut args = std::env::args().skip(1);
    let root = args.next().expect("usage: embed_probe <model_cache_root> [text_file]");
    let embedder = match GemmaEmbedder::load_variant(
        &askfaro_core::search::models::EMBEDDINGGEMMA_FP16,
        std::path::Path::new(&root),
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load failed: {e}");
            std::process::exit(1);
        }
    };

    if let Some(path) = args.next() {
        let text = std::fs::read_to_string(&path).expect("readable text file");
        report(&embedder, &format!("file {path}"), &text);
        return;
    }

    // Prose, because token-per-char ratio is what decides whether truncation
    // is even reached.
    let sentence = "The quarterly review is on Thursday and the deck needs the \
                    revenue split by region before then. ";
    for chars in [200usize, 3_521, 8_000, 40_000] {
        let mut text = String::new();
        while text.len() < chars {
            text.push_str(sentence);
        }
        text.truncate(chars);
        report(&embedder, &format!("{chars} chars of prose"), &text);
    }

    // Content that is not prose at all: a long unbroken token (base64, a URL,
    // a minified blob) is the case where chars and tokens come apart.
    let blob = "aGVsbG8td29ybGQtdGhpcy1pcy1ub3QtcHJvc2U".repeat(90);
    report(&embedder, &format!("{} chars of base64", blob.len()), &blob);

    let cjk = "四半期レビューは木曜日です。".repeat(270);
    report(&embedder, &format!("{} chars of Japanese", cjk.len()), &cjk);
}

#[cfg(feature = "embeddinggemma")]
fn report(embedder: &impl askfaro_core::search::EmbedEngine, label: &str, text: &str) {
    let out = embedder.embed_documents(&[text]);
    match out.first().and_then(|v| v.as_ref()) {
        Some(v) => println!("{label}: ok, {} dims", v.len()),
        None => println!("{label}: FAILED (reason logged above)"),
    }
}

#[cfg(not(feature = "embeddinggemma"))]
fn main() {
    eprintln!("build with --features embeddinggemma");
}
