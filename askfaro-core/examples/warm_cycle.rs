//! How expensive is it to give the weights back and take them again?
//!
//! The idle-unload policy lives or dies on this number. The cold load is ~60s
//! for E4B's 5.15 GB, and if a RELOAD costs the same then unloading is a trap:
//! it trades memory the user is not short of for a minute of waiting at the
//! exact moment they came back to the app.
//!
//! The reason to expect otherwise is that llama.cpp mmaps the GGUF, so an
//! unload drops the mapping while the OS page cache may still hold the bytes.
//! If that is right, a reload is a remap rather than a re-read from disk.
//!
//! ```sh
//! cargo run --release --features llama-cpp --example warm_cycle -- <model.gguf> [drafter.gguf]
//! ```
use askfaro_core::generation::{LlamaCppConfig, LlamaCppEngine};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(model) = args.next() else {
        eprintln!("usage: warm_cycle <model.gguf> [drafter.gguf]");
        std::process::exit(2);
    };
    let draft = args.next();

    let mut engine = LlamaCppEngine::new(LlamaCppConfig {
        model_path: model.into(),
        draft_path: draft.map(Into::into),
        ..Default::default()
    });

    let cold = engine.warm().expect("cold warm");
    println!("cold load      : {cold} ms");

    for i in 1..=3 {
        assert!(engine.unload(), "expected weights to be resident");
        let t = std::time::Instant::now();
        let ms = engine.warm().expect("re-warm");
        println!(
            "reload #{i}      : {ms} ms (wall {} ms)",
            t.elapsed().as_millis()
        );
    }
    println!("\nIf reloads are far cheaper than the cold load, the page cache is");
    println!("absorbing them and idle-unload is safe. If not, do not unload.");
}
