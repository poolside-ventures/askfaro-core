//! Does a CPU buffer-type override actually move a tensor off the accelerator?
//!
//! Written because the answer was assumed on a phone and the assumption was
//! wrong: adding `per_layer_token_embd` to `cpu_tensor_overrides` changed the
//! reported Metal buffer by exactly zero bytes (3,179.26 MiB before and after).
//! That leaves two very different possibilities — the override never reached
//! the engine, or it reached it and did not match — and they have different
//! fixes, so this probe settles it on a machine with a fast edit-run loop
//! instead of a 20-minute device cycle.
//!
//! Reads the same GGUF the device runs. Watch llama.cpp's own
//! `load_tensors: ... model buffer size` lines: the Metal figure is the one
//! that has to fall.
//!
//!   cargo run --release --example buft_override_probe --features llama-cpp \
//!     -p askfaro-core -- [pattern ...]
//!
//! With no arguments it runs the baseline (no overrides) so the two can be
//! compared in one place.

use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
};
use std::path::PathBuf;

fn main() {
    let patterns: Vec<String> = std::env::args().skip(1).collect();

    let model = PathBuf::from(std::env::var("BRAIN_MODEL").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!(
            "{home}/Library/Application Support/com.getscopy.desktop/models/\
             gemma-4-e2b-it-qat-q4_0/gemma-4-E2B_q4_0-it.gguf"
        )
    }));
    if !model.exists() {
        eprintln!("model not found: {}", model.display());
        eprintln!("set BRAIN_MODEL to the GGUF path");
        std::process::exit(2);
    }

    println!("model     : {}", model.display());
    println!(
        "overrides : {}",
        if patterns.is_empty() {
            "(none — baseline)".to_string()
        } else {
            patterns.join(", ")
        }
    );
    println!("--- llama.cpp load log follows; watch the buffer sizes ---");

    let cfg = LlamaCppConfig {
        model_path: model,
        draft_path: None,
        // Match the device run so the numbers are comparable.
        n_ctx: 4096,
        n_ubatch: Some(128),
        n_slots: 1,
        enable_thinking: false,
        cpu_tensor_overrides: patterns,
        // The untried axis: with mmap ON, weights NOT offloaded stay file-backed
        // and evictable, so trimming layers shrinks the wired Metal buffer
        // without converting the remainder into anonymous memory.
        n_gpu_layers: std::env::var("N_GPU_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000),
        // Overrides are a no-op under mmap; this probe exists to show that.
        use_mmap: std::env::var("USE_MMAP").map(|v| v != "0").unwrap_or(true),
        ..Default::default()
    };

    let mut engine = LlamaCppEngine::new(cfg);
    match engine.warm() {
        Ok(ms) => println!("--- loaded in {ms} ms ---"),
        Err(e) => {
            eprintln!("load failed: {e}");
            std::process::exit(1);
        }
    }

    // One short generation, so the probe also proves the split still computes a
    // real answer rather than merely allocating differently.
    let req = GenerateRequest {
        system: "Answer in exactly one short sentence.".into(),
        messages: vec![Msg::user("What is the capital of France?")],
        ..Default::default()
    };
    match engine.generate(req) {
        Ok(r) => println!(
            "generate  : {} ms (prefill {} / decode {}) -> {:?}",
            r.model_ms,
            r.timings.prefill_ms,
            r.timings.decode_ms,
            r.text.trim()
        ),
        Err(e) => eprintln!("generate failed: {e}"),
    }
}
