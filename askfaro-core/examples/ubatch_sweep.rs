//! Prefill throughput vs `n_ubatch`, on real weights.
//!
//! `n_ubatch` is the PHYSICAL Metal dispatch size during prefill; llama.cpp
//! defaults it to 512 no matter what `n_batch` says, so an engine that only
//! sizes `n_batch` (as this one did until Aug 2026) still prefills in
//! 512-token kernel launches. Community numbers on Apple Silicon claim 2-3x
//! prompt eval at 2048; this measures it on OUR engine, OUR prompt shape and
//! the deployment-floor hardware instead of trusting a blog.
//!
//! ```sh
//! cargo run --release --features llama-cpp --example ubatch_sweep -- \
//!   <model.gguf> [ubatch ...]     # default sweep: 512 1024 2048
//! ```
//!
//! Each setting loads a fresh engine (a context's ubatch is fixed at
//! creation), prefills the same ~8K-token prompt cold, and reports the split.
//! Watch the `compute buffer size` lines in the llama.cpp log for the memory
//! price of each setting; that is the axis the numbers here do not show.

use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| {
        eprintln!("usage: ubatch_sweep <model.gguf> [ubatch ...]");
        std::process::exit(2);
    });
    let sweep: Vec<u32> = {
        let rest: Vec<u32> = args.filter_map(|a| a.parse().ok()).collect();
        if rest.is_empty() { vec![512, 1024, 2048] } else { rest }
    };

    // ~8K tokens of plausible prose (each sentence tokenizes to ~40; 200 of
    // them land well inside the 16K window). Content is irrelevant to prefill
    // compute; the counter keeps the tokenizer from collapsing repetition.
    let mut long = String::new();
    for i in 0..200 {
        long.push_str(&format!(
            "Meeting note {i}: the quarterly review moved to Thursday at ten, \
             the follow-up with the design team is still unscheduled, and the \
             billing migration remains blocked on the vendor's sandbox. "
        ));
    }

    println!("ubatch,load_ms,prompt_tokens,prefill_ms,prefill_tok_s,decode_ms,output_tokens");
    for ub in sweep {
        let mut engine = LlamaCppEngine::new(LlamaCppConfig {
            model_path: model.clone().into(),
            n_ctx: 16384,
            n_slots: 1,
            n_ubatch: Some(ub),
            enable_thinking: false,
            // No drafter and no prefix cache: this measures raw prefill, so
            // nothing may skip or split it.
            ..Default::default()
        });
        let resp = engine
            .generate(GenerateRequest {
                system: "Reply with the single word OK.".into(),
                messages: vec![Msg::user(format!("{long}\n\nSay OK."))],
                enable_thinking: Some(false),
                ..Default::default()
            })
            .map_err(|e| {
                // No process::exit here: the engine must DROP before the
                // process ends, or ggml-metal's exit assert turns a clean
                // error report into a SIGABRT (see Loaded's field-order docs).
                eprintln!("ubatch {ub}: generate failed: {e}");
            });
        let Ok(resp) = resp else { continue };
        let t = resp.timings;
        println!(
            "{ub},{},{},{},{:.0},{},{}",
            t.load_ms,
            t.prompt_tokens,
            t.prefill_ms,
            t.prompt_tokens as f64 / (t.prefill_ms.max(1) as f64 / 1000.0),
            t.decode_ms,
            t.output_tokens,
        );
    }
}
