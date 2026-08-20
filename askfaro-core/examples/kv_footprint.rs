//! What does giving the background workload its own window actually save?
//!
//! Builds the two shapes side by side over the SAME weights and lets llama.cpp
//! report its own KV cache sizes:
//!
//!   shared : one context, two slots of `--big` tokens each  (today)
//!   split  : one context of `--big` + one of `--small`      (with `extra_contexts`)
//!
//! Read the `llama_kv_cache: size = ... MiB ( N cells, M layers, S/S seqs)`
//! lines under each heading, and the `compute buffer size` lines with them:
//!
//! ```sh
//! cargo run --release --features llama-cpp --example kv_footprint -- \
//!     <model.gguf> [big] [small] [big_ubatch] [small_ubatch] 2>&1 \
//!   | grep -E '===|kv_cache: size|compute buffer size|slot windows|read off disk'
//! ```
//!
//! **Do not measure this with process RSS**: the weights are mmapped, so
//! `load_tensors: model buffer size` prints 0.00 MiB and RSS for one identical
//! configuration has been observed at both 6.44 and 1.66 GiB across two runs on
//! the same machine.
//!
//! The compute buffers are not a footnote. A second context allocates its OWN,
//! sized by `n_ubatch` and not by the window: measured on E4B q4_0, the split
//! shape gives back exactly 768 MiB of KV and then spends 522 MiB of it on a
//! second compute buffer at the default ubatch of 512. Hand the background
//! context a smaller one — that is what `ContextSpec::n_ubatch` is for, and it
//! is the difference between a ~246 MiB win and a ~620 MiB one.
use askfaro_core::generation::{ContextSpec, LlamaCppConfig, LlamaCppEngine};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(model) = args.next() else {
        eprintln!(
            "usage: kv_footprint <model.gguf> [big_n_ctx] [small_n_ctx] \
             [big_ubatch] [small_ubatch]"
        );
        std::process::exit(2);
    };
    let big: u32 = args.next().map_or(65536, |a| a.parse().expect("big n_ctx"));
    let small: u32 = args.next().map_or(16384, |a| a.parse().expect("small n_ctx"));
    let ubatch: Option<u32> = args.next().map(|a| a.parse().expect("big n_ubatch"));
    // The background context's own micro-batch, which is the other half of the
    // saving; `None` inherits the big one, as `ContextSpec` does.
    let small_ubatch: Option<u32> = args
        .next()
        .map(|a| a.parse().expect("small n_ubatch"))
        .or(ubatch);

    let base = LlamaCppConfig {
        model_path: model.into(),
        n_ubatch: ubatch,
        // Thinking off and no drafter: this measures cache shapes, and the
        // drafter shares the target's KV memory, which would muddy the lines.
        enable_thinking: false,
        ..Default::default()
    };

    println!("\n=== shared: one context, two slots of {big} ===\n");
    let mut shared = LlamaCppEngine::new(LlamaCppConfig {
        n_ctx: big,
        n_slots: 2,
        ..base.clone()
    });
    let ms = shared.warm().expect("shared warm");
    println!("\nwarm in {ms}ms, slot windows: {:?}", shared.slot_windows());
    // Released before the split shape is built, so the two never contend for
    // the accelerator's budget and each set of lines describes itself alone.
    shared.unload();

    println!("\n=== split: one context of {big} + one of {small} ===\n");
    let mut split = LlamaCppEngine::new(LlamaCppConfig {
        n_ctx: big,
        n_slots: 1,
        extra_contexts: vec![ContextSpec {
            n_ctx: small,
            n_slots: 1,
            n_ubatch: small_ubatch,
        }],
        ..base
    });
    let ms = split.warm().expect("split warm");
    println!("\nwarm in {ms}ms, slot windows: {:?}", split.slot_windows());
    println!("weights read off disk this process: {}", LlamaCppEngine::model_loads());

    println!(
        "\nAdd up the `llama_kv_cache: size` AND `compute buffer size` lines under\n\
         each heading. The KV should differ by one slot at {big} minus one at\n\
         {small}; against that, the split shape pays one extra compute buffer,\n\
         sized by the small context's n_ubatch. `weights read off disk` must be 2\n\
         (one per shape) rather than 3 — the split shape's two contexts share a\n\
         single read of the weights."
    );
}
