//! Does a persisted KV prefix that no longer matches the prompt get DISCARDED?
//!
//! This is the hazard the persisted prefix introduces, and it is worse than the
//! ones around it because it does not fail: a stale KV cache produces fluent,
//! confident, wrong text. Nothing throws, no timing looks odd, and the tool
//! calls still parse. `llama_bench` cannot see it: every case there shares one
//! system block, which is the state this failure needs to leave.
//!
//! The check is a fact that lives ONLY in the system prompt. Two system blocks
//! differ in their first line, one carrying passphrase `ALPHA-ROSEWOOD-11` and the
//! other `OMEGA-BLUEJAY-42`, and are otherwise identical filler. Save the
//! prefix under the first, reload so the cache can only have come off disk, then
//! ask under the second. If the stale state is reused, or trimmed back to the
//! common point instead of dropped whole, the model answers with the passphrase
//! it can still see in the cache. There is no way to pass this by accident.
//!
//! Divergence is put in the FIRST line on purpose. Gemma 4 attends over a
//! sliding window, so the SWA half of the restored cache holds only the
//! positions just before where the save ended; a partial trim to a point that
//! far back leaves those layers reading cells nobody restored. That is the
//! specific shape of "plausible but wrong" this defends.
//!
//! ```sh
//! cargo run --release --features llama-cpp --example prefix_invalidation -- <model.gguf>
//! ```
use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
};

/// Long enough that reusing the cache is worth thousands of milliseconds, so a
/// broken invalidation is a temptation the engine actually has to refuse.
fn system_with(passphrase: &str) -> String {
    let mut s = format!("The passphrase is {passphrase}. Remember it exactly.\n\n");
    for i in 0..400 {
        s.push_str(&format!(
            "Fact {i}: item {i} belongs to group {}, with weight {} and a short note.\n",
            i % 7,
            i * 3 % 11
        ));
    }
    s
}

fn ask(engine: &mut LlamaCppEngine, system: &str) -> (String, u64) {
    let out = engine
        .generate(GenerateRequest {
            system: system.into(),
            messages: vec![Msg {
                role: "user".into(),
                content: "What is the passphrase? Reply with just the passphrase.".into(),
            }],
            ..Default::default()
        })
        .expect("generate");
    (out.text, out.timings.prefill_ms)
}

/// Drop the weights and get them back, so the next turn's cache can only have
/// come from the file. Building and reading a prefix in one context would prove
/// nothing about the file at all.
fn relaunch(engine: &mut LlamaCppEngine) {
    engine.unload();
    engine.warm().expect("reload");
}

fn main() {
    let Some(model) = std::env::args().nth(1) else {
        eprintln!("usage: prefix_invalidation <model.gguf>");
        std::process::exit(2);
    };
    let dir = std::env::temp_dir().join("faro-prefix-invalidation");
    let _ = std::fs::remove_dir_all(&dir);

    let mut engine = LlamaCppEngine::new(LlamaCppConfig {
        model_path: model.into(),
        prefix_cache_dir: Some(dir.clone()),
        state_key: "prefix_invalidation".into(),
        ..Default::default()
    });

    let alpha = system_with("ALPHA-ROSEWOOD-11");
    let omega = system_with("OMEGA-BLUEJAY-42");
    let mut failures: Vec<String> = Vec::new();

    // 1. First turn under ALPHA builds and saves the prefix.
    let (a1, _) = ask(&mut engine, &alpha);
    println!("alpha, cold        : {a1:?}");

    // 2. Relaunch and ask again, unchanged. The restore must be USED.
    relaunch(&mut engine);
    let (a2, a2_prefill) = ask(&mut engine, &alpha);
    println!("alpha, restored    : {a2:?} ({a2_prefill}ms prefill)");
    if !a2.contains("ALPHA-ROSEWOOD-11") {
        failures.push(format!(
            "after restoring its own prefix the model answered {a2:?}, which does not contain \
             ALPHA-ROSEWOOD-11. The restored cache does not match the tokens it was saved with."
        ));
    }
    if a2_prefill > 1_500 {
        failures.push(format!(
            "a matching prefix cost {a2_prefill}ms of prefill, so it was not reused. The restore \
             is being discarded when it should be kept."
        ));
    }

    // 3. Relaunch and ask under OMEGA. The restored ALPHA prefix is stale from
    //    its first line on, and must be dropped WHOLE rather than reused or
    //    trimmed. This is the assertion the whole example exists for.
    relaunch(&mut engine);
    let (b1, b1_prefill) = ask(&mut engine, &omega);
    println!("omega, stale prefix: {b1:?} ({b1_prefill}ms prefill)");
    if b1.contains("ALPHA-ROSEWOOD-11") || !b1.contains("OMEGA-BLUEJAY-42") {
        failures.push(format!(
            "with a stale prefix on disk the model answered {b1:?}, but this prompt says the \
             passphrase is OMEGA-BLUEJAY-42. A KV cache belonging to a different prompt was \
             reused, and the result is confident wrong text rather than an error."
        ));
    }

    // 4. The stale one must have been REPLACED, not merely ignored, or every
    //    launch from here pays a cold prefill and rewrites the same file.
    relaunch(&mut engine);
    let (b2, b2_prefill) = ask(&mut engine, &omega);
    println!("omega, restored    : {b2:?} ({b2_prefill}ms prefill)");
    if b2_prefill > 1_500 {
        failures.push(format!(
            "the rebuilt prefix cost {b2_prefill}ms on the next launch, so it was never saved. \
             Invalidation discarded the stale state without writing the new one."
        ));
    }
    if !b2.contains("OMEGA-BLUEJAY-42") {
        failures.push(format!(
            "after restoring the rebuilt prefix the model answered {b2:?} rather than \
             OMEGA-BLUEJAY-42."
        ));
    }

    if failures.is_empty() {
        println!("\npassed: a stale prefix is discarded and rebuilt; a matching one is reused");
    } else {
        eprintln!("\nFAILED:");
        for f in &failures {
            eprintln!("  - {f}");
        }
        std::process::exit(1);
    }
}
