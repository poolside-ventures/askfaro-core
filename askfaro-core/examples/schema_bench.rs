//! Parse-rate bench for grammar-constrained (`json_schema`) one-shots.
//!
//! The existing harnesses (`llama_generate`, `llama_bench`, `multistep_bench`)
//! all drive TOOLS; none of them sets `json_schema`, so the response-format
//! path had no bench at all. That mattered once a product feature's ONLY route
//! became a schema-constrained local generation: a schema the model cannot
//! satisfy fails as an unparseable string, which the caller drops silently, so
//! the user just never gets the thing and nothing anywhere reports why.
//!
//! Reads a prompts file (the EXACT `{system, user, schema}` the server hands
//! the device, dumped from the real builders, so this benches the shipping
//! prompt rather than a paraphrase of it) and reports, per case: parse rate,
//! required-key conformance, latency and token counts.
//!
//! ```sh
//! cargo run --release --features llama-cpp,metal --example schema_bench -- \
//!     /path/to/model.gguf prompts.json [--draft /path/to/drafter.gguf] [--iterations 3]
//! ```
//!
//! Needs multi-gigabyte GGUF weights, so it is an example and never CI.

use std::path::PathBuf;
use std::time::Instant;

use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
};
use serde_json::Value;

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    system: String,
    user: String,
    /// Absent for free-prose cases (draft_reply), which have no grammar at all.
    /// Those are scored on length and latency, not on parsing.
    #[serde(default)]
    schema: Option<Value>,
}

/// Top-level `required` keys the reply must actually carry. Grammar sampling
/// should make this unrepresentable-if-wrong, which is exactly why it is
/// asserted: a silent divergence between "the grammar was applied" and "the
/// grammar was applied to OUR schema" is the failure this bench exists to catch.
fn missing_required(schema: &Value, got: &Value) -> Vec<String> {
    let Some(req) = schema.get("required").and_then(|r| r.as_array()) else {
        return vec![];
    };
    req.iter()
        .filter_map(|k| k.as_str())
        .filter(|k| got.get(*k).is_none())
        .map(|k| k.to_string())
        .collect()
}

fn pct(v: &mut Vec<u64>, p: f64) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    let idx = (((v.len() - 1) as f64) * p).round() as usize;
    v[idx]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path: PathBuf = args
        .next()
        .unwrap_or_else(|| {
            eprintln!("usage: schema_bench <model.gguf> <prompts.json> [--draft <d.gguf>] [--iterations N]");
            std::process::exit(2);
        })
        .into();
    let prompts_path: PathBuf = args
        .next()
        .unwrap_or_else(|| {
            eprintln!("usage: schema_bench <model.gguf> <prompts.json> [--draft <d.gguf>] [--iterations N]");
            std::process::exit(2);
        })
        .into();

    let mut draft_path: Option<PathBuf> = None;
    let mut iterations: usize = 3;
    // Mirrors the shipping caller: a request carrying a schema runs with
    // thinking OFF, because the engine drops the reasoning budget and the close
    // bias whenever a grammar is present. `--think` reproduces the old
    // behaviour (thinking on, unbounded) for a before/after on one prompt set.
    let mut thinking = false;
    let mut show_sample = false;
    let mut dump_path: Option<PathBuf> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--draft" => draft_path = args.next().map(PathBuf::from),
            "--think" => thinking = true,
            "--sample" => show_sample = true,
            "--dump" => dump_path = args.next().map(PathBuf::from),
            "--iterations" => iterations = args.next().and_then(|n| n.parse().ok()).unwrap_or(3),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    let cases: Vec<Case> = serde_json::from_slice(
        &std::fs::read(&prompts_path).expect("read prompts file"),
    )
    .expect("parse prompts file");

    // Defaults to the DESKTOP shipping configuration, so a pass here means a
    // pass there: 16K window, MTP drafter, windowed SWA, the app's ubatch.
    //
    // `MOBILE=1` switches to the phone's configuration instead, which differs
    // in ways that can move quality rather than only speed: a 4K window, a
    // 128-token ubatch, one KV slot and no drafter. Worth a separate run — the
    // numbers this bench is usually quoted for were taken at the desktop
    // settings, and quoting them for mobile would be quoting a different
    // experiment.
    let mobile = std::env::var("MOBILE").map(|v| v == "1").unwrap_or(false);
    let cfg = if mobile {
        LlamaCppConfig {
            model_path,
            draft_path: None,
            n_ctx: 4096,
            n_ubatch: Some(128),
            n_slots: 1,
            ..Default::default()
        }
    } else {
        LlamaCppConfig {
            model_path,
            draft_path,
            n_ctx: 16384,
            ..Default::default()
        }
    };
    if mobile {
        println!("config: MOBILE (n_ctx 4096, n_ubatch 128, 1 slot, no drafter)");
    }
    println!(
        "availability: {:?}",
        LlamaCppEngine::availability_for(&cfg.model_path)
    );
    let load = Instant::now();
    let mut engine = LlamaCppEngine::new(cfg);
    println!("cases: {} x {iterations} iterations", cases.len());

    let mut total = 0usize;
    let mut parsed_ok = 0usize;
    let mut conformed = 0usize;
    let mut all_ms: Vec<u64> = vec![];
    let mut first_failure: Option<(String, String)> = None;
    // Every reply, for offline scoring. Parse rate cannot tell you whether the
    // output is any GOOD, and one printed sample per case is not an evaluation.
    let mut dump: Vec<Value> = vec![];

    for case in &cases {
        let mut ok = 0usize;
        let mut ms: Vec<u64> = vec![];
        let mut ptoks = 0u32;
        let mut otoks = 0u32;
        for i in 0..iterations {
            let req = GenerateRequest {
                system: case.system.clone(),
                messages: vec![Msg::user(case.user.clone())],
                json_schema: case.schema.clone(),
                // Desktop reserves slot 0 for the agent loop and runs background
                // one-shots on slot 1, so the bench matches that. Mobile has one
                // workload and therefore one slot: asking for slot 1 against a
                // single-sequence context fails as `invalid seq_id ... >= 1`,
                // surfacing as "n_tokens == 0" rather than anything about slots.
                slot: if mobile { 0 } else { 1 },
                enable_thinking: Some(thinking),
                ..Default::default()
            };
            let started = Instant::now();
            let res = match engine.generate(req) {
                Ok(r) => r,
                Err(e) => {
                    total += 1;
                    if first_failure.is_none() {
                        first_failure = Some((case.name.clone(), format!("ENGINE ERROR: {e}")));
                    }
                    println!("  {} #{i}: engine error: {e}", case.name);
                    continue;
                }
            };
            let elapsed = started.elapsed().as_millis() as u64;
            if dump_path.is_some() {
                dump.push(serde_json::json!({
                    "case": case.name,
                    "iteration": i,
                    "text": res.text.trim(),
                    "reasoning_chars": res.reasoning.len(),
                    "ms": elapsed,
                    "prompt_tokens": res.timings.prompt_tokens,
                    "output_tokens": res.timings.output_tokens,
                    "truncated": res.timings.truncated,
                }));
            }
            total += 1;
            ms.push(elapsed);
            all_ms.push(elapsed);
            ptoks = res.timings.prompt_tokens;
            otoks = res.timings.output_tokens;

            // No schema: free prose. Nothing to parse; count a non-empty reply.
            if case.schema.is_none() {
                let text = res.text.trim();
                if text.is_empty() {
                    println!("  {} #{i}: EMPTY reply", case.name);
                } else {
                    parsed_ok += 1;
                    conformed += 1;
                    ok += 1;
                    if ok == 1 && show_sample {
                        println!(
                            "  --- {} sample --- ({} chars, reasoning {} chars, {} out tokens)\n{}",
                            case.name, text.len(), res.reasoning.len(),
                            res.timings.output_tokens, text,
                        );
                    }
                }
                continue;
            }
            match serde_json::from_str::<Value>(res.text.trim()) {
                Ok(v) => {
                    parsed_ok += 1;
                    let missing = missing_required(case.schema.as_ref().unwrap(), &v);
                    if missing.is_empty() {
                        conformed += 1;
                        ok += 1;
                        // Parse rate is a proxy. Print one accepted reply per
                        // case so the CONTENT can be judged too: a schema-valid
                        // profile that describes the wrong person still passes
                        // every assertion here.
                        if ok == 1 && show_sample {
                            // reasoning is a SEPARATE field from text: if the
                            // token count dwarfs the printed object, the model
                            // is thinking despite the grammar, and that is the
                            // thing to see rather than infer.
                            println!(
                                "  --- {} sample --- (text {} chars, reasoning {} chars, {} out tokens)\n{}",
                                case.name,
                                res.text.trim().len(),
                                res.reasoning.len(),
                                res.timings.output_tokens,
                                res.text.trim(),
                            );
                            if !res.reasoning.is_empty() {
                                println!("  --- {} REASONING ---\n{}", case.name,
                                    res.reasoning.chars().take(600).collect::<String>());
                            }
                        }
                    } else {
                        println!("  {} #{i}: missing required {:?}", case.name, missing);
                        if first_failure.is_none() {
                            first_failure =
                                Some((case.name.clone(), res.text.chars().take(400).collect()));
                        }
                    }
                }
                Err(e) => {
                    println!(
                        "  {} #{i}: UNPARSEABLE ({e}); truncated={} out_tokens={}",
                        case.name, res.timings.truncated, res.timings.output_tokens
                    );
                    if first_failure.is_none() {
                        first_failure =
                            Some((case.name.clone(), res.text.chars().take(400).collect()));
                    }
                }
            }
        }
        println!(
            "{:<28} {}/{} ok  p50 {}ms  prompt {} tok  out {} tok",
            case.name,
            ok,
            iterations,
            pct(&mut ms, 0.5),
            ptoks,
            otoks
        );
    }

    println!("\n--- summary ---");
    println!("total runs        {total}");
    println!(
        "parsed as JSON    {parsed_ok}/{total} ({:.0}%)",
        100.0 * parsed_ok as f64 / total.max(1) as f64
    );
    println!(
        "schema-conformant {conformed}/{total} ({:.0}%)",
        100.0 * conformed as f64 / total.max(1) as f64
    );
    println!(
        "latency           p50 {}ms  p95 {}ms",
        pct(&mut all_ms, 0.5),
        pct(&mut all_ms, 0.95)
    );
    println!("wall clock        {:.1}s", load.elapsed().as_secs_f64());
    if let Some(path) = dump_path {
        std::fs::write(&path, serde_json::to_string_pretty(&dump).expect("serialize dump"))
            .expect("write dump");
        println!("dumped {} replies to {}", dump.len(), path.display());
    }
    if let Some((name, sample)) = first_failure {
        println!("\nfirst failure ({name}):\n{sample}");
    }
}
