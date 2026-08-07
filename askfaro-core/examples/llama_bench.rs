//! Run Scope's tool-calling bench cases through the in-process engine.
//!
//! This is the parity gate. `bench.mjs` measures the Ollama transport; this
//! measures `LlamaCppEngine` on the same cases, the same registry and the same
//! system prompt, so the only thing that differs is the engine. It emits JSONL
//! for `grade-local.mjs` to feed into the EXISTING grader, because re-implementing
//! grading here would make the two numbers incomparable, which is the whole
//! failure this bench was rebuilt to stop.
//!
//! ```sh
//! cargo run --release --features llama-cpp,metal --example llama_bench -- \
//!   <model.gguf> <scope-repo-root> > results.jsonl
//! ```
//!
//! Prints progress on stderr and JSONL on stdout, so the caller can pipe cleanly.
//!
//! It also gates the **persisted KV prefix**: the prefix is built, the engine is
//! unloaded and reloaded so the KV cache can only have come off disk, and then
//! the FIRST case is held to the same prefill budget as every later one. A
//! restore that quietly does nothing is invisible in every other number here.
//! The state file lands in `$TMPDIR/faro-bench-prefix/`; delete that directory
//! to force a rebuild.

use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg, ToolSchema,
};
use serde_json::{json, Value};

fn read_json(path: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(2);
    }))
    .unwrap_or_else(|e| {
        eprintln!("cannot parse {path}: {e}");
        std::process::exit(2);
    })
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (model_path, repo) = match (args.next(), args.next()) {
        (Some(m), Some(r)) => (m, r),
        _ => {
            eprintln!("usage: llama_bench <model.gguf> <scope-repo-root>");
            std::process::exit(2);
        }
    };

    let platform = format!("{repo}/frontend/src/platform");
    let registry = read_json(&format!("{platform}/scope_tools.json"));
    let prompt_consts = read_json(&format!("{platform}/scopy_prompt.json"));
    let index = read_json(&format!("{platform}/scopy_instructions_index.json"));
    let cases = read_json(&format!(
        "{repo}/desktop/spikes/f7-tool-calling/cases.json"
    ));

    // The app's real system prompt, assembled as desktop.ts does. `now` is
    // deliberately real: without a date the model cannot resolve "next Monday",
    // which is what made the old bench's number an artifact of its own prompt.
    let style_rules = prompt_consts["styleRules"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    // STABLE ONLY, mirroring the app. Nothing here may change between turns.
    //
    // This bench used to put `scopy_context` in the system block with a FROZEN
    // clock, which is how it reported 233ms warm prefill while the app paid
    // 27,495ms on every turn: the app's clock has minute resolution, so the
    // prefix diverged at about token 20 and re-prefilled all ~7,700 tokens. The
    // one variable that mattered was the one the harness held still. The clock
    // now moves per case, below, so this class of regression is measurable here
    // instead of arriving as "the app feels slow" weeks later.
    let system = format!(
        "You are Scopy, a fast on-device assistant.\n\n\
         response_style: {style} {rules}\n\n{loop_guidance}\n\n{index}",
        style = prompt_consts["responseStyle"].as_str().unwrap_or(""),
        rules = style_rules,
        loop_guidance = prompt_consts["toolLoopGuidance"].as_str().unwrap_or(""),
        index = index["index"].as_str().unwrap_or(""),
    );

    // The full registry, matching `--no-selector` on the JS bench. Selection is a
    // separate variable; mixing it in here would confound the engine comparison.
    let tools: Vec<ToolSchema> = registry
        .as_array()
        .expect("registry is an array")
        .iter()
        .map(|t| ToolSchema {
            name: t["function"]["name"].as_str().unwrap_or_default().to_string(),
            description: t["function"]["description"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            parameters: t["function"]["parameters"].clone(),
        })
        .collect();
    eprintln!("tools {} | cases {}", tools.len(), cases.as_array().map(|a| a.len()).unwrap_or(0));

    // Optional third arg: the MTP drafter. With it, decode goes through the
    // speculative path, which must produce IDENTICAL output to without it.
    let draft = args.next();
    let draft_configured = draft.is_some();
    if let Some(d) = &draft {
        eprintln!("drafter: {d}");
    }
    // The persisted KV prefix, on by default so the standard invocation
    // exercises it. In a temp directory rather than beside the weights, which is
    // where the APP puts it: the bench is usually pointed straight at Ollama's
    // blob store and has no business writing into someone else's model cache.
    let model_path: std::path::PathBuf = model_path.into();
    let prefix_dir = std::env::temp_dir().join("faro-bench-prefix");
    let mut engine = LlamaCppEngine::new(LlamaCppConfig {
        model_path: model_path.clone(),
        draft_path: draft.map(Into::into),
        prefix_cache_dir: Some(prefix_dir.clone()),
        state_key: "llama_bench".into(),
        ..Default::default()
    });

    // --- persist the prefix, then prove it survives a new context ---------
    //
    // Two phases on purpose. Building it in memory and then measuring prefill
    // would only re-measure the prefill that just happened; the claim under test
    // is that a KV cache written to disk restores into a context that never saw
    // the prompt. So the engine is UNLOADED and brought back, which is what a
    // relaunch does, and only then are the cases run.
    let prefix_req = GenerateRequest {
        system: system.clone(),
        tools: tools.clone(),
        ..Default::default()
    };
    match engine.ensure_prefix(&prefix_req) {
        Ok(r) => eprintln!(
            "prefix: {} tokens, {:.1} MB on disk, rebuilt={} ({}ms)\n  {}",
            r.tokens,
            r.bytes as f64 / 1e6,
            r.rebuilt,
            r.ms,
            r.path
        ),
        Err(e) => {
            eprintln!("FAILED: could not build the persisted prefix: {e}");
            std::process::exit(1);
        }
    }
    engine.unload();
    match engine.warm() {
        Ok(ms) => eprintln!("reloaded in {ms}ms; the KV prefix below comes off disk"),
        Err(e) => {
            eprintln!("FAILED: reload after unload: {e}");
            std::process::exit(1);
        }
    }

    // Collected for the ASSERTIONS at the end. A bench that only prints numbers
    // cannot fail, and every on-device bug this year has been a component
    // degrading to a working-looking fallback: the numbers were all there, in a
    // log nobody diffs.
    let mut warm_prefills: Vec<u64> = Vec::new();
    let mut leaked_marker: Vec<String> = Vec::new();
    let mut drafted_total: u32 = 0;
    let mut first_prefill: u64 = 0;

    let case_list = cases.as_array().expect("cases is an array");
    for (i, c) in case_list.iter().enumerate() {
        let id = c["id"].as_str().unwrap_or("?");
        let prompt = c["prompt"].as_str().unwrap_or("");
        eprint!("\r  {}/{} {id:<24}", i + 1, case_list.len());

        // Volatile context on the USER turn, which the template renders after
        // the tools, so the stable prefix survives. The minute ticks per case on
        // purpose: a real session's clock moves, and a bench that freezes it
        // cannot see the most expensive prompt bug there is.
        let req = GenerateRequest {
            system: system.clone(),
            messages: vec![Msg::user(format!(
                "scopy_context: {{\"now\":\"2026-07-30 08:{:02}\",\"timezone\":\"UTC\"}}\n\n{prompt}",
                i % 60,
            ))],
            tools: tools.clone(),
            slot: 0,
            ..Default::default()
        };

        let out = match engine.generate(req) {
            Ok(o) => o,
            Err(e) => {
                println!("{}", json!({"id": id, "error": e.to_string()}));
                continue;
            }
        };

        // EVERY case, the first one included. It used to be excluded because it
        // paid the cold prefill; with a persisted prefix there is no cold turn,
        // so case 0 is now the case that proves the restore worked and is the
        // one assertion 3 below is about.
        warm_prefills.push(out.timings.prefill_ms);
        if i == 0 {
            first_prefill = out.timings.prefill_ms;
        }
        if out.text.contains("<turn|>") || out.text.contains("<|turn>") {
            leaked_marker.push(id.to_string());
        }
        drafted_total += out.timings.draft_proposed;

        // One line per case; the grader wants tool name + decoded args.
        let call = out.tool_calls.first();
        println!(
            "{}",
            json!({
                "id": id,
                "toolName": call.map(|c| c.name.clone()),
                "args": call.map(|c| c.arguments.clone()).unwrap_or(Value::Null),
                "text": out.text,
                "reasoningChars": out.reasoning.len(),
                // The reasoning ITSELF, not just its length. On a thinking model
                // the choice between two near-synonymous tools is argued out here
                // and nowhere else; a length tells you the argument happened and
                // withholds what it said.
                "reasoning": out.reasoning,
                "ms": out.model_ms,
                "promptTokens": out.timings.prompt_tokens,
                "outputTokens": out.timings.output_tokens,
                "prefillMs": out.timings.prefill_ms,
                "decodeMs": out.timings.decode_ms,
                "truncated": out.timings.truncated,
            })
        );
    }
    eprintln!("\ndone");

    // Release the engine BEFORE the assertions. `std::process::exit` does not run
    // destructors, so exiting with the engine still alive leaves ggml-metal's
    // atexit check to find unreleased resource sets and abort: the process dies
    // with SIGABRT (134) instead of the intended 1, and a caller reading the exit
    // code learns "crashed" where the truth is "an assertion failed". The same
    // drop-order rule the engine documents internally, applied to the harness.
    drop(engine);

    // --- assertions ------------------------------------------------------
    //
    // These exist because every failure this year was silent. The engine kept
    // answering, the bench kept printing plausible numbers, and the defect
    // surfaced days later as "the app feels slow" or a stray token in a reply.
    // A threshold that fails is worth more than a number that is merely true.
    let mut failures: Vec<String> = Vec::new();

    // 1. Prefix reuse. The single most expensive regression available here: a
    //    volatile prompt head, or a second workload sharing the cache, takes
    //    warm prefill from ~230ms to ~27,000ms and nothing else changes.
    warm_prefills.sort_unstable();
    let p50 = warm_prefills.get(warm_prefills.len() / 2).copied().unwrap_or(0);
    const WARM_PREFILL_BUDGET_MS: u64 = 1_500;
    if p50 > WARM_PREFILL_BUDGET_MS {
        failures.push(format!(
            "warm prefill p50 {p50}ms exceeds {WARM_PREFILL_BUDGET_MS}ms: the KV prefix is not \
             being reused. Something volatile is at the HEAD of the prompt, or another \
             workload shares this cache slot."
        ));
    }

    // 2. Template markers must never reach `content`. Upstream's parser leaves
    //    the end-of-turn token in on purpose; trimming it is the caller's job,
    //    and when that broke, all 20 cases still passed because nothing here
    //    asserted on answer TEXT.
    if !leaked_marker.is_empty() {
        failures.push(format!(
            "end-of-turn marker leaked into content on {} case(s): {}",
            leaked_marker.len(),
            leaked_marker.join(", ")
        ));
    }

    // 3. The FIRST turn after a reload must be warm too. This is the persisted
    //    prefix's entire claim, and it fails in the direction that looks fine:
    //    a restore that silently does nothing leaves every number below correct
    //    and only the cold turn, the one a user meets on every launch, slow.
    if first_prefill > WARM_PREFILL_BUDGET_MS {
        failures.push(format!(
            "first-turn prefill {first_prefill}ms exceeds {WARM_PREFILL_BUDGET_MS}ms after a \
             reload: the persisted KV prefix was not restored, or was restored and did not \
             match the prompt. Check the `llama_cpp: restored`/`persisted` lines above."
        ));
    }

    // 4. A configured drafter must actually draft. Speculation fails OPEN: the
    //    drafter loads, contributes nothing, and the run completes at exactly
    //    the unspeculated rate. Twice in one day, once from a missing
    //    `--spec-type` and once from a context built without `ctx_other`.
    if draft_configured && drafted_total == 0 {
        failures.push(
            "a drafter was configured but proposed ZERO tokens: speculation silently \
             degraded to plain decode."
                .to_string(),
        );
    }

    if failures.is_empty() {
        eprintln!(
            "assertions passed (prefill p50 {p50}ms, first turn {first_prefill}ms, \
             drafted {drafted_total} tokens)"
        );
    } else {
        eprintln!("\nFAILED:");
        for f in &failures {
            eprintln!("  - {f}");
        }
        std::process::exit(1);
    }
}
