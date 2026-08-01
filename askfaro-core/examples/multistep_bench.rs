//! Run Scope's tool-calling cases through the MULTI-STEP agent loop.
//!
//! `llama_bench` stops at the first model call. The product does not: it runs
//! `generateText` with `maxSteps: 8`, feeds each tool's result back as a
//! `role: "tool"` message, and calls the model again until it answers in text.
//! Every number we have about this engine describes the first call of that loop.
//! This measures the whole of it.
//!
//! ```sh
//! cargo run --release --features llama-cpp --example multistep_bench -- \
//!   <model.gguf> <scope-repo-root> [drafter.gguf] > results.jsonl
//! node desktop/spikes/f7-tool-calling/grade-local.mjs results.jsonl \
//!   --cases desktop/spikes/f7-tool-calling/cases-multistep.json
//! ```
//!
//! Cases come from `cases-multistep.json`, which carries a synthetic result per
//! tool. Nothing is executed: the point is to measure the ENGINE across a
//! continuation, so the results are fixed and the same on every run.
//!
//! ## The transcript mapping is copied, not invented
//!
//! What the model sees on step 2 is whatever `shared/src/agent/local-provider.ts`
//! renders, so this reproduces that function. It matters that the two move
//! together: the first run of this bench measured 0% correct continuation calls
//! because the provider was writing tool use as PROSE (`[called NAME({args})]`,
//! `[NAME result: ...]`) into ordinary message text, and the model, shown bracket
//! prose where the template has real `<|tool_call>` and `<|tool_response>`
//! tokens, wrote bracket prose back: fabricated `[tool_response]` blocks, and our
//! own `[called ...]` syntax handed to the user as their answer.
//!
//! Both sides now send structured calls and results, so this uses
//! `Msg::assistant_calls` / `Msg::tool_result` rather than formatting strings.
//! If the provider ever goes back to prose, change this in the same commit or the
//! bench stops describing the product.
//!
//! ## What it asserts
//!
//! The headline risk is a silent re-prefill. Step 2's prompt is step 1's prompt
//! plus an assistant turn and a tool turn, so it EXTENDS the cached tokens and
//! should cost a few hundred tokens of prefill. If the persisted prefix does not
//! survive a transcript containing tool results, step 2 pays the full ~7,700
//! tokens again at ~280 tok/s and the only symptom is that the app feels slow.
//! Nothing errors. That is assertion 1.

use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg, ToolCall, ToolSchema,
};
use serde_json::{json, Map, Value};

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

/// How tool use is written into the transcript.
///
/// Two mappings, selectable, because the difference between them is the whole
/// finding and a claim about it should be re-runnable rather than trusted. The
/// first measurement of `Prose` and the first of `Structured` happened hours
/// apart, and another session regenerated `scopy_prompt.json` and
/// `scope_tools.json` in between, which moved the prefix by 45 tokens. Two
/// numbers taken over a moved prompt do not compare, and the fix for that is a
/// switch, not a promise to be careful.
#[derive(Clone, Copy, PartialEq)]
enum Mapping {
    /// What `local-provider.ts` sent until 2026-08-01: tool use as message TEXT.
    /// Kept so the defect stays reproducible after the code that caused it is
    /// gone. `--prose` selects it.
    Prose,
    /// Structured `tool_calls` / `tool_name` / `tool_call_id`, which is what the
    /// template's tool branches actually read. The default.
    Structured,
}

/// The assistant turn that made the calls, mirroring `local-provider.ts`.
fn assistant_msg(mapping: Mapping, text: &str, calls: &[ToolCall]) -> Msg {
    match mapping {
        Mapping::Structured => Msg::assistant_calls(text, calls.to_vec()),
        Mapping::Prose => {
            let mut parts: Vec<String> = Vec::new();
            if !text.is_empty() {
                parts.push(text.to_string());
            }
            for c in calls {
                parts.push(format!("[called {}({})]", c.name, c.arguments));
            }
            Msg::assistant(parts.join(" "))
        }
    }
}

/// The tool-result turns, mirroring `local-provider.ts`.
///
/// Structured emits one message PER result, each naming its tool: a tool turn
/// names the tool it came from, so folding two results into one message would
/// have to drop a name. Prose concatenated them into a single `tool` message and
/// wrapped each in the AI SDK's `{"type":"json","value":...}` envelope, which is
/// what `JSON.stringify(part.output)` produces on a tagged union.
fn tool_msgs(mapping: Mapping, results: &[(ToolCall, Value)]) -> Vec<Msg> {
    match mapping {
        Mapping::Structured => results
            .iter()
            .map(|(call, out)| Msg::tool_result(&call.name, &call.id, out.to_string()))
            .collect(),
        Mapping::Prose => vec![Msg {
            role: "tool".into(),
            content: results
                .iter()
                .map(|(call, out)| {
                    format!(
                        "[{} result: {}]",
                        call.name,
                        json!({"type": "json", "value": out})
                    )
                })
                .collect::<Vec<_>>()
                .join(" "),
            ..Default::default()
        }],
    }
}

/// The synthetic result for a call, by tool name, falling back to a generic ok.
///
/// A tool the case did not anticipate still gets an answer rather than ending
/// the run: the model going somewhere unexpected is a finding to record, not a
/// reason to stop measuring.
fn result_for(case: &Value, name: &str) -> Value {
    case["results"]
        .get(name)
        .cloned()
        .unwrap_or_else(|| json!({"ok": true, "note": "no synthetic result for this tool"}))
}

fn main() {
    // `--prose` anywhere in argv selects the old mapping; everything else is
    // positional as before.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mapping = if argv.iter().any(|a| a == "--prose") {
        Mapping::Prose
    } else {
        Mapping::Structured
    };
    let mut args = argv.into_iter().filter(|a| a != "--prose");
    let (model_path, repo) = match (args.next(), args.next()) {
        (Some(m), Some(r)) => (m, r),
        _ => {
            eprintln!(
                "usage: multistep_bench <model.gguf> <scope-repo-root> [drafter.gguf] [--prose]"
            );
            std::process::exit(2);
        }
    };

    let platform = format!("{repo}/frontend/src/platform");
    let registry = read_json(&format!("{platform}/scope_tools.json"));
    let prompt_consts = read_json(&format!("{platform}/scopy_prompt.json"));
    let index = read_json(&format!("{platform}/scopy_instructions_index.json"));
    let cases = read_json(&format!(
        "{repo}/desktop/spikes/f7-tool-calling/cases-multistep.json"
    ));

    // Assembled exactly as `onDeviceSystemPrompt()` does, and for the same
    // reason llama_bench does it: a bench with its own prompt measures its own
    // prompt. STABLE ONLY -- the clock goes on the user turn, below.
    let style_rules = prompt_consts["styleRules"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(" "))
        .unwrap_or_default();
    let system = format!(
        "You are Scopy, a fast on-device assistant.\n\n\
         response_style: {style} {rules}\n\n{loop_guidance}\n\n{index}",
        style = prompt_consts["responseStyle"].as_str().unwrap_or(""),
        rules = style_rules,
        loop_guidance = prompt_consts["toolLoopGuidance"].as_str().unwrap_or(""),
        index = index["index"].as_str().unwrap_or(""),
    );

    let tools: Vec<ToolSchema> = registry
        .as_array()
        .expect("registry is an array")
        .iter()
        .map(|t| ToolSchema {
            name: t["function"]["name"].as_str().unwrap_or_default().to_string(),
            description: t["function"]["description"].as_str().unwrap_or_default().to_string(),
            parameters: t["function"]["parameters"].clone(),
        })
        .collect();

    let draft = args.next();
    let draft_configured = draft.is_some();
    eprintln!(
        "tools {} | cases {} | drafter {} | mapping {}",
        tools.len(),
        cases.as_array().map(|a| a.len()).unwrap_or(0),
        draft.as_deref().unwrap_or("none"),
        if mapping == Mapping::Prose { "PROSE (pre-2026-08-01)" } else { "structured" },
    );

    // The SAME state_key and prefix directory as `llama_bench`, deliberately.
    // The prefix is a function of the system block and tool set, which are
    // identical here, so the two benches should share one file. If they ever
    // stop sharing it, that is the signal that one of them has drifted from the
    // app's prompt and it should be loud rather than a second 145 MB file.
    let prefix_dir = std::env::temp_dir().join("faro-bench-prefix");
    let mut engine = LlamaCppEngine::new(LlamaCppConfig {
        model_path: model_path.into(),
        draft_path: draft.map(Into::into),
        prefix_cache_dir: Some(prefix_dir),
        state_key: "llama_bench".into(),
        ..Default::default()
    });

    let prefix_req = GenerateRequest {
        system: system.clone(),
        tools: tools.clone(),
        ..Default::default()
    };
    match engine.ensure_prefix(&prefix_req) {
        Ok(r) => eprintln!(
            "prefix: {} tokens, {:.1} MB, rebuilt={} ({}ms)",
            r.tokens,
            r.bytes as f64 / 1e6,
            r.rebuilt,
            r.ms
        ),
        Err(e) => {
            eprintln!("FAILED: could not build the persisted prefix: {e}");
            std::process::exit(1);
        }
    }
    // Unload/reload so the KV below can only have come off disk, exactly as
    // llama_bench does: a restore that silently does nothing is invisible in
    // every other number here.
    engine.unload();
    if let Err(e) = engine.warm() {
        eprintln!("FAILED: reload after unload: {e}");
        std::process::exit(1);
    }

    // The app's ceiling. A turn that needs more than this has not converged.
    const MAX_STEPS: usize = 8;

    let mut first_prefills: Vec<u64> = Vec::new();
    let mut cont_prefills: Vec<u64> = Vec::new();
    let mut leaked_marker: Vec<String> = Vec::new();
    let mut never_answered: Vec<String> = Vec::new();
    let mut exceeded_calls: Vec<String> = Vec::new();
    let mut drafted_total: u32 = 0;
    // Answers that leaked the transcript's own tool syntax back to the user, or
    // wrote a tool-response block themselves. Both are the signature of a model
    // being shown prose where structure belongs, and both READ FINE: nothing
    // errors, the reply is fluent, and the content is invented.
    let mut echoed_syntax: Vec<String> = Vec::new();

    let case_list = cases.as_array().expect("cases is an array");
    for (ci, case) in case_list.iter().enumerate() {
        let id = case["id"].as_str().unwrap_or("?");
        let prompt = case["prompt"].as_str().unwrap_or("");
        eprint!("\r  {}/{} {id:<26}", ci + 1, case_list.len());

        // Volatile context on the user turn, minute ticking per case. Same
        // reasoning as llama_bench: a frozen clock hides the single most
        // expensive prompt bug there is, and the app's clock has minute
        // resolution.
        //
        // Built ONCE and never rebuilt across steps, which is also what the app
        // does: `volatileContext` is prepended to the last user message before
        // the loop starts, so every step of one turn replays a byte-identical
        // user turn. If it were re-stamped per step the cache would break here
        // and nowhere else.
        let user_content = format!(
            "scopy_context: {{\"now\":\"2026-07-30 08:{:02}\",\"timezone\":\"UTC\"}}\n\n{prompt}",
            ci % 60,
        );
        let mut messages = vec![Msg::user(user_content)];

        let mut calls_made: Vec<Value> = Vec::new();
        let mut step_rows: Vec<Value> = Vec::new();
        let mut final_text = String::new();
        let mut total_ms: u64 = 0;
        let mut first_call: Option<ToolCall> = None;
        let mut first_step_prefill = 0u64;
        let mut first_step_prompt_tokens = 0u32;
        let mut first_step_decode = 0u64;
        let mut first_step_output = 0u32;
        let mut hard_error: Option<String> = None;
        let mut answered = false;

        for step in 0..MAX_STEPS {
            let out = match engine.generate(GenerateRequest {
                system: system.clone(),
                messages: messages.clone(),
                tools: tools.clone(),
                slot: 0,
            }) {
                Ok(o) => o,
                Err(e) => {
                    hard_error = Some(e.to_string());
                    break;
                }
            };
            total_ms += out.model_ms;
            drafted_total += out.timings.draft_proposed;
            if step == 0 {
                first_prefills.push(out.timings.prefill_ms);
                first_step_prefill = out.timings.prefill_ms;
                first_step_prompt_tokens = out.timings.prompt_tokens;
                first_step_decode = out.timings.decode_ms;
                first_step_output = out.timings.output_tokens;
            } else {
                cont_prefills.push(out.timings.prefill_ms);
            }
            if out.text.contains("<turn|>") || out.text.contains("<|turn>") {
                leaked_marker.push(format!("{id}#{step}"));
            }

            step_rows.push(json!({
                "step": step,
                "prefillMs": out.timings.prefill_ms,
                "decodeMs": out.timings.decode_ms,
                "promptTokens": out.timings.prompt_tokens,
                "outputTokens": out.timings.output_tokens,
                "calls": out.tool_calls.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
                "textChars": out.text.len(),
            }));

            if out.tool_calls.is_empty() {
                // Text (or an abstention) ends the turn, exactly as the harness's
                // stop condition does.
                final_text = out.text.clone();
                answered = true;
                break;
            }

            let calls: Vec<ToolCall> = out
                .tool_calls
                .iter()
                .cloned()
                .collect();
            if first_call.is_none() {
                first_call = calls.first().cloned();
            }
            for c in &calls {
                calls_made.push(json!({"name": c.name, "args": c.arguments}));
            }

            let results: Vec<(ToolCall, Value)> = calls
                .iter()
                .map(|c| (c.clone(), result_for(case, &c.name)))
                .collect();
            messages.push(assistant_msg(mapping, &out.text, &calls));
            messages.extend(tool_msgs(mapping, &results));
        }

        if hard_error.is_none() && !answered {
            never_answered.push(id.to_string());
        }
        // The regression check for THE bug this bench found. When tool use went
        // into the transcript as text, the model answered in the same register:
        // it echoed `[called scopy_task({...})]` verbatim as its reply and wrote
        // its own `[tool_response]` blocks full of records that never existed.
        // Assert on the answer, because every other number stayed green through
        // all of it.
        for marker in ["[called ", "[tool_response]", "[/tool_response]", "[waiting for tool"] {
            if final_text.contains(marker) {
                echoed_syntax.push(format!("{id} ({marker:?})"));
                break;
            }
        }
        let max_calls = case["maxCalls"].as_u64().unwrap_or(MAX_STEPS as u64) as usize;
        if calls_made.len() > max_calls {
            exceeded_calls.push(format!("{id} ({} > {max_calls})", calls_made.len()));
        }

        // One row per case, in the shape grade-local.mjs already reads:
        // `toolName`/`args` are the FIRST call, so the existing grader scores
        // step one identically to the single-step bench and the two runs stay
        // comparable. Everything multi-step is additive.
        let mut row = Map::new();
        row.insert("id".into(), json!(id));
        row.insert("toolName".into(), json!(first_call.as_ref().map(|c| c.name.clone())));
        row.insert(
            "args".into(),
            first_call.as_ref().map(|c| c.arguments.clone()).unwrap_or(Value::Null),
        );
        row.insert("text".into(), json!(final_text));
        row.insert("ms".into(), json!(total_ms));
        row.insert("promptTokens".into(), json!(first_step_prompt_tokens));
        row.insert("outputTokens".into(), json!(first_step_output));
        row.insert("prefillMs".into(), json!(first_step_prefill));
        row.insert("decodeMs".into(), json!(first_step_decode));
        row.insert("calls".into(), Value::Array(calls_made));
        row.insert("steps".into(), Value::Array(step_rows));
        row.insert("answered".into(), json!(answered));
        row.insert(
            "mapping".into(),
            json!(if mapping == Mapping::Prose { "prose" } else { "structured" }),
        );
        if let Some(e) = hard_error {
            row.insert("error".into(), json!(e));
        }
        println!("{}", Value::Object(row));
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
    let mut failures: Vec<String> = Vec::new();
    let p50 = |v: &mut Vec<u64>| {
        v.sort_unstable();
        v.get(v.len() / 2).copied().unwrap_or(0)
    };
    let first_p50 = p50(&mut first_prefills);
    let cont_p50 = p50(&mut cont_prefills);

    // 1. THE assertion this bench exists for. A continuation's prompt is the
    //    previous one plus an assistant turn and a tool turn, so it extends the
    //    cache rather than replacing it and should prefill a few hundred tokens.
    //    If a transcript containing tool results breaks the match, step 2 re-reads
    //    all ~7,700 tokens and nothing errors: the turn is simply 27 seconds
    //    slower. A tighter budget than the first-call one on purpose, because a
    //    continuation has strictly LESS new text to read than the turn before it.
    const CONT_PREFILL_BUDGET_MS: u64 = 1_000;
    if cont_prefills.is_empty() {
        failures.push(
            "no continuation steps ran at all, so the multi-step path was never exercised: \
             every case ended on its first model call."
                .to_string(),
        );
    } else if cont_p50 > CONT_PREFILL_BUDGET_MS {
        failures.push(format!(
            "continuation prefill p50 {cont_p50}ms exceeds {CONT_PREFILL_BUDGET_MS}ms across \
             {} steps: feeding a tool result back is invalidating the KV prefix. The transcript \
             the assistant/tool turns render into is not extending the cached prompt.",
            cont_prefills.len()
        ));
    }

    // 2. The first call must still be warm, which is llama_bench's assertion 3
    //    repeated here because this bench restores the same file: if the two
    //    disagree, the prefix is sensitive to something one of them does.
    const WARM_PREFILL_BUDGET_MS: u64 = 1_500;
    if first_p50 > WARM_PREFILL_BUDGET_MS {
        failures.push(format!(
            "first-call prefill p50 {first_p50}ms exceeds {WARM_PREFILL_BUDGET_MS}ms: the \
             persisted prefix was not restored, or did not match."
        ));
    }

    // 3. Every turn must END. `maxSteps: 8` is a backstop, not a plan; a case
    //    that calls tools eight times and never speaks is a hang the user sees
    //    as a spinner, and it scores as a perfectly good first call.
    if !never_answered.is_empty() {
        failures.push(format!(
            "{} case(s) never produced a text answer within {MAX_STEPS} steps: {}",
            never_answered.len(),
            never_answered.join(", ")
        ));
    }

    // 4. Call-count ceiling per case. Catches the loop that converges but takes
    //    a scenic route, which costs real seconds and no correctness score.
    if !exceeded_calls.is_empty() {
        failures.push(format!(
            "{} case(s) exceeded their expected call budget: {}",
            exceeded_calls.len(),
            exceeded_calls.join(", ")
        ));
    }

    // 5 and 6. As llama_bench: markers must not leak into content, and a
    //    configured drafter must actually draft rather than fail open.
    if !leaked_marker.is_empty() {
        failures.push(format!(
            "end-of-turn marker leaked into content on {} step(s): {}",
            leaked_marker.len(),
            leaked_marker.join(", ")
        ));
    }
    // 7. The answer must never contain the transcript's own scaffolding. This is
    //    the one assertion that would have caught the multi-step defect on the
    //    day it shipped: first-call accuracy, prefill, tool-selection and latency
    //    were all perfect throughout, and the only evidence was in the text.
    if !echoed_syntax.is_empty() {
        failures.push(format!(
            "{} case(s) echoed transcript syntax into the ANSWER: {}. The model is imitating \
             the shape of the conversation it was shown instead of reading it, which is what \
             happens when tool calls and results are rendered as prose rather than through the \
             template's tool branches (see Msg::assistant_calls / Msg::tool_result).",
            echoed_syntax.len(),
            echoed_syntax.join(", ")
        ));
    }

    if draft_configured && drafted_total == 0 {
        failures.push(
            "a drafter was configured but proposed ZERO tokens: speculation silently degraded \
             to plain decode."
                .to_string(),
        );
    }

    if failures.is_empty() {
        eprintln!(
            "assertions passed (first-call prefill p50 {first_p50}ms, continuation p50 \
             {cont_p50}ms over {} steps, drafted {drafted_total} tokens)",
            cont_prefills.len()
        );
    } else {
        eprintln!("\nFAILED:");
        for f in &failures {
            eprintln!("  - {f}");
        }
        std::process::exit(1);
    }
}
