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
    let system = format!(
        "You are Scopy, a fast on-device assistant.\n\n\
         scopy_context: {{\"now\":\"{now}\",\"timezone\":\"UTC\"}}\n\n\
         response_style: {style} {rules}\n\n{loop_guidance}\n\n{index}",
        now = "2026-07-30 08:00",
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
    if let Some(d) = &draft {
        eprintln!("drafter: {d}");
    }
    let mut engine = LlamaCppEngine::new(LlamaCppConfig {
        model_path: model_path.into(),
        draft_path: draft.map(Into::into),
        ..Default::default()
    });

    let case_list = cases.as_array().expect("cases is an array");
    for (i, c) in case_list.iter().enumerate() {
        let id = c["id"].as_str().unwrap_or("?");
        let prompt = c["prompt"].as_str().unwrap_or("");
        eprint!("\r  {}/{} {id:<24}", i + 1, case_list.len());

        let req = GenerateRequest {
            system: system.clone(),
            messages: vec![Msg {
                role: "user".into(),
                content: prompt.to_string(),
            }],
            tools: tools.clone(),
        };

        let out = match engine.generate(req) {
            Ok(o) => o,
            Err(e) => {
                println!("{}", json!({"id": id, "error": e.to_string()}));
                continue;
            }
        };

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
}
