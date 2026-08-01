//! Does the end-of-turn marker leak into `content`, and when?
//!
//! Observed in the desktop app: a plain chat turn came back as
//! `"Today is July 31, 2026.<turn|>"`. Only 9 tokens were decoded, so generation
//! stopped correctly; the marker survived the PARSE.
//!
//! Two candidate causes, which this separates:
//!
//!   1. The decode loop stops only on `is_eog_token` and ignores the
//!      `additional_stops` the template itself declares, so the marker is
//!      emitted as ordinary text.
//!   2. With NO tools, Gemma 4 may not resolve to the `peg-gemma4` format, so
//!      `ctx->last.parser` is empty, the PEG matcher has nothing to match, and
//!      raw text falls through into `content`. Every bench case carries the full
//!      registry, which would explain why this never showed up there.
//!
//! If only the no-tools arm leaks, it is (2). If both leak, it is (1).
//! The drafter is also toggled, to rule out the speculative path.
//!
//! ```sh
//! cargo run --release --features llama-cpp --example turn_marker -- <model.gguf> [drafter.gguf]
//! ```
use askfaro_core::generation::{
    GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg, ToolSchema,
};
use serde_json::json;

const PROMPT: &str = "What day is it today?";

fn tool() -> ToolSchema {
    ToolSchema {
        name: "scopy_task_list".into(),
        description: "List the user's tasks.".into(),
        parameters: json!({
            "type": "object",
            "properties": { "status": { "type": "string", "description": "Filter by status." } },
        }),
    }
}

fn run(engine: &mut LlamaCppEngine, label: &str, tools: Vec<ToolSchema>) {
    let req = GenerateRequest {
        system: "You are Scopy, a fast on-device assistant.\n\n\
                 scopy_context: {\"now\":\"2026-07-31 16:00\",\"timezone\":\"UTC\"}"
            .into(),
        messages: vec![Msg::user(PROMPT)],
        tools,
        slot: 0,
    };
    match engine.generate(req) {
        Ok(out) => {
            let leaked = out.text.contains("<turn|>") || out.text.contains("<|turn>");
            println!(
                "{label:<22} leaked={:<5} out_tok={:<4} content={:?}",
                leaked, out.timings.output_tokens, out.text
            );
            if !out.reasoning.is_empty() {
                println!("{:<22} reasoning={:?}", "", out.reasoning);
            }
        }
        Err(e) => println!("{label:<22} ERROR {e}"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(model) = args.next() else {
        eprintln!("usage: turn_marker <model.gguf> [drafter.gguf]");
        std::process::exit(2);
    };
    let draft = args.next();

    // Fresh engine per arm: the chat context caches the LAST apply, and the
    // whole question is whether that apply had tools in it.
    for (arm, tools) in [("no-tools", vec![]), ("with-tools", vec![tool()])] {
        let mut engine = LlamaCppEngine::new(LlamaCppConfig {
            model_path: model.clone().into(),
            draft_path: None,
            ..Default::default()
        });
        run(&mut engine, &format!("plain/{arm}"), tools);
    }

    if let Some(d) = draft {
        for (arm, tools) in [("no-tools", vec![]), ("with-tools", vec![tool()])] {
            let mut engine = LlamaCppEngine::new(LlamaCppConfig {
                model_path: model.clone().into(),
                draft_path: Some(d.clone().into()),
                ..Default::default()
            });
            run(&mut engine, &format!("mtp/{arm}"), tools);
        }
    }
}
