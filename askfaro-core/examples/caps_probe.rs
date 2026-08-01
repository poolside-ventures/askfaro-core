//! What does this model's chat template actually support?
//!
//! Run this FIRST when wiring up a new model family, before trusting a
//! transcript to it. The jinja layer computes these flags when it parses the
//! template and llama.cpp has always exposed them; not reading them is how Scope
//! shipped an agent loop whose tool calls were silently dropped for months,
//! found eventually by benchmark rather than by anything saying so.
//!
//! ```sh
//! cargo run --release --features llama-cpp --example caps_probe -- <model.gguf>
//! ```
//!
//! The two that decide whether a tool-calling host will work at all:
//!
//! - `supports_tools` false: tool schemas never reach the prompt, so the model
//!   cannot call anything, and it will say so in fluent prose rather than error.
//! - `supports_tool_calls` false: a REPLAYED assistant tool call is dropped, so
//!   the model sees a conversation in which it never called the tool whose
//!   result it is being shown. This is the one that bit us.
//!
//! `supports_object_arguments` is NOT a problem to solve: upstream converts our
//! JSON-string arguments itself when a template wants objects. It is printed
//! here because it is informative, not because it needs action.
//!
//! The engine warns about a genuine mismatch on its own (see `warn_unsupported`)
//! and `LlamaCppEngine::template_caps()` returns the same map, so this example
//! is the standalone version for when you have a GGUF and a question.

use askfaro_core::generation::{LlamaCppConfig, LlamaCppEngine};

fn main() {
    let Some(model) = std::env::args().nth(1) else {
        eprintln!("usage: caps_probe <model.gguf>");
        std::process::exit(2);
    };

    let mut engine = LlamaCppEngine::new(LlamaCppConfig {
        model_path: model.into(),
        ..Default::default()
    });
    // Loads the weights, because the template comes out of the GGUF. There is no
    // cheaper way to read it: the file IS the source of truth for the template.
    if let Err(e) = engine.warm() {
        eprintln!("FAILED: could not load the model: {e}");
        std::process::exit(1);
    }

    let caps = engine.template_caps();
    if caps.is_empty() {
        eprintln!(
            "this build reported no template capabilities at all. That is not a model \
             problem: an upstream without `common_chat_templates_get_caps` cannot answer, \
             and the engine treats every capability as present in that case."
        );
        std::process::exit(1);
    }

    println!("template capabilities:");
    for (name, value) in &caps {
        println!("  {name:<34} {value}");
    }

    // The verdict, rather than leaving the reader to know which flags matter.
    let get = |k: &str| caps.get(k).copied().unwrap_or(true);
    println!();
    match (get("supports_tools"), get("supports_tool_calls")) {
        (true, true) => {
            println!("usable for a multi-step agent loop: tools render, and replayed tool calls survive.")
        }
        (true, false) => println!(
            "SINGLE-CALL ONLY. Tools render, but a replayed assistant tool call is dropped, so \
             the model cannot see what it already did. A loop over this template will answer \
             from a transcript with its own actions missing."
        ),
        (false, _) => {
            println!("NOT usable for tool calling. Tool schemas do not reach the prompt at all.")
        }
    }
}
