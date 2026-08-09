//! Build the persisted KV prefix as a distributable ARTIFACT.
//!
//! The release pipeline runs this once per (app prompt, weights) pair so a
//! fresh install can DOWNLOAD the prefix instead of spending ~40s of GPU
//! computing it during onboarding. It assembles the exact on-device prompt the
//! way `llama_bench` does (same platform JSONs, same registry), builds the
//! prefix through the same `ensure_prefix` the app calls, and copies the
//! resulting state file to the requested output path.
//!
//! ```sh
//! cargo run --release --features llama-cpp --example build_prefix -- \
//!   <model.gguf> <scope-repo-root> <drafter.gguf> <out.kv>
//! ```
//!
//! The engine config below MUST mirror the desktop's `brain_config` for every
//! field that shapes the state (n_ctx, n_slots, swa_full, drafter): a state
//! built under a different shape deserializes into plausible nonsense rather
//! than an error, which is why the app also validates the token list against
//! its first real prompt before trusting anything it downloaded.

use askfaro_core::generation::{GenerateRequest, LlamaCppConfig, LlamaCppEngine, ToolSchema};
use serde_json::Value;

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
    let (model_path, repo, draft, out) = match (args.next(), args.next(), args.next(), args.next())
    {
        (Some(m), Some(r), Some(d), Some(o)) => (m, r, d, o),
        _ => {
            eprintln!("usage: build_prefix <model.gguf> <scope-repo-root> <drafter.gguf> <out.kv>");
            std::process::exit(2);
        }
    };

    let platform = format!("{repo}/frontend/src/platform");
    let registry = read_json(&format!("{platform}/scope_tools.json"));
    let prompt_consts = read_json(&format!("{platform}/scopy_prompt.json"));
    let index = read_json(&format!("{platform}/scopy_instructions_index.json"));

    // Identical assembly to llama_bench and desktop.ts `onDeviceSystemPrompt`.
    // A prefix built from a different prompt is a 178MB file the app validates
    // once, discards, and rebuilds over: worse than no artifact.
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

    let staging = std::env::temp_dir().join("faro-prefix-artifact");
    let _ = std::fs::remove_dir_all(&staging);
    let cfg = LlamaCppConfig {
        model_path: model_path.clone().into(),
        draft_path: Some(draft.into()),
        prefix_cache_dir: Some(staging.clone()),
        // Only the file NAME depends on this; the artifact is renamed by the
        // installing host anyway. Content depends on the fields above.
        state_key: "prefix-artifact".into(),
        ..Default::default()
    };
    let artifact_src = LlamaCppEngine::prefix_artifact_path(&cfg).expect("cache dir configured");

    let mut engine = LlamaCppEngine::new(cfg);
    let req = GenerateRequest { system, tools, ..Default::default() };
    match engine.ensure_prefix(&req) {
        Ok(r) => eprintln!(
            "prefix: {} tokens, {:.1} MB, {}ms",
            r.tokens,
            r.bytes as f64 / 1e6,
            r.ms
        ),
        Err(e) => {
            eprintln!("FAILED to build the prefix: {e}");
            std::process::exit(1);
        }
    }
    drop(engine);

    std::fs::copy(&artifact_src, &out).unwrap_or_else(|e| {
        eprintln!("cannot copy {} -> {out}: {e}", artifact_src.display());
        std::process::exit(1);
    });
    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!("{out} ({:.1} MB)", bytes as f64 / 1e6);
}
