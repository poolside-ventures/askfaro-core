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
//!   <model.gguf> <scope-repo-root> <drafter.gguf> <out.kv> <n_ctx> <n_slots>
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

    // `--print-shape <n_ctx> <n_slots>`: the shape half of the publish path,
    // without weights, without a GPU and without a prompt. The pipeline needs
    // the URL BEFORE it decides whether to spend a 5 GB download and a cold
    // prefill on rebuilding what is already published, and the shape is a pure
    // function of the config. Reimplementing that hash in the workflow would be
    // a second statement of the spec, which is the drift this whole pipeline is
    // built to avoid.
    //
    // Both arguments are REQUIRED, for the reason `n_ctx` became required below:
    // a default that happens to agree with its caller is a bug waiting for the
    // caller to change. `n_slots` was left defaulted when `n_ctx` was fixed on
    // 2026-08-20 and agreed with the desktop for exactly one day, until the
    // background workload moved to its own context and the app dropped its base
    // context to one slot. `prefix_shape_id` hashes BOTH, so that published
    // every artifact under a shape id no consumer would ever ask for.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("--print-shape") {
        let mut shape = argv.iter().skip(1).map(|v| v.parse::<u32>());
        let (Some(Ok(n_ctx)), Some(Ok(n_slots))) = (shape.next(), shape.next()) else {
            eprintln!("usage: build_prefix --print-shape <n_ctx> <n_slots>");
            std::process::exit(2);
        };
        println!(
            "{}",
            LlamaCppEngine::prefix_shape_id(&LlamaCppConfig {
                n_ctx,
                n_slots,
                ..Default::default()
            })
        );
        return;
    }

    let (model_path, repo, draft, out) = match (args.next(), args.next(), args.next(), args.next())
    {
        (Some(m), Some(r), Some(d), Some(o)) => (m, r, d, o),
        _ => {
            eprintln!(
                "usage: build_prefix <model.gguf> <scope-repo-root> <drafter.gguf> <out.kv> \
                 <n_ctx> <n_slots>"
            );
            std::process::exit(2);
        }
    };
    // REQUIRED, and deliberately not defaulted. It used to inherit
    // `LlamaCppConfig::default()`, which was 16,384 and happened to equal the
    // desktop's window — until the desktop moved to 65,536 on 2026-08-20 and
    // this kept publishing a 16k-shaped state under a URL that could not say
    // so. A default here is a silent agreement with a caller that is free to
    // change, so the caller states it and a mismatch is a missing argument
    // rather than a wrong artifact.
    let n_ctx: u32 = match args.next().map(|v| v.parse()) {
        Some(Ok(v)) => v,
        _ => {
            eprintln!(
                "error: pass n_ctx explicitly; it must equal the app's \
                 GEMMA4_E4B.contextWindow (shared/src/agent/model-profile.ts)"
            );
            std::process::exit(2);
        }
    };
    // Same rule, and the same trap one field over. `--print-shape` above was
    // taught to take this and the BUILD below was not, so the two disagreed:
    // the workflow addressed the artifact at a one-slot shape and this built a
    // two-slot state to put there. Required, for the reason `n_ctx` is.
    let n_slots: u32 = match args.next().map(|v| v.parse()) {
        Some(Ok(v)) => v,
        _ => {
            eprintln!(
                "error: pass n_slots explicitly; it must equal the app's base \
                 context slot count (workload::AGENT_SLOTS)"
            );
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
        n_ctx,
        n_slots,
        // Only the file NAME depends on this; the artifact is renamed by the
        // installing host anyway. Content depends on the fields above.
        state_key: "prefix-artifact".into(),
        ..Default::default()
    };
    let shape = LlamaCppEngine::prefix_shape_id(&cfg);
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

    // The publish subpath, printed for the pipeline so publisher and consumer
    // derive it from ONE place (these constants and this function; the
    // installing app computes the identical string from the same ones).
    //
    // It carries the WEIGHTS identity and the state's SHAPE, not just the
    // prompt fingerprint the pipeline appends. Both are mismatches the engine's
    // token check cannot catch, because in both cases the tokens are right and
    // the cached VALUES belong to something else: to a different model in the
    // weights case, and to a differently laid-out cache in the shape case.
    use askfaro_core::generation::models::{GEMMA4_E4B_IT_QAT_Q4_0, GEMMA4_E4B_MTP_DRAFTER_Q4_0};
    println!(
        "artifact-subpath: {}-{}+{}/{shape}",
        GEMMA4_E4B_IT_QAT_Q4_0.id,
        &GEMMA4_E4B_IT_QAT_Q4_0.files[0].sha256[..12],
        &GEMMA4_E4B_MTP_DRAFTER_Q4_0.files[0].sha256[..12],
    );
}
