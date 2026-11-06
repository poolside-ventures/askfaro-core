//! Build the persisted KV prefix as a distributable ARTIFACT.
//!
//! The release pipeline runs this once per (app prompt, weights) pair so a
//! fresh install can DOWNLOAD the prefix instead of spending ~40s of GPU
//! computing it during onboarding. It takes the exact on-device prompt from a
//! file the caller writes, builds the prefix through the same `ensure_prefix`
//! the app calls, and copies the resulting state file to the requested output
//! path.
//!
//! ```sh
//! cargo run --release --features llama-cpp --example build_prefix -- \
//!   <model.gguf> <prompt.json> <drafter.gguf> <out.kv> <n_ctx> <n_slots> <thinking on|off>
//! ```
//!
//! The engine config below MUST mirror the desktop's `brain_config` for every
//! field that shapes the state (n_ctx, n_slots, swa_full, drafter): a state
//! built under a different shape deserializes into plausible nonsense rather
//! than an error, which is why the app also validates the token list against
//! its first real prompt before trusting anything it downloaded.

use askfaro_core::generation::{GenerateRequest, LlamaCppConfig, LlamaCppEngine, ToolSchema};
use serde_json::Value;

/// `on` / `off`, the agent's thinking state. Required for the reason `n_ctx` is:
/// it renders into the prompt and is hashed into the shape, and the two brains
/// differ (E4B's agent thinks, E2B's does not).
fn thinking_arg(v: Option<&str>) -> bool {
    match v {
        Some("on") => true,
        Some("off") => false,
        _ => {
            eprintln!(
                "error: pass the agent's thinking state, on|off; it must equal the brain's \
                 `agent_thinks` (desktop lib.rs) / profile `agentThinking`"
            );
            std::process::exit(2);
        }
    }
}

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
            eprintln!("usage: build_prefix --print-shape <n_ctx> <n_slots> <thinking on|off>");
            std::process::exit(2);
        };
        let enable_thinking = thinking_arg(argv.get(3).map(String::as_str));
        println!(
            "{}",
            LlamaCppEngine::prefix_shape_id(&LlamaCppConfig {
                n_ctx,
                n_slots,
                enable_thinking,
                ..Default::default()
            })
        );
        return;
    }

    let (model_path, prompt_path, draft, out) =
        match (args.next(), args.next(), args.next(), args.next()) {
            (Some(m), Some(p), Some(d), Some(o)) => (m, p, d, o),
            _ => {
                eprintln!(
                    "usage: build_prefix <model.gguf> <prompt.json> <drafter.gguf> <out.kv> \
                     <n_ctx> <n_slots> <thinking on|off>"
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
                "error: pass n_ctx explicitly; it must equal the brain profile's \
                 contextWindow (shared/src/agent/model-profile.ts)"
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
    let enable_thinking = thinking_arg(args.next().as_deref());

    // The prompt arrives whole, `{system, tools: [{name, description,
    // parameters}]}`, written by the caller from the code its app runs (Scope's
    // `scripts/prefix_fingerprint.mjs --inputs`). This used to assemble it from
    // the caller's registry and index files, and that copy drifted twice: it
    // always read the full set's index, and it never had the tool the app's
    // harness adds on top of the registry. Each drift made every published
    // prefix one the app discards whole on first use. Taking the prompt instead
    // of rebuilding it leaves nothing here to drift.
    let prompt = read_json(&prompt_path);
    let system = prompt["system"].as_str().unwrap_or_else(|| {
        eprintln!("{prompt_path} has no string `system`");
        std::process::exit(2);
    });
    let system = system.to_string();
    let tools: Vec<ToolSchema> = prompt["tools"]
        .as_array()
        .unwrap_or_else(|| {
            eprintln!("{prompt_path} has no array `tools`");
            std::process::exit(2);
        })
        .iter()
        .map(|t| ToolSchema {
            name: t["name"].as_str().unwrap_or_default().to_string(),
            description: t["description"].as_str().unwrap_or_default().to_string(),
            parameters: t["parameters"].clone(),
        })
        .collect();

    let staging = std::env::temp_dir().join("faro-prefix-artifact");
    let _ = std::fs::remove_dir_all(&staging);
    let cfg = LlamaCppConfig {
        model_path: model_path.clone().into(),
        draft_path: Some(draft.clone().into()),
        prefix_cache_dir: Some(staging.clone()),
        n_ctx,
        n_slots,
        enable_thinking,
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
    //
    // The brain is recognised by the files it was given, so an E2B build can
    // never print E4B's subpath. Anything else is refused: a subpath this
    // cannot name is one no install would ever ask for.
    use askfaro_core::generation::models::{
        GEMMA4_E2B_IT_QAT_Q4_0, GEMMA4_E2B_MTP_DRAFTER_Q4_0, GEMMA4_E4B_IT_QAT_Q4_0,
        GEMMA4_E4B_MTP_DRAFTER_Q4_0,
    };
    let file_name = |p: &str| std::path::Path::new(p).file_name().map(|n| n.to_string_lossy().into_owned());
    let (brain, drafter) = [
        (&GEMMA4_E4B_IT_QAT_Q4_0, &GEMMA4_E4B_MTP_DRAFTER_Q4_0),
        (&GEMMA4_E2B_IT_QAT_Q4_0, &GEMMA4_E2B_MTP_DRAFTER_Q4_0),
    ]
    .into_iter()
    .find(|(b, d)| {
        file_name(&model_path).as_deref() == Some(b.files[0].name)
            && file_name(&draft).as_deref() == Some(d.files[0].name)
    })
    .unwrap_or_else(|| {
        eprintln!("no brain spec names {model_path} with drafter {draft}; refusing to print a subpath");
        std::process::exit(1);
    });
    println!(
        "artifact-subpath: {}-{}+{}/{shape}",
        brain.id,
        &brain.files[0].sha256[..12],
        &drafter.files[0].sha256[..12],
    );
}
