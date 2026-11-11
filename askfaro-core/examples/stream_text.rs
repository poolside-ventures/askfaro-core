//! Does a streamed reply add up to the reply `generate` returns?
//!
//! Runs one plain request (a reply draft: no tools, thinking off) through
//! `generate_streaming`, then the same request through `generate` on a fresh
//! engine, and checks: the pieces arrive over time, they contain no control
//! token text, and joined and trimmed they equal the returned text, which in
//! turn equals the plain call's.
//!
//! `<prompt.json>` is `{system, user}`.
//!
//! ```sh
//! cargo run --release --features metal --example stream_text -- \
//!   <model.gguf> <prompt.json> [<add-on name> <add-on.gguf>]
//! ```
use askfaro_core::generation::{
    AdapterSpec, GenerateRequest, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(model), Some(prompt)) = (args.next(), args.next()) else {
        eprintln!("usage: stream_text <model.gguf> <prompt.json> [<add-on name> <add-on.gguf>]");
        std::process::exit(2);
    };
    let adapter = args.next().zip(args.next());
    let p: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&prompt).expect("read")).expect("parse");
    let cfg = || LlamaCppConfig {
        model_path: model.clone().into(),
        n_ctx: 8192,
        n_slots: 1,
        enable_thinking: false,
        adapters: adapter
            .iter()
            .map(|(name, path)| AdapterSpec { name: name.clone(), path: path.into() })
            .collect(),
        ..Default::default()
    };
    let req = || GenerateRequest {
        system: p["system"].as_str().expect("system").into(),
        messages: vec![Msg::user(p["user"].as_str().expect("user"))],
        enable_thinking: Some(false),
        adapter: adapter.as_ref().map(|(name, _)| name.clone()),
        ..Default::default()
    };

    let pieces: Arc<Mutex<Vec<(u128, String)>>> = Arc::default();
    let start = Instant::now();
    let sink_pieces = pieces.clone();
    let mut engine = LlamaCppEngine::new(cfg());
    let streamed = engine
        .generate_streaming(
            req(),
            Box::new(move |s| sink_pieces.lock().unwrap().push((start.elapsed().as_millis(), s.to_string()))),
        )
        .expect("streaming generate");
    let done_ms = start.elapsed().as_millis();
    drop(engine);
    let mut plain_engine = LlamaCppEngine::new(cfg());
    let plain = plain_engine.generate(req()).expect("generate");
    drop(plain_engine);

    let pieces = pieces.lock().unwrap();
    let joined: String = pieces.iter().map(|(_, s)| s.as_str()).collect();
    let mut failed = false;
    let mut check = |ok: bool, what: String| {
        println!("{} {what}", if ok { "PASS" } else { "FAIL" });
        failed |= !ok;
    };
    check(pieces.len() > 3, format!("{} pieces arrived", pieces.len()));
    if let (Some(first), Some(last)) = (pieces.first(), pieces.last()) {
        println!(
            "first piece at {}ms, last at {}ms, call returned at {done_ms}ms (prefill {}ms)",
            first.0, last.0, streamed.timings.prefill_ms
        );
    }
    check(!joined.contains("<|") && !joined.contains("|>"), "no control token text in the stream".into());
    check(joined.trim() == streamed.text, "the joined pieces, trimmed, are the returned text".into());
    check(streamed.text == plain.text, "streaming changes nothing about the reply".into());
    println!("---\n{}", streamed.text);
    std::process::exit(i32::from(failed));
}
