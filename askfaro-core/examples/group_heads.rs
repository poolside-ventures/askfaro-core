//! Does a request kind keep its head across other kinds on the same slot?
//!
//! A background slot alternates kinds of job, and the slot's single shared
//! head cannot survive that: it is learned from consecutive prompts and
//! restored over cells that a different job has already replaced. With
//! `GenerateRequest::cache_group` each kind keeps a full-state head of its
//! own. This alternates two kinds (A: the reply drafts in `<drafts.json>`, B: a
//! second, unrelated instruction block) on slot 1 and checks:
//!
//!  - every answer equals a fresh engine's answer to the same prompt, so a
//!    restored head changes nothing about what is said;
//!  - from the third A on, A reads only what follows its head.
//!
//! `<drafts.json>` is `[{system, user}, ...]`, three or more prompts of kind A.
//!
//! ```sh
//! cargo run --release --features metal --example group_heads -- \
//!   <model.gguf> <drafts.json> [<add-on name> <add-on.gguf>]
//! ```
use askfaro_core::generation::{
    AdapterSpec, GenerateRequest, GenerateResponse, GenerationEngine, LlamaCppConfig, LlamaCppEngine, Msg,
};

const B_SYSTEM: &str = "You sort one email into exactly one category. Categories: \
action_needed (the sender is waiting on the owner), fyi (information, nothing to do), \
newsletter (a mailing to many people), receipt (a payment or order confirmation), \
calendar (an invitation or schedule change), other. Answer with the category name only, \
lower case, nothing else. When an email fits two categories, prefer the one that asks \
something of the owner. Mail from a person the owner wrote to before is never a \
newsletter. Automated notifications about the owner's own accounts are fyi unless they \
ask for an action with a deadline, which makes them action_needed. Read the whole email, \
including any quoted text, before you decide.";

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(model), Some(drafts)) = (args.next(), args.next()) else {
        eprintln!("usage: group_heads <model.gguf> <drafts.json> [<add-on name> <add-on.gguf>]");
        std::process::exit(2);
    };
    let adapter = args.next().zip(args.next());
    let a: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&drafts).expect("read")).expect("parse");
    assert!(a.len() >= 3, "need three or more prompts of kind A");
    let b_users = [
        "From: shop@example.com\nYour order #4411 has shipped and arrives Thursday.",
        "From: ana@client.es\nCould you send the signed contract back by Friday?",
        "From: news@weekly.io\nThis week in AI: ten links you should read.",
    ];
    let cfg = || LlamaCppConfig {
        model_path: model.clone().into(),
        n_ctx: 8192,
        n_slots: 2,
        enable_thinking: false,
        adapters: adapter
            .iter()
            .map(|(name, path)| AdapterSpec { name: name.clone(), path: path.into() })
            .collect(),
        ..Default::default()
    };
    let ask = |e: &mut LlamaCppEngine, system: &str, user: &str, group: Option<&str>, addon: bool| -> GenerateResponse {
        e.generate(GenerateRequest {
            system: system.into(),
            messages: vec![Msg::user(user)],
            slot: 1,
            enable_thinking: Some(false),
            adapter: if addon { adapter.as_ref().map(|(n, _)| n.clone()) } else { None },
            cache_group: group.map(str::to_string),
            ..Default::default()
        })
        .expect("generate")
    };

    // The sequence a background slot sees: kinds alternating.
    let mut order: Vec<(&str, String, String, bool)> = Vec::new();
    for (i, p) in a.iter().take(3).enumerate() {
        order.push(("A", p["system"].as_str().unwrap().into(), p["user"].as_str().unwrap().into(), true));
        order.push(("B", B_SYSTEM.into(), b_users[i].into(), false));
    }

    let mut grouped = LlamaCppEngine::new(cfg());
    let mut ungrouped = LlamaCppEngine::new(cfg());
    let mut failed = false;
    let mut a_seen = 0;
    println!("kind  prompt  ungrouped-ms  grouped-ms  same-answer");
    for (kind, system, user, addon) in &order {
        let with = ask(&mut grouped, system, user, Some(kind), *addon);
        let without = ask(&mut ungrouped, system, user, None, *addon);
        // The reference: a fresh engine, nothing restored.
        let mut fresh = LlamaCppEngine::new(cfg());
        let reference = ask(&mut fresh, system, user, None, *addon);
        drop(fresh);
        let same = with.text == reference.text && without.text == reference.text;
        failed |= !same;
        println!(
            "{kind}     {:6}  {:12}  {:10}  {}",
            with.timings.prompt_tokens, without.timings.prefill_ms, with.timings.prefill_ms,
            if same { "yes" } else { "NO" },
        );
        if *kind == "A" {
            a_seen += 1;
            if a_seen >= 3 && with.timings.cached_tokens < 64 {
                println!("FAIL the third A reused no head ({} cached tokens)", with.timings.cached_tokens);
                failed = true;
            }
        }
    }
    drop(grouped);
    drop(ungrouped);
    println!("{}", if failed { "FAIL" } else { "PASS every answer equals a fresh engine's" });
    std::process::exit(i32::from(failed));
}
