//! Do the free tools ACCEPT what they are documented to accept, and what the
//! model actually sends?
//!
//! Every other check stops one step short. `llama_bench` grades which tool the
//! model NAMES and never executes one, so an intent the model invented and the
//! tool rejects scores a perfect 100%. Measured: five `scopy_compute` cases all
//! graded correct, and four of the five intents behind them were rejected.
//!
//! Two halves, and the second is the gate:
//!
//! 1. **What the model sent**, verbatim from a real `llama_bench` run
//!    (2026-08-01). Recorded, not asserted: it is evidence about what an
//!    untutored model guesses, and the fix for a rejection here is the
//!    DESCRIPTION, not this file.
//! 2. **What `free_tools::describe()` documents**, which MUST execute. That
//!    text is what reaches the model, so a shape documented and not accepted is
//!    worse than no documentation at all: it is a confident instruction to do
//!    the wrong thing. This half fails the build.
//!
//! ```sh
//! cargo run --features free-tools --example free_tool_intents
//! ```
use askfaro_core::free_tools;
use serde_json::json;

/// Runs an intent and reports whether the envelope came back an error.
fn run(cap: &str, intent: &serde_json::Value) -> (bool, String) {
    let out = free_tools::execute(cap, intent.clone());
    let json = serde_json::to_value(&out).unwrap_or(serde_json::Value::Null);
    let errored = json.get("error").map(|e| !e.is_null()).unwrap_or(false)
        || json.get("status").and_then(|s| s.as_str()) == Some("failed");
    let note = if errored {
        json.get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .unwrap_or("rejected")
            .to_string()
    } else {
        json.get("summary")
            .and_then(|s| s.as_str())
            .unwrap_or("ok")
            .to_string()
    };
    (errored, note)
}

fn main() {
    // --- 1. what an untutored model guessed -----------------------------
    let guessed: Vec<(&str, serde_json::Value, &str)> = vec![
        ("calc", json!({"expression": "17% of 4200"}), "What's 17% of 4,200?"),
        ("calc", json!({"expression": "1847 * 236"}), "1847 multiplied by 236"),
        ("units", json!({"from": "4.7 mi", "to": "km"}), "4.7 miles in km"),
        (
            "datetime",
            json!({"from": "2026-03-03", "to": "2026-11-19", "operation": "difference"}),
            "days between two dates",
        ),
        (
            "timezone",
            json!({"from": "15:00", "from_zone": "Europe/Madrid", "to_zone": "Asia/Tokyo"}),
            "15:00 Madrid in Tokyo",
        ),
    ];
    println!("what the model guessed, before being told the shape:");
    let mut guessed_ok = 0;
    for (cap, intent, asked) in &guessed {
        let (errored, note) = run(cap, intent);
        if !errored {
            guessed_ok += 1;
        }
        println!("  {:9} {:5}  {asked}\n            {note}", cap, if errored { "REJ" } else { "ok" });
    }
    println!("  -> {guessed_ok}/{} accepted\n", guessed.len());

    // --- 1b. what the model sends once it HAS the contract ---------------
    //
    // Verbatim from a `llama_bench` run after `describe()` reached the tool
    // description (2026-08-01). These are asserted, not merely recorded: they
    // are the end-to-end claim, that a model reading only the shipped
    // description produces something the tool accepts. Every other check in
    // this repo tests one side of that.
    let informed: Vec<(&str, serde_json::Value, &str)> = vec![
        ("calc", json!({"operation": "evaluate", "expression": "0.17 * 4200"}), "17% of 4,200"),
        ("calc", json!({"operation": "evaluate", "expression": "1847 * 236"}), "1847 x 236"),
        (
            "units",
            json!({"value": 4.7, "from_unit": "mile", "to_unit": "kilometer"}),
            "4.7 miles in km",
        ),
        (
            "datetime",
            json!({"operation": "diff", "start": "2026-03-03", "end": "2026-11-19"}),
            "days between two dates",
        ),
        (
            // No `operation`: inferred from the zone pair, exactly as `units`
            // and `calc` have always inferred theirs.
            "timezone",
            json!({
                "datetime": "2026-07-30T15:00:00",
                "from_timezone": "Europe/Madrid",
                "to_timezone": "Asia/Tokyo"
            }),
            "15:00 Madrid in Tokyo",
        ),
    ];
    println!("what the model sends WITH the contract in front of it:");
    let mut informed_failures = Vec::new();
    for (cap, intent, asked) in &informed {
        let (errored, note) = run(cap, intent);
        println!("  {:9} {:5}  {asked}\n            {note}", cap, if errored { "REJ" } else { "ok" });
        if errored {
            informed_failures.push(format!(
                "{cap} rejected what the model produced FROM the published contract \
                 for \"{asked}\": {intent} -> {note}"
            ));
        }
    }
    println!("  -> {}/{} accepted\n", informed.len() - informed_failures.len(), informed.len());

    // --- 2. what the description tells it to send -----------------------
    //
    // One per operation named in `describe()`. If a shape is documented, it has
    // to work; the model has nothing else to go on.
    let documented: Vec<(&str, serde_json::Value)> = vec![
        ("calc", json!({"operation": "evaluate", "expression": "0.17 * 4200"})),
        ("calc", json!({"operation": "evaluate", "expression": "1847 * 236"})),
        ("calc", json!({"operation": "base_convert", "value": "255", "from_base": 10, "to_base": 16})),
        ("calc", json!({"operation": "roman", "value": 2026})),
        ("units", json!({"operation": "convert", "value": 4.7, "from_unit": "mile", "to_unit": "kilometer"})),
        ("units", json!({"operation": "list_units", "dimension": "length"})),
        ("datetime", json!({"operation": "diff", "start": "2026-03-03", "end": "2026-11-19"})),
        ("datetime", json!({"operation": "add", "base": "2026-03-03", "days": 30})),
        ("datetime", json!({"operation": "weekday", "date": "2026-03-03"})),
        ("datetime", json!({"operation": "business_days", "start": "2026-03-03", "end": "2026-03-31"})),
        ("timezone", json!({"operation": "current_time", "timezone": "Europe/Madrid"})),
        ("timezone", json!({
            "operation": "convert",
            "datetime": "2026-03-03T15:00:00",
            "from_timezone": "Europe/Madrid",
            "to_timezone": "Asia/Tokyo"
        })),
        ("timezone", json!({"operation": "zone_info", "timezone": "Asia/Tokyo"})),
        ("astronomy", json!({"operation": "sun", "latitude": 40.4, "longitude": -3.7, "date": "2026-03-03"})),
        ("astronomy", json!({"operation": "moon", "date": "2026-03-03"})),
        ("encoding", json!({"operation": "hash", "text": "hello", "algorithm": "sha256"})),
        ("encoding", json!({"operation": "uuid", "count": 1})),
        ("encoding", json!({"operation": "url", "text": "a b", "action": "encode"})),
        ("phone", json!({"operation": "parse", "number": "+34 600 000 000"})),
        ("random", json!({"operation": "integer", "min": 1, "max": 6})),
        ("random", json!({"operation": "string", "length": 12})),
    ];

    println!("what the description tells it to send:");
    let mut failures = informed_failures;
    for (cap, intent) in &documented {
        let (errored, note) = run(cap, intent);
        println!("  {:9} {:5}  {}\n            {note}", cap, if errored { "REJ" } else { "ok" }, intent);
        if errored {
            failures.push(format!("{cap} rejected a DOCUMENTED shape {intent}: {note}"));
        }
    }

    // Every namespace must be described, or a capability the model can select
    // arrives with no shape at all.
    let described: Vec<&str> = free_tools::describe().into_iter().map(|(n, _)| n).collect();
    for name in free_tools::available() {
        if !described.contains(&name) {
            failures.push(format!("{name} is available but has no describe() entry"));
        }
    }

    if failures.is_empty() {
        println!("\npassed: every documented shape executes, and every capability is described");
    } else {
        eprintln!("\nFAILED: {} documented shape(s) do not work", failures.len());
        for f in &failures {
            eprintln!("  - {f}");
        }
        eprintln!(
            "\nA shape that is documented and rejected is worse than none: the model has \
             nothing else to go on, so it follows the description into a guaranteed error."
        );
        std::process::exit(1);
    }
}
