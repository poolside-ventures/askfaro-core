//! Locks the on-device free-tool surface exposed by askfaro-core (the `free-tools`
//! feature). These exercise the public re-exports a host binding actually calls,
//! independently of the vendored crate's own internal tests.
#![cfg(feature = "free-tools")]

use askfaro_core::free_tools;

#[test]
fn available_lists_the_local_catalog() {
    let names = free_tools::available();
    for expected in [
        "astronomy", "calc", "datetime", "encoding", "phone", "random", "timer",
        "timezone", "units",
    ] {
        assert!(names.contains(&expected), "missing free tool: {expected}");
    }
}

#[test]
fn execute_returns_the_canonical_envelope() {
    let result = free_tools::execute("calc", serde_json::json!({"expression": "2 + 2 * 3"}));
    let v: serde_json::Value = serde_json::from_str(&result.to_json().unwrap()).unwrap();
    assert_eq!(v["status"], "success");
    assert_eq!(v["result"]["data"]["result"], 8);
    // Free tools are zero-cost — that is what makes them local.
    assert_eq!(v["meta"]["credits_charged"], 0.0);
}

#[test]
fn execute_json_is_the_binding_entry_point() {
    let out = free_tools::execute_free_tool_json(
        "units",
        r#"{"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"}"#,
    );
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["status"], "success");
    assert_eq!(v["result"]["data"]["result"], 212);
}

#[test]
fn unknown_tool_fails_closed_without_panicking() {
    let out = free_tools::execute_free_tool_json("does-not-exist", "{}");
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["status"], "failed");
    assert_eq!(v["error"]["code"], "not_found");
}
