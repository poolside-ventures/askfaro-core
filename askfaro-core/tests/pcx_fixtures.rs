//! Parity tests against REAL `askfaro-progressive-context` builder output — not
//! hand-written JSON — so the Rust pcx port and the Python builder can't drift
//! silently.
//!
//! Two fixtures, both pcx 0.2:
//! - `scope-tools.pcx.4k.json`: built by the Python CLI from
//!   `tests/fixtures/scope-tools.json`. Regenerate with:
//!   `uv run pcx build tests/fixtures/scope-tools.json --kind tools \
//!    --budgets 4k --fake --cross-links --out <dir>`
//!   (run inside the faro-progressive-context repo; `--fake` is the offline
//!   descriptor model, `--cross-links` infers the see-also links).
//! - `faro-catalog.pcx.4k.json`: the server-built production catalog
//!   (faro-api `scripts/build_pcx.py`), which carries `facets` and
//!   model-generated link why-phrases the CLI tools adapter doesn't emit.

use askfaro_core::progressive::pcx::PcxManifest;
use askfaro_core::progressive::{SelectBudget, Selector};
use serde_json::Value;

const CLI_FIXTURE: &str = include_str!("fixtures/scope-tools.pcx.4k.json");
const SERVER_FIXTURE: &str = include_str!("fixtures/faro-catalog.pcx.4k.json");

/// Assert every key/value present in `original` appears identically in `ours`,
/// recursively. `ours` may carry EXTRA keys (typed defaults like `tokens: 0`
/// become explicit on re-serialization); it may never drop or alter one.
fn assert_superset(ours: &Value, original: &Value, path: &str) {
    match (original, ours) {
        (Value::Object(orig), Value::Object(got)) => {
            for (k, v) in orig {
                let sub = got
                    .get(k)
                    .unwrap_or_else(|| panic!("re-serialized manifest dropped {path}/{k}"));
                assert_superset(sub, v, &format!("{path}/{k}"));
            }
        }
        (Value::Array(orig), Value::Array(got)) => {
            assert_eq!(got.len(), orig.len(), "array length changed at {path}");
            for (i, (o, g)) in orig.iter().zip(got).enumerate() {
                assert_superset(g, o, &format!("{path}[{i}]"));
            }
        }
        _ => assert_eq!(ours, original, "value changed at {path}"),
    }
}

fn roundtrip(raw: &str) -> PcxManifest {
    let manifest: PcxManifest = serde_json::from_str(raw).expect("builder output parses");
    assert!(manifest.version_supported());

    // Struct-level round trip: serialize -> reparse -> identical.
    let ours = serde_json::to_string(&manifest).unwrap();
    let back: PcxManifest = serde_json::from_str(&ours).unwrap();
    assert_eq!(back, manifest, "round trip must be lossless");

    // Wire-level parity: nothing the Python builder wrote is dropped or mangled.
    let original: Value = serde_json::from_str(raw).unwrap();
    let reserialized: Value = serde_json::from_str(&ours).unwrap();
    assert_superset(&reserialized, &original, "");

    manifest
}

#[test]
fn cli_built_manifest_roundtrips() {
    let manifest = roundtrip(CLI_FIXTURE);
    assert_eq!(manifest.pcx_version, "0.2");
    assert_eq!(manifest.source.kind, "tools");
    assert!(manifest.usage.is_some(), "builder ships the usage protocol");

    // The 0.2 additions came through: the cross-linked leaf pair.
    let draft = &manifest.nodes["mail-draft"];
    assert!(draft.is_leaf());
    assert_eq!(draft.links.len(), 1);
    assert_eq!(draft.links[0].to, "crm-contact-upsert");
    assert!(!draft.links[0].why.is_empty());
    // Branches cross-link too.
    assert!(manifest.nodes["crm"].links.iter().any(|l| l.to == "mail"));
}

#[test]
fn cli_built_manifest_drives_the_selector() {
    let manifest = roundtrip(CLI_FIXTURE);
    let selector = Selector::load(manifest).expect("selector loads CLI output");

    let budget = SelectBudget::default();
    let tools = selector.select("draft an email to someone", &budget);
    assert!(!tools.is_empty());
    assert_eq!(tools[0].name, "mail.draft");

    // See-also follows the builder-inferred link between leaves.
    let related = selector.related("mail-draft");
    assert_eq!(related.len(), 1);
    assert_eq!(related[0].0.name, "crm.contact_upsert");
    assert!(related[0].1.starts_with("related:"));
}

#[test]
fn server_built_catalog_roundtrips_with_facets() {
    let manifest = roundtrip(SERVER_FIXTURE);
    assert_eq!(manifest.pcx_version, "0.2");

    // Facets and links exist at production scale.
    let with_facets = manifest.nodes.values().filter(|n| !n.facets.is_empty()).count();
    let with_links = manifest.nodes.values().filter(|n| !n.links.is_empty()).count();
    assert!(with_facets > 10, "expected facetted nodes, got {with_facets}");
    assert!(with_links > 5, "expected linked nodes, got {with_links}");

    // Domain extras are flattened beside schema keys (Python `Node.meta`
    // semantics), not nested under a "meta" object.
    let skill = &manifest.nodes["skill-video"];
    assert_eq!(skill.facets.get("kind").map(String::as_str), Some("skill"));
    assert!(skill.meta.contains_key("skill_id"), "flattened extra key lands in meta");
}

#[test]
fn server_built_catalog_selects_with_facet_filter() {
    let manifest: PcxManifest = serde_json::from_str(SERVER_FIXTURE).unwrap();
    let selector = Selector::load(manifest).expect("selector loads server catalog");

    let budget = SelectBudget { top_k: 50, window: 100_000, reserve: 0 };
    let all = selector.select("company information", &budget);
    let filtered = selector.select_filtered(
        "company information",
        &[("category", "Business & Companies")],
        &budget,
    );
    assert!(!filtered.is_empty());
    assert!(
        filtered.len() < all.len(),
        "facet filter must narrow the candidate set ({} vs {})",
        filtered.len(),
        all.len()
    );

    // Production cross-links resolve leaf-to-leaf.
    let related = selector.related("skill-company-data");
    assert!(
        related.iter().any(|(schema, why)| {
            schema.name.to_lowercase().contains("filings") && !why.is_empty()
        }),
        "expected the SEC filings see-also, got {:?}",
        related.iter().map(|(s, _)| &s.name).collect::<Vec<_>>()
    );
}
