//! The progressive-context (`pcx`) manifest types, modelled to round-trip with
//! `askfaro-progressive-context`'s pcx 0.1 schema
//! (`schema/pcx-0.1.schema.json`). A catalog authored by that Python library (or
//! served by faro-api's `/pcx/*` endpoints) deserializes here unchanged, so
//! catalogs are portable between the server builder and this on-device selector.
//!
//! This crate *accesses* a catalog; it does not build one. The consumer supplies
//! the manifest (e.g. from a bundled file or a cached HTTP fetch).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The pcx schema version this crate reads/writes.
pub const PCX_VERSION: &str = "0.1";

/// A complete progressive-context manifest: a tree of [`Node`]s addressed by id,
/// built for one token `budget`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcxManifest {
    #[serde(default = "default_pcx_version")]
    pub pcx_version: String,
    /// Self-description of the navigation protocol (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<String>,
    pub source: Source,
    pub variant: Variant,
    /// Token cost of expanding the entire tree (planning hint).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_tokens: Option<u64>,
    pub root: Node,
    /// All nodes by id (includes the root's descendants; the root itself is in
    /// [`PcxManifest::root`]).
    pub nodes: HashMap<String, Node>,
}

fn default_pcx_version() -> String {
    PCX_VERSION.to_string()
}

/// What the manifest was built from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub id: String,
    /// e.g. `"tools"`, `"skills"`, `"docs"`, `"website"`, `"memory"`, `"file"`.
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

/// The budget variant this manifest was generated for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    /// Target context window (tokens) this variant targets.
    pub budget: u64,
    /// Token cost of the always-loaded baseline (root + immediate children
    /// descriptors). Estimated when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_tokens: Option<u64>,
    /// Other budgets this source was also built at.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub siblings: Vec<u64>,
}

/// One unit of content. A node is either a **branch** (has `children`) or a
/// **leaf** (has `payload`) — never both (pcx schema `oneOf`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Stable id. Often omitted inside the `nodes` map (the map key is the id);
    /// [`PcxManifest`] consumers should prefer the map key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Depth tier (root = 0). Drives progressive disclosure ordering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// One line: what is behind this node (names the artifact). Required.
    pub what: String,
    /// One line: what user goal makes this the right branch. Required.
    pub when: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
    /// Cost of showing THIS node's descriptor in a frontier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub desc_tokens: Option<u32>,
    /// Cost to expand this node's direct full payload (0 for branches).
    #[serde(default)]
    pub tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary_tokens: Option<u32>,
    /// Cost to expand everything under this node to full leaves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtree_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    /// Child node ids (branch). Mutually exclusive with `payload`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<String>>,
    /// Leaf payload pointer. Mutually exclusive with `children`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<Payload>,
    /// Domain-specific attributes preserved verbatim. For a `tools` catalog this
    /// carries the tool's `parameters` JSON Schema under `meta["parameters"]`.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub meta: Map<String, Value>,
}

impl Node {
    /// A branch node reveals children; a leaf node holds a payload.
    pub fn is_branch(&self) -> bool {
        self.children.is_some()
    }

    /// A leaf node carries an expandable payload (a tool, a doc section, etc.).
    pub fn is_leaf(&self) -> bool {
        self.payload.is_some()
    }
}

/// Pointer to a leaf's verbatim content (never inlined into the manifest).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Payload {
    /// Content reference, e.g. `node://<id>`.
    #[serde(rename = "ref")]
    pub ref_: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Available render levels, e.g. `["full"]` or `["full", "summary"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render: Option<Vec<String>>,
}
