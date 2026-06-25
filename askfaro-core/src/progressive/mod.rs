//! # askfaro-core::progressive
//!
//! On-device catalog **selector**: given a user query and a token budget, return
//! the tool subset an on-device model should be shown this turn. It combines two
//! requirements, both always applied:
//!
//! 1. **Embedded suggestions** — relevance ranking over the catalog, via
//!    the `search` module (FTS5 lexical + bag-of-words semantic today; the
//!    EmbeddingGemma vector path drops in unchanged when the on-device embedder
//!    lands, since this crate just consumes [`SearchIndex`](crate::search::SearchIndex)).
//! 2. **Progressive access** — tiered expansion of the pcx manifest under a hard
//!    token budget, conforming to `askfaro-progressive-context`'s pcx schema (see
//!    [`pcx`]). The model never sees more tools than the budget allows.
//!
//! No LLM call happens here. The returned [`ToolSchema`]s are exactly the type
//! the `generation` module consumes, so selection feeds generation directly.
//!
//! ```no_run
//! use askfaro_core::progressive::{Selector, SelectBudget};
//! # use askfaro_core::progressive::pcx::PcxManifest;
//! # fn load() -> PcxManifest { unimplemented!() }
//! let selector = Selector::load(load()).unwrap();
//! let tools = selector.select("Mark task t_8f3a as completed", &SelectBudget::default());
//! ```

pub mod pcx;

use crate::generation::ToolSchema;
use crate::search::sqlite::SqliteBackend;
use crate::search::{BowEmbedder, IndexDoc, SearchIndex, SearchParams};
use serde_json::{json, Value};

use pcx::{Node, PcxManifest};

/// Apple Foundation Models' context window is 4,096 tokens. The default budget
/// keeps the returned subset well under it so the prompt + the model's own
/// reasoning have room.
pub const APPLE_FM_SAFE_MAX_TOKENS: usize = 3_800;

/// The embedding space the in-memory index uses (single space; identity is the
/// bag-of-words embedder).
const SPACE: &str = "default";

/// Heuristic chars-per-token, matching `askfaro-progressive-context`'s
/// `estimate_tokens` so token math agrees with the server-built manifest.
const CHARS_PER_TOKEN: usize = 4;

/// How much to select. `max_tokens` is a HARD ceiling on the returned subset's
/// token cost; `top_k` caps the count.
#[derive(Debug, Clone)]
pub struct SelectBudget {
    pub top_k: usize,
    pub max_tokens: usize,
}

impl Default for SelectBudget {
    fn default() -> Self {
        SelectBudget {
            top_k: 8,
            max_tokens: APPLE_FM_SAFE_MAX_TOKENS,
        }
    }
}

/// Errors from building a selector.
#[derive(Debug, thiserror::Error)]
pub enum SelectError {
    /// The in-memory search backend failed to open or index.
    #[error("search backend error: {0}")]
    Backend(String),
    /// The manifest was structurally unusable.
    #[error("invalid manifest: {0}")]
    Manifest(String),
}

/// A loaded catalog ready to answer [`select`](Selector::select) queries. Build
/// once (indexing is the cost); query repeatedly.
pub struct Selector {
    index: SearchIndex<BowEmbedder>,
    /// node id -> the tool it resolves to, with its pcx token cost.
    tools: std::collections::HashMap<String, ToolEntry>,
}

struct ToolEntry {
    schema: ToolSchema,
    /// Full-render token cost (pcx `tokens`, or estimated from the schema).
    tokens: usize,
    /// Tier (for stable ordering among equally-relevant tools).
    tier: u32,
}

impl Selector {
    /// Index a catalog. Every leaf node becomes a searchable tool; branches are
    /// the progressive tiers walked to reach them.
    pub fn load(catalog: PcxManifest) -> Result<Self, SelectError> {
        let backend =
            SqliteBackend::open_in_memory(&[SPACE]).map_err(|e| SelectError::Backend(e.to_string()))?;
        let index = SearchIndex::new(backend, BowEmbedder::new(SPACE));

        let mut tools = std::collections::HashMap::new();
        let mut docs: Vec<IndexDoc> = Vec::new();

        for (id, node) in &catalog.nodes {
            if !node.is_leaf() {
                continue; // branches are tiers, not selectable tools
            }
            let schema = tool_schema(id, node);
            let tokens = leaf_tokens(node, &schema);
            // Index over name + description + keywords so both lexical and
            // semantic retrieval have signal.
            let body = format!(
                "{}\n{}\n{}",
                node.what,
                node.when,
                node.keywords.join(" ")
            );
            docs.push(IndexDoc::leaf("tool", id, &schema.name, &body));
            tools.insert(
                id.clone(),
                ToolEntry {
                    schema,
                    tokens,
                    tier: node.tier.unwrap_or(0),
                },
            );
        }

        if tools.is_empty() {
            return Err(SelectError::Manifest(
                "catalog has no leaf (tool) nodes".into(),
            ));
        }

        index
            .upsert_many(&docs)
            .map_err(|e| SelectError::Backend(e.to_string()))?;

        Ok(Selector { index, tools })
    }

    /// Select the tool subset for `query` under `budget`.
    ///
    /// Relevance ranking orders the candidates; then tools are admitted in rank
    /// order while the cumulative pcx token cost stays within `budget.max_tokens`
    /// (hard) and the count stays within `budget.top_k`. A too-large tool is
    /// skipped, not truncated, so the budget is never exceeded.
    pub fn select(&self, query: &str, budget: &SelectBudget) -> Vec<ToolSchema> {
        // Pull a generous candidate list, then apply the budget ourselves.
        let params = SearchParams {
            k: self.tools.len().max(budget.top_k),
            ..SearchParams::default()
        };

        let mut ranked: Vec<&ToolEntry> = match self.index.search(query, &params) {
            Ok(hits) => hits
                .iter()
                .filter_map(|h| self.tools.get(&h.object_id))
                .collect(),
            // Search failures degrade to tier-ordered selection rather than
            // returning nothing.
            Err(_) => {
                let mut all: Vec<&ToolEntry> = self.tools.values().collect();
                all.sort_by_key(|t| (t.tier, t.schema.name.clone()));
                all
            }
        };

        // Ensure determinism for tools that tied (or that search omitted): append
        // any not already ranked, tier-ordered.
        if ranked.len() < self.tools.len() {
            let mut seen: std::collections::HashSet<&str> =
                ranked.iter().map(|t| t.schema.name.as_str()).collect();
            let mut rest: Vec<&ToolEntry> = self
                .tools
                .values()
                .filter(|t| !seen.contains(t.schema.name.as_str()))
                .collect();
            rest.sort_by_key(|t| (t.tier, t.schema.name.clone()));
            for t in rest {
                seen.insert(t.schema.name.as_str());
                ranked.push(t);
            }
        }

        let mut selected = Vec::new();
        let mut spent = 0usize;
        for entry in ranked {
            if selected.len() >= budget.top_k {
                break;
            }
            if spent + entry.tokens > budget.max_tokens {
                continue; // hard budget: skip, keep scanning for smaller tools
            }
            spent += entry.tokens;
            selected.push(entry.schema.clone());
        }
        selected
    }
}

/// Build a [`ToolSchema`] from a leaf node: name from `title` (fallback id),
/// description from `what`, parameters from `meta["parameters"]` (fallback an
/// empty object schema).
fn tool_schema(id: &str, node: &Node) -> ToolSchema {
    let name = node.title.clone().unwrap_or_else(|| id.to_string());
    let parameters = node
        .meta
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
    ToolSchema {
        name,
        description: node.what.clone(),
        parameters,
    }
}

/// Full-render token cost of a leaf: the manifest's `tokens` if present, else an
/// estimate of the serialized tool schema (the bytes the model actually sees).
fn leaf_tokens(node: &Node, schema: &ToolSchema) -> usize {
    if node.tokens > 0 {
        return node.tokens as usize;
    }
    let text = serde_json::to_string(&serialized_tool(schema)).unwrap_or_default();
    estimate_tokens(&text)
}

/// The OpenAI function-tool wire shape, for cost estimation.
fn serialized_tool(schema: &ToolSchema) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        }
    })
}

/// `max(1, ceil(len / 4))` — matches `askfaro-progressive-context`'s heuristic.
fn estimate_tokens(text: &str) -> usize {
    ((text.len() + CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pcx::{Node, Payload, Source, Variant};
    use serde_json::{json, Map};
    use std::collections::HashMap;

    /// A leaf (tool) node with parameters in `meta`, mirroring a `tools` catalog.
    fn tool_node(
        title: &str,
        what: &str,
        when: &str,
        keywords: &[&str],
        params: Value,
        tokens: u32,
    ) -> Node {
        let mut meta = Map::new();
        meta.insert("parameters".into(), params);
        Node {
            id: None,
            tier: Some(2),
            title: Some(title.into()),
            what: what.into(),
            when: when.into(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
            desc_tokens: None,
            tokens,
            summary_tokens: None,
            subtree_tokens: None,
            content_hash: None,
            children: None,
            payload: Some(Payload {
                ref_: format!("node://{title}"),
                format: Some("json".into()),
                render: Some(vec!["full".into()]),
            }),
            meta,
        }
    }

    /// A Scope-shaped catalog mirroring the F-7 bench tools.
    fn scope_catalog() -> PcxManifest {
        let mut nodes: HashMap<String, Node> = HashMap::new();
        nodes.insert(
            "scope_task".into(),
            tool_node(
                "scope_task",
                "Update an existing task: mark it completed or cancelled, change its priority, reschedule, or delete it.",
                "User refers to an existing task by id and wants to change or complete it.",
                &["task", "complete", "completed", "done", "mark", "status", "priority", "reschedule", "delete"],
                json!({
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "The task id, e.g. t_8f3a"},
                        "status": {"type": "string", "enum": ["completed", "in_progress", "cancelled"]}
                    },
                    "required": ["task_id"]
                }),
                140,
            ),
        );
        nodes.insert(
            "scope_task_create".into(),
            tool_node(
                "scope_task_create",
                "Create a brand-new task with a title, optional priority and schedule.",
                "User wants to create a new task that does not exist yet.",
                &["task", "create", "new", "add", "todo", "follow up"],
                json!({
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"]
                }),
                120,
            ),
        );
        nodes.insert(
            "scope_contact".into(),
            tool_node(
                "scope_contact",
                "Create or update a CRM contact record (a person).",
                "User wants to add or edit a person in the CRM.",
                &["contact", "person", "crm", "create", "update"],
                json!({
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"]
                }),
                110,
            ),
        );
        nodes.insert(
            "scope_email".into(),
            tool_node(
                "scope_email",
                "Draft an email to a recipient about a subject.",
                "User wants to compose or draft an email.",
                &["email", "draft", "compose", "message", "send"],
                json!({
                    "type": "object",
                    "properties": {"to": {"type": "string"}},
                    "required": ["to"]
                }),
                115,
            ),
        );

        let root = Node {
            id: Some("r".into()),
            tier: Some(0),
            title: Some("Scope".into()),
            what: "Scope assistant capabilities: tasks, CRM, email.".into(),
            when: "Consult to act on tasks, contacts, or email.".into(),
            keywords: vec![],
            desc_tokens: None,
            tokens: 0,
            summary_tokens: None,
            subtree_tokens: None,
            content_hash: None,
            children: Some(vec![
                "scope_task".into(),
                "scope_task_create".into(),
                "scope_contact".into(),
                "scope_email".into(),
            ]),
            payload: None,
            meta: Map::new(),
        };

        PcxManifest {
            pcx_version: pcx::PCX_VERSION.into(),
            usage: None,
            source: Source {
                id: "scope-tools".into(),
                kind: "tools".into(),
                generated_at: None,
                content_hash: None,
            },
            variant: Variant {
                budget: 4096,
                manifest_tokens: Some(120),
                siblings: vec![],
            },
            full_tokens: None,
            root,
            nodes,
        }
    }

    #[test]
    fn task_query_selects_task_tool_within_budget() {
        let selector = Selector::load(scope_catalog()).expect("load catalog");
        let budget = SelectBudget {
            top_k: 3,
            max_tokens: 300,
        };
        let tools = selector.select("Mark task t_8f3a as completed", &budget);

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"scope_task"),
            "expected scope_task in selection, got {names:?}"
        );
        // scope_task is the most relevant, so it leads the selection.
        assert_eq!(names[0], "scope_task");
        // Budget is honoured: count under top_k (token ceiling covered by
        // `budget_is_enforced_hard`).
        assert!(!tools.is_empty() && tools.len() <= budget.top_k);
    }

    #[test]
    fn budget_is_enforced_hard() {
        let selector = Selector::load(scope_catalog()).expect("load catalog");
        // Only room for one ~140-token tool.
        let budget = SelectBudget {
            top_k: 10,
            max_tokens: 150,
        };
        let tools = selector.select("Mark task t_8f3a as completed", &budget);
        assert_eq!(tools.len(), 1, "only one tool fits in 150 tokens");
        assert_eq!(tools[0].name, "scope_task");
    }

    #[test]
    fn manifest_roundtrips_through_json() {
        let manifest = scope_catalog();
        let s = serde_json::to_string(&manifest).unwrap();
        let back: PcxManifest = serde_json::from_str(&s).unwrap();
        assert_eq!(back.pcx_version, "0.1");
        assert_eq!(back.nodes.len(), 4);
        assert!(back.nodes["scope_task"].is_leaf());
        assert!(back.root.is_branch());
    }

    #[test]
    fn empty_catalog_is_rejected() {
        let mut manifest = scope_catalog();
        manifest.nodes.clear();
        manifest.root.children = Some(vec![]);
        assert!(Selector::load(manifest).is_err());
    }
}
