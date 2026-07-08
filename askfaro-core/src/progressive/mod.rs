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

use pcx::{Link, Node, PcxManifest};
use std::collections::BTreeMap;

/// Apple Foundation Models' context window is 4,096 tokens — and it covers
/// EVERYTHING: instructions, transcript, tool schemas, and the model's own
/// output.
pub const APPLE_FM_CONTEXT_TOKENS: usize = 4_096;

/// Default host reserve: the window share kept free for the system prompt,
/// transcript, and the model's generated output. What remains
/// (`window - reserve`, ~2.3k by default) is the tool subset's hard ceiling —
/// comfortably above the ~2k a `top_k = 8` selection of real catalog leaves
/// (~110-250 tokens each) costs, so the reserve, not luck, is what keeps
/// prompts inside the window.
pub const DEFAULT_HOST_RESERVE_TOKENS: usize = 1_800;

/// The embedding space the in-memory index uses (single space; identity is the
/// bag-of-words embedder).
const SPACE: &str = "default";

/// Heuristic chars-per-token, matching `askfaro-progressive-context`'s
/// `estimate_tokens` so token math agrees with the server-built manifest.
const CHARS_PER_TOKEN: usize = 4;

/// How much to select, stated as the real constraint: a total token `window`
/// minus a host `reserve`, mirroring `askfaro-progressive-context`'s
/// `Runtime(budget, reserve)`. The tool subset's HARD ceiling is
/// [`tool_tokens`](SelectBudget::tool_tokens) = `window - reserve`; `top_k`
/// caps the count.
///
/// The split keeps the semantics honest: the window is the model's whole
/// context (prompt + transcript + tools + output), so a budget that only
/// capped the tool subset at "under the window" would leave no room for
/// anything else.
#[derive(Debug, Clone)]
pub struct SelectBudget {
    pub top_k: usize,
    /// Total token window the selection targets (e.g. the model's context
    /// window).
    pub window: usize,
    /// Host headroom reserved out of `window`: system prompt, transcript, and
    /// the model's generated output.
    pub reserve: usize,
}

impl SelectBudget {
    /// A budget for `window` total tokens with `reserve` kept for the host.
    ///
    /// # Panics
    /// Panics when `reserve >= window` — a zero-token tool budget is a
    /// misconfiguration, and it should fail loudly at construction (the Python
    /// runtime raises on `reserve >= budget` the same way), not surface as a
    /// silently empty selection. Constructing the struct literally skips this
    /// check.
    pub fn for_window(window: usize, reserve: usize) -> Self {
        assert!(
            reserve < window,
            "SelectBudget reserve ({reserve}) must be smaller than the window ({window}): \
             nothing would be left for tools"
        );
        SelectBudget {
            top_k: 8,
            window,
            reserve,
        }
    }

    /// The HARD ceiling on the returned subset's token cost.
    pub fn tool_tokens(&self) -> usize {
        self.window.saturating_sub(self.reserve)
    }
}

impl Default for SelectBudget {
    fn default() -> Self {
        SelectBudget {
            top_k: 8,
            window: APPLE_FM_CONTEXT_TOKENS,
            reserve: DEFAULT_HOST_RESERVE_TOKENS,
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
    /// Orthogonal facets (pcx 0.2), for filter-before-rank narrowing.
    facets: BTreeMap<String, String>,
    /// Lateral see-also links (pcx 0.2) out of this node.
    links: Vec<Link>,
}

impl ToolEntry {
    /// True when every `(key, value)` pair is present verbatim in this tool's
    /// facets. An empty filter matches everything (including 0.1 catalogs,
    /// where no node has facets).
    fn matches_facets(&self, facets: &[(&str, &str)]) -> bool {
        facets
            .iter()
            .all(|(k, v)| self.facets.get(*k).is_some_and(|have| have == v))
    }
}

impl Selector {
    /// Index a catalog. Every leaf node becomes a searchable tool; branches are
    /// the progressive tiers walked to reach them.
    pub fn load(catalog: PcxManifest) -> Result<Self, SelectError> {
        if !catalog.version_supported() {
            return Err(SelectError::Manifest(format!(
                "unsupported pcx_version {:?} (this crate reads {:?})",
                catalog.pcx_version,
                pcx::SUPPORTED_PCX_VERSIONS
            )));
        }
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
            // Index over name + description + keywords + facet values so both
            // lexical and semantic retrieval have signal (a query naming a
            // facet value, e.g. a category, still ranks the right tools).
            let facet_text = node
                .facets
                .values()
                .cloned()
                .collect::<Vec<_>>()
                .join(" ");
            let body = format!(
                "{}\n{}\n{}\n{}",
                node.what,
                node.when,
                node.keywords.join(" "),
                facet_text
            );
            docs.push(IndexDoc::leaf("tool", id, &schema.name, &body));
            tools.insert(
                id.clone(),
                ToolEntry {
                    schema,
                    tokens,
                    tier: node.tier.unwrap_or(0),
                    facets: node.facets.clone(),
                    links: node.links.clone(),
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
    /// order while the cumulative pcx token cost stays within
    /// [`budget.tool_tokens()`](SelectBudget::tool_tokens) (hard) and the count
    /// stays within `budget.top_k`. A too-large tool is skipped, not truncated,
    /// so the budget is never exceeded.
    pub fn select(&self, query: &str, budget: &SelectBudget) -> Vec<ToolSchema> {
        self.select_filtered(query, &[], budget)
    }

    /// [`select`](Selector::select) with a facet pre-filter (pcx 0.2): only
    /// tools whose `facets` contain every given `(key, value)` pair are
    /// candidates. Filter-before-rank is the manifest's own guidance — it cuts
    /// the space cheaply so ranking runs over a handful of candidates, not the
    /// whole catalog. An empty filter behaves exactly like `select` (and 0.1
    /// catalogs, which have no facets, are only ever matched by empty filters).
    pub fn select_filtered(
        &self,
        query: &str,
        facets: &[(&str, &str)],
        budget: &SelectBudget,
    ) -> Vec<ToolSchema> {
        // Pull a generous candidate list, then apply the budget ourselves.
        let params = SearchParams {
            k: self.tools.len().max(budget.top_k),
            ..SearchParams::default()
        };

        let mut ranked: Vec<&ToolEntry> = match self.index.search(query, &params) {
            Ok(hits) => hits
                .iter()
                .filter_map(|h| self.tools.get(&h.object_id))
                .filter(|t| t.matches_facets(facets))
                .collect(),
            // Search failures degrade to tier-ordered selection rather than
            // returning nothing.
            Err(_) => {
                let mut all: Vec<&ToolEntry> = self
                    .tools
                    .values()
                    .filter(|t| t.matches_facets(facets))
                    .collect();
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
                .filter(|t| !seen.contains(t.schema.name.as_str()) && t.matches_facets(facets))
                .collect();
            rest.sort_by_key(|t| (t.tier, t.schema.name.clone()));
            for t in rest {
                seen.insert(t.schema.name.as_str());
                ranked.push(t);
            }
        }

        let max_tokens = budget.tool_tokens();
        let mut selected = Vec::new();
        let mut spent = 0usize;
        for entry in ranked {
            if selected.len() >= budget.top_k {
                break;
            }
            if spent + entry.tokens > max_tokens {
                continue; // hard budget: skip, keep scanning for smaller tools
            }
            spent += entry.tokens;
            selected.push(entry.schema.clone());
        }
        selected
    }

    /// See-also tools for `node_id` (pcx 0.2 `links`): the lateral move when a
    /// selected tool is close-but-not-exact. Returns each linked tool with its
    /// why-phrase. Links to nodes that are not leaf tools in this catalog
    /// (branches, dangling ids) are skipped; 0.1 catalogs always return empty.
    pub fn related(&self, node_id: &str) -> Vec<(ToolSchema, String)> {
        let Some(entry) = self.tools.get(node_id) else {
            return Vec::new();
        };
        entry
            .links
            .iter()
            .filter_map(|link| {
                self.tools
                    .get(&link.to)
                    .map(|target| (target.schema.clone(), link.why.clone()))
            })
            .collect()
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
        category: &str,
    ) -> Node {
        let mut meta = Map::new();
        meta.insert("parameters".into(), params);
        let mut facets = BTreeMap::new();
        facets.insert("kind".to_string(), "tool".to_string());
        facets.insert("category".to_string(), category.to_string());
        Node {
            id: None,
            tier: Some(2),
            title: Some(title.into()),
            what: what.into(),
            when: when.into(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
            links: vec![],
            facets,
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
                "tasks",
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
                "tasks",
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
                "crm",
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
                "communication",
            ),
        );
        // A lateral see-also: drafting an email to someone often follows from
        // (or needs) their CRM contact record.
        nodes.get_mut("scope_email").unwrap().links = vec![Link {
            to: "scope_contact".into(),
            why: "related: the recipient is a CRM contact".into(),
        }];

        let root = Node {
            id: Some("r".into()),
            tier: Some(0),
            title: Some("Scope".into()),
            what: "Scope assistant capabilities: tasks, CRM, email.".into(),
            when: "Consult to act on tasks, contacts, or email.".into(),
            keywords: vec![],
            links: vec![],
            facets: BTreeMap::new(),
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
            window: 300,
            reserve: 0,
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
            window: 150,
            reserve: 0,
        };
        let tools = selector.select("Mark task t_8f3a as completed", &budget);
        assert_eq!(tools.len(), 1, "only one tool fits in 150 tokens");
        assert_eq!(tools[0].name, "scope_task");
    }

    #[test]
    fn reserve_shrinks_the_tool_ceiling() {
        let selector = Selector::load(scope_catalog()).expect("load catalog");
        // Same 150-token effective ceiling as `budget_is_enforced_hard`, but
        // expressed as window minus reserve.
        let budget = SelectBudget::for_window(4096, 3946);
        assert_eq!(budget.tool_tokens(), 150);
        let tools = selector.select("Mark task t_8f3a as completed", &budget);
        assert_eq!(tools.len(), 1, "only one tool fits in the unreserved 150 tokens");
        assert_eq!(tools[0].name, "scope_task");

        // The default budget leaves real room for prompt + output: the tool
        // ceiling is far below the window, yet fits a full top_k selection.
        let default = SelectBudget::default();
        assert_eq!(default.window, APPLE_FM_CONTEXT_TOKENS);
        assert!(default.tool_tokens() <= default.window - 1_500);
    }

    #[test]
    #[should_panic(expected = "must be smaller than the window")]
    fn reserve_swallowing_the_window_fails_loudly() {
        let _ = SelectBudget::for_window(4096, 4096);
    }

    #[test]
    fn manifest_roundtrips_through_json() {
        let manifest = scope_catalog();
        let s = serde_json::to_string(&manifest).unwrap();
        let back: PcxManifest = serde_json::from_str(&s).unwrap();
        assert_eq!(back.pcx_version, "0.2");
        assert_eq!(back.nodes.len(), 4);
        assert!(back.nodes["scope_task"].is_leaf());
        assert!(back.root.is_branch());
        // 0.2 additions survive the round trip.
        assert_eq!(back.nodes["scope_email"].links, manifest.nodes["scope_email"].links);
        assert_eq!(back.nodes["scope_task"].facets, manifest.nodes["scope_task"].facets);
        // meta is flattened (Python parity): `parameters` sits beside `what`,
        // there is no nested "meta" key on the wire.
        let raw: Value = serde_json::from_str(&s).unwrap();
        let task = &raw["nodes"]["scope_task"];
        assert!(task.get("parameters").is_some());
        assert!(task.get("meta").is_none());
        assert_eq!(back, manifest);
    }

    #[test]
    fn v01_manifest_without_links_or_facets_loads() {
        // A 0.1 manifest is 0.2 minus links/facets: strip them and re-declare.
        let mut manifest = scope_catalog();
        manifest.pcx_version = "0.1".into();
        for node in manifest.nodes.values_mut() {
            node.links.clear();
            node.facets.clear();
        }
        let selector = Selector::load(manifest).expect("0.1 catalog loads");
        let tools = selector.select("Mark task t_8f3a as completed", &SelectBudget::default());
        assert!(!tools.is_empty());
        // No facets anywhere: a facet filter matches nothing.
        assert!(selector
            .select_filtered("task", &[("kind", "tool")], &SelectBudget::default())
            .is_empty());
    }

    #[test]
    fn unsupported_version_is_rejected() {
        let mut manifest = scope_catalog();
        manifest.pcx_version = "9.9".into();
        let err = match Selector::load(manifest) {
            Ok(_) => panic!("future version must not load"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("9.9"), "got: {err}");
    }

    #[test]
    fn facet_filter_narrows_candidates() {
        let selector = Selector::load(scope_catalog()).expect("load catalog");
        let budget = SelectBudget::default();
        // "create" is ambiguous between tasks and CRM; the facet disambiguates.
        let names: Vec<String> = selector
            .select_filtered("create a new record", &[("category", "crm")], &budget)
            .iter()
            .map(|t| t.name.clone())
            .collect();
        assert_eq!(names, vec!["scope_contact"]);
        // A facet value no node carries selects nothing.
        assert!(selector
            .select_filtered("create", &[("category", "nope")], &budget)
            .is_empty());
    }

    #[test]
    fn related_follows_see_also_links() {
        let selector = Selector::load(scope_catalog()).expect("load catalog");
        let related = selector.related("scope_email");
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].0.name, "scope_contact");
        assert_eq!(related[0].1, "related: the recipient is a CRM contact");
        // Unlinked and unknown ids return empty, never error.
        assert!(selector.related("scope_task").is_empty());
        assert!(selector.related("no_such_node").is_empty());
    }

    #[test]
    fn empty_catalog_is_rejected() {
        let mut manifest = scope_catalog();
        manifest.nodes.clear();
        manifest.root.children = Some(vec![]);
        assert!(Selector::load(manifest).is_err());
    }
}
