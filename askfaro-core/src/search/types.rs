//! Core data types — the Rust mirror of the Python lib's `types.py`. Field names
//! and the `index_text()` composition are part of the contract: a document
//! embedded on the server and on the device must hash to the same text.

/// Node kinds form the tiering vocabulary. Leaves are raw objects; summary and
/// cluster nodes are enrichment rows that live in the same flat pool and are
/// retrieved by the same top-k (no tree traversal at query time).
pub const NODE_KIND_LEAF: &str = "leaf";

/// One indexable unit: an `(object_type, object_id, node_kind)` triple.
#[derive(Debug, Clone)]
pub struct IndexDoc {
    pub object_type: String,
    pub object_id: String,
    pub node_kind: String,
    pub title: Option<String>,
    pub body: Option<String>,
    pub partition: Option<String>,
    /// Opaque display metadata stored alongside the row (JSON string).
    pub payload: Option<String>,
    /// Structured fields to filter on at query time (JSON object string).
    pub attrs: Option<String>,
    /// Which embedding spaces to populate (None = every configured space).
    pub embed_spaces: Option<Vec<String>>,
    pub source_updated_at: String,
}

impl IndexDoc {
    /// A leaf document with the common fields; matches `IndexDoc(...)` defaults.
    pub fn leaf(object_type: &str, object_id: &str, title: &str, body: &str) -> Self {
        IndexDoc {
            object_type: object_type.to_string(),
            object_id: object_id.to_string(),
            node_kind: NODE_KIND_LEAF.to_string(),
            title: Some(title.to_string()),
            body: Some(body.to_string()),
            partition: None,
            payload: None,
            attrs: None,
            embed_spaces: None,
            source_updated_at: String::new(),
        }
    }

    /// The text fed to the embedder: title + body joined by newline (non-empty
    /// parts only). Contract-critical — must match the Python `index_text()`.
    pub fn index_text(&self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if let Some(t) = self.title.as_deref() {
            if !t.is_empty() {
                parts.push(t);
            }
        }
        if let Some(b) = self.body.as_deref() {
            if !b.is_empty() {
                parts.push(b);
            }
        }
        parts.join("\n")
    }
}

/// A candidate row as returned by a backend retriever (pre-fusion).
#[derive(Debug, Clone)]
pub struct RawHit {
    pub object_type: String,
    pub object_id: String,
    pub node_kind: String,
    pub partition: Option<String>,
    pub title: Option<String>,
    pub payload: Option<String>,
    /// Populated by the semantic retriever only.
    pub sim: Option<f32>,
}

impl RawHit {
    /// The `(object_type, object_id, node_kind)` fusion key.
    pub fn key(&self) -> (String, String, String) {
        (
            self.object_type.clone(),
            self.object_id.clone(),
            self.node_kind.clone(),
        )
    }
}

/// How a result matched: keyword-only, semantic-only, or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchType {
    Keyword,
    Semantic,
    Hybrid,
}

impl MatchType {
    pub fn as_str(self) -> &'static str {
        match self {
            MatchType::Keyword => "keyword",
            MatchType::Semantic => "semantic",
            MatchType::Hybrid => "hybrid",
        }
    }
}

/// A fused, ranked result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub object_type: String,
    pub object_id: String,
    pub node_kind: String,
    pub partition: Option<String>,
    pub title: Option<String>,
    pub payload: Option<String>,
    pub score: f64,
    pub match_type: MatchType,
    pub lexical_rank: Option<usize>,
    pub semantic_rank: Option<usize>,
    pub semantic_score: Option<f32>,
    pub matched_node_kinds: Vec<String>,
}

/// Backend-agnostic query filters.
#[derive(Debug, Clone, Default)]
pub struct Filters {
    pub partition: Option<String>,
    pub object_types: Option<Vec<String>>,
    pub node_kinds: Option<Vec<String>>,
    /// Equality match on stored `IndexDoc.attrs` (key -> JSON-encoded value).
    pub attrs: Option<Vec<(String, String)>>,
}
