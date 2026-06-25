//! # askfaro-core::search
//!
//! On-device embedded vector search — the single retrieval engine that powers
//! all local search in consuming apps (agent tool selection, skills, notes,
//! contacts, emails, tasks). It reads the SQLite shard contract exported by the
//! Python server lib and ranks it **identically**: FTS5 lexical + exact cosine
//! semantic, fused by Reciprocal Rank Fusion (K=60).
//!
//! ```no_run
//! use askfaro_core::search::{SearchIndex, SearchParams, BowEmbedder, IndexDoc};
//! use askfaro_core::search::sqlite::SqliteBackend;
//!
//! let backend = SqliteBackend::open_in_memory(&["default"])?;
//! let index = SearchIndex::new(backend, BowEmbedder::new("default"));
//! index.upsert_many(&[IndexDoc::leaf("note", "n1", "Grocery list", "milk eggs bread")])?;
//! let hits = index.search("groceries", &SearchParams::default())?;
//! # Ok::<(), askfaro_core::search::sqlite::BackendError>(())
//! ```
//!
//! The default build is light (rusqlite + a bag-of-words embedder). The
//! EmbeddingGemma default embedder is opt-in behind the `embeddinggemma` feature.

pub mod embed;
pub mod fusion;
pub mod models;
pub mod sqlite;
pub mod types;

#[cfg(feature = "embeddinggemma")]
pub mod gemma;

pub use embed::{BowEmbedder, EmbedEngine};
pub use sqlite::{BackendError, SqliteBackend, UpsertRow};
pub use types::{Filters, IndexDoc, MatchType, RawHit, SearchResult};

/// Floor below which a cosine match is noise, not signal (carried over from
/// Faro's production tuning; matches the Python `DEFAULT_MIN_SEMANTIC_SCORE`).
pub const DEFAULT_MIN_SEMANTIC_SCORE: f32 = 0.20;

/// The retrieval **contract**: the numbers and text-preprocessing any backend
/// (device SQLite, or a server's Postgres/Elastic/whatever) must honour to rank
/// identically. The core deliberately does NOT own a server database — it owns
/// this contract plus [`fuse`]. A server retrieves candidates from its own store
/// using these constants, then fuses here; parity is asserted by the conformance
/// suite, not by sharing database code. Exposed so servers read these instead of
/// re-hardcoding (the drift that broke parity before).
pub mod contract {
    /// Canonical RRF constant (Cormack et al. 2009). Ranks, not scores.
    pub use crate::search::fusion::RRF_K;

    /// Cosine floor a semantic hit must clear to enter fusion.
    pub const DEFAULT_MIN_SEMANTIC_SCORE: f32 = super::DEFAULT_MIN_SEMANTIC_SCORE;

    /// Candidates pulled per retriever before fusion (RRF benefits from a fuller
    /// list than `k`): `max(k*3, 30)`.
    pub fn candidate_count(k: usize) -> usize {
        (k * 3).max(30)
    }

    /// EmbeddingGemma query prompt — the shared space's text preprocessing.
    pub fn query_prompt(text: &str) -> String {
        format!("task: search result | query: {text}")
    }

    /// EmbeddingGemma document prompt.
    pub fn document_prompt(text: &str) -> String {
        format!("title: none | text: {text}")
    }
}

/// Query-time knobs. `Default` mirrors the Python `search(...)` defaults.
pub struct SearchParams {
    pub k: usize,
    pub filters: Filters,
    pub min_semantic_score: f32,
    pub collapse: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            k: 10,
            filters: Filters::default(),
            min_semantic_score: DEFAULT_MIN_SEMANTIC_SCORE,
            collapse: true,
        }
    }
}

/// Wire a backend to an embedder and get incremental upsert plus hybrid search.
/// All ranking logic lives here and in [`fusion`], so any two backends return
/// identically-ranked results for the same corpus and query.
pub struct SearchIndex<E: EmbedEngine> {
    backend: SqliteBackend,
    embedder: E,
}

impl<E: EmbedEngine> SearchIndex<E> {
    pub fn new(backend: SqliteBackend, embedder: E) -> Self {
        SearchIndex { backend, embedder }
    }

    pub fn backend(&self) -> &SqliteBackend {
        &self.backend
    }

    /// Index documents incrementally — one upsert per doc, no rebuilds. A doc
    /// can opt out of the space via `embed_spaces`; embedding failure is
    /// non-fatal (the row is written lexical-only).
    pub fn upsert_many(&self, docs: &[IndexDoc]) -> Result<(), BackendError> {
        let space = self.embedder.space();
        let texts: Vec<String> = docs.iter().map(|d| d.index_text()).collect();

        // Embed only the docs that opt into this space.
        let opted: Vec<usize> = docs
            .iter()
            .enumerate()
            .filter(|(_, d)| {
                d.embed_spaces
                    .as_ref()
                    .map_or(true, |s| s.iter().any(|x| x == space))
            })
            .map(|(i, _)| i)
            .collect();
        let to_embed: Vec<&str> = opted.iter().map(|&i| texts[i].as_str()).collect();
        let embedded = self.embedder.embed_documents(&to_embed);

        let mut vectors: Vec<Option<Vec<f32>>> = vec![None; docs.len()];
        for (j, &i) in opted.iter().enumerate() {
            vectors[i] = embedded.get(j).cloned().flatten();
        }

        for (i, doc) in docs.iter().enumerate() {
            let embeddings = vec![(space.to_string(), vectors[i].clone())];
            let indexed = vectors[i].is_some();
            self.backend.upsert(&UpsertRow {
                object_type: &doc.object_type,
                object_id: &doc.object_id,
                node_kind: &doc.node_kind,
                partition: doc.partition.as_deref(),
                title: doc.title.as_deref(),
                body: doc.body.as_deref(),
                payload: doc.payload.as_deref(),
                attrs: doc.attrs.as_deref(),
                source_updated_at: &doc.source_updated_at,
                embedding_indexed_at: if indexed {
                    Some(doc.source_updated_at.as_str())
                } else {
                    None
                },
                embeddings: &embeddings,
            })?;
        }
        Ok(())
    }

    /// Hybrid search: embeds the query with this index's embedder, then fuses
    /// lexical + semantic candidates by RRF, optionally collapsing to one row per
    /// object, truncated to `k`.
    pub fn search(
        &self,
        query: &str,
        params: &SearchParams,
    ) -> Result<Vec<SearchResult>, BackendError> {
        let query_vec = self.embedder.embed_query(query);
        hybrid_search(
            &self.backend,
            query,
            query_vec.as_deref(),
            self.embedder.space(),
            params,
        )
    }
}

/// The retrieval orchestration, decoupled from how the query vector is obtained.
///
/// On-device, [`SearchIndex::search`] supplies the vector from a Rust
/// [`EmbedEngine`]. The server (PyO3 binding) supplies a vector it computed with
/// its own Python embedder — same ranking either way, because the core owns it.
/// `query_vec = None` degrades cleanly to lexical-only.
pub fn hybrid_search(
    backend: &SqliteBackend,
    query: &str,
    query_vec: Option<&[f32]>,
    space: &str,
    params: &SearchParams,
) -> Result<Vec<SearchResult>, BackendError> {
    let query = query.trim();
    if query.is_empty() {
        return Ok(Vec::new());
    }
    let candidates = contract::candidate_count(params.k);
    let lexical = backend.lexical_search(query, &params.filters, candidates)?;
    let mut semantic = Vec::new();
    if let Some(qv) = query_vec {
        semantic =
            backend.semantic_search(qv, space, &params.filters, candidates, params.min_semantic_score)?;
    }
    Ok(fuse(&lexical, &semantic, params.k, params.collapse))
}

/// Fuse caller-supplied candidate lists into the canonical ranking — the
/// DB-agnostic half of [`hybrid_search`]. `lexical` and `semantic` are each in
/// retriever-rank order (list position == rank); semantic hits carry their cosine
/// in [`RawHit::sim`]. A server retrieves candidates from its own store (honouring
/// [`contract`]) and calls this; the device path supplies them from SQLite. Either
/// way the ranking is identical because this is the only fusion that exists.
pub fn fuse(
    lexical: &[RawHit],
    semantic: &[RawHit],
    k: usize,
    collapse: bool,
) -> Vec<SearchResult> {
    let mut results = fusion::rrf_fuse(lexical, semantic);
    if collapse {
        results = fusion::collapse_objects(results);
    }
    results.truncate(k);
    results
}
