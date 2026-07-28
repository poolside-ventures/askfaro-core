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
pub mod shard;
pub mod sqlite;
pub mod types;

#[cfg(feature = "embeddinggemma")]
pub mod gemma;

pub use embed::{BowEmbedder, EmbedEngine};
pub use shard::{apply_delta, apply_full, row_count, verify_shard, ShardError};
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
///
/// The embedder is **optional**: with one, queries and upserts get the semantic
/// half; without one (`new_lexical`), the index runs keyword-only (FTS5/BM25).
/// This is the graceful degrade a consumer wants before the (large) embedding
/// model is downloaded — `hybrid_search` already collapses to lexical when the
/// query vector is `None`, so lexical-only search is the same ranking minus the
/// semantic candidates, not a different code path.
pub struct SearchIndex<E: EmbedEngine> {
    backend: SqliteBackend,
    embedder: Option<E>,
}

impl<E: EmbedEngine> SearchIndex<E> {
    pub fn new(backend: SqliteBackend, embedder: E) -> Self {
        SearchIndex {
            backend,
            embedder: Some(embedder),
        }
    }

    /// A keyword-only index — no embedder, so no model download required.
    /// Queries and upserts run lexical-only.
    pub fn new_lexical(backend: SqliteBackend) -> Self {
        SearchIndex {
            backend,
            embedder: None,
        }
    }

    pub fn backend(&self) -> &SqliteBackend {
        &self.backend
    }

    /// Whether this index has an embedder (semantic search available).
    pub fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    /// Index documents incrementally — one upsert per doc, no rebuilds. A doc
    /// can opt out of the space via `embed_spaces`; embedding failure is
    /// non-fatal (the row is written lexical-only). With no embedder, every
    /// row is written lexical-only.
    pub fn upsert_many(&self, docs: &[IndexDoc]) -> Result<(), BackendError> {
        let Some(embedder) = self.embedder.as_ref() else {
            return self.upsert_lexical(docs);
        };
        let space = embedder.space();
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
        let embedded = embedder.embed_documents(&to_embed);

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

    /// Hybrid search: embeds the query with this index's embedder (when it has
    /// one), then fuses lexical + semantic candidates by RRF, optionally
    /// collapsing to one row per object, truncated to `k`. With no embedder the
    /// query is lexical-only.
    pub fn search(
        &self,
        query: &str,
        params: &SearchParams,
    ) -> Result<Vec<SearchResult>, BackendError> {
        let query_vec = self.embedder.as_ref().and_then(|e| e.embed_query(query));
        // `space` is only consulted for the semantic half; with no query vector
        // it is unused, so an empty name is fine for the lexical-only path.
        let space = self.embedder.as_ref().map_or("", |e| e.space());
        hybrid_search(&self.backend, query, query_vec.as_deref(), space, params)
    }

    /// Filter-only listing over this index (see the free [`list`] function).
    pub fn list(&self, params: &SearchParams) -> Result<Vec<SearchResult>, BackendError> {
        list(&self.backend, params)
    }

    /// Upsert rows with no vectors (the no-embedder path).
    fn upsert_lexical(&self, docs: &[IndexDoc]) -> Result<(), BackendError> {
        for doc in docs {
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
                embedding_indexed_at: None,
                embeddings: &[],
            })?;
        }
        Ok(())
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

/// Filter-only retrieval: the rows matching `params.filters`, newest first.
///
/// [`hybrid_search`] returns nothing for an empty query, which is right for
/// search and leaves listings unserved: "my pending tasks" is entirely filters
/// and has no search terms to rank against. Consumers were left either
/// synthesising a query string, which ranks on words the user never typed, or
/// going back to the network for data already on the device.
///
/// Ordering is `source_updated_at` descending, not a relevance score, and hits
/// come back with [`MatchType::Filter`] so a caller can tell an unranked listing
/// from a BM25 result. `params.min_semantic_score` is ignored; nothing here is
/// scored.
pub fn list(
    backend: &SqliteBackend,
    params: &SearchParams,
) -> Result<Vec<SearchResult>, BackendError> {
    let rows = backend.list(&params.filters, params.k, params.collapse)?;
    Ok(rows
        .into_iter()
        .map(|h| SearchResult {
            object_type: h.object_type,
            object_id: h.object_id,
            node_kind: h.node_kind.clone(),
            partition: h.partition,
            title: h.title,
            payload: h.payload,
            score: 0.0,
            match_type: MatchType::Filter,
            lexical_rank: None,
            semantic_rank: None,
            semantic_score: None,
            matched_node_kinds: vec![h.node_kind],
        })
        .collect())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::sqlite::SqliteBackend;

    /// Two notes, one archived, both indexed with an `archived` attr.
    fn seeded_index() -> SearchIndex<BowEmbedder> {
        let backend = SqliteBackend::open_in_memory(&["default"]).unwrap();
        let index: SearchIndex<BowEmbedder> = SearchIndex::new_lexical(backend);
        let mut live = IndexDoc::leaf("note", "n1", "Groceries", "milk eggs bread");
        live.attrs = Some(r#"{"archived": false}"#.into());
        live.source_updated_at = "2026-07-01T00:00:00Z".into();
        let mut old = IndexDoc::leaf("note", "n2", "Todo", "call the dentist");
        old.attrs = Some(r#"{"archived": true}"#.into());
        old.source_updated_at = "2026-06-01T00:00:00Z".into();
        index.upsert_many(&[live, old]).unwrap();
        index
    }

    #[test]
    fn list_returns_rows_with_no_query_text() {
        let hits = seeded_index().list(&SearchParams::default()).unwrap();
        assert_eq!(hits.len(), 2);
        // Newest first, and marked as unranked rather than as a keyword hit.
        assert_eq!(hits[0].object_id, "n1");
        assert!(matches!(hits[0].match_type, MatchType::Filter));
    }

    #[test]
    fn list_applies_attrs_equality() {
        let params = SearchParams {
            filters: Filters {
                attrs: Some(vec![("archived".into(), "true".into())]),
                ..Filters::default()
            },
            ..SearchParams::default()
        };
        let hits = seeded_index().list(&params).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].object_id, "n2");
    }

    #[test]
    fn list_matches_string_attrs_bare_or_json_quoted() {
        let backend = SqliteBackend::open_in_memory(&["default"]).unwrap();
        let index: SearchIndex<BowEmbedder> = SearchIndex::new_lexical(backend);
        let mut doc = IndexDoc::leaf("task", "t1", "Ship it", "");
        doc.attrs = Some(r#"{"status": "pending"}"#.into());
        index.upsert_many(&[doc]).unwrap();

        for value in ["pending", r#""pending""#] {
            let params = SearchParams {
                filters: Filters {
                    attrs: Some(vec![("status".into(), value.into())]),
                    ..Filters::default()
                },
                ..SearchParams::default()
            };
            assert_eq!(index.list(&params).unwrap().len(), 1, "value {value:?}");
        }
    }

    #[test]
    fn list_matches_numeric_attrs() {
        let backend = SqliteBackend::open_in_memory(&["default"]).unwrap();
        let index: SearchIndex<BowEmbedder> = SearchIndex::new_lexical(backend);
        let mut doc = IndexDoc::leaf("task", "t1", "Ship it", "");
        doc.attrs = Some(r#"{"attempts": 3}"#.into());
        index.upsert_many(&[doc]).unwrap();

        let params = SearchParams {
            filters: Filters {
                attrs: Some(vec![("attempts".into(), "3".into())]),
                ..Filters::default()
            },
            ..SearchParams::default()
        };
        assert_eq!(index.list(&params).unwrap().len(), 1);
    }

    #[test]
    fn list_honours_k() {
        let params = SearchParams {
            k: 1,
            ..SearchParams::default()
        };
        assert_eq!(seeded_index().list(&params).unwrap().len(), 1);
    }

    #[test]
    fn list_skips_tombstoned_rows() {
        let index = seeded_index();
        index.backend().delete("note", "n1", None).unwrap();
        let hits = index.list(&SearchParams::default()).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].object_id, "n2");
    }

    #[test]
    fn an_attrs_key_cannot_inject_sql() {
        // The key is interpolated into a json_extract path, so it has to be
        // rejected rather than bound. Callers now pass these through from tool
        // arguments, which is to say from a model.
        let params = SearchParams {
            filters: Filters {
                attrs: Some(vec![("a') = 1 OR json_extract(si.attrs, '$.b".into(), "1".into())]),
                ..Filters::default()
            },
            ..SearchParams::default()
        };
        let err = seeded_index().list(&params).unwrap_err();
        assert!(matches!(err, BackendError::InvalidAttrKey(_)), "{err:?}");
    }

    #[test]
    fn hybrid_search_still_rejects_a_bad_attrs_key() {
        let params = SearchParams {
            filters: Filters {
                attrs: Some(vec![("bad key".into(), "1".into())]),
                ..Filters::default()
            },
            ..SearchParams::default()
        };
        let err = seeded_index().search("bread", &params).unwrap_err();
        assert!(matches!(err, BackendError::InvalidAttrKey(_)), "{err:?}");
    }

    #[test]
    fn lexical_index_searches_without_an_embedder() {
        let backend = SqliteBackend::open_in_memory(&["default"]).unwrap();
        let index: SearchIndex<BowEmbedder> = SearchIndex::new_lexical(backend);
        assert!(!index.has_embedder());

        index
            .upsert_many(&[
                IndexDoc::leaf("note", "n1", "Groceries", "milk eggs bread"),
                IndexDoc::leaf("note", "n2", "Todo", "call the dentist"),
            ])
            .unwrap();

        // Keyword (FTS/BM25) ranking works with no model present.
        let hits = index.search("bread", &SearchParams::default()).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].object_id, "n1");
        assert!(matches!(hits[0].match_type, MatchType::Keyword));
    }
}
