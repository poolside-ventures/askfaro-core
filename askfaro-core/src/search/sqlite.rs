//! SQLite backend — the on-device store and the shard interchange format.
//!
//! **The schema of this file IS the shard format.** It is copied verbatim from
//! the Python lib's `backends/sqlite.py` so a shard exported by the server
//! (Postgres → SQLite export) is read and ranked identically here: FTS5
//! (`porter unicode61`, bm25) lexical + an exact cosine scan over packed float32
//! blobs, fused by RRF K=60.
//!
//! Embedding **spaces**: a row can carry a vector from more than one model — each
//! space is its own `embedding_<space>` BLOB column plus an `_dim` column.
//! Vectors from different models are never compared.

use rusqlite::{params_from_iter, Connection};

use crate::search::types::{Filters, RawHit};

/// The default single-space name (matches the Python `DEFAULT_SPACE`).
pub const DEFAULT_SPACE: &str = "default";

/// Base schema — no vector columns (those are added per space). Byte-for-byte
/// the Python contract: the `porter` stemmer and the external-content FTS
/// triggers must match so device and server rank the same corpus identically.
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS search_index (
    id INTEGER PRIMARY KEY,
    partition_key TEXT,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    node_kind TEXT NOT NULL DEFAULT 'leaf',
    title TEXT,
    body TEXT,
    payload TEXT,
    attrs TEXT,
    source_updated_at TEXT NOT NULL,
    embedding_indexed_at TEXT,
    deleted_at TEXT,
    updated_seq INTEGER NOT NULL,
    UNIQUE (object_type, object_id, node_kind)
);
CREATE INDEX IF NOT EXISTS ix_si_partition
    ON search_index (partition_key, object_type) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS ix_si_seq ON search_index (updated_seq);
CREATE TABLE IF NOT EXISTS search_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS search_fts
    USING fts5(title, body, content='search_index', content_rowid='id',
               tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS search_index_ai AFTER INSERT ON search_index BEGIN
    INSERT INTO search_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END;
CREATE TRIGGER IF NOT EXISTS search_index_ad AFTER DELETE ON search_index BEGIN
    INSERT INTO search_fts(search_fts, rowid, title, body)
        VALUES ('delete', old.id, old.title, old.body);
END;
CREATE TRIGGER IF NOT EXISTS search_index_au AFTER UPDATE ON search_index BEGIN
    INSERT INTO search_fts(search_fts, rowid, title, body)
        VALUES ('delete', old.id, old.title, old.body);
    INSERT INTO search_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END;
"#;

/// Errors from the SQLite backend.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("invalid embedding space name {0:?}: must match [a-z][a-z0-9_]*")]
    InvalidSpace(String),
}

fn validate_space(name: &str) -> Result<(), BackendError> {
    let mut chars = name.chars();
    let ok = matches!(chars.next(), Some(c) if c.is_ascii_lowercase())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if ok {
        Ok(())
    } else {
        Err(BackendError::InvalidSpace(name.to_string()))
    }
}

fn col(space: &str) -> String {
    format!("embedding_{space}")
}

fn pack(vec: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vec.len() * 4);
    for v in vec {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn unpack(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Quote each token so user input can't break FTS5 syntax (AND semantics,
/// matching `plainto_tsquery` on the Postgres side). Mirrors `_fts_query`.
fn fts_query(query: &str) -> Option<String> {
    let tokens = crate::search::embed::tokenize_lower(query);
    if tokens.is_empty() {
        None
    } else {
        Some(
            tokens
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(" "),
        )
    }
}

/// Cosine similarity in f32, mirroring the Python `_cosine` (numpy float32).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

/// One document to write, with its per-space vectors.
pub struct UpsertRow<'a> {
    pub object_type: &'a str,
    pub object_id: &'a str,
    pub node_kind: &'a str,
    pub partition: Option<&'a str>,
    pub title: Option<&'a str>,
    pub body: Option<&'a str>,
    pub payload: Option<&'a str>,
    pub attrs: Option<&'a str>,
    pub source_updated_at: &'a str,
    pub embedding_indexed_at: Option<&'a str>,
    /// (space, vector) for each configured space; `None` vector = not embedded.
    pub embeddings: &'a [(String, Option<Vec<f32>>)],
}

pub struct SqliteBackend {
    conn: Connection,
    spaces: Vec<String>,
}

impl SqliteBackend {
    /// Open (or create) a shard at `path` with the given embedding spaces.
    pub fn open(path: &str, spaces: &[&str]) -> Result<Self, BackendError> {
        for s in spaces {
            validate_space(s)?;
        }
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        Self::init(conn, spaces)
    }

    /// Open an in-memory shard (tests).
    pub fn open_in_memory(spaces: &[&str]) -> Result<Self, BackendError> {
        for s in spaces {
            validate_space(s)?;
        }
        let conn = Connection::open_in_memory()?;
        Self::init(conn, spaces)
    }

    fn init(conn: Connection, spaces: &[&str]) -> Result<Self, BackendError> {
        conn.execute_batch(SCHEMA)?;
        let backend = SqliteBackend {
            conn,
            spaces: spaces.iter().map(|s| s.to_string()).collect(),
        };
        backend.ensure_columns()?;
        Ok(backend)
    }

    pub fn spaces(&self) -> &[String] {
        &self.spaces
    }

    fn ensure_columns(&self) -> Result<(), BackendError> {
        let existing: Vec<String> = {
            let mut stmt = self.conn.prepare("PRAGMA table_info(search_index)")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(1))?;
            rows.collect::<Result<_, _>>()?
        };
        for space in &self.spaces {
            let c = col(space);
            if !existing.contains(&c) {
                self.conn
                    .execute(&format!("ALTER TABLE search_index ADD COLUMN {c} BLOB"), [])?;
                self.conn.execute(
                    &format!("ALTER TABLE search_index ADD COLUMN {c}_dim INTEGER"),
                    [],
                )?;
            }
        }
        Ok(())
    }

    fn next_seq(&self) -> Result<i64, BackendError> {
        let cur: Option<i64> = self
            .conn
            .query_row(
                "SELECT value FROM search_meta WHERE key = 'seq'",
                [],
                |r| r.get::<_, String>(0),
            )
            .ok()
            .and_then(|s| s.parse().ok());
        let seq = cur.unwrap_or(0) + 1;
        self.conn.execute(
            "INSERT INTO search_meta (key, value) VALUES ('seq', ?) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [seq.to_string()],
        )?;
        Ok(seq)
    }

    /// Insert or update one row (incremental — one upsert per doc, no rebuild).
    pub fn upsert(&self, row: &UpsertRow) -> Result<(), BackendError> {
        let seq = self.next_seq()?;
        let mut cols: Vec<String> = vec![
            "partition_key".into(),
            "object_type".into(),
            "object_id".into(),
            "node_kind".into(),
            "title".into(),
            "body".into(),
            "payload".into(),
            "attrs".into(),
            "source_updated_at".into(),
            "embedding_indexed_at".into(),
            "deleted_at".into(),
            "updated_seq".into(),
        ];
        let opt = |s: Option<&str>| -> rusqlite::types::Value { s.map(String::from).into() };
        let mut vals: Vec<rusqlite::types::Value> = vec![
            opt(row.partition),
            row.object_type.to_string().into(),
            row.object_id.to_string().into(),
            row.node_kind.to_string().into(),
            opt(row.title),
            opt(row.body),
            opt(row.payload),
            opt(row.attrs),
            row.source_updated_at.to_string().into(),
            opt(row.embedding_indexed_at),
            rusqlite::types::Value::Null, // deleted_at
            seq.into(),
        ];
        for space in &self.spaces {
            let vec = row
                .embeddings
                .iter()
                .find(|(s, _)| s == space)
                .and_then(|(_, v)| v.as_ref());
            cols.push(col(space));
            cols.push(format!("{}_dim", col(space)));
            match vec {
                Some(v) => {
                    vals.push(pack(v).into());
                    vals.push((v.len() as i64).into());
                }
                None => {
                    vals.push(rusqlite::types::Value::Null);
                    vals.push(rusqlite::types::Value::Null);
                }
            }
        }

        let placeholders = vec!["?"; cols.len()].join(", ");
        let updatable: Vec<String> = cols
            .iter()
            .filter(|c| !matches!(c.as_str(), "object_type" | "object_id" | "node_kind"))
            .map(|c| format!("{c} = excluded.{c}"))
            .collect();
        let sql = format!(
            "INSERT INTO search_index ({}) VALUES ({placeholders}) \
             ON CONFLICT (object_type, object_id, node_kind) DO UPDATE SET {}",
            cols.join(", "),
            updatable.join(", "),
        );
        self.conn.execute(&sql, params_from_iter(vals))?;
        Ok(())
    }

    /// Tombstone a row (or all node kinds of an object): mark `deleted_at` and
    /// strip its text + vectors, but keep the row so a future sync can propagate
    /// the delete. Mirrors the Python `delete_row`.
    pub fn delete(
        &self,
        object_type: &str,
        object_id: &str,
        node_kind: Option<&str>,
    ) -> Result<(), BackendError> {
        let seq = self.next_seq()?;
        let null_vecs: String = self
            .spaces
            .iter()
            .map(|s| format!("{c} = NULL, {c}_dim = NULL", c = col(s)))
            .collect::<Vec<_>>()
            .join(", ");
        let null_vecs = if null_vecs.is_empty() {
            String::new()
        } else {
            format!("{null_vecs}, ")
        };
        let mut sql = format!(
            "UPDATE search_index SET deleted_at = '', title = NULL, body = NULL, \
             {null_vecs}updated_seq = ? \
             WHERE object_type = ? AND object_id = ? AND deleted_at IS NULL"
        );
        let mut params: Vec<rusqlite::types::Value> = vec![
            seq.into(),
            object_type.to_string().into(),
            object_id.to_string().into(),
        ];
        if let Some(kind) = node_kind {
            sql.push_str(" AND node_kind = ?");
            params.push(kind.to_string().into());
        }
        self.conn.execute(&sql, params_from_iter(params))?;
        Ok(())
    }

    fn filter_sql(&self, filters: &Filters) -> (String, Vec<rusqlite::types::Value>) {
        let mut clauses: Vec<String> = Vec::new();
        let mut params: Vec<rusqlite::types::Value> = Vec::new();
        if let Some(p) = &filters.partition {
            clauses.push("si.partition_key = ?".into());
            params.push(p.clone().into());
        }
        if let Some(types) = &filters.object_types {
            if !types.is_empty() {
                clauses.push(format!(
                    "si.object_type IN ({})",
                    vec!["?"; types.len()].join(",")
                ));
                params.extend(types.iter().map(|t| t.clone().into()));
            }
        }
        if let Some(kinds) = &filters.node_kinds {
            if !kinds.is_empty() {
                clauses.push(format!(
                    "si.node_kind IN ({})",
                    vec!["?"; kinds.len()].join(",")
                ));
                params.extend(kinds.iter().map(|k| k.clone().into()));
            }
        }
        if let Some(attrs) = &filters.attrs {
            for (key, value) in attrs {
                clauses.push(format!("json_extract(si.attrs, '$.{key}') = ?"));
                params.push(value.clone().into());
            }
        }
        if clauses.is_empty() {
            (String::new(), params)
        } else {
            (format!(" AND {}", clauses.join(" AND ")), params)
        }
    }

    /// FTS5 / bm25 lexical retrieval (best first).
    pub fn lexical_search(
        &self,
        query: &str,
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<RawHit>, BackendError> {
        let Some(matchq) = fts_query(query) else {
            return Ok(Vec::new());
        };
        let (where_sql, fparams) = self.filter_sql(filters);
        let sql = format!(
            "SELECT si.object_type, si.object_id, si.node_kind, si.partition_key, \
                    si.title, si.payload \
             FROM search_fts \
             JOIN search_index si ON si.id = search_fts.rowid \
             WHERE search_fts MATCH ? AND si.deleted_at IS NULL{where_sql} \
             ORDER BY bm25(search_fts) ASC LIMIT ?"
        );
        let mut params: Vec<rusqlite::types::Value> = vec![matchq.into()];
        params.extend(fparams);
        params.push((limit as i64).into());

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(params), |r| {
            Ok(RawHit {
                object_type: r.get(0)?,
                object_id: r.get(1)?,
                node_kind: r.get(2)?,
                partition: r.get(3)?,
                title: r.get(4)?,
                payload: r.get(5)?,
                sim: None,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Exact cosine scan over one space (best first), filtered by `min_score`.
    pub fn semantic_search(
        &self,
        query_vec: &[f32],
        space: &str,
        filters: &Filters,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<RawHit>, BackendError> {
        if !self.spaces.iter().any(|s| s == space) {
            return Ok(Vec::new());
        }
        let c = col(space);
        let (where_sql, params) = self.filter_sql(filters);
        let sql = format!(
            "SELECT si.object_type, si.object_id, si.node_kind, si.partition_key, \
                    si.title, si.payload, si.{c} \
             FROM search_index si \
             WHERE si.{c} IS NOT NULL AND si.deleted_at IS NULL{where_sql}"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut scored: Vec<(f32, RawHit)> = stmt
            .query_map(params_from_iter(params), |r| {
                let blob: Vec<u8> = r.get(6)?;
                let sim = cosine(query_vec, &unpack(&blob));
                Ok((
                    sim,
                    RawHit {
                        object_type: r.get(0)?,
                        object_id: r.get(1)?,
                        node_kind: r.get(2)?,
                        partition: r.get(3)?,
                        title: r.get(4)?,
                        payload: r.get(5)?,
                        sim: Some(sim),
                    },
                ))
            })?
            .collect::<Result<_, _>>()?;
        scored.retain(|(sim, _)| *sim >= min_score);
        // Sort by similarity descending; stable to mirror Python's list.sort.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(limit).map(|(_, h)| h).collect())
    }
}
