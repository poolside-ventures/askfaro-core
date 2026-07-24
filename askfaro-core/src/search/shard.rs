//! Shard sync — apply a server-exported shard (or delta) to the local shard.
//!
//! The cloud→device retrieval pattern is: the server indexes documents and
//! exports a per-partition SQLite shard (whose schema IS [the shard
//! format](super::sqlite)); the device downloads it and ranks locally. This
//! module owns the *apply* half — verify, atomic full-replace, and trigger-safe
//! delta merge — so every consumer (Scope desktop/mobile, third-party Faro
//! apps) shares one correct implementation instead of re-deriving it.
//!
//! These functions are pure file/byte operations with **no network**: the
//! consumer owns the fetch, auth, and cursor policy (their backend contract);
//! the core owns the shard-format-coupled apply. That boundary is deliberate —
//! Faro ships no opinion about any consumer's export endpoint.

use std::path::Path;

use rusqlite::Connection;

use crate::model::sha256_file;
use crate::search::sqlite::SqliteBackend;

/// A shard-apply failure.
#[derive(Debug, thiserror::Error)]
pub enum ShardError {
    #[error("shard: sha256 mismatch (expected {expected}, got {got})")]
    Sha256Mismatch { expected: String, got: String },
    #[error("shard i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("shard sqlite: {0}")]
    Sqlite(String),
    #[error("shard: path not valid utf-8")]
    NonUtf8Path,
}

/// Verify a downloaded shard file against its expected lowercase-hex sha256.
pub fn verify_shard(path: &Path, expected_sha256: &str) -> Result<(), ShardError> {
    let got = sha256_file(path)?;
    if got != expected_sha256 {
        return Err(ShardError::Sha256Mismatch {
            expected: expected_sha256.to_string(),
            got,
        });
    }
    Ok(())
}

/// Atomically replace the local shard with a freshly downloaded full export.
///
/// Verifies `incoming` (when a hash is given), then removes the old shard and
/// its `-wal`/`-shm` sidecars and renames `incoming` into place. `incoming`
/// must sit on the same filesystem as `shard` for the rename to be atomic
/// (both are the consumer's app-data dir in practice).
pub fn apply_full(
    shard: &Path,
    incoming: &Path,
    expected_sha256: Option<&str>,
) -> Result<(), ShardError> {
    if let Some(sha) = expected_sha256 {
        verify_shard(incoming, sha)?;
    }
    for suffix in ["", "-wal", "-shm"] {
        let sidecar = with_suffix(shard, suffix);
        if sidecar.exists() {
            std::fs::remove_file(&sidecar)?;
        }
    }
    std::fs::rename(incoming, shard)?;
    Ok(())
}

/// Merge a downloaded delta shard into the local shard, then delete the delta.
///
/// Verifies `incoming` (when a hash is given) and applies [`merge_delta`].
/// `spaces` are the embedding spaces the shard carries — used to materialise
/// the contract schema (+ vector columns) if this is the first delta on a
/// fresh install.
pub fn apply_delta(
    shard: &Path,
    incoming: &Path,
    spaces: &[&str],
    expected_sha256: Option<&str>,
) -> Result<(), ShardError> {
    if let Some(sha) = expected_sha256 {
        verify_shard(incoming, sha)?;
    }
    merge_delta(shard, incoming, spaces)?;
    if incoming.exists() {
        std::fs::remove_file(incoming)?;
    }
    Ok(())
}

/// Rows currently in a shard (excludes tombstones) — for a consumer's "N items
/// indexed" UI. Returns 0 if the shard doesn't exist yet.
pub fn row_count(shard: &Path) -> Result<i64, ShardError> {
    if !shard.exists() {
        return Ok(0);
    }
    let conn = Connection::open_with_flags(shard, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| ShardError::Sqlite(e.to_string()))?;
    conn.query_row(
        "SELECT count(*) FROM search_index WHERE deleted_at IS NULL",
        [],
        |r| r.get::<_, i64>(0),
    )
    .map_err(|e| ShardError::Sqlite(e.to_string()))
}

/// Merge a delta shard into the local shard with a column-intersection upsert.
///
/// `ON CONFLICT DO UPDATE` (never `INSERT OR REPLACE`) so the FTS
/// external-content triggers fire on both the insert and the update path and
/// the lexical index stays consistent without `recursive_triggers`. Kept
/// byte-identical to the reference implementation it was lifted from.
fn merge_delta(shard: &Path, delta: &Path, spaces: &[&str]) -> Result<(), ShardError> {
    let shard_str = shard.to_str().ok_or(ShardError::NonUtf8Path)?;
    // Opening through the backend creates the contract schema (+ the spaces'
    // vector columns) when this is the first delta on a fresh install.
    drop(SqliteBackend::open(shard_str, spaces).map_err(|e| ShardError::Sqlite(e.to_string()))?);

    let conn = Connection::open(shard).map_err(|e| ShardError::Sqlite(e.to_string()))?;
    conn.execute(
        "ATTACH DATABASE ?1 AS delta",
        [delta.to_str().ok_or(ShardError::NonUtf8Path)?],
    )
    .map_err(|e| ShardError::Sqlite(e.to_string()))?;

    let result = merge_attached(&conn);
    let detach = conn.execute("DETACH DATABASE delta", []);
    result?;
    detach.map_err(|e| ShardError::Sqlite(e.to_string()))?;
    Ok(())
}

fn merge_attached(conn: &Connection) -> Result<(), ShardError> {
    let columns = |table: &str| -> Result<Vec<String>, ShardError> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA {table}.table_info(search_index)"))
            .map_err(|e| ShardError::Sqlite(e.to_string()))?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(1))
            .map_err(|e| ShardError::Sqlite(e.to_string()))?;
        rows.collect::<Result<_, _>>()
            .map_err(|e| ShardError::Sqlite(e.to_string()))
    };
    let main_cols = columns("main")?;
    // Everything both sides know, minus the local rowid (fresh ids keep the
    // FTS content_rowid mapping intact).
    let cols: Vec<String> = columns("delta")?
        .into_iter()
        .filter(|c| c != "id" && main_cols.contains(c))
        .collect();
    let col_list = cols.join(", ");
    let set_list = cols
        .iter()
        .filter(|c| !matches!(c.as_str(), "object_type" | "object_id" | "node_kind"))
        .map(|c| format!("{c} = excluded.{c}"))
        .collect::<Vec<_>>()
        .join(", ");
    conn.execute_batch(&format!(
        "BEGIN;
         INSERT INTO main.search_index ({col_list})
             SELECT {col_list} FROM delta.search_index WHERE true
             ON CONFLICT(object_type, object_id, node_kind) DO UPDATE SET {set_list};
         COMMIT;"
    ))
    .map_err(|e| ShardError::Sqlite(e.to_string()))
}

fn with_suffix(path: &Path, suffix: &str) -> std::path::PathBuf {
    if suffix.is_empty() {
        return path.to_path_buf();
    }
    let mut s = path.as_os_str().to_os_string();
    s.push(suffix);
    std::path::PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::sqlite::{SqliteBackend, UpsertRow};

    const SPACE: &str = "test_space";

    fn write_shard(path: &Path, rows: &[(&str, &str, &str, &str)]) {
        let backend = SqliteBackend::open(path.to_str().unwrap(), &[SPACE]).unwrap();
        for (object_type, object_id, title, body) in rows {
            backend
                .upsert(&UpsertRow {
                    object_type,
                    object_id,
                    node_kind: "leaf",
                    partition: None,
                    title: Some(title),
                    body: Some(body),
                    payload: None,
                    attrs: None,
                    source_updated_at: "2026-01-01T00:00:00Z",
                    embedding_indexed_at: None,
                    embeddings: &[],
                })
                .unwrap();
        }
        drop(backend);
    }

    fn fts_matches(shard: &Path, query: &str) -> i64 {
        let conn = Connection::open(shard).unwrap();
        conn.query_row(
            "SELECT count(*) FROM search_fts WHERE search_fts MATCH ?",
            [query],
            |r| r.get::<_, i64>(0),
        )
        .unwrap()
    }

    #[test]
    fn delta_merge_upserts_and_keeps_fts_consistent() {
        let dir = std::env::temp_dir().join(format!("shard-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("shard.sqlite3");
        let delta = dir.join("delta.sqlite3");

        // Base: one note. Delta: update that note's body + add a second.
        write_shard(&shard, &[("note", "n1", "Groceries", "milk eggs")]);
        write_shard(
            &delta,
            &[
                ("note", "n1", "Groceries", "milk eggs bread cheese"),
                ("note", "n2", "Todo", "call the dentist"),
            ],
        );

        apply_delta(&shard, &delta, &[SPACE], None).unwrap();

        // Two rows, no duplicate for n1 (conflict updated in place).
        assert_eq!(row_count(&shard).unwrap(), 2);
        // FTS reflects the *updated* body (trigger fired on the update path) and
        // the newly inserted row.
        assert_eq!(fts_matches(&shard, "bread"), 1);
        assert_eq!(fts_matches(&shard, "dentist"), 1);
        // Delta file consumed.
        assert!(!delta.exists());

        std::fs::remove_dir_all(&dir).ok();
    }
}
