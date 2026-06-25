//! Reciprocal Rank Fusion — the Rust mirror of the Python lib's `fusion.py`.
//!
//! Fusion operates on **ranks, never on raw scores** (SQLite `bm25()` and a
//! cosine scan are not on comparable scales, but their orderings are). This is
//! what makes retrieval semantics identical across the server (Postgres) and
//! on-device (SQLite) backends. `RRF_K = 60` is the canonical constant from
//! Cormack et al. (2009) and is part of the contract.

use std::collections::HashMap;

use crate::search::types::{MatchType, RawHit, SearchResult};

/// Canonical RRF constant. Changing it breaks server/device rank parity.
pub const RRF_K: usize = 60;

/// Fuse two ranked candidate lists into one, ordered by RRF score.
///
/// Insertion order mirrors Python's `dict.setdefault`: lexical hits first (in
/// lexical-rank order), then semantic-only keys (in semantic-rank order). The
/// final sort is stable, so equal scores keep that order — exactly as CPython's
/// stable `list.sort` does.
pub fn rrf_fuse(lexical: &[RawHit], semantic: &[RawHit]) -> Vec<SearchResult> {
    let lexical_ranks: HashMap<_, usize> = lexical
        .iter()
        .enumerate()
        .map(|(i, h)| (h.key(), i + 1))
        .collect();
    let semantic_ranks: HashMap<_, usize> = semantic
        .iter()
        .enumerate()
        .map(|(i, h)| (h.key(), i + 1))
        .collect();
    let semantic_scores: HashMap<_, f32> = semantic
        .iter()
        .filter_map(|h| h.sim.map(|s| (h.key(), s)))
        .collect();

    // Preserve first-seen insertion order (lexical, then semantic-only).
    let mut order: Vec<(String, String, String)> = Vec::new();
    let mut by_key: HashMap<(String, String, String), &RawHit> = HashMap::new();
    for hit in lexical.iter().chain(semantic.iter()) {
        by_key.entry(hit.key()).or_insert_with(|| {
            order.push(hit.key());
            hit
        });
    }

    let mut fused: Vec<SearchResult> = Vec::with_capacity(order.len());
    for key in &order {
        let hit = by_key[key];
        let lex_rank = lexical_ranks.get(key).copied();
        let sem_rank = semantic_ranks.get(key).copied();
        let mut score = 0.0f64;
        if let Some(r) = lex_rank {
            score += 1.0 / (RRF_K + r) as f64;
        }
        if let Some(r) = sem_rank {
            score += 1.0 / (RRF_K + r) as f64;
        }
        let match_type = match (lex_rank.is_some(), sem_rank.is_some()) {
            (true, true) => MatchType::Hybrid,
            (true, false) => MatchType::Keyword,
            _ => MatchType::Semantic,
        };
        fused.push(SearchResult {
            object_type: hit.object_type.clone(),
            object_id: hit.object_id.clone(),
            node_kind: hit.node_kind.clone(),
            partition: hit.partition.clone(),
            title: hit.title.clone(),
            payload: hit.payload.clone(),
            score,
            match_type,
            lexical_rank: lex_rank,
            semantic_rank: sem_rank,
            semantic_score: semantic_scores.get(key).copied(),
            matched_node_kinds: Vec::new(),
        });
    }

    // Stable sort by score descending (ties keep insertion order).
    fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

/// Collapse multiple node kinds of the same object into its best-scoring row,
/// recording which kinds matched. Mirrors Python's `collapse_objects`.
pub fn collapse_objects(results: Vec<SearchResult>) -> Vec<SearchResult> {
    let mut best: HashMap<(String, String), SearchResult> = HashMap::new();
    let mut order: Vec<(String, String)> = Vec::new();
    for mut r in results {
        let obj_key = (r.object_type.clone(), r.object_id.clone());
        match best.get_mut(&obj_key) {
            None => {
                r.matched_node_kinds = vec![r.node_kind.clone()];
                order.push(obj_key.clone());
                best.insert(obj_key, r);
            }
            Some(existing) => {
                existing.matched_node_kinds.push(r.node_kind.clone());
                if r.score > existing.score {
                    r.matched_node_kinds = existing.matched_node_kinds.clone();
                    *existing = r;
                }
            }
        }
    }
    order.into_iter().map(|k| best.remove(&k).unwrap()).collect()
}
