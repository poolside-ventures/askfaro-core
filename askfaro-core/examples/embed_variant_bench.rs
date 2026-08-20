//! Measure what a QUANTIZED EmbeddingGemma export costs in retrieval quality,
//! against the device's own shard rather than a synthetic corpus.
//!
//! One process per variant on purpose: the resident footprint of an `ort`
//! session is the number this exists to produce, and loading two graphs into one
//! process makes that number unattributable. The driver script loops.
//!
//! ```text
//! cargo run --release --features embeddinggemma --example embed_variant_bench -- \
//!   --shard "$HOME/Library/Application Support/com.getscopy.desktop/search/shard.sqlite3" \
//!   --label q8 --dir <model_dir> --graph model_quantized.onnx \
//!   --queries 200 --docs 4000 --out /tmp/q8.json
//! ```
//!
//! What it reports, per variant:
//!  - **footprint**: `vmmap` physical footprint and the dirty/clean split of the
//!    regions backed by the weight file, before and after load. That is the only
//!    way to tell an `ort` session that mmaps its external data from one that
//!    copies it into the heap, and RSS cannot: mmapped clean pages count in RSS
//!    and cost nothing under pressure.
//!  - **known-item retrieval**: a document's own title, used as the query,
//!    should retrieve that document. Ground truth with no labelling, identical
//!    difficulty across variants, so a drop in MRR is a real drop.
//!  - **cross-precision** (query embedded by the variant, documents still the
//!    server's fp32 vectors) vs **same-precision** (both sides the variant).
//!    Only the second needs a new space and a backfill, so the gap between them
//!    is what decides whether the migration is necessary at all.

#[cfg(not(feature = "embeddinggemma"))]
fn main() {
    eprintln!("build with --features embeddinggemma");
}

#[cfg(feature = "embeddinggemma")]
fn main() {
    real_main();
}

#[cfg(feature = "embeddinggemma")]
fn real_main() {
    use askfaro_core::search::gemma::{GemmaEmbedder, GemmaOptions};
    use askfaro_core::search::EmbedEngine;
    use std::time::Instant;

    let args = Args::parse();

    // Footprint before anything is loaded, so the delta is the model's.
    let before = footprint();

    let t0 = Instant::now();
    let opts = GemmaOptions {
        cpu_arena: args.arena,
        memory_pattern: args.mempattern,
        intra_threads: args.threads,
    };
    let embedder = GemmaEmbedder::load_with(&args.dir, &args.graph, "bench", &opts)
        .unwrap_or_else(|e| panic!("load {}/{}: {e}", args.dir, args.graph));
    let load_ms = t0.elapsed().as_millis();
    let after_load = footprint();

    // ---- footprint-only: no corpus in the process at all -----------------
    //
    // The shard costs ~100 MiB of vectors and text to hold, identical for every
    // variant but large enough to swamp the difference between two of them. This
    // arm loads the model, embeds a fixed list of query-shaped strings, and
    // reports what the MODEL costs.
    if args.footprint_only {
        let mut ms = Vec::new();
        for i in 0..args.queries {
            let t = Instant::now();
            let _ = embedder.embed_query(&format!(
                "{} {}",
                FOOTPRINT_QUERIES[i % FOOTPRINT_QUERIES.len()],
                i
            ));
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let after = footprint_for(Some(&format!("{}_data", args.graph)));
        let out = serde_json::json!({
            "label": args.label,
            "graph": args.graph,
            "weightsBytes": weights_bytes(&args.dir, &args.graph),
            "loadMs": load_ms,
            "opts": { "cpuArena": args.arena, "memoryPattern": args.mempattern, "intraThreads": args.threads },
            "queries": args.queries,
            "queryMs": stats(&ms),
            "footprint": {
                "before": before,
                "afterLoad": footprint_for(Some(&format!("{}_data", args.graph))),
                "afterQueries": after,
            },
        });
        std::fs::write(&args.out, serde_json::to_string(&out).unwrap()).expect("write out");
        eprintln!(
            "{}: {} MiB weights, phys {:.0} MiB after {} queries -> {}",
            args.label,
            weights_bytes(&args.dir, &args.graph) / 1048576,
            after["physFootprintMiB"].as_f64().unwrap_or(0.0),
            args.queries,
            args.out
        );
        return;
    }

    // ---- corpus ----------------------------------------------------------
    let conn = rusqlite::Connection::open_with_flags(
        &args.shard,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .expect("open shard");

    // Deterministic sample: order by id, take a fixed stride. Every variant sees
    // the identical corpus and the identical queries.
    let mut stmt = conn
        .prepare(
            "SELECT id, object_type, title, body, embedding_embeddinggemma_300m_fp32
               FROM search_index
              WHERE deleted_at IS NULL
                AND embedding_embeddinggemma_300m_fp32 IS NOT NULL
                AND title IS NOT NULL AND length(title) > 12
              ORDER BY id",
        )
        .expect("prepare");
    let all: Vec<Doc> = stmt
        .query_map([], |r| {
            Ok(Doc {
                id: r.get(0)?,
                object_type: r.get(1)?,
                title: r.get(2)?,
                body: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                fp32: unpack(&r.get::<_, Vec<u8>>(4)?),
            })
        })
        .expect("query")
        .filter_map(Result::ok)
        .collect();
    drop(stmt);

    let stride = if args.docs == 0 { 1 } else { (all.len() / args.docs).max(1) };
    let take = if args.docs == 0 { all.len() } else { args.docs };
    let docs: Vec<&Doc> = all.iter().step_by(stride).take(take).collect();
    let q_stride = (docs.len() / args.queries).max(1);
    let queries: Vec<&Doc> = docs.iter().copied().step_by(q_stride).take(args.queries).collect();
    eprintln!(
        "{}: corpus {} docs (of {} eligible), {} known-item queries",
        args.label,
        docs.len(),
        all.len(),
        queries.len()
    );

    // ---- embed the queries ----------------------------------------------
    let mut q_vecs = Vec::with_capacity(queries.len());
    let mut q_ms = Vec::with_capacity(queries.len());
    for (i, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let v = embedder.embed_query(&q.title);
        q_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        q_vecs.push(v);
        if i % 50 == 0 {
            eprintln!("  {}: query {i}/{}", args.label, queries.len());
        }
    }
    let after_queries = footprint();

    // ---- re-embed the corpus with THIS variant (same-precision arm) ------
    //
    // Skipped by default: it is ~500x the cost of the query pass, and it only
    // decides anything if the cross-precision arm already looks survivable.
    let mut d_vecs: Vec<Option<Vec<f32>>> = Vec::with_capacity(docs.len());
    let mut d_ms = Vec::with_capacity(docs.len());
    let t_docs = Instant::now();
    for (i, d) in docs.iter().enumerate() {
        if !args.reembed {
            d_vecs.push(None);
            continue;
        }
        let text = index_text(d);
        let t = Instant::now();
        let v = embedder.embed_documents(&[&text]).into_iter().next().flatten();
        d_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        d_vecs.push(v);
        if i % 250 == 0 {
            eprintln!(
                "  {}: doc {i}/{} ({:.0}s elapsed)",
                args.label,
                docs.len(),
                t_docs.elapsed().as_secs_f64()
            );
        }
    }
    let after_docs = footprint();

    // ---- scoring ---------------------------------------------------------
    let cross = rank(&queries, &q_vecs, &docs, |d, _| Some(&d.fp32));
    let same = if args.reembed {
        Some(rank(&queries, &q_vecs, &docs, |_, i| d_vecs[i].as_deref()))
    } else {
        None
    };

    let out = serde_json::json!({
        "label": args.label,
        "dir": args.dir,
        "graph": args.graph,
        "weightsBytes": weights_bytes(&args.dir, &args.graph),
        "loadMs": load_ms,
        "opts": { "cpuArena": args.arena, "memoryPattern": args.mempattern, "intraThreads": args.threads },
        "footprint": {
            "before": before,
            "afterLoad": after_load,
            "afterQueries": after_queries,
            "afterDocs": after_docs,
        },
        "queryMs": stats(&q_ms),
        "docMs": stats(&d_ms),
        "corpus": docs.len(),
        "queries": queries.len(),
        "embedFailures": {
            "query": q_vecs.iter().filter(|v| v.is_none()).count(),
            "doc": d_vecs.iter().filter(|v| v.is_none()).count(),
        },
        // Cross-precision: variant query against the server's fp32 doc vectors.
        // This is the arm that needs NO migration.
        "crossPrecision": cross.json(),
        // Same-precision: both sides this variant. Needs a new space + backfill.
        "samePrecision": same.as_ref().map(|s| s.json()),
        // Raw query vectors so the comparison step can measure cosine agreement
        // against the fp32 reference run without re-embedding anything.
        "queryVectors": q_vecs,
        "queryIds": queries.iter().map(|q| q.id).collect::<Vec<_>>(),
        "queryTypes": queries.iter().map(|q| q.object_type.clone()).collect::<Vec<_>>(),
        "docIds": docs.iter().map(|d| d.id).collect::<Vec<_>>(),
        "topDocIds": cross.top_ids,
    });
    std::fs::write(&args.out, serde_json::to_string(&out).unwrap()).expect("write out");
    eprintln!(
        "{}: cross MRR {:.4} R@1 {:.3} R@10 {:.3} | same {} -> {}",
        args.label,
        cross.mrr,
        cross.r1,
        cross.r10,
        same.as_ref().map_or("(skipped)".to_string(), |s| format!(
            "MRR {:.4} R@1 {:.3} R@10 {:.3}",
            s.mrr, s.r1, s.r10
        )),
        args.out
    );
}

/// Query-shaped strings for the footprint arm — the lengths a user actually
/// types, not the 2,048-token documents the memory sidecar sends.
#[cfg(feature = "embeddinggemma")]
const FOOTPRINT_QUERIES: &[&str] = &[
    "invoice from last quarter",
    "what did anna say about the pricing deck",
    "meetings with the design team next week",
    "unpaid tasks assigned to me",
    "the thread where we agreed the launch date",
    "contract renewal reminder",
    "notes from the offsite",
    "emails I have not replied to",
];

#[cfg(feature = "embeddinggemma")]
struct Doc {
    id: i64,
    object_type: String,
    title: String,
    body: String,
    fp32: Vec<f32>,
}

/// Mirrors `IndexDoc::index_text` — title + body, newline-joined. The server
/// embedded these rows through that composition, so re-embedding through any
/// other one would measure the difference in text, not in the model.
#[cfg(feature = "embeddinggemma")]
fn index_text(d: &Doc) -> String {
    if d.body.is_empty() {
        d.title.clone()
    } else {
        format!("{}\n{}", d.title, d.body)
    }
}

#[cfg(feature = "embeddinggemma")]
struct Ranked {
    mrr: f64,
    r1: f64,
    r5: f64,
    r10: f64,
    /// Rank of each query's own document (0-based; `usize::MAX` = not in top 50).
    ranks: Vec<usize>,
    top_ids: Vec<Vec<i64>>,
}

#[cfg(feature = "embeddinggemma")]
impl Ranked {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "mrr": self.mrr, "r1": self.r1, "r5": self.r5, "r10": self.r10,
            "ranks": self.ranks.iter().map(|&r| if r == usize::MAX { -1i64 } else { r as i64 }).collect::<Vec<_>>(),
        })
    }
}

/// Semantic-only known-item ranking: does a document's own title retrieve it?
/// `doc_vec` picks which side of the precision question this arm measures.
#[cfg(feature = "embeddinggemma")]
fn rank<'a, F>(queries: &[&Doc], q_vecs: &[Option<Vec<f32>>], docs: &[&'a Doc], doc_vec: F) -> Ranked
where
    F: Fn(&'a Doc, usize) -> Option<&'a [f32]>,
{
    const TOP: usize = 50;
    let mut ranks = Vec::with_capacity(queries.len());
    let mut top_ids = Vec::with_capacity(queries.len());
    for (qi, q) in queries.iter().enumerate() {
        let Some(qv) = q_vecs[qi].as_deref() else {
            ranks.push(usize::MAX);
            top_ids.push(Vec::new());
            continue;
        };
        let mut scored: Vec<(f32, i64)> = docs
            .iter()
            .enumerate()
            .filter_map(|(i, d)| doc_vec(d, i).map(|dv| (cosine(qv, dv), d.id)))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let r = scored.iter().position(|&(_, id)| id == q.id).unwrap_or(usize::MAX);
        ranks.push(r);
        top_ids.push(scored.iter().take(TOP).map(|&(_, id)| id).collect());
    }
    let n = ranks.len() as f64;
    let hit = |k: usize| ranks.iter().filter(|&&r| r < k).count() as f64 / n;
    Ranked {
        mrr: ranks
            .iter()
            .map(|&r| if r == usize::MAX { 0.0 } else { 1.0 / (r as f64 + 1.0) })
            .sum::<f64>()
            / n,
        r1: hit(1),
        r5: hit(5),
        r10: hit(10),
        ranks,
        top_ids,
    }
}

#[cfg(feature = "embeddinggemma")]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(feature = "embeddinggemma")]
fn unpack(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[cfg(feature = "embeddinggemma")]
fn stats(xs: &[f64]) -> serde_json::Value {
    if xs.is_empty() {
        return serde_json::json!({});
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |p: f64| s[((s.len() - 1) as f64 * p) as usize];
    serde_json::json!({ "p50": at(0.5), "p90": at(0.9), "mean": s.iter().sum::<f64>() / s.len() as f64 })
}

#[cfg(feature = "embeddinggemma")]
fn weights_bytes(dir: &str, graph: &str) -> u64 {
    let base = std::path::Path::new(dir);
    let mut total = 0;
    // External data is `model.onnx_data` from onnx-community's exporter and
    // `model.onnx.data` from torch's; a size that silently reads 3 MiB because
    // it guessed the wrong one is worse than no size at all.
    for name in [
        graph.to_string(),
        format!("{graph}_data"),
        format!("{graph}.data"),
    ] {
        if let Ok(m) = std::fs::metadata(base.join(&name)) {
            total += m.len();
        }
    }
    total
}

/// Physical footprint and the dirty/clean split, from `vmmap` on our own pid.
///
/// RSS is not usable for this question — the weight file is mmapped, so its
/// pages are resident but clean, and two runs of the identical configuration
/// reported 6.44 and 1.66 GiB. `vmmap` separates them: `dirty` is what the
/// machine actually loses, `clean` is file-backed and evictable under pressure.
#[cfg(feature = "embeddinggemma")]
fn footprint() -> serde_json::Value {
    footprint_for(None)
}

/// `weight_file` attributes the mmapped half: how much of the weight file is
/// actually resident, and how much of that the process has dirtied. A weight
/// page that is resident and clean is file-backed — it evicts under pressure
/// with no swap write — so it is not the same kind of cost as a dirty one, and
/// a single "resident" number that adds them together hides the whole question.
#[cfg(feature = "embeddinggemma")]
fn footprint_for(weight_file: Option<&str>) -> serde_json::Value {
    let pid = std::process::id();
    let mapped = weight_file.map(|f| mapped_file_regions(pid, f));
    let out = std::process::Command::new("/usr/bin/vmmap")
        .args(["--summary", &pid.to_string()])
        .output();
    let mut v = footprint_summary(out);
    if let (Some(m), Some(obj)) = (mapped, v.as_object_mut()) {
        obj.insert("weightFile".into(), m);
    }
    v
}

/// Sum the vmmap regions backed by `name`: `REGION_TYPE START-END [ VSIZE RSDNT
/// DIRTY SWAP] ... path`.
#[cfg(feature = "embeddinggemma")]
fn mapped_file_regions(pid: u32, name: &str) -> serde_json::Value {
    let Ok(out) = std::process::Command::new("/usr/bin/vmmap")
        .arg(pid.to_string())
        .output()
    else {
        return serde_json::json!({ "error": "vmmap unavailable" });
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let (mut virt, mut res, mut dirty, mut swap) = (0u64, 0u64, 0u64, 0u64);
    for line in text.lines() {
        if !line.contains(name) {
            continue;
        }
        // The bracketed quadruple is the only place four sizes appear together.
        let Some(open) = line.find('[') else { continue };
        let Some(close) = line[open..].find(']') else { continue };
        let cols: Vec<&str> = line[open + 1..open + close].split_whitespace().collect();
        if cols.len() < 4 {
            continue;
        }
        virt += parse_size(cols[0]);
        res += parse_size(cols[1]);
        dirty += parse_size(cols[2]);
        swap += parse_size(cols[3]);
    }
    serde_json::json!({
        "mappedMiB": virt as f64 / 1048576.0,
        "residentMiB": res as f64 / 1048576.0,
        "dirtyMiB": dirty as f64 / 1048576.0,
        "swappedMiB": swap as f64 / 1048576.0,
    })
}

#[cfg(feature = "embeddinggemma")]
fn footprint_summary(out: std::io::Result<std::process::Output>) -> serde_json::Value {
    let Ok(out) = out else {
        return serde_json::json!({ "error": "vmmap unavailable" });
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut phys = 0u64;
    let mut dirty = 0u64;
    let mut clean_file = 0u64;
    let mut swapped = 0u64;
    for line in text.lines() {
        let l = line.trim();
        if let Some(v) = l.strip_prefix("Physical footprint:") {
            phys = parse_size(v);
        } else if l.starts_with("TOTAL") && !l.starts_with("TOTAL,") {
            // Columns: REGION TYPE / VIRTUAL / RESIDENT / DIRTY / SWAPPED / ...
            let cols: Vec<&str> = l.split_whitespace().collect();
            if cols.len() >= 5 && cols[0] == "TOTAL" {
                let resident = parse_size(cols[2]);
                dirty = parse_size(cols[3]);
                swapped = parse_size(cols[4]);
                clean_file = resident.saturating_sub(dirty);
            }
        }
    }
    serde_json::json!({
        "physFootprintMiB": phys as f64 / 1048576.0,
        "dirtyMiB": dirty as f64 / 1048576.0,
        "cleanResidentMiB": clean_file as f64 / 1048576.0,
        "swappedMiB": swapped as f64 / 1048576.0,
    })
}

#[cfg(feature = "embeddinggemma")]
fn parse_size(s: &str) -> u64 {
    let s = s.trim();
    let (num, mult) = if let Some(n) = s.strip_suffix('G') {
        (n, 1u64 << 30)
    } else if let Some(n) = s.strip_suffix('M') {
        (n, 1u64 << 20)
    } else if let Some(n) = s.strip_suffix('K') {
        (n, 1u64 << 10)
    } else {
        (s, 1)
    };
    num.trim().parse::<f64>().map(|v| (v * mult as f64) as u64).unwrap_or(0)
}

#[cfg(feature = "embeddinggemma")]
struct Args {
    shard: String,
    label: String,
    dir: String,
    graph: String,
    out: String,
    queries: usize,
    docs: usize,
    reembed: bool,
    arena: bool,
    mempattern: bool,
    threads: Option<usize>,
    footprint_only: bool,
}

#[cfg(feature = "embeddinggemma")]
impl Args {
    fn parse() -> Self {
        let mut a = Args {
            shard: String::new(),
            label: "variant".into(),
            dir: String::new(),
            graph: "model.onnx".into(),
            out: "/tmp/embed_variant.json".into(),
            queries: 200,
            docs: 4000,
            reembed: false,
            arena: false,
            mempattern: false,
            threads: None,
            footprint_only: false,
        };
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i + 1 < argv.len() {
            let v = argv[i + 1].clone();
            match argv[i].as_str() {
                "--shard" => a.shard = v,
                "--label" => a.label = v,
                "--dir" => a.dir = v,
                "--graph" => a.graph = v,
                "--out" => a.out = v,
                "--queries" => a.queries = v.parse().expect("--queries"),
                "--docs" => a.docs = v.parse().expect("--docs"),
                "--reembed" => a.reembed = v == "1" || v == "true",
                "--arena" => a.arena = v == "1",
                "--mempattern" => a.mempattern = v == "1",
                "--threads" => a.threads = Some(v.parse().expect("--threads")),
                "--footprint-only" => a.footprint_only = v == "1",
                other => panic!("unknown flag {other}"),
            }
            i += 2;
        }
        assert!(a.footprint_only || !a.shard.is_empty(), "--shard is required");
        assert!(!a.dir.is_empty(), "--dir is required");
        a
    }
}
