//! The pluggable embedder (RFC §6). `model identity == space name` (the hard
//! rule, §7): an `EmbedEngine` declares the space it writes/queries, and vectors
//! are only ever compared within one space.
//!
//! The default build ships [`BowEmbedder`] — a deterministic bag-of-words hash
//! embedder (token overlap → cosine). It is the Rust twin of the Python lib's
//! test embedder, which makes it the substrate for the retrieval-parity
//! conformance suite, and a zero-dependency fallback. The real default,
//! EmbeddingGemma, lives behind the `embeddinggemma` feature (see `gemma.rs`).

/// A per-device embedder. Documents and queries embed through distinct methods
/// because some models (EmbeddingGemma) use different prompt prefixes for each.
pub trait EmbedEngine {
    /// The embedding space this engine reads/writes. Model identity.
    fn space(&self) -> &str;

    /// Embed document texts. `None` marks a per-text failure (non-fatal: the row
    /// is written lexical-only and gains semantic retrieval after a backfill).
    fn embed_documents(&self, texts: &[&str]) -> Vec<Option<Vec<f32>>>;

    /// Embed a single query string.
    fn embed_query(&self, text: &str) -> Option<Vec<f32>>;
}

/// Deterministic bag-of-words hash embedder. Mirrors the Python lib's
/// `tests/conftest.py::bow_vector` exactly (md5 token bucketing, DIM=512, L2
/// normalize) so a corpus indexed by either ranks identically.
pub struct BowEmbedder {
    space: String,
}

/// Wide enough that md5-bucket collisions between unrelated tokens are
/// vanishingly rare (matches the Python reference).
pub const BOW_DIM: usize = 512;

impl BowEmbedder {
    pub fn new(space: &str) -> Self {
        BowEmbedder {
            space: space.to_string(),
        }
    }

    fn vector(text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; BOW_DIM];
        for token in tokenize_lower(text) {
            let digest = md5::compute(token.as_bytes());
            let bucket = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize
                % BOW_DIM;
            vec[bucket] += 1.0;
        }
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut vec {
                *x /= norm;
            }
        }
        vec
    }
}

impl EmbedEngine for BowEmbedder {
    fn space(&self) -> &str {
        &self.space
    }
    fn embed_documents(&self, texts: &[&str]) -> Vec<Option<Vec<f32>>> {
        texts.iter().map(|t| Some(Self::vector(t))).collect()
    }
    fn embed_query(&self, text: &str) -> Option<Vec<f32>> {
        Some(Self::vector(text))
    }
}

/// Tokenize on runs of word characters, lowercased — the analogue of Python's
/// `re.findall(r"\w+", text.lower())`.
pub(crate) fn tokenize_lower(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    lower
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}
