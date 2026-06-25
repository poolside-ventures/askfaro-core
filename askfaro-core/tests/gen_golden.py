"""Generate golden rankings from the Python contract lib (the source of truth).

Indexes a curated corpus with the deterministic bag-of-words embedder + the
SQLite backend, runs a set of queries, and dumps each query's ranked
[(object_id, match_type)] to golden.json. The Rust `conformance` test must
reproduce this exactly — that is the executable form of "index server-side,
retrieve on-device, get the same results."

Run:  PYTHONPATH=/Users/vjong/Development/faro-embedded-search \
      python3 core-search/tests/gen_golden.py
"""
import asyncio, hashlib, json, math, re, os, sys

sys.path.insert(0, "/Users/vjong/Development/faro-embedded-search")

from askfaro_embedded_search import IndexDoc, SearchIndex, CallableEmbedder
from askfaro_embedded_search.backends.sqlite import SQLiteBackend

DIM = 512

def bow_vector(text):
    vec = [0.0] * DIM
    for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.md5(token.encode()).digest()
        vec[int.from_bytes(digest[:4], "big") % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec

# Curated so each query has an unambiguous order (robust to f32 tie noise).
CORPUS = [
    ("note", "n1", "Grocery list", "milk eggs bread butter", {"folder": "home"}),
    ("note", "n2", "Quantum entanglement", "spooky action at a distance physics", {"folder": "science"}),
    ("note", "n3", "Bread recipe", "flour water yeast salt baking oven", {"folder": "home"}),
    ("task", "t1", "Fix login bug", "oauth redirect loop authentication", {"folder": "work"}),
    ("task", "t2", "Quarterly planning", "roadmap milestones objectives", {"folder": "work"}),
    ("contact", "c1", "Ada Lovelace", "mathematician analytical engine", {"folder": "people"}),
    ("contact", "c2", "Grace Hopper", "compiler nanosecond navy programming", {"folder": "people"}),
    ("email", "e1", "Invoice due", "payment reminder client overdue", {"folder": "mail"}),
    ("tool", "tl1", "Stripe payments", "charge card customer checkout", {"folder": "finance"}),
]
QUERIES = [
    {"query": "groceries"},
    {"query": "bread baking yeast"},
    {"query": "quantum physics distance"},
    {"query": "roadmap planning", "object_types": ["task"]},
    {"query": "yeast", "attrs": {"folder": "home"}},
    {"query": "oauth authentication login"},
    {"query": "compiler programming"},
    {"query": "card payment checkout"},
]

async def main():
    backend = SQLiteBackend(":memory:", spaces=("default",))
    index = SearchIndex(backend, CallableEmbedder(lambda ts: [bow_vector(t) for t in ts]))
    await index.upsert_many([
        IndexDoc(object_type=o, object_id=i, title=t, body=b, attrs=a)
        for o, i, t, b, a in CORPUS
    ])
    out = []
    for q in QUERIES:
        res = await index.search(**q, k=5)
        out.append({"query": q, "ranked": [[r.object_id, r.match_type] for r in res]})
    await index.close()

    path = os.path.join(os.path.dirname(__file__), "golden.json")
    json.dump({"corpus": CORPUS, "queries": out}, open(path, "w"), indent=2)
    for row in out:
        print(row["query"]["query"], "->", row["ranked"])
    print("wrote", path)

asyncio.run(main())
