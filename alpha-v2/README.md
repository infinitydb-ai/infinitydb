# InfinityDB Next — v2 alpha workspace

This is an isolated alpha workspace integrated alongside the existing InfinityDB 1.x implementation. It does not replace the stable `infinitydb` Python package or `infinitydb-js` Node SDK.

## Included capabilities

- SQLite WAL-backed tenant-scoped memory records, validity windows and relationship edges.
- FTS5 lexical retrieval with AND-first matching and a substring fallback when FTS5 is unavailable.
- Tenant-scoped exact cosine vector retrieval for caller-supplied embeddings, with temporal and kind filters.
- Hybrid lexical/vector retrieval using deterministic reciprocal-rank fusion (RRF), preserving tenant, temporal and kind filters across both candidate channels.
- Provider-neutral text embedding protocol and optional local Ollama adapter using the standard-library HTTP client.
- Atomic batch ingestion, evidence capture and SHA-256 evidence-chain verification.
- Context-pack compilation with a conservative character-based token estimate.
- Python reference package, optional REST adapter (see `src/infinitydb_next/api.py`), TypeScript HTTP client and dependency-free Rust score-fusion primitives.

## Install and test

Python:
```bash
cd alpha-v2
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

TypeScript SDK:
```bash
cd alpha-v2/sdk/typescript
npm install
npm test
```

## Explicit alpha limitations

This workspace does **not** implement approximate-nearest-neighbor (ANN) indexing, managed embedding lifecycle/model migration, predictive prefetch, autonomous consolidation, multi-agent federation, production authentication, encryption-at-rest or MCP transport. Exact vector search scans eligible stored embeddings and is intended as a correctness baseline, not a large-scale ANN solution. Callers must generate embeddings and keep model/version consistency themselves. The REST adapter is a development-only interface and must remain bound to localhost until authentication and transport security are implemented. The token estimate is not tokenizer-accurate. Evidence hash chains detect alteration but do not prove authorship.

Benchmark artifacts are workload-specific synthetic measurements; they are not competitor comparisons or production guarantees. See `docs/BENCHMARKS.md` before publishing performance claims.


## Text embeddings (optional local Ollama)

The core package has no mandatory model or network dependency. The Ollama adapter
contacts a local Ollama server only when `embed()` is called:

```python
from infinitydb_next import MemoryStore, OllamaEmbeddingProvider

provider = OllamaEmbeddingProvider("nomic-embed-text")
vectors = provider.embed(["A durable fact to remember", "A second memory"])
print(provider.model_info)  # provider, model and observed dimensions

with MemoryStore("demo.sqlite3") as store:
    memory_id = store.add_memory("A durable fact to remember", tenant_id="demo")
    store.set_embedding(\n        memory_id, vectors[0], tenant_id="demo",\n        provider=provider.model_info.provider, model=provider.model_info.model,\n    )
    print(store.search_vector(vectors[0], tenant_id="demo"))
```

Pull the model separately with `ollama pull nomic-embed-text`. The adapter
uses Ollama's `POST /api/embed` endpoint. You can inject a transport for tests
or custom deployments. Model identity is exposed as metadata but is not yet
persisted alongside each stored vector; do not mix vectors from different
models or revisions in one search space.


### Automatic embedding ingestion

Embedding is opt-in: existing `add_memory()` and `add_memories()` remain
model-free and make no provider calls. To create memories and their vectors
together, use the explicit provider-backed methods:

```python
from infinitydb_next import MemoryStore, OllamaEmbeddingProvider

provider = OllamaEmbeddingProvider("nomic-embed-text")
with MemoryStore("demo.sqlite3") as store:
    memory_id = store.add_memory_with_embedding(
        "A fact to remember", provider, tenant_id="demo",
        model_revision="nomic-embed-text:latest",
    )
    ids = store.add_memories_with_embeddings(
        [
            {"content": "First batch fact", "tenant_id": "demo"},
            {"content": "Second batch fact", "tenant_id": "demo"},
        ],
        provider,
        model_revision="nomic-embed-text:latest",
    )
```

The provider is called before opening the SQLite write transaction. Batch
vectors are validated for cardinality, finite non-zero values and consistent
dimensions; memories, vectors and FTS rows are then committed atomically.
Provider failures therefore create no partial memories. Batches use one
provider call. Model revision is caller-supplied because providers may not
expose an immutable revision identifier. Provider-backed ingestion is
synchronous and does not add automatic retries or background queues.


### Persistent embedding cache

Provider-backed ingestion now caches vectors by tenant, SHA-256 content hash,
provider, model, caller-supplied model revision and preprocessing/configuration
fingerprint. Repeated content within the same tenant and model configuration
reuses the cached vector; identical content in another tenant is embedded
separately. Batch ingestion deduplicates repeated cache misses before calling
the provider. Cache rows are committed atomically with memory and embedding
rows and survive database reopen. Use a new `model_revision` or
`config_fingerprint` whenever model weights or text preprocessing changes.
The cache stores hashes and vectors, not a second copy of raw content.
Concurrent writers may both compute the same cache miss; SQLite uniqueness
keeps one cache row, but provider calls are not single-flight coordinated.


### Bounded graph retrieval

`MemoryStore.traverse_graph(seed_ids, tenant_id=..., max_hops=2)` performs
tenant-scoped breadth-first traversal over graph edges in either direction.
It returns each reachable memory once, at its shortest hop distance, with an
explainable memory-ID path, relation sequence and multiplicative edge score.
Optional relation and valid-time filters are applied during traversal. The
default is a two-hop bounded traversal; this is a graph expansion primitive,
not a claim of learned relevance ranking. Cycles are guarded by a visited set.


### Temporal fact supersession

Use `MemoryStore.supersede_memory(memory_id, new_content, tenant_id=...)`
when a fact is explicitly replaced. The operation atomically sets the old
record's `valid_to`, creates the successor at the same valid-time boundary,
and adds a tenant-local `supersedes` edge. Historical queries before the
boundary can still retrieve the old fact; queries at or after it see the
successor. The original evidence remains intact, and optional new evidence is
attached to the successor. This is explicit versioning, not automatic
contradiction detection: callers decide when two facts conflict.
