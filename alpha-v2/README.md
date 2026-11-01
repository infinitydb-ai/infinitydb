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
when an application has determined that a fact should be replaced. The
operation runs in one SQLite transaction:

1. Verify that the old memory exists in the requested tenant and is still open.
2. Set the old memory's `valid_to` to the replacement timestamp.
3. Insert the successor with `valid_from` set to that same timestamp.
4. Add a tenant-local `supersedes` edge from the old memory to the successor.
5. Optionally attach raw evidence to the successor.

Example:

```python
from infinitydb_next import MemoryStore

with MemoryStore("agent-memory.sqlite3") as store:
    old_id = store.add_memory(
        "The service runs in northstar",
        tenant_id="acme",
        valid_from=1_700_000_000,
    )
    new_id = store.supersede_memory(
        old_id,
        "The service now runs in southstar",
        tenant_id="acme",
        at=1_710_000_000,
        raw_evidence="Approved deployment record #42",
        source_id="deployment-42",
    )

    # Before the boundary, the previous fact remains valid.
    historical = store.search(
        "northstar", tenant_id="acme", as_of=1_705_000_000
    )

    # At the boundary, the successor is valid and the old fact is not.
    current = store.search(
        "southstar", tenant_id="acme", as_of=1_710_000_000
    )
```

The method rejects missing memories, memories that already have a `valid_to`,
non-finite timestamps, and replacement times earlier than the original
`valid_from`. If any write (including evidence capture) fails, the transaction
rolls back, leaving the old memory unchanged.

This is explicit, caller-directed versioning. It does not infer contradictions,
decide which statement is true, or automatically supersede memories. The
application must make that decision and provide the replacement timestamp.
