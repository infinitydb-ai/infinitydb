import tempfile
import time
from pathlib import Path

import pytest

from infinitydb_next import MemoryStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as directory:
        instance = MemoryStore(Path(directory) / "test.sqlite")
        yield instance
        instance.close()


def test_round_trip_and_tenant_isolation(store):
    memory_id = store.add_memory("local-first AI infrastructure", tenant_id="tenant-a")
    assert store.get(memory_id, tenant_id="tenant-a").content == "local-first AI infrastructure"
    assert store.get(memory_id, tenant_id="tenant-b") is None
    assert store.search("local-first", tenant_id="tenant-a")
    assert store.search("local-first", tenant_id="tenant-b") == []


def test_evidence_chain_verifies(store):
    memory_id = store.add_memory("deployment succeeded", raw_evidence="stdout: success")
    assert store.evidence_for(memory_id)
    assert store.verify_evidence_chain()


def test_temporal_validity(store):
    now = time.time()
    old_id = store.add_memory("employer is Alpha", valid_from=now - 100, valid_to=now - 10)
    new_id = store.add_memory("employer is Beta", valid_from=now - 9)
    assert store.search("employer", as_of=now - 50)[0].memory.id == old_id
    assert store.search("employer", as_of=now)[0].memory.id == new_id


def test_context_budget(store):
    store.add_memory("short retrieval note")
    store.add_memory("long retrieval note " * 100)
    pack = store.context_pack("retrieval", token_budget=10)
    assert pack.estimated_tokens <= 10


def test_validation_and_atomic_batch(store):
    with pytest.raises(ValueError):
        store.add_memory(" ")
    with pytest.raises(ValueError):
        store.add_memory("x", confidence=2)
    with pytest.raises(ValueError):
        store.add_memories([{"content": "valid"}, {"content": " "}])
    assert store.stats()["memories"] == 0


def test_batch_ingestion_and_kind_filter(store):
    ids = store.add_memories([
        {"content": "Rust SIMD retrieval engine", "tenant_id": "batch", "kind": "fact"},
        {"content": "SQLite temporal memory", "tenant_id": "batch", "kind": "episode"},
    ])
    hits = store.search("Rust retrieval", tenant_id="batch", kinds=["fact"])
    assert hits and hits[0].memory.id == ids[0]
    assert all(hit.memory.kind == "fact" for hit in hits)


def test_edges_require_both_endpoints_in_same_tenant(store):
    source = store.add_memory("source fact", tenant_id="tenant-a")
    destination = store.add_memory("destination fact", tenant_id="tenant-a")
    foreign = store.add_memory("foreign fact", tenant_id="tenant-b")

    store.add_edge(source, destination, "supports", tenant_id="tenant-a", weight=0.5)
    assert store.stats(tenant_id="tenant-a")["edges"] == 1

    with pytest.raises(ValueError, match="both edge endpoints"):
        store.add_edge(source, foreign, "leaks", tenant_id="tenant-a")
    with pytest.raises(ValueError, match="both edge endpoints"):
        store.add_edge(source, "missing-id", "dangling", tenant_id="tenant-a")
    assert store.stats(tenant_id="tenant-a")["edges"] == 1
    assert store.stats(tenant_id="tenant-b")["edges"] == 0


@pytest.mark.parametrize("weight", [-0.1, float("inf"), float("-inf"), float("nan")])
def test_edges_reject_invalid_weights(store, weight):
    source = store.add_memory("source")
    destination = store.add_memory("destination")
    with pytest.raises(ValueError, match="finite non-negative"):
        store.add_edge(source, destination, "related", weight=weight)
    assert store.stats()["edges"] == 0


def test_edges_reject_empty_relation(store):
    source = store.add_memory("source")
    destination = store.add_memory("destination")
    with pytest.raises(ValueError, match="relation must not be empty"):
        store.add_edge(source, destination, " ")
    assert store.stats()["edges"] == 0


def test_reopen_repairs_partial_fts_index():
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "repair.sqlite"
        first = MemoryStore(database)
        present_id = first.add_memory(
            "rebuildable tenant search phrase", tenant_id="tenant-a"
        )
        missing_id = first.add_memory(
            "recover missing index entry", tenant_id="tenant-b"
        )
        first.db.execute(
            "DELETE FROM memory_fts_v2 WHERE memory_id=?", (missing_id,)
        )
        first.db.commit()
        first.close()

        reopened = MemoryStore(database)
        try:
            assert reopened.search(
                "recover missing", tenant_id="tenant-b"
            )[0].memory.id == missing_id
            assert reopened.search(
                "rebuildable tenant", tenant_id="tenant-a"
            )[0].memory.id == present_id
            assert reopened.db.execute(
                "SELECT count(*) FROM memory_fts_v2"
            ).fetchone()[0] == 2
        finally:
            reopened.close()


def test_exact_vector_search_is_tenant_scoped_and_ranked(store):
    near = store.add_memory("near semantic match", tenant_id="tenant-a")
    far = store.add_memory("far semantic match", tenant_id="tenant-a")
    foreign = store.add_memory("foreign semantic match", tenant_id="tenant-b")
    store.set_embedding(near, [1.0, 0.0], tenant_id="tenant-a")
    store.set_embedding(far, [0.0, 1.0], tenant_id="tenant-a")
    store.set_embedding(foreign, [1.0, 0.0], tenant_id="tenant-b")

    hits = store.search_vector([0.9, 0.1], tenant_id="tenant-a")
    assert [hit.memory.id for hit in hits] == [near, far]
    assert hits[0].matched_by == ["exact_vector"]
    assert store.search_vector([1.0, 0.0], tenant_id="tenant-b")[0].memory.id == foreign


def test_vector_search_respects_temporal_and_kind_filters(store):
    expired_at = time.time()
    expired = store.add_memory(
        "expired vector memory", kind="fact",
        valid_from=expired_at - 20, valid_to=expired_at - 1,
    )
    current = store.add_memory("current vector memory", kind="episode")
    store.set_embedding(expired, [1.0, 0.0])
    store.set_embedding(current, [1.0, 0.0])

    # Capture the query time after insertion: add_memory assigns valid_from
    # using its own clock, so an earlier timestamp can precede this memory.
    as_of = time.time()
    hits = store.search_vector([1.0, 0.0], kinds=["episode"], as_of=as_of)
    assert [hit.memory.id for hit in hits] == [current]


@pytest.mark.parametrize("vector", [[], [0.0, 0.0], [float("nan")], [float("inf")]])
def test_vector_validation_rejects_invalid_embeddings(store, vector):
    memory_id = store.add_memory("vector target")
    with pytest.raises(ValueError, match="embedding"):
        store.set_embedding(memory_id, vector)


def test_embedding_cannot_be_written_across_tenants(store):
    memory_id = store.add_memory("private memory", tenant_id="tenant-a")
    with pytest.raises(ValueError, match="specified tenant"):
        store.set_embedding(memory_id, [1.0, 0.0], tenant_id="tenant-b")
    assert store.search_vector([1.0, 0.0], tenant_id="tenant-a") == []


def test_vector_embeddings_persist_after_reopen():
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "vectors.sqlite"
        first = MemoryStore(database)
        memory_id = first.add_memory("persistent vector memory", tenant_id="persist")
        first.set_embedding(memory_id, [0.25, 0.75], tenant_id="persist")
        first.close()

        reopened = MemoryStore(database)
        try:
            hits = reopened.search_vector([0.25, 0.75], tenant_id="persist")
            assert len(hits) == 1
            assert hits[0].memory.id == memory_id
        finally:
            reopened.close()


def test_hybrid_search_fuses_lexical_and_vector_rankings(store):
    lexical = store.add_memory("alpha lexical retrieval result", tenant_id="tenant-a")
    semantic = store.add_memory("unrelated wording", tenant_id="tenant-a")
    foreign = store.add_memory("alpha lexical retrieval result", tenant_id="tenant-b")
    store.set_embedding(lexical, [0.0, 1.0], tenant_id="tenant-a")
    store.set_embedding(semantic, [1.0, 0.0], tenant_id="tenant-a")
    store.set_embedding(foreign, [1.0, 0.0], tenant_id="tenant-b")

    hits = store.search_hybrid(
        "alpha lexical", [1.0, 0.0], tenant_id="tenant-a", limit=10
    )
    by_id = {hit.memory.id: hit for hit in hits}
    assert set(by_id) == {lexical, semantic}
    assert by_id[lexical].matched_by == ["exact_vector", "fts5_and"]
    assert by_id[semantic].matched_by == ["exact_vector"]
    assert foreign not in by_id


def test_hybrid_search_applies_temporal_and_kind_filters_to_both_channels(store):
    expired_at = time.time()
    store.add_memory(
        "expired lexical semantic", kind="fact",
        valid_from=expired_at - 20, valid_to=expired_at - 1,
    )
    current = store.add_memory("current lexical semantic", kind="episode")
    store.set_embedding(current, [1.0, 0.0])
    hits = store.search_hybrid(
        "current lexical", [1.0, 0.0], kinds=["episode"], as_of=time.time()
    )
    assert [hit.memory.id for hit in hits] == [current]


def test_hybrid_search_validates_fusion_parameters(store):
    with pytest.raises(ValueError, match="rrf_k"):
        store.search_hybrid("query", [1.0, 0.0], rrf_k=0)
    with pytest.raises(ValueError, match="candidate_limit"):
        store.search_hybrid("query", [1.0, 0.0], limit=5, candidate_limit=2)


def test_embedding_metadata_filters_incompatible_model_spaces(store):
    first = store.add_memory("first model vector", tenant_id="models")
    second = store.add_memory("second model vector", tenant_id="models")
    store.set_embedding(
        first, [1.0, 0.0], tenant_id="models",
        provider="ollama", model="nomic-embed-text", model_revision="v1",
    )
    store.set_embedding(
        second, [0.0, 1.0], tenant_id="models",
        provider="ollama", model="mxbai-embed-large", model_revision="v2",
    )

    hits = store.search_vector(
        [1.0, 0.0], tenant_id="models",
        provider="ollama", model="nomic-embed-text", model_revision="v1",
    )
    assert [hit.memory.id for hit in hits] == [first]
    assert store.search_vector(
        [1.0, 0.0], tenant_id="models", provider="ollama",
        model="missing-model",
    ) == []


def test_embedding_revision_filter_requires_provider_and_model(store):
    with pytest.raises(ValueError, match="provider and model"):
        store.search_vector([1.0, 0.0], model_revision="v1")


def test_embedding_metadata_migration_is_additive_and_idempotent():
    import sqlite3

    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "legacy.sqlite"
        connection = sqlite3.connect(database)
        connection.executescript("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, kind TEXT NOT NULL,
                content TEXT NOT NULL, created_at REAL NOT NULL, valid_from REAL NOT NULL,
                valid_to REAL, metadata_json TEXT NOT NULL, source_id TEXT,
                confidence REAL NOT NULL, content_hash TEXT NOT NULL
            );
            CREATE TABLE embeddings (
                memory_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL, vector_json TEXT NOT NULL
            );
            INSERT INTO memories VALUES(
                'legacy-id','legacy','fact','legacy vector',1,1,NULL,'{}',NULL,1,'hash'
            );
            INSERT INTO embeddings VALUES('legacy-id','legacy',2,'[1.0,0.0]');
        """)
        connection.commit()
        connection.close()

        first = MemoryStore(database)
        try:
            columns = {row["name"] for row in first.db.execute("PRAGMA table_info(embeddings)")}
            assert {"provider", "model", "model_revision"} <= columns
            metadata = first.db.execute(
                "SELECT provider,model,model_revision FROM embeddings WHERE memory_id='legacy-id'"
            ).fetchone()
            assert tuple(metadata) == ("unknown", "unknown", None)
        finally:
            first.close()

        second = MemoryStore(database)
        try:
            columns = [row["name"] for row in second.db.execute("PRAGMA table_info(embeddings)")]
            assert columns.count("provider") == 1
            assert second.search_vector([1.0, 0.0], tenant_id="legacy")
        finally:
            second.close()


class FakeEmbeddingProvider:
    def __init__(self, vectors, *, model="test-embed"):
        from infinitydb_next import EmbeddingModel
        self.vectors = vectors
        self.calls = []
        self._model_info = EmbeddingModel("fake", model)

    @property
    def model_info(self):
        return self._model_info

    def embed(self, texts):
        self.calls.append(list(texts))
        if isinstance(self.vectors, Exception):
            raise self.vectors
        return self.vectors


def test_add_memory_with_embedding_persists_vector_and_model_atomically(store):
    provider = FakeEmbeddingProvider([[1.0, 0.0]])
    memory_id = store.add_memory_with_embedding(
        "automatically embedded memory", provider, tenant_id="auto",
        raw_evidence="source transcript",
    )
    assert provider.calls == [["automatically embedded memory"]]
    hits = store.search_vector(
        [1.0, 0.0], tenant_id="auto", provider="fake", model="test-embed",
    )
    assert [hit.memory.id for hit in hits] == [memory_id]
    assert store.evidence_for(memory_id, tenant_id="auto")


def test_add_memories_with_embeddings_batches_provider_call_and_persists(store):
    provider = FakeEmbeddingProvider([[1.0, 0.0], [0.0, 1.0]])
    ids = store.add_memories_with_embeddings([
        {"content": "first batch memory", "tenant_id": "batch-auto"},
        {"content": "second batch memory", "tenant_id": "batch-auto"},
    ], provider)
    assert provider.calls == [["first batch memory", "second batch memory"]]
    hits = store.search_vector(
        [1.0, 0.0], tenant_id="batch-auto", provider="fake", model="test-embed",
    )
    assert {hit.memory.id for hit in hits} == set(ids)


def test_batch_embedding_provider_failure_writes_no_memories(store):
    provider = FakeEmbeddingProvider(RuntimeError("provider unavailable"))
    with pytest.raises(RuntimeError, match="provider unavailable"):
        store.add_memories_with_embeddings([{"content": "must not persist"}], provider)
    assert store.stats()["memories"] == 0


@pytest.mark.parametrize("vectors", [[], [[1.0, 0.0], [1.0]]])
def test_batch_embedding_rejects_bad_cardinality_or_dimensions_atomically(store, vectors):
    provider = FakeEmbeddingProvider(vectors)
    with pytest.raises(ValueError, match="exactly one|inconsistent dimensions"):
        store.add_memories_with_embeddings([
            {"content": "first invalid batch"},
            {"content": "second invalid batch"},
        ], provider)
    assert store.stats()["memories"] == 0


def test_single_embedding_failure_does_not_create_memory(store):
    provider = FakeEmbeddingProvider(RuntimeError("offline"))
    with pytest.raises(RuntimeError, match="offline"):
        store.add_memory_with_embedding("not persisted", provider)
    assert store.stats()["memories"] == 0


def test_embedding_cache_reuses_vector_for_same_tenant_and_model(store):
    provider = FakeEmbeddingProvider([[0.2, 0.8]])
    first = store.add_memory_with_embedding("reusable fact", provider, tenant_id="cache-a")
    second = store.add_memory_with_embedding("reusable fact", provider, tenant_id="cache-a")
    assert first != second
    assert provider.calls == [["reusable fact"]]
    assert store.search_vector([0.2, 0.8], tenant_id="cache-a",
                               provider="fake", model="test-embed")
    assert store.db.execute("SELECT count(*) FROM embedding_cache").fetchone()[0] == 1


def test_embedding_cache_is_tenant_isolated(store):
    provider = FakeEmbeddingProvider([[0.2, 0.8]])
    store.add_memory_with_embedding("private reusable fact", provider, tenant_id="tenant-a")
    store.add_memory_with_embedding("private reusable fact", provider, tenant_id="tenant-b")
    assert provider.calls == [["private reusable fact"], ["private reusable fact"]]
    assert store.db.execute("SELECT count(*) FROM embedding_cache").fetchone()[0] == 2


def test_embedding_cache_is_model_revision_and_config_scoped(store):
    provider = FakeEmbeddingProvider([[1.0, 0.0]])
    store.add_memory_with_embedding("versioned fact", provider, tenant_id="versions",
                                    model_revision="r1")
    store.add_memory_with_embedding("versioned fact", provider, tenant_id="versions",
                                    model_revision="r2")
    store.add_memory_with_embedding("versioned fact", provider, tenant_id="versions",
                                    model_revision="r2", config_fingerprint="normalized-v2")
    assert len(provider.calls) == 3
    assert store.db.execute("SELECT count(*) FROM embedding_cache").fetchone()[0] == 3


def test_batch_embedding_cache_deduplicates_within_tenant_but_not_across_tenants(store):
    provider = FakeEmbeddingProvider([[1.0, 0.0], [0.0, 1.0]])
    ids = store.add_memories_with_embeddings([
        {"content": "duplicate batch item", "tenant_id": "same", "valid_from": 1.0},
        {"content": "duplicate batch item", "tenant_id": "same", "valid_from": 2.0},
        {"content": "duplicate batch item", "tenant_id": "different", "valid_from": 3.0},
    ], provider)
    assert len(ids) == 3
    assert provider.calls == [["duplicate batch item", "duplicate batch item"]]
    assert store.db.execute("SELECT count(*) FROM embedding_cache").fetchone()[0] == 2


def test_embedding_cache_survives_reopen():
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "cache.sqlite"
        first = MemoryStore(database)
        provider = FakeEmbeddingProvider([[0.3, 0.7]])
        first.add_memory_with_embedding("persist cached vector", provider, tenant_id="persist")
        first.close()
        second = MemoryStore(database)
        try:
            again = FakeEmbeddingProvider([[0.9, 0.1]])
            second.add_memory_with_embedding("persist cached vector", again, tenant_id="persist")
            assert again.calls == []
            hit = second.search_vector([0.3, 0.7], tenant_id="persist",
                                       provider="fake", model="test-embed")
            assert len(hit) == 2
        finally:
            second.close()
