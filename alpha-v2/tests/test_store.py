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



def test_supersede_memory_preserves_history_and_links_successor(store):
    original = store.add_memory("The service runs in northstar", tenant_id="history",
                                valid_from=10.0)
    successor = store.supersede_memory(
        original, "The service now runs in southstar", tenant_id="history",
        at=20.0, raw_evidence="deployment record 42",
    )

    assert store.get(original, tenant_id="history").valid_to == 20.0
    current = store.get(successor, tenant_id="history")
    assert current.content == "The service now runs in southstar"
    assert current.valid_from == 20.0
    assert store.search("northstar", tenant_id="history", as_of=19.0)
    assert not store.search("northstar", tenant_id="history", as_of=20.0)
    assert store.search("southstar", tenant_id="history", as_of=20.0)
    edge = store.db.execute(
        "SELECT relation FROM edges WHERE tenant_id=? AND src_id=? AND dst_id=?",
        ("history", original, successor),
    ).fetchone()
    assert edge["relation"] == "supersedes"
    assert store.evidence_for(successor, tenant_id="history")[0]["raw_content"] == "deployment record 42"


def test_supersede_memory_rejects_missing_closed_and_invalid_time(store):
    with pytest.raises(ValueError, match="does not exist"):
        store.supersede_memory("missing", "replacement", tenant_id="history")

    original = store.add_memory("original fact", tenant_id="history", valid_from=10.0)
    store.supersede_memory(original, "replacement", tenant_id="history", at=20.0)
    with pytest.raises(ValueError, match="already superseded"):
        store.supersede_memory(original, "another replacement", tenant_id="history", at=30.0)

    other = store.add_memory("future fact", tenant_id="history", valid_from=50.0)
    with pytest.raises(ValueError, match="cannot precede"):
        store.supersede_memory(other, "too early", tenant_id="history", at=40.0)

    with pytest.raises(ValueError, match="finite"):
        store.supersede_memory(other, "not a timestamp", tenant_id="history", at=float("nan"))


def test_supersede_memory_rolls_back_all_writes_on_failure(store, monkeypatch):
    original = store.add_memory("rollback baseline", tenant_id="history")
    def fail(*args, **kwargs):
        raise RuntimeError("evidence write failed")
    monkeypatch.setattr(store, "_append_evidence", fail)
    with pytest.raises(RuntimeError, match="evidence write failed"):
        store.supersede_memory(original, "successorphantom", tenant_id="history",
                               raw_evidence="failure")
    assert store.get(original, tenant_id="history").valid_to is None
    assert store.search("successorphantom", tenant_id="history") == []

def test_bounded_graph_traversal_returns_explainable_shortest_paths(store):
    seed = store.add_memory("seed", tenant_id="graph")
    middle = store.add_memory("middle", tenant_id="graph")
    target = store.add_memory("target", tenant_id="graph")
    store.add_edge(seed, middle, "supports", tenant_id="graph", weight=0.8)
    store.add_edge(middle, target, "causes", tenant_id="graph", weight=0.5)

    hits = store.traverse_graph([seed], tenant_id="graph", max_hops=2)
    assert [(hit.memory.id, hit.depth) for hit in hits] == [(middle, 1), (target, 2)]
    target_hit = hits[1]
    assert target_hit.path == [seed, middle, target]
    assert target_hit.relations == ["supports", "causes"]
    assert target_hit.score == pytest.approx(0.4)


def test_graph_traversal_filters_relations_and_temporal_memories(store):
    seed = store.add_memory("graph seed", tenant_id="graph-filter", valid_from=1.0)
    included = store.add_memory("included node", tenant_id="graph-filter", valid_from=1.0)
    excluded = store.add_memory("excluded node", tenant_id="graph-filter", valid_from=1.0)
    expired = store.add_memory("expired node", tenant_id="graph-filter",
                               valid_from=1.0, valid_to=2.0)
    store.add_edge(seed, included, "supports", tenant_id="graph-filter")
    store.add_edge(seed, excluded, "mentions", tenant_id="graph-filter")
    store.add_edge(seed, expired, "supports", tenant_id="graph-filter")

    hits = store.traverse_graph([seed], tenant_id="graph-filter", max_hops=1,
                                relations=["supports"], as_of=3.0)
    assert [hit.memory.id for hit in hits] == [included]


def test_graph_traversal_is_tenant_scoped_and_cycle_safe(store):
    seed = store.add_memory("tenant A seed", tenant_id="tenant-a")
    neighbor = store.add_memory("tenant A neighbor", tenant_id="tenant-a")
    foreign = store.add_memory("tenant B node", tenant_id="tenant-b")
    store.add_edge(seed, neighbor, "related", tenant_id="tenant-a")
    foreign_peer = store.add_memory("tenant B peer", tenant_id="tenant-b")
    store.add_edge(foreign, foreign_peer, "related", tenant_id="tenant-b")
    store.add_edge(neighbor, seed, "back", tenant_id="tenant-a")

    hits = store.traverse_graph([seed], tenant_id="tenant-a", max_hops=5)
    assert [hit.memory.id for hit in hits] == [neighbor]
    assert all(hit.memory.tenant_id == "tenant-a" for hit in hits)


def test_graph_traversal_validates_hops_and_empty_seeds(store):
    with pytest.raises(ValueError, match="max_hops"):
        store.traverse_graph(["missing"], max_hops=-1)
    assert store.traverse_graph([], max_hops=2) == []
    assert store.traverse_graph(["missing"], max_hops=2) == []


def test_vector_top_k_is_exact_and_deterministic_for_ties(store):
    ids = store.add_memories([
        {"content": "vector candidate c", "tenant_id": "top-k", "memory_id": "c", "created_at": 10.0, "valid_from": 1.0},
        {"content": "vector candidate b", "tenant_id": "top-k", "memory_id": "b", "created_at": 10.0, "valid_from": 1.0},
        {"content": "vector candidate a", "tenant_id": "top-k", "memory_id": "a", "created_at": 10.0, "valid_from": 1.0},
    ])
    for memory_id in ids:
        store.set_embedding(memory_id, [1.0, 0.0], tenant_id="top-k", provider="fake", model="ranker")

    hits = store.search_vector(
        [1.0, 0.0], tenant_id="top-k", limit=2, provider="fake", model="ranker"
    )
    assert [hit.memory.id for hit in hits] == ["a", "b"]
    assert [hit.score for hit in hits] == [1.0, 1.0]


def test_vector_top_k_respects_model_revision_and_tenant_filters(store):
    first = store.add_memory("revision one vector", tenant_id="revision", valid_from=1.0)
    second = store.add_memory("revision two vector", tenant_id="revision", valid_from=1.0)
    foreign = store.add_memory("foreign tenant vector", tenant_id="other", valid_from=1.0)
    store.set_embedding(first, [1.0, 0.0], tenant_id="revision",
                        provider="fake", model="ranker", model_revision="r1")
    store.set_embedding(second, [0.9, 0.1], tenant_id="revision",
                        provider="fake", model="ranker", model_revision="r2")
    store.set_embedding(foreign, [1.0, 0.0], tenant_id="other",
                        provider="fake", model="ranker", model_revision="r1")

    hits = store.search_vector(
        [1.0, 0.0], tenant_id="revision", provider="fake", model="ranker",
        model_revision="r2", limit=10,
    )
    assert [hit.memory.id for hit in hits] == [second]

class FakeHNSWIndex:
    init_calls = 0
    load_calls = 0

    def __init__(self, space, dim):
        assert space == "cosine"
        self.dim = dim
        self.vectors = {}

    def init_index(self, max_elements, ef_construction, M):
        type(self).init_calls += 1
        self.max_elements = max_elements

    def resize_index(self, max_elements):
        self.max_elements = max_elements

    def add_items(self, vectors, labels, num_threads=1):
        for vector, label in zip(vectors, labels):
            self.vectors[int(label)] = list(vector)

    def set_ef(self, ef):
        self.ef = ef

    def save_index(self, path):
        import json

        Path(path).write_text(json.dumps(self.vectors), encoding="utf-8")

    def load_index(self, path, max_elements=None):
        import json

        type(self).load_calls += 1
        self.vectors = {
            int(label): vector
            for label, vector in json.loads(Path(path).read_text(encoding="utf-8")).items()
        }

    def knn_query(self, queries, k, num_threads=1):
        import math

        query = list(queries[0])
        qnorm = math.sqrt(sum(value * value for value in query))
        ranked = []
        for label, vector in self.vectors.items():
            norm = math.sqrt(sum(value * value for value in vector))
            cosine = sum(a * b for a, b in zip(query, vector)) / (qnorm * norm)
            ranked.append((1.0 - cosine, label))
        ranked.sort()
        selected = [label for _, label in ranked[:k]]
        return [selected], [[distance for distance, _ in ranked[:k]]]


def test_ann_search_isolated_by_tenant_and_embedding_model(store, monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    a_best = store.add_memory("tenant a nearest", tenant_id="ann-a")
    a_other = store.add_memory("tenant a other model", tenant_id="ann-a")
    b_best = store.add_memory("tenant b nearest", tenant_id="ann-b")
    store.set_embedding(a_best, [1.0, 0.0], tenant_id="ann-a", provider="p", model="m")
    store.set_embedding(a_other, [0.0, 1.0], tenant_id="ann-a", provider="p", model="other")
    store.set_embedding(b_best, [1.0, 0.0], tenant_id="ann-b", provider="p", model="m")

    hits_a = store.search_vector_ann(
        [1.0, 0.0], tenant_id="ann-a", provider="p", model="m", limit=5
    )
    hits_b = store.search_vector_ann(
        [1.0, 0.0], tenant_id="ann-b", provider="p", model="m", limit=5
    )
    assert [hit.memory.id for hit in hits_a] == [a_best]
    assert [hit.memory.id for hit in hits_b] == [b_best]
    assert hits_a[0].matched_by == ["ann_vector"]


def test_ann_index_is_invalidated_when_embedding_changes(store, monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    memory_id = store.add_memory("mutable vector", tenant_id="ann-refresh")
    store.set_embedding(
        memory_id, [0.0, 1.0], tenant_id="ann-refresh", provider="p", model="m"
    )
    first = store.search_vector_ann(
        [1.0, 0.0], tenant_id="ann-refresh", provider="p", model="m"
    )
    assert first[0].score == pytest.approx(0.0)

    store.set_embedding(
        memory_id, [1.0, 0.0], tenant_id="ann-refresh", provider="p", model="m"
    )
    second = store.search_vector_ann(
        [1.0, 0.0], tenant_id="ann-refresh", provider="p", model="m"
    )
    assert second[0].score == pytest.approx(1.0)


def test_ann_search_applies_current_temporal_and_kind_filters(store, monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    old = store.add_memory(
        "old fact", tenant_id="ann-filters", kind="fact",
        valid_from=1.0, valid_to=5.0,
    )
    current = store.add_memory(
        "current episode", tenant_id="ann-filters", kind="episode",
        valid_from=5.0,
    )
    store.set_embedding(old, [1.0, 0.0], tenant_id="ann-filters", provider="p", model="m")
    store.set_embedding(current, [0.9, 0.1], tenant_id="ann-filters", provider="p", model="m")

    hits = store.search_vector_ann(
        [1.0, 0.0], tenant_id="ann-filters", provider="p", model="m",
        kinds=["episode"], as_of=10.0, limit=5,
    )
    assert [hit.memory.id for hit in hits] == [current]

def test_ann_cache_invalidated_by_provider_backed_single_and_batch_ingestion(store, monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    first = store.add_memory_with_embedding(
        "first provider-backed memory",
        FakeEmbeddingProvider([[0.0, 1.0]]),
        tenant_id="ann-ingest",
    )
    before = store.search_vector_ann(
        [1.0, 0.0], tenant_id="ann-ingest", provider="fake", model="test-embed"
    )
    assert [hit.memory.id for hit in before] == [first]

    second = store.add_memory_with_embedding(
        "second provider-backed memory",
        FakeEmbeddingProvider([[1.0, 0.0]]),
        tenant_id="ann-ingest",
    )
    after_single = store.search_vector_ann(
        [1.0, 0.0], tenant_id="ann-ingest", provider="fake", model="test-embed"
    )
    assert [hit.memory.id for hit in after_single][0] == second

    ids = store.add_memories_with_embeddings(
        [{"content": "third provider-backed memory", "tenant_id": "ann-ingest"}],
        FakeEmbeddingProvider([[0.8, 0.2]]),
    )
    after_batch = store.search_vector_ann(
        [1.0, 0.0], tenant_id="ann-ingest", provider="fake", model="test-embed"
    )
    assert ids[0] in [hit.memory.id for hit in after_batch]




def test_persistent_ann_index_loads_after_store_reopen(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    FakeHNSWIndex.load_calls = 0;
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "persistent.sqlite"
        index_dir = Path(directory) / "ann"
        first = MemoryStore(database, ann_index_dir=index_dir)
        memory_id = first.add_memory("persistent ANN memory", tenant_id="persistent")
        first.set_embedding(
            memory_id, [1.0, 0.0], tenant_id="persistent", provider="p", model="m"
        )
        first.search_vector_ann(
            [1.0, 0.0], tenant_id="persistent", provider="p", model="m"
        )
        assert list(index_dir.glob("*.hnsw"))
        assert list(index_dir.glob("*.json"))
        first.close()

        second = MemoryStore(database, ann_index_dir=index_dir)
        hits = second.search_vector_ann(
            [1.0, 0.0], tenant_id="persistent", provider="p", model="m"
        )
        assert [hit.memory.id for hit in hits] == [memory_id]
        assert FakeHNSWIndex.load_calls == 1
        second.close()


def test_persistent_ann_rebuilds_when_sqlite_embeddings_change(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "stale.sqlite"
        index_dir = Path(directory) / "ann"
        first = MemoryStore(database, ann_index_dir=index_dir)
        memory_id = first.add_memory("stale ANN memory", tenant_id="stale")
        first.set_embedding(
            memory_id, [0.0, 1.0], tenant_id="stale", provider="p", model="m"
        )
        first.search_vector_ann(
            [1.0, 0.0], tenant_id="stale", provider="p", model="m"
        )
        first.close()

        second = MemoryStore(database, ann_index_dir=index_dir)
        second.set_embedding(
            memory_id, [1.0, 0.0], tenant_id="stale", provider="p", model="m"
        )
        hits = second.search_vector_ann(
            [1.0, 0.0], tenant_id="stale", provider="p", model="m"
        )
        assert hits[0].score == pytest.approx(1.0)
        second.close()


def test_persistent_ann_rebuilds_corrupt_index_file(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "corrupt.sqlite"
        index_dir = Path(directory) / "ann"
        first = MemoryStore(database, ann_index_dir=index_dir)
        memory_id = first.add_memory("corrupt ANN memory", tenant_id="corrupt")
        first.set_embedding(
            memory_id, [1.0, 0.0], tenant_id="corrupt", provider="p", model="m"
        )
        first.search_vector_ann(
            [1.0, 0.0], tenant_id="corrupt", provider="p", model="m"
        )
        first.close()
        index_file = next(index_dir.glob("*.hnsw"))
        index_file.write_text("corrupted", encoding="utf-8")

        second = MemoryStore(database, ann_index_dir=index_dir)
        hits = second.search_vector_ann(
            [1.0, 0.0], tenant_id="corrupt", provider="p", model="m"
        )
        assert [hit.memory.id for hit in hits] == [memory_id]
        second.close()



def test_single_embedding_ingestion_updates_warm_ann_index_incrementally(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    FakeHNSWIndex.init_calls = 0
    with tempfile.TemporaryDirectory() as directory:
        store = MemoryStore(Path(directory) / "incremental.sqlite")
        first = store.add_memory_with_embedding(
            "initial warm index item", FakeEmbeddingProvider([[0.0, 1.0]]),
            tenant_id="incremental",
        )
        store.search_vector_ann(
            [1.0, 0.0], tenant_id="incremental", provider="fake", model="test-embed"
        )
        assert FakeHNSWIndex.init_calls == 1

        second = store.add_memory_with_embedding(
            "incremental appended item", FakeEmbeddingProvider([[1.0, 0.0]]),
            tenant_id="incremental",
        )
        hits = store.search_vector_ann(
            [1.0, 0.0], tenant_id="incremental", provider="fake", model="test-embed"
        )
        assert hits[0].memory.id == second
        assert first in [hit.memory.id for hit in hits]
        assert FakeHNSWIndex.init_calls == 1
        store.close()


def test_batch_embedding_ingestion_updates_warm_ann_index_incrementally(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    FakeHNSWIndex.init_calls = 0
    with tempfile.TemporaryDirectory() as directory:
        store = MemoryStore(Path(directory) / "batch-incremental.sqlite")
        store.add_memory_with_embedding(
            "batch initial item", FakeEmbeddingProvider([[0.0, 1.0]]),
            tenant_id="batch-incremental",
        )
        store.search_vector_ann(
            [1.0, 0.0], tenant_id="batch-incremental",
            provider="fake", model="test-embed",
        )
        appended = store.add_memories_with_embeddings(
            [
                {"content": "batch appended one", "tenant_id": "batch-incremental"},
                {"content": "batch appended two", "tenant_id": "batch-incremental"},
            ],
            FakeEmbeddingProvider([[1.0, 0.0], [0.8, 0.2]]),
        )
        hits = store.search_vector_ann(
            [1.0, 0.0], tenant_id="batch-incremental",
            provider="fake", model="test-embed",
        )
        assert set(appended).issubset({hit.memory.id for hit in hits})
        assert FakeHNSWIndex.init_calls == 1
        store.close()



def test_incrementally_persisted_ann_keeps_label_mapping_after_restart(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    FakeHNSWIndex.load_calls = 0
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "incremental-persist.sqlite"
        index_dir = Path(directory) / "ann"
        first_store = MemoryStore(database, ann_index_dir=index_dir)
        first_id = first_store.add_memory_with_embedding(
            "z initial vector", FakeEmbeddingProvider([[0.0, 1.0]]),
            tenant_id="persist-incremental",
        )
        first_store.search_vector_ann(
            [1.0, 0.0], tenant_id="persist-incremental",
            provider="fake", model="test-embed",
        )
        second_id = first_store.add_memory_with_embedding(
            "a appended vector", FakeEmbeddingProvider([[1.0, 0.0]]),
            tenant_id="persist-incremental",
        )
        first_store.close()

        second_store = MemoryStore(database, ann_index_dir=index_dir)
        hits = second_store.search_vector_ann(
            [1.0, 0.0], tenant_id="persist-incremental",
            provider="fake", model="test-embed",
        )
        assert hits[0].memory.id == second_id
        assert {first_id, second_id}.issubset({hit.memory.id for hit in hits})
        assert FakeHNSWIndex.load_calls == 1
        second_store.close()



def test_delete_embedding_is_tenant_scoped_and_preserves_memory_and_evidence(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    with tempfile.TemporaryDirectory() as directory:
        store = MemoryStore(Path(directory) / "delete-vector.sqlite")
        memory_id = store.add_memory(
            "keep the source memory", tenant_id="delete-tenant",
            raw_evidence="source record",
        )
        store.set_embedding(
            memory_id, [1.0, 0.0], tenant_id="delete-tenant", provider="p", model="m"
        )
        store.search_vector_ann(
            [1.0, 0.0], tenant_id="delete-tenant", provider="p", model="m"
        )

        assert store.delete_embedding(memory_id, tenant_id="other-tenant") is False
        assert store.delete_embedding(memory_id, tenant_id="delete-tenant") is True
        assert store.delete_embedding(memory_id, tenant_id="delete-tenant") is False
        assert store.get(memory_id, tenant_id="delete-tenant") is not None
        assert len(store.evidence_for(memory_id, tenant_id="delete-tenant")) == 1
        assert store.search_vector(
            [1.0, 0.0], tenant_id="delete-tenant", provider="p", model="m"
        ) == []
        assert store.search_vector_ann(
            [1.0, 0.0], tenant_id="delete-tenant", provider="p", model="m"
        ) == []
        store.close()


def test_ann_compaction_rebuilds_partition_and_persists_only_live_vectors(monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    FakeHNSWIndex.init_calls = 0
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "compact.sqlite"
        index_dir = Path(directory) / "ann"
        store = MemoryStore(database, ann_index_dir=index_dir)
        first = store.add_memory("first vector", tenant_id="compact")
        second = store.add_memory("second vector", tenant_id="compact")
        store.set_embedding(first, [1.0, 0.0], tenant_id="compact", provider="p", model="m")
        store.set_embedding(second, [0.0, 1.0], tenant_id="compact", provider="p", model="m")
        store.search_vector_ann(
            [1.0, 0.0], tenant_id="compact", provider="p", model="m"
        )
        assert store.delete_embedding(first, tenant_id="compact") is True

        result = store.compact_ann_index(
            tenant_id="compact", dimensions=2, provider="p", model="m"
        )
        assert result == {"vectors": 1, "removed": False}
        assert FakeHNSWIndex.init_calls == 2
        assert store._ann_cache[("compact", 2, "p", "m", None)][1] == [second]
        manifest_path = next(index_dir.glob("*.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["labels"] == [second]

        hits = store.search_vector_ann(
            [0.0, 1.0], tenant_id="compact", provider="p", model="m"
        )
        assert [hit.memory.id for hit in hits] == [second]
        store.close()


def test_ann_compaction_removes_empty_persistent_partition(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    with tempfile.TemporaryDirectory() as directory:
        store = MemoryStore(
            Path(directory) / "empty-compact.sqlite",
            ann_index_dir=Path(directory) / "ann",
        )
        memory_id = store.add_memory("only vector", tenant_id="empty-compact")
        store.set_embedding(
            memory_id, [1.0, 0.0], tenant_id="empty-compact", provider="p", model="m"
        )
        store.search_vector_ann(
            [1.0, 0.0], tenant_id="empty-compact", provider="p", model="m"
        )
        assert list((Path(directory) / "ann").glob("*.hnsw"))
        store.delete_embedding(memory_id, tenant_id="empty-compact")
        result = store.compact_ann_index(
            tenant_id="empty-compact", dimensions=2, provider="p", model="m"
        )
        assert result == {"vectors": 0, "removed": True}
        assert list((Path(directory) / "ann").glob("*.hnsw")) == []
        assert list((Path(directory) / "ann").glob("*.json")) == []
        store.close()



def test_ann_cache_refreshes_after_another_sqlite_connection_commits(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "multi-connection.sqlite"
        first = MemoryStore(database)
        second = MemoryStore(database)
        try:
            first_id = first.add_memory("first process memory", tenant_id="shared")
            first.set_embedding(
                first_id, [0.0, 1.0], tenant_id="shared", provider="p", model="m"
            )
            first.search_vector_ann(
                [1.0, 0.0], tenant_id="shared", provider="p", model="m"
            )
            second_id = second.add_memory("second process memory", tenant_id="shared")
            second.set_embedding(
                second_id, [1.0, 0.0], tenant_id="shared", provider="p", model="m"
            )

            hits = first.search_vector_ann(
                [1.0, 0.0], tenant_id="shared", provider="p", model="m"
            )
            assert hits[0].memory.id == second_id
            assert {first_id, second_id}.issubset({hit.memory.id for hit in hits})
        finally:
            first.close()
            second.close()


def test_ann_cold_build_uses_sqlite_write_lock_for_shared_manifest(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=FakeHNSWIndex))
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "shared-build.sqlite"
        index_dir = Path(directory) / "ann"
        first = MemoryStore(database, ann_index_dir=index_dir)
        second = MemoryStore(database, ann_index_dir=index_dir)
        try:
            memory_id = first.add_memory("shared persistent vector", tenant_id="shared-build")
            first.set_embedding(
                memory_id, [1.0, 0.0], tenant_id="shared-build", provider="p", model="m"
            )
            first_hits = first.search_vector_ann(
                [1.0, 0.0], tenant_id="shared-build", provider="p", model="m"
            )
            second_hits = second.search_vector_ann(
                [1.0, 0.0], tenant_id="shared-build", provider="p", model="m"
            )
            assert [hit.memory.id for hit in first_hits] == [memory_id]
            assert [hit.memory.id for hit in second_hits] == [memory_id]
        finally:
            first.close()
            second.close()
