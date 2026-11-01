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
    now = time.time()
    expired = store.add_memory(
        "expired vector memory", kind="fact", valid_from=now - 20, valid_to=now - 1
    )
    current = store.add_memory("current vector memory", kind="episode")
    store.set_embedding(expired, [1.0, 0.0])
    store.set_embedding(current, [1.0, 0.0])
    hits = store.search_vector([1.0, 0.0], kinds=["episode"], as_of=now)
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
