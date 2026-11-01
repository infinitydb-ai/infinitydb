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
