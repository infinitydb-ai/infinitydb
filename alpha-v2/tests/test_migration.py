import pytest

from infinitydb_next import EmbeddingMigrationManager, MemoryStore
from infinitydb_next.embeddings import EmbeddingModel


class FakeProvider:
    model_info = EmbeddingModel("fake", "new-model", 2)

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


def test_migration_is_resumable_and_preserves_old_vectors_until_activation(tmp_path):
    store = MemoryStore(tmp_path / "migration.sqlite")
    try:
        a = store.add_memory("alpha", tenant_id="acme")
        b = store.add_memory("bravo", tenant_id="acme")
        store.set_embedding(a, [1, 0], tenant_id="acme", provider="old", model="old-model")
        store.set_embedding(b, [0, 1], tenant_id="acme", provider="old", model="old-model")
        provider = FakeProvider()
        manager = EmbeddingMigrationManager(store)
        migration = manager.begin(provider, tenant_id="acme", model_revision="r2")

        assert manager.status(migration)["remaining"] == 2
        assert manager.migrate_batch(migration, provider, limit=1)["completed"] == 1
        assert store.db.execute(
            "SELECT provider FROM embeddings WHERE memory_id=?", (a,)
        ).fetchone()[0] == "old"
        assert manager.migrate_batch(migration, provider, limit=10)["completed"] == 2
        assert manager.activate(migration)["status"] == "active"
        rows = store.db.execute(
            "SELECT provider,model,model_revision FROM embeddings WHERE tenant_id='acme'"
        ).fetchall()
        assert {(r["provider"], r["model"], r["model_revision"]) for r in rows} == {
            ("fake", "new-model", "r2")
        }
        assert manager.rollback(migration)["status"] == "rolled_back"
        restored = store.db.execute(
            "SELECT memory_id,provider,model FROM embeddings WHERE tenant_id='acme'"
        ).fetchall()
        assert {(r["memory_id"], r["provider"], r["model"]) for r in restored} == {
            (a, "old", "old-model"), (b, "old", "old-model")
        }
    finally:
        store.close()


def test_migration_rejects_wrong_provider_and_incomplete_activation(tmp_path):
    store = MemoryStore(tmp_path / "incomplete.sqlite")
    try:
        store.add_memory("one", tenant_id="tenant")
        provider = FakeProvider()
        manager = EmbeddingMigrationManager(store)
        migration = manager.begin(provider, tenant_id="tenant")
        with pytest.raises(ValueError, match="identity"):
            manager.migrate_batch(migration, type("Wrong", (), {
                "model_info": EmbeddingModel("other", "model"),
                "embed": lambda self, texts: [[1, 0] for _ in texts],
            })())
        with pytest.raises(ValueError, match="incomplete"):
            manager.activate(migration)
    finally:
        store.close()


def test_migration_rejects_new_memory_after_snapshot(tmp_path):
    store = MemoryStore(tmp_path / "snapshot.sqlite")
    try:
        store.add_memory("initial", tenant_id="tenant")
        provider = FakeProvider()
        manager = EmbeddingMigrationManager(store)
        migration = manager.begin(provider, tenant_id="tenant")
        manager.migrate_batch(migration, provider)
        store.add_memory("late arrival", tenant_id="tenant")
        with pytest.raises(ValueError, match="changed"):
            manager.activate(migration)
    finally:
        store.close()
