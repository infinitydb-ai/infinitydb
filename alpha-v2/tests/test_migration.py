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


class ActiveProvider:
    model_info = EmbeddingModel("active", "old-model", 2)

    def embed(self, texts):
        return [[1.0, float(len(text))] for text in texts]


class FailingProvider(FakeProvider):
    def embed(self, texts):
        raise RuntimeError("target provider unavailable")


def test_dual_write_keeps_migration_complete_for_new_memories(tmp_path):
    store = MemoryStore(tmp_path / "dual-write.sqlite")
    try:
        first = store.add_memory("first", tenant_id="tenant")
        store.set_embedding(first, [1, 0], tenant_id="tenant",
                            provider="active", model="old-model")
        target = FakeProvider()
        manager = EmbeddingMigrationManager(store)
        migration = manager.begin(
            target, tenant_id="tenant", model_revision="r2",
            config_fingerprint="target-config",
        )
        manager.migrate_batch(migration, target)

        new_id = manager.add_memory_with_dual_embedding(
            migration, "arrived during migration", ActiveProvider(), target,
            tenant_id="tenant", migration_model_revision="r2",
            migration_config_fingerprint="target-config",
        )
        status = manager.status(migration)
        assert status["total"] == 2
        assert status["completed"] == 2
        assert status["remaining"] == 0
        assert store.db.execute(
            "SELECT provider,model FROM embeddings WHERE memory_id=?", (new_id,)
        ).fetchone()[:] == ("active", "old-model")
        assert manager.activate(migration)["status"] == "active"
        target_rows = store.db.execute(
            "SELECT provider,model,model_revision FROM embeddings WHERE tenant_id='tenant'"
        ).fetchall()
        assert {(row["provider"], row["model"], row["model_revision"])
                for row in target_rows} == {("fake", "new-model", "r2")}
    finally:
        store.close()


def test_dual_write_provider_failure_persists_nothing(tmp_path):
    store = MemoryStore(tmp_path / "dual-write-failure.sqlite")
    try:
        target = FakeProvider()
        manager = EmbeddingMigrationManager(store)
        migration = manager.begin(target, tenant_id="tenant")
        with pytest.raises(RuntimeError, match="unavailable"):
            manager.add_memory_with_dual_embedding(
                migration, "must not persist", ActiveProvider(), FailingProvider(),
                tenant_id="tenant",
            )
        assert store.stats(tenant_id="tenant")["memories"] == 0
        assert manager.status(migration)["total"] == 0
        assert manager.status(migration)["completed"] == 0
    finally:
        store.close()
