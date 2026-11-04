"""Crash/reopen recovery tests for persistent embedding migrations."""
from infinitydb_next import EmbeddingMigrationManager, MemoryStore
from infinitydb_next.embeddings import EmbeddingModel


class RecoveryProvider:
    model_info = EmbeddingModel("recovery-test", "model-v1", 2)

    def embed(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


def test_migration_resumes_after_store_reopen(tmp_path):
    database = tmp_path / "resume.sqlite"
    store = MemoryStore(database)
    ids = [store.add_memory(f"resume-{i}", tenant_id="resume-tenant") for i in range(3)]
    provider = RecoveryProvider()
    manager = EmbeddingMigrationManager(store)
    migration = manager.begin(provider, tenant_id="resume-tenant", model_revision="r2")
    assert manager.migrate_batch(migration, provider, limit=1)["completed"] == 1
    store.close()

    reopened = MemoryStore(database)
    try:
        resumed = EmbeddingMigrationManager(reopened)
        assert resumed.status(migration)["completed"] == 1
        assert resumed.migrate_batch(migration, provider, limit=10)["completed"] == 3
        assert resumed.activate(migration)["status"] == "active"
        actual = reopened.db.execute(
            "SELECT count(*) FROM embeddings WHERE tenant_id='resume-tenant' AND provider='recovery-test'"
        ).fetchone()[0]
        assert actual == len(ids)
        assert resumed.rollback(migration)["status"] == "rolled_back"
    finally:
        reopened.close()
