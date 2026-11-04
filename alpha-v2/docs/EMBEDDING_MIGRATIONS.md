# Embedding model migrations



### Versioned embedding migrations

Use `EmbeddingMigrationManager` to prepare a replacement model space without
mutating the currently searchable vectors. Provider calls run outside SQLite
write transactions; batches are resumable and tenant-scoped.

```python
from infinitydb_next import EmbeddingMigrationManager, MemoryStore, OllamaEmbeddingProvider

with MemoryStore("agent-memory.sqlite3") as store:
    provider = OllamaEmbeddingProvider("nomic-embed-text")
    migrations = EmbeddingMigrationManager(store)
    migration_id = migrations.begin(
        provider, tenant_id="acme",
        model_revision="nomic-embed-text:latest",
    )
    while migrations.status(migration_id)["remaining"]:
        migrations.migrate_batch(migration_id, provider, limit=100)
    migrations.activate(migration_id)
    # If validation or downstream checks fail:
    # migrations.rollback(migration_id)
```

The migration snapshots the tenant's memory IDs at creation. Activation is
rejected if the tenant's memory set changes or any snapshot memory lacks a
replacement vector. Activation atomically replaces that tenant's canonical
vectors and saves the prior vectors for rollback. Rollback restores those
vectors atomically. A migration is not a distributed zero-downtime router:
applications should validate the target model and coordinate query-embedding
switch-over with activation. Keep backups until the new space has been
validated; rollback is available while the migration record and backup remain.
