"""Resumable shadow migrations between embedding model spaces.

The canonical embeddings table is untouched while a migration is prepared.
Activation and rollback are single SQLite transactions; provider calls happen
outside write transactions and can safely be retried batch by batch.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any

from .embeddings import EmbeddingProvider


class EmbeddingMigrationManager:
    """Manage tenant-scoped, resumable embedding migrations for a MemoryStore."""

    def __init__(self, store: Any) -> None:
        self.store = store
        with store._lock, store.db:
            store.db.executescript("""
                CREATE TABLE IF NOT EXISTS embedding_migrations (
                    migration_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    model_revision TEXT,
                    config_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_memories INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    previous_space_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_embedding_migrations_tenant
                    ON embedding_migrations(tenant_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS embedding_migration_vectors (
                    migration_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    PRIMARY KEY(migration_id, memory_id),
                    FOREIGN KEY(migration_id) REFERENCES embedding_migrations(migration_id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS embedding_migration_backup (
                    migration_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    model_revision TEXT,
                    PRIMARY KEY(migration_id, memory_id),
                    FOREIGN KEY(migration_id) REFERENCES embedding_migrations(migration_id)
                        ON DELETE CASCADE
                );
            """)

    def begin(self, provider: EmbeddingProvider, *, tenant_id: str,
              model_revision: str | None = None,
              config_fingerprint: str = "default") -> str:
        info = provider.model_info
        if not tenant_id or not info.provider.strip() or not info.model.strip():
            raise ValueError("tenant, provider and model are required")
        if model_revision is not None and not model_revision.strip():
            raise ValueError("model_revision must be non-empty")
        if not config_fingerprint.strip():
            raise ValueError("config_fingerprint must be non-empty")
        now, migration_id = time.time(), str(uuid.uuid4())
        with self.store._lock, self.store.db:
            pending = self.store.db.execute(
                """SELECT 1 FROM embedding_migrations
                   WHERE tenant_id=? AND status IN ('preparing','ready') LIMIT 1""",
                (tenant_id,),
            ).fetchone()
            if pending:
                raise ValueError("tenant already has an unactivated embedding migration")
            total = self.store.db.execute(
                "SELECT count(*) FROM memories WHERE tenant_id=?", (tenant_id,)
            ).fetchone()[0]
            self.store.db.execute(
                """INSERT INTO embedding_migrations
                   (migration_id,tenant_id,provider,model,model_revision,
                    config_fingerprint,status,total_memories,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,'preparing',?,?,?)""",
                (migration_id, tenant_id, info.provider, info.model, model_revision,
                 config_fingerprint, total, now, now),
            )
        return migration_id

    def migrate_batch(self, migration_id: str, provider: EmbeddingProvider,
                      *, limit: int = 100) -> dict[str, Any]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self.store._lock:
            migration = self._get(migration_id)
            if migration["status"] != "preparing":
                raise ValueError("migration is not preparing")
            expected = (migration["provider"], migration["model"])
            actual = (provider.model_info.provider, provider.model_info.model)
            if expected != actual:
                raise ValueError("provider identity does not match migration target")
            rows = self.store.db.execute(
                """SELECT id,content FROM memories
                   WHERE tenant_id=? AND id NOT IN (
                     SELECT memory_id FROM embedding_migration_vectors WHERE migration_id=?
                   ) ORDER BY id LIMIT ?""",
                (migration["tenant_id"], migration_id, limit),
            ).fetchall()
        if rows:
            vectors = provider.embed([row["content"] for row in rows])
            if len(vectors) != len(rows):
                raise ValueError("provider must return one vector per memory")
            validated = [self.store._validated_vector(v) for v in vectors]
            dims = {len(v) for v in validated}
            if len(dims) != 1:
                raise ValueError("provider returned inconsistent vector dimensions")
            with self.store._lock, self.store.db:
                current = self._get(migration_id)
                if current["status"] != "preparing":
                    raise ValueError("migration state changed while embedding")
                self.store.db.executemany(
                    """INSERT OR IGNORE INTO embedding_migration_vectors
                       (migration_id,memory_id,dimensions,vector_json) VALUES(?,?,?,?)""",
                    [(migration_id, row["id"], len(vector),
                      json.dumps(vector, separators=(",", ":")))
                     for row, vector in zip(rows, validated)],
                )
                self.store.db.execute(
                    "UPDATE embedding_migrations SET updated_at=? WHERE migration_id=?",
                    (time.time(), migration_id),
                )
        return self.status(migration_id)

    def status(self, migration_id: str) -> dict[str, Any]:
        with self.store._lock:
            row = self._get(migration_id)
            completed = self.store.db.execute(
                "SELECT count(*) FROM embedding_migration_vectors WHERE migration_id=?",
                (migration_id,),
            ).fetchone()[0]
            return {
                "migration_id": migration_id, "tenant_id": row["tenant_id"],
                "provider": row["provider"], "model": row["model"],
                "model_revision": row["model_revision"], "status": row["status"],
                "total": row["total_memories"], "completed": completed,
                "remaining": row["total_memories"] - completed,
            }

    def activate(self, migration_id: str) -> dict[str, Any]:
        with self.store._lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                row = self._get(migration_id)
                if row["status"] != "preparing":
                    raise ValueError("only a preparing migration can be activated")
                count = self.store.db.execute(
                    "SELECT count(*) FROM embedding_migration_vectors WHERE migration_id=?",
                    (migration_id,),
                ).fetchone()[0]
                live = self.store.db.execute(
                    "SELECT count(*) FROM memories WHERE tenant_id=?", (row["tenant_id"],)
                ).fetchone()[0]
                if count != row["total_memories"] or live != row["total_memories"]:
                    raise ValueError("migration is incomplete or tenant memories changed")
                prior = self.store.db.execute(
                    """SELECT provider,model,model_revision FROM embeddings
                       WHERE tenant_id=? ORDER BY memory_id LIMIT 1""",
                    (row["tenant_id"],),
                ).fetchone()
                previous = dict(prior) if prior else None
                self.store.db.execute(
                    """INSERT INTO embedding_migration_backup
                       SELECT ?,memory_id,tenant_id,dimensions,vector_json,provider,model,model_revision
                       FROM embeddings WHERE tenant_id=?""",
                    (migration_id, row["tenant_id"]),
                )
                self.store.db.execute("DELETE FROM embeddings WHERE tenant_id=?",
                                      (row["tenant_id"],))
                self.store.db.execute(
                    """INSERT INTO embeddings
                       (memory_id,tenant_id,dimensions,vector_json,provider,model,model_revision)
                       SELECT v.memory_id,m.tenant_id,v.dimensions,v.vector_json,?,?,?
                       FROM embedding_migration_vectors v
                       JOIN memories m ON m.id=v.memory_id
                       WHERE v.migration_id=? AND m.tenant_id=?""",
                    (row["provider"], row["model"], row["model_revision"],
                     migration_id, row["tenant_id"]),
                )
                self.store.db.execute(
                    """UPDATE embedding_migrations SET status='active',
                       previous_space_json=?,updated_at=? WHERE migration_id=?""",
                    (json.dumps(previous) if previous else None, time.time(), migration_id),
                )
                self.store._invalidate_ann([row["tenant_id"]])
                self.store.db.commit()
            except Exception:
                self.store.db.rollback()
                raise
        return self.status(migration_id)

    def rollback(self, migration_id: str) -> dict[str, Any]:
        with self.store._lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                row = self._get(migration_id)
                if row["status"] != "active":
                    raise ValueError("only an active migration can be rolled back")
                self.store.db.execute("DELETE FROM embeddings WHERE tenant_id=?",
                                      (row["tenant_id"],))
                self.store.db.execute(
                    """INSERT INTO embeddings
                       (memory_id,tenant_id,dimensions,vector_json,provider,model,model_revision)
                       SELECT memory_id,tenant_id,dimensions,vector_json,provider,model,model_revision
                       FROM embedding_migration_backup WHERE migration_id=?""",
                    (migration_id,),
                )
                self.store.db.execute(
                    "UPDATE embedding_migrations SET status='rolled_back',updated_at=? WHERE migration_id=?",
                    (time.time(), migration_id),
                )
                self.store._invalidate_ann([row["tenant_id"]])
                self.store.db.commit()
            except Exception:
                self.store.db.rollback()
                raise
        return self.status(migration_id)


    def add_memory_with_dual_embedding(
        self,
        migration_id: str,
        content: str,
        active_provider: EmbeddingProvider,
        migration_provider: EmbeddingProvider,
        *,
        tenant_id: str,
        kind: str = "fact",
        metadata: dict[str, Any] | None = None,
        source_id: str | None = None,
        raw_evidence: str | None = None,
        confidence: float = 1.0,
        valid_from: float | None = None,
        valid_to: float | None = None,
        memory_id: str | None = None,
        active_model_revision: str | None = None,
        active_config_fingerprint: str = "default",
        migration_model_revision: str | None = None,
        migration_config_fingerprint: str = "default",
    ) -> str:
        """Atomically ingest a memory into the active and preparing embedding spaces.

        Both provider calls finish before the SQLite write transaction starts.
        If either embedding fails, no memory or vector is persisted.
        """
        if not content or not content.strip():
            raise ValueError("content must not be empty")
        if not tenant_id:
            raise ValueError("tenant_id must not be empty")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if migration_model_revision != (migration_model_revision or "").strip() and migration_model_revision is not None:
            raise ValueError("migration_model_revision must be non-empty")
        if not migration_config_fingerprint.strip() or not active_config_fingerprint.strip():
            raise ValueError("config fingerprints must be non-empty")

        active_name, active_model, _ = self.store._provider_identity(active_provider)
        target_name, target_model, _ = self.store._provider_identity(migration_provider)
        with self.store._lock:
            migration = self._get(migration_id)
            if migration["tenant_id"] != tenant_id or migration["status"] != "preparing":
                raise ValueError("migration must be preparing for the specified tenant")
            if (migration["provider"], migration["model"]) != (target_name, target_model):
                raise ValueError("migration provider identity does not match target")
            if migration["model_revision"] != migration_model_revision:
                raise ValueError("migration model revision does not match target")
            if migration["config_fingerprint"] != migration_config_fingerprint:
                raise ValueError("migration config fingerprint does not match target")

        entries = [(tenant_id, content)]
        active_vectors = self.store._cached_or_embed(
            entries, active_provider, provider_name=active_name, model_name=active_model,
            revision=active_model_revision, config_fingerprint=active_config_fingerprint,
        )
        target_vectors = self.store._cached_or_embed(
            entries, migration_provider, provider_name=target_name, model_name=target_model,
            revision=migration_model_revision, config_fingerprint=migration_config_fingerprint,
        )
        digest = __import__("hashlib").sha256(content.encode()).hexdigest()
        active_vector = active_vectors[(tenant_id, digest)]
        target_vector = target_vectors[(tenant_id, digest)]
        now = time.time()
        mid = memory_id or str(uuid.uuid4())
        starts = now if valid_from is None else float(valid_from)
        active_cache = self.store._cache_rows(
            entries, active_vectors, provider_name=active_name, model_name=active_model,
            revision=active_model_revision, config_fingerprint=active_config_fingerprint, now=now,
        )
        target_cache = self.store._cache_rows(
            entries, target_vectors, provider_name=target_name, model_name=target_model,
            revision=migration_model_revision, config_fingerprint=migration_config_fingerprint, now=now,
        )
        with self.store._lock, self.store.db:
            current = self._get(migration_id)
            if current["status"] != "preparing":
                raise ValueError("migration state changed during dual embedding")
            self.store.db.executemany(
                """INSERT OR IGNORE INTO embedding_cache
                   (tenant_id,content_hash,provider,model,model_revision,
                    config_fingerprint,dimensions,vector_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                active_cache + target_cache,
            )
            self.store.db.execute(
                "INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (mid, tenant_id, kind, content, now, starts, valid_to,
                 json.dumps(metadata or {}, sort_keys=True), source_id, confidence, digest),
            )
            if self.store._fts:
                self.store.db.execute(
                    "INSERT INTO memory_fts_v2 VALUES(?,?,?)",
                    (mid, self.store._tenant_key(tenant_id), content),
                )
            self.store.db.execute(
                """INSERT INTO embeddings
                   (memory_id,tenant_id,dimensions,vector_json,provider,model,model_revision)
                   VALUES(?,?,?,?,?,?,?)""",
                (mid, tenant_id, len(active_vector),
                 json.dumps(active_vector, separators=(",", ":")),
                 active_name, active_model, active_model_revision),
            )
            self.store.db.execute(
                """INSERT INTO embedding_migration_vectors
                   (migration_id,memory_id,dimensions,vector_json) VALUES(?,?,?,?)""",
                (migration_id, mid, len(target_vector),
                 json.dumps(target_vector, separators=(",", ":"))),
            )
            if raw_evidence is not None:
                self.store._append_evidence(tenant_id, mid, raw_evidence, now)
            self.store.db.execute(
                """UPDATE embedding_migrations
                   SET total_memories=total_memories+1,updated_at=?
                   WHERE migration_id=?""",
                (now, migration_id),
            )
            self.store._refresh_ann_after_insert([tenant_id])
        return mid

    def _get(self, migration_id: str) -> Any:
        row = self.store.db.execute(
            "SELECT * FROM embedding_migrations WHERE migration_id=?", (migration_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown embedding migration")
        return row
