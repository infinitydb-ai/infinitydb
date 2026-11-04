from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from .embeddings import EmbeddingProvider
from .models import ContextPack, GraphHit, Memory, SearchHit

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS memories (
 id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, kind TEXT NOT NULL,
 content TEXT NOT NULL, created_at REAL NOT NULL, valid_from REAL NOT NULL,
 valid_to REAL, metadata_json TEXT NOT NULL, source_id TEXT,
 confidence REAL NOT NULL, content_hash TEXT NOT NULL,
 UNIQUE(tenant_id, content_hash, valid_from)
);
CREATE INDEX IF NOT EXISTS idx_mem_tenant_time ON memories(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mem_tenant_kind ON memories(tenant_id, kind);
CREATE INDEX IF NOT EXISTS idx_mem_temporal ON memories(tenant_id, valid_from, valid_to);
CREATE TABLE IF NOT EXISTS embeddings (
 memory_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL,
 dimensions INTEGER NOT NULL, vector_json TEXT NOT NULL,
 provider TEXT NOT NULL DEFAULT 'unknown', model TEXT NOT NULL DEFAULT 'unknown',
 model_revision TEXT,
 FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_embeddings_tenant ON embeddings(tenant_id, dimensions);
CREATE TABLE IF NOT EXISTS embedding_cache (
 tenant_id TEXT NOT NULL, content_hash TEXT NOT NULL,
 provider TEXT NOT NULL, model TEXT NOT NULL, model_revision TEXT NOT NULL,
 config_fingerprint TEXT NOT NULL, dimensions INTEGER NOT NULL,
 vector_json TEXT NOT NULL, created_at REAL NOT NULL,
 PRIMARY KEY(tenant_id,content_hash,provider,model,model_revision,config_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_tenant
 ON embedding_cache(tenant_id,provider,model);
CREATE TABLE IF NOT EXISTS evidence (
 id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, memory_id TEXT NOT NULL,
 raw_content TEXT NOT NULL, created_at REAL NOT NULL, sha256 TEXT NOT NULL,
 previous_hash TEXT, chain_hash TEXT NOT NULL,
 FOREIGN KEY(memory_id) REFERENCES memories(id)
);
CREATE INDEX IF NOT EXISTS idx_evidence_memory ON evidence(memory_id);
CREATE TABLE IF NOT EXISTS edges (
 tenant_id TEXT NOT NULL, src_id TEXT NOT NULL, dst_id TEXT NOT NULL,
 relation TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0,
 PRIMARY KEY(tenant_id, src_id, dst_id, relation)
);
"""


class MemoryStore:
    """Deterministic SQLite alpha store; no model or external API calls are made."""

    def __init__(
        self, path: str | Path = "infinitydb-next.sqlite3", *,
        ann_index_dir: str | Path | None = None,
    ) -> None:
        self.path = str(path)
        # Persistence is opt-in: callers choose where derived native index files live.
        self.ann_index_dir = Path(ann_index_dir) if ann_index_dir is not None else None
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.RLock()
        self._ann_cache: dict[tuple[str, int, str, str, str | None], tuple[Any, list[str]]] = {}
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA temp_store=MEMORY")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self._migrate_embedding_metadata()
        # Legacy databases gain these columns in the migration above; only then
        # can the composite index referencing them be created safely.
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_tenant_model "
            "ON embeddings(tenant_id, dimensions, provider, model, model_revision)"
        )
        self._fts = self._init_fts()

    def _migrate_embedding_metadata(self) -> None:
        """Additive, idempotent migration for databases created before model metadata."""
        columns = {
            row["name"] for row in self.db.execute("PRAGMA table_info(embeddings)")
        }
        additions = (
            ("provider", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("model", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("model_revision", "TEXT"),
        )
        with self.db:
            for name, definition in additions:
                if name not in columns:
                    self.db.execute(f"ALTER TABLE embeddings ADD COLUMN {name} {definition}")

    @staticmethod
    def _tenant_key(tenant_id: str) -> str:
        return "tenant_" + hashlib.sha256(tenant_id.encode()).hexdigest()[:24]

    def _invalidate_ann(self, tenant_ids: Iterable[str]) -> None:
        affected = set(tenant_ids)
        if affected:
            self._ann_cache = {
                key: cached for key, cached in self._ann_cache.items()
                if key[0] not in affected
            }

    def _refresh_ann_after_insert(self, tenant_ids: Iterable[str]) -> None:
        """Append newly committed vectors to warm ANN partitions without a full rebuild.

        If the optional HNSW backend cannot resize/update an index, discard that
        cache entry; the next query safely reconstructs it from canonical SQLite.
        """
        affected = set(tenant_ids)
        if not affected:
            return
        for key, cached in list(self._ann_cache.items()):
            tenant, dimensions, provider, model, revision = key
            if tenant not in affected:
                continue
            index, labels = cached
            rows = self.db.execute(
                """SELECT memory_id,vector_json FROM embeddings
                   WHERE tenant_id=? AND dimensions=? AND provider=? AND model=?
                     AND model_revision IS ?
                   ORDER BY memory_id""",
                key,
            ).fetchall()
            current_ids = [row["memory_id"] for row in rows]
            known = set(labels)
            new_rows = [row for row in rows if row["memory_id"] not in known]
            if not new_rows:
                continue
            try:
                resize = getattr(index, "resize_index", None)
                if resize is None:
                    raise RuntimeError("HNSW backend does not support index resizing")
                resize(len(current_ids))
                new_labels = list(range(len(labels), len(labels) + len(new_rows)))
                index.add_items(
                    [json.loads(row["vector_json"]) for row in new_rows],
                    new_labels,
                    num_threads=1,
                )
                labels.extend(row["memory_id"] for row in new_rows)
                fingerprint = self._ann_source_fingerprint(rows)
                self._save_persisted_ann(index, key, fingerprint, labels)
                self._ann_cache[key] = (index, labels)
            except Exception:
                self._ann_cache.pop(key, None)


    def _ann_partition_paths(
        self, key: tuple[str, int, str, str, str | None]
    ) -> tuple[Path, Path]:
        if self.ann_index_dir is None:
            raise RuntimeError("persistent ANN storage is not enabled")
        identity = json.dumps(key, separators=(",", ":"), ensure_ascii=False)
        partition = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return (
            self.ann_index_dir / f"{partition}.hnsw",
            self.ann_index_dir / f"{partition}.json",
        )

    @staticmethod
    def _ann_source_fingerprint(rows: list[sqlite3.Row]) -> str:
        digest = hashlib.sha256()
        for row in rows:
            digest.update(row["memory_id"].encode("utf-8"))
            digest.update(bytes([0]))
            digest.update(row["vector_json"].encode("utf-8"))
            digest.update(bytes([10]))
        return digest.hexdigest()

    def _load_persisted_ann(
        self, hnswlib: Any, key: tuple[str, int, str, str, str | None],
        fingerprint: str, expected_labels: list[str],
    ) -> tuple[Any, list[str]] | None:
        if self.ann_index_dir is None:
            return None
        index_path, manifest_path = self._ann_partition_paths(key)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("format_version") != 1
                or manifest.get("dimensions") != key[1]
                or manifest.get("source_fingerprint") != fingerprint
                or not isinstance(manifest.get("labels"), list)
                or len(manifest["labels"]) != len(expected_labels)
                or len(set(manifest["labels"])) != len(expected_labels)
                or set(manifest["labels"]) != set(expected_labels)
                or not index_path.is_file()
            ):
                return None
            checksum = hashlib.sha256(index_path.read_bytes()).hexdigest()
            if checksum != manifest.get("index_sha256"):
                return None
            index = hnswlib.Index(space="cosine", dim=key[1])
            index.load_index(str(index_path), max_elements=len(expected_labels))
            return index, manifest["labels"]
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError, TypeError):
            return None

    def _save_persisted_ann(
        self, index: Any, key: tuple[str, int, str, str, str | None],
        fingerprint: str, labels: list[str],
    ) -> None:
        if self.ann_index_dir is None:
            return
        self.ann_index_dir.mkdir(parents=True, exist_ok=True)
        index_path, manifest_path = self._ann_partition_paths(key)
        index_fd, index_temp_name = tempfile.mkstemp(
            prefix=".ann-index-", suffix=".tmp", dir=self.ann_index_dir
        )
        os.close(index_fd)
        manifest_fd, manifest_temp_name = tempfile.mkstemp(
            prefix=".ann-manifest-", suffix=".tmp", dir=self.ann_index_dir
        )
        try:
            index.save_index(index_temp_name)
            index_checksum = hashlib.sha256(Path(index_temp_name).read_bytes()).hexdigest()
            manifest = {
                "format_version": 1,
                "dimensions": key[1],
                "source_fingerprint": fingerprint,
                "labels": labels,
                "index_sha256": index_checksum,
            }
            with os.fdopen(manifest_fd, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream, separators=(",", ":"), ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(index_temp_name, index_path)
            os.replace(manifest_temp_name, manifest_path)
        except Exception:
            try:
                os.close(manifest_fd)
            except OSError:
                pass
            raise
        finally:
            for temporary in (index_temp_name, manifest_temp_name):
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def _init_fts(self) -> bool:
        try:
            self.db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts_v2
                USING fts5(memory_id UNINDEXED, tenant_key, content, tokenize='unicode61')""")
            # The memories table is canonical. Rebuild the derived FTS index on
            # open so partial/stale indexes from interrupted older versions recover.
            with self.db:
                self.db.execute("DELETE FROM memory_fts_v2")
                rows = self.db.execute(
                    "SELECT id,tenant_id,content FROM memories ORDER BY rowid"
                ).fetchall()
                self.db.executemany(
                    "INSERT INTO memory_fts_v2 VALUES(?,?,?)",
                    [(row["id"], self._tenant_key(row["tenant_id"]), row["content"])
                     for row in rows],
                )
            return True
        except sqlite3.OperationalError:
            return False

    def close(self) -> None:
        with self._lock:
            self._ann_cache.clear()
            self.db.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def add_memory(
        self, content: str, *, tenant_id: str = "default", kind: str = "fact",
        metadata: dict[str, Any] | None = None, source_id: str | None = None,
        raw_evidence: str | None = None, confidence: float = 1.0,
        valid_from: float | None = None, valid_to: float | None = None,
        memory_id: str | None = None,
    ) -> str:
        if not content or not content.strip():
            raise ValueError("content must not be empty")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        now = time.time()
        mid = memory_id or str(uuid.uuid4())
        created = now
        starts = now if valid_from is None else valid_from
        digest = hashlib.sha256(content.encode()).hexdigest()
        with self._lock, self.db:
            self.db.execute(
                "INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (mid, tenant_id, kind, content, created, starts, valid_to,
                 json.dumps(metadata or {}, sort_keys=True), source_id, confidence, digest),
            )
            if self._fts:
                self.db.execute("INSERT INTO memory_fts_v2 VALUES(?,?,?)",
                                (mid, self._tenant_key(tenant_id), content))
            if raw_evidence is not None:
                self._append_evidence(tenant_id, mid, raw_evidence, now)
        return mid

    def supersede_memory(
        self, memory_id: str, new_content: str, *,
        tenant_id: str = "default", kind: str = "fact",
        metadata: dict[str, Any] | None = None, source_id: str | None = None,
        raw_evidence: str | None = None, confidence: float = 1.0,
        at: float | None = None,
    ) -> str:
        """Atomically close an active memory and create its explicit successor.

        The old record remains queryable before its valid_to boundary. The new
        record starts at that same boundary and is linked by a 'supersedes' edge.
        """
        if not new_content or not new_content.strip():
            raise ValueError("new_content must not be empty")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        timestamp = time.time() if at is None else float(at)
        if not math.isfinite(timestamp):
            raise ValueError("at must be a finite timestamp")
        new_id = str(uuid.uuid4())
        digest = hashlib.sha256(new_content.encode()).hexdigest()
        now = time.time()
        with self._lock, self.db:
            previous = self.db.execute(
                "SELECT valid_from,valid_to FROM memories WHERE id=? AND tenant_id=?",
                (memory_id, tenant_id),
            ).fetchone()
            if previous is None:
                raise ValueError("memory does not exist in the specified tenant")
            if previous["valid_to"] is not None:
                raise ValueError("memory is already superseded or expired")
            if timestamp < previous["valid_from"]:
                raise ValueError("supersession time cannot precede the memory valid_from")
            self.db.execute(
                "UPDATE memories SET valid_to=? WHERE id=? AND tenant_id=? AND valid_to IS NULL",
                (timestamp, memory_id, tenant_id),
            )
            self.db.execute(
                "INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (new_id, tenant_id, kind, new_content, now, timestamp, None,
                 json.dumps(metadata or {}, sort_keys=True), source_id, confidence, digest),
            )
            self.db.execute(
                "INSERT INTO edges(tenant_id,src_id,dst_id,relation,weight) VALUES(?,?,?,?,?)",
                (tenant_id, memory_id, new_id, "supersedes", 1.0),
            )
            if self._fts:
                self.db.execute("INSERT INTO memory_fts_v2 VALUES(?,?,?)",
                                (new_id, self._tenant_key(tenant_id), new_content))
            if raw_evidence is not None:
                self._append_evidence(tenant_id, new_id, raw_evidence, now)
        return new_id

    def add_memories(self, items: Iterable[dict[str, Any]]) -> list[str]:
        now = time.time()
        prepared: list[tuple[Any, ...]] = []
        fts_rows: list[tuple[str, str, str]] = []
        ids: list[str] = []
        for item in items:
            content = str(item.get("content", ""))
            confidence = float(item.get("confidence", 1.0))
            if not content.strip():
                raise ValueError("content must not be empty")
            if not 0 <= confidence <= 1:
                raise ValueError("confidence must be between 0 and 1")
            tenant = str(item.get("tenant_id", "default"))
            mid = str(item.get("memory_id") or uuid.uuid4())
            created = float(item.get("created_at") or now)
            starts = float(item.get("valid_from") or created)
            digest = hashlib.sha256(content.encode()).hexdigest()
            prepared.append((mid, tenant, str(item.get("kind", "fact")), content, created,
                             starts, item.get("valid_to"),
                             json.dumps(item.get("metadata") or {}, sort_keys=True),
                             item.get("source_id"), confidence, digest))
            fts_rows.append((mid, self._tenant_key(tenant), content))
            ids.append(mid)
        with self._lock, self.db:
            self.db.executemany("INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?)", prepared)
            if self._fts and fts_rows:
                self.db.executemany("INSERT INTO memory_fts_v2 VALUES(?,?,?)", fts_rows)
        return ids

    @staticmethod
    def _provider_identity(provider: EmbeddingProvider) -> tuple[str, str, str | None]:
        info = provider.model_info
        if not info.provider.strip() or not info.model.strip():
            raise ValueError("embedding provider must expose non-empty provider and model identity")
        return info.provider, info.model, None

    def _cached_or_embed(
        self, entries: list[tuple[str, str]], provider: EmbeddingProvider, *,
        provider_name: str, model_name: str, revision: str | None,
        config_fingerprint: str,
    ) -> dict[tuple[str, str], list[float]]:
        """Resolve (tenant, content) vectors, calling the provider only for cache misses."""
        revision_key = revision or ""
        resolved: dict[tuple[str, str], list[float]] = {}
        missing: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        with self._lock:
            for tenant, content in entries:
                digest = hashlib.sha256(content.encode()).hexdigest()
                key = (tenant, digest)
                if key in seen:
                    continue
                seen.add(key)
                row = self.db.execute(
                    """SELECT vector_json FROM embedding_cache
                       WHERE tenant_id=? AND content_hash=? AND provider=? AND model=?
                         AND model_revision=? AND config_fingerprint=?""",
                    (tenant, digest, provider_name, model_name, revision_key, config_fingerprint),
                ).fetchone()
                if row is not None:
                    resolved[key] = self._validated_vector(json.loads(row["vector_json"]))
                else:
                    missing.append((tenant, digest, content))
        if missing:
            # A single provider call handles all unique cache misses in this request.
            vectors = provider.embed([content for _, _, content in missing])
            if len(vectors) != len(missing):
                raise ValueError("embedding provider must return exactly one vector per cache miss")
            validated = [self._validated_vector(vector) for vector in vectors]
            dimensions = {len(vector) for vector in validated}
            if len(dimensions) > 1:
                raise ValueError("embedding provider returned inconsistent dimensions")
            for (tenant, digest, _), vector in zip(missing, validated):
                resolved[(tenant, digest)] = vector
        return resolved

    @staticmethod
    def _cache_rows(
        entries: list[tuple[str, str]], vectors: dict[tuple[str, str], list[float]], *,
        provider_name: str, model_name: str, revision: str | None,
        config_fingerprint: str, now: float,
    ) -> list[tuple[Any, ...]]:
        rows: dict[tuple[str, str], tuple[Any, ...]] = {}
        for tenant, content in entries:
            digest = hashlib.sha256(content.encode()).hexdigest()
            key = (tenant, digest)
            vector = vectors[key]
            rows[key] = (
                tenant, digest, provider_name, model_name, revision or "",
                config_fingerprint, len(vector),
                json.dumps(vector, separators=(",", ":")), now,
            )
        return list(rows.values())

    def add_memory_with_embedding(
        self, content: str, provider: EmbeddingProvider, *,
        tenant_id: str = "default", kind: str = "fact",
        metadata: dict[str, Any] | None = None, source_id: str | None = None,
        raw_evidence: str | None = None, confidence: float = 1.0,
        valid_from: float | None = None, valid_to: float | None = None,
        memory_id: str | None = None, model_revision: str | None = None,
        config_fingerprint: str = "default",
    ) -> str:
        """Embed and persist one memory, reusing a tenant/model-scoped cached vector."""
        if not content or not content.strip():
            raise ValueError("content must not be empty")
        provider_name, model_name, _ = self._provider_identity(provider)
        if model_revision is not None and not model_revision.strip():
            raise ValueError("model_revision must be non-empty when provided")
        if not config_fingerprint.strip():
            raise ValueError("config_fingerprint must be non-empty")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        entries = [(tenant_id, content)]
        vectors = self._cached_or_embed(
            entries, provider, provider_name=provider_name, model_name=model_name,
            revision=model_revision, config_fingerprint=config_fingerprint,
        )
        vector = vectors[(tenant_id, hashlib.sha256(content.encode()).hexdigest())]
        now = time.time()
        mid = memory_id or str(uuid.uuid4())
        starts = now if valid_from is None else valid_from
        digest = hashlib.sha256(content.encode()).hexdigest()
        cache_rows = self._cache_rows(
            entries, vectors, provider_name=provider_name, model_name=model_name,
            revision=model_revision, config_fingerprint=config_fingerprint, now=now,
        )
        with self._lock, self.db:
            self.db.executemany(
                """INSERT OR IGNORE INTO embedding_cache
                   (tenant_id,content_hash,provider,model,model_revision,
                    config_fingerprint,dimensions,vector_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""", cache_rows,
            )
            self.db.execute(
                "INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (mid, tenant_id, kind, content, now, starts, valid_to,
                 json.dumps(metadata or {}, sort_keys=True), source_id, confidence, digest),
            )
            if self._fts:
                self.db.execute("INSERT INTO memory_fts_v2 VALUES(?,?,?)",
                                (mid, self._tenant_key(tenant_id), content))
            self.db.execute(
                """INSERT INTO embeddings(
                       memory_id,tenant_id,dimensions,vector_json,provider,model,model_revision
                   ) VALUES(?,?,?,?,?,?,?)""",
                (mid, tenant_id, len(vector), json.dumps(vector, separators=(",", ":")),
                 provider_name, model_name, model_revision),
            )
            if raw_evidence is not None:
                self._append_evidence(tenant_id, mid, raw_evidence, now)
            self._refresh_ann_after_insert([tenant_id])
        return mid

    def add_memories_with_embeddings(
        self, items: Iterable[dict[str, Any]], provider: EmbeddingProvider, *,
        model_revision: str | None = None, config_fingerprint: str = "default",
    ) -> list[str]:
        """Batch-ingest memories, embedding only unique tenant-scoped cache misses."""
        batch = list(items)
        if not batch:
            return []
        contents = [str(item.get("content", "")) for item in batch]
        if any(not content.strip() for content in contents):
            raise ValueError("content must not be empty")
        provider_name, model_name, _ = self._provider_identity(provider)
        if model_revision is not None and not model_revision.strip():
            raise ValueError("model_revision must be non-empty when provided")
        if not config_fingerprint.strip():
            raise ValueError("config_fingerprint must be non-empty")
        tenants = [str(item.get("tenant_id", "default")) for item in batch]
        entries = list(zip(tenants, contents))
        vectors = self._cached_or_embed(
            entries, provider, provider_name=provider_name, model_name=model_name,
            revision=model_revision, config_fingerprint=config_fingerprint,
        )
        now = time.time()
        memory_rows: list[tuple[Any, ...]] = []
        embedding_rows: list[tuple[Any, ...]] = []
        fts_rows: list[tuple[str, str, str]] = []
        ids: list[str] = []
        for item, tenant, content in zip(batch, tenants, contents):
            confidence = float(item.get("confidence", 1.0))
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("confidence must be between 0 and 1")
            mid = str(item.get("memory_id") or uuid.uuid4())
            created = float(item.get("created_at") or now)
            starts = float(item.get("valid_from") or created)
            digest = hashlib.sha256(content.encode()).hexdigest()
            vector = vectors[(tenant, digest)]
            memory_rows.append((
                mid, tenant, str(item.get("kind", "fact")), content, created,
                starts, item.get("valid_to"),
                json.dumps(item.get("metadata") or {}, sort_keys=True),
                item.get("source_id"), confidence, digest,
            ))
            embedding_rows.append((
                mid, tenant, len(vector), json.dumps(vector, separators=(",", ":")),
                provider_name, model_name, model_revision,
            ))
            fts_rows.append((mid, self._tenant_key(tenant), content))
            ids.append(mid)
        cache_rows = self._cache_rows(
            entries, vectors, provider_name=provider_name, model_name=model_name,
            revision=model_revision, config_fingerprint=config_fingerprint, now=now,
        )
        with self._lock, self.db:
            self.db.executemany(
                """INSERT OR IGNORE INTO embedding_cache
                   (tenant_id,content_hash,provider,model,model_revision,
                    config_fingerprint,dimensions,vector_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""", cache_rows,
            )
            self.db.executemany("INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?)", memory_rows)
            self.db.executemany(
                """INSERT INTO embeddings(
                       memory_id,tenant_id,dimensions,vector_json,provider,model,model_revision
                   ) VALUES(?,?,?,?,?,?,?)""",
                embedding_rows,
            )
            if self._fts:
                self.db.executemany("INSERT INTO memory_fts_v2 VALUES(?,?,?)", fts_rows)
            self._refresh_ann_after_insert(tenants)
        return ids

    def _append_evidence(self, tenant: str, mid: str, raw: str, now: float) -> str:
        evidence_id = str(uuid.uuid4())
        digest = hashlib.sha256(raw.encode()).hexdigest()
        prior = self.db.execute(
            "SELECT chain_hash FROM evidence WHERE tenant_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1",
            (tenant,),
        ).fetchone()
        previous = prior[0] if prior else "0" * 64
        chain = hashlib.sha256((previous + digest + evidence_id + mid).encode()).hexdigest()
        self.db.execute("INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?)",
                        (evidence_id, tenant, mid, raw, now, digest, previous, chain))
        return evidence_id

    def add_edge(self, src_id: str, dst_id: str, relation: str, *,
                 tenant_id: str = "default", weight: float = 1.0) -> None:
        if not src_id or not dst_id:
            raise ValueError("source and destination IDs must not be empty")
        if not relation or not relation.strip():
            raise ValueError("relation must not be empty")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weight must be a finite non-negative number")
        with self._lock, self.db:
            rows = self.db.execute(
                "SELECT id FROM memories WHERE tenant_id=? AND id IN (?,?)",
                (tenant_id, src_id, dst_id),
            ).fetchall()
            if len({row["id"] for row in rows}) != len({src_id, dst_id}):
                raise ValueError("both edge endpoints must exist in the specified tenant")
            self.db.execute("INSERT OR REPLACE INTO edges VALUES(?,?,?,?,?)",
                            (tenant_id, src_id, dst_id, relation, weight))

    def traverse_graph(
        self, seed_ids: Iterable[str], *, tenant_id: str = "default",
        max_hops: int = 2, limit: int = 50,
        relations: Iterable[str] | None = None, as_of: float | None = None,
    ) -> list[GraphHit]:
        """Traverse tenant-local edges breadth-first and return explainable paths.

        Edges are treated as traversable in either direction. Each memory is
        returned once at its shortest hop distance; equal-depth paths use the
        highest product of edge weights, then a stable ID/path ordering.
        """
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        if limit < 1:
            return []
        seeds = list(dict.fromkeys(str(value) for value in seed_ids if str(value)))
        if not seeds:
            return []
        relation_values = list(dict.fromkeys(relations or []))
        relation_clause = (" AND relation IN (" + ",".join("?" for _ in relation_values) + ")") if relation_values else ""
        now = time.time() if as_of is None else as_of
        with self._lock:
            placeholders = ",".join("?" for _ in seeds)
            seed_rows = self.db.execute(
                f"SELECT * FROM memories WHERE tenant_id=? AND id IN ({placeholders}) AND valid_from<=? AND (valid_to IS NULL OR valid_to>?)",
                [tenant_id, *seeds, now, now],
            ).fetchall()
            valid_seeds = {row["id"] for row in seed_rows}
            if not valid_seeds:
                return []
            visited = set(valid_seeds)
            frontier = [(seed, (seed,), (), 1.0) for seed in sorted(valid_seeds)]
            results: dict[str, tuple[int, tuple[str, ...], tuple[str, ...], float]] = {}
            for depth in range(1, max_hops + 1):
                candidates: dict[str, tuple[tuple[str, ...], tuple[str, ...], float]] = {}
                for current, path, path_relations, score in frontier:
                    edge_rows = self.db.execute(
                        f"""SELECT src_id,dst_id,relation,weight FROM edges
                            WHERE tenant_id=? AND (src_id=? OR dst_id=?) {relation_clause}
                            ORDER BY relation,src_id,dst_id""",
                        [tenant_id, current, current, *relation_values],
                    ).fetchall()
                    for edge in edge_rows:
                        neighbor = edge["dst_id"] if edge["src_id"] == current else edge["src_id"]
                        if neighbor in visited:
                            continue
                        next_path = path + (neighbor,)
                        next_relations = path_relations + (edge["relation"],)
                        next_score = score * float(edge["weight"])
                        previous = candidates.get(neighbor)
                        candidate = (next_path, next_relations, next_score)
                        if previous is None or (-next_score, next_path, next_relations) < (-previous[2], previous[0], previous[1]):
                            candidates[neighbor] = candidate
                if not candidates:
                    break
                ids = sorted(candidates)
                marks = ",".join("?" for _ in ids)
                memory_rows = self.db.execute(
                    f"SELECT * FROM memories WHERE tenant_id=? AND id IN ({marks}) AND valid_from<=? AND (valid_to IS NULL OR valid_to>?)",
                    [tenant_id, *ids, now, now],
                ).fetchall()
                allowed = {row["id"] for row in memory_rows}
                visited.update(candidates)
                for memory_id in sorted(allowed):
                    path, path_relations, score = candidates[memory_id]
                    results[memory_id] = (depth, path, path_relations, score)
                frontier = [(memory_id, *candidates[memory_id]) for memory_id in sorted(allowed)]
                if len(results) >= limit:
                    break
            if not results:
                return []
            ids = list(results)
            marks = ",".join("?" for _ in ids)
            rows = self.db.execute(
                f"SELECT * FROM memories WHERE tenant_id=? AND id IN ({marks})",
                [tenant_id, *ids],
            ).fetchall()
        by_id = {row["id"]: row for row in rows}
        ordered = sorted(results.items(), key=lambda item: (item[1][0], -item[1][3], item[0]))[:limit]
        return [GraphHit(self._to_memory(by_id[memory_id]), details[0], list(details[1]),
                         list(details[2]), details[3]) for memory_id, details in ordered]

    @staticmethod
    def _validated_vector(vector: Iterable[float]) -> list[float]:
        values = [float(value) for value in vector]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("embedding must contain finite numeric values")
        if math.sqrt(sum(value * value for value in values)) == 0:
            raise ValueError("embedding must have a non-zero norm")
        return values

    def set_embedding(self, memory_id: str, vector: Iterable[float], *,
                      tenant_id: str = "default", provider: str = "unknown",
                      model: str = "unknown", model_revision: str | None = None) -> None:
        values = self._validated_vector(vector)
        if not provider.strip() or not model.strip():
            raise ValueError("embedding provider and model must not be empty")
        if model_revision is not None and not model_revision.strip():
            raise ValueError("model_revision must be non-empty when provided")
        with self._lock, self.db:
            exists = self.db.execute(
                "SELECT 1 FROM memories WHERE id=? AND tenant_id=?",
                (memory_id, tenant_id),
            ).fetchone()
            if exists is None:
                raise ValueError("memory does not exist in the specified tenant")
            self.db.execute(
                """INSERT INTO embeddings(
                       memory_id,tenant_id,dimensions,vector_json,provider,model,model_revision
                   ) VALUES(?,?,?,?,?,?,?) ON CONFLICT(memory_id) DO UPDATE SET
                   tenant_id=excluded.tenant_id, dimensions=excluded.dimensions,
                   vector_json=excluded.vector_json, provider=excluded.provider,
                   model=excluded.model, model_revision=excluded.model_revision""",
                (memory_id, tenant_id, len(values), json.dumps(values, separators=(",", ":")),
                 provider, model, model_revision),
            )
            self._invalidate_ann([tenant_id])

    def search_vector_ann(
        self,
        query_embedding: Iterable[float],
        *,
        tenant_id: str,
        provider: str,
        model: str,
        model_revision: str | None = None,
        limit: int = 10,
        kinds: Iterable[str] | None = None,
        as_of: float | None = None,
        ef_search: int = 64,
        candidate_multiplier: int = 8,
    ) -> list[SearchHit]:
        """Search a tenant/model-partitioned HNSW index.

        Install the optional ANN extra with pip install infinitydb-next[ann].
        HNSW supplies candidates; SQLite applies current temporal/kind filters
        and exact cosine rescoring. Set ann_index_dir on the store to persist
        recoverable index files; source fingerprints detect stale files after
        database mutations.
        """
        if limit < 1:
            return []
        if not tenant_id or not provider.strip() or not model.strip():
            raise ValueError("tenant_id, provider and model are required for ANN search")
        if model_revision is not None and not model_revision.strip():
            raise ValueError("model_revision must be non-empty when provided")
        if ef_search < 1 or candidate_multiplier < 1:
            raise ValueError("ef_search and candidate_multiplier must be positive")

        query = self._validated_vector(query_embedding)
        try:
            import hnswlib
        except ImportError as exc:
            raise RuntimeError(
                "HNSW support is optional; install with pip install infinitydb-next[ann]"
            ) from exc

        key = (tenant_id, len(query), provider, model, model_revision)
        now = time.time() if as_of is None else as_of
        kind_values = list(kinds or [])
        kind_clause = (
            " AND m.kind IN (" + ",".join("?" for _ in kind_values) + ")"
            if kind_values else ""
        )

        with self._lock:
            cached = self._ann_cache.get(key)
            if cached is None:
                source_rows = self.db.execute(
                    """SELECT memory_id,vector_json FROM embeddings
                       WHERE tenant_id=? AND dimensions=? AND provider=? AND model=?
                         AND model_revision IS ?
                       ORDER BY memory_id""",
                    key,
                ).fetchall()
                if not source_rows:
                    return []
                labels = [row["memory_id"] for row in source_rows]
                fingerprint = self._ann_source_fingerprint(source_rows)
                cached = self._load_persisted_ann(hnswlib, key, fingerprint, labels)
                if cached is None:
                    vectors = [json.loads(row["vector_json"]) for row in source_rows]
                    index = hnswlib.Index(space="cosine", dim=len(query))
                    index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
                    index.add_items(vectors, list(range(len(vectors))), num_threads=1)
                    cached = (index, labels)
                    # SQLite is authoritative; native indexes are recoverable derivatives.
                    self._save_persisted_ann(index, key, fingerprint, labels)
                index, labels = cached
                index.set_ef(max(ef_search, min(len(labels), 64)))
                self._ann_cache[key] = cached

            index, labels = cached
            total = len(labels)
            candidate_count = min(total, max(limit * candidate_multiplier, 64, ef_search))
            query_norm = math.sqrt(sum(value * value for value in query))
            scored: list[tuple[sqlite3.Row, float]] = []

            while True:
                index.set_ef(max(ef_search, candidate_count))
                found, _ = index.knn_query([query], k=candidate_count, num_threads=1)
                memory_ids = [labels[int(label)] for label in found[0]]
                eligible_rows: list[sqlite3.Row] = []
                for offset in range(0, len(memory_ids), 400):
                    chunk = memory_ids[offset:offset + 400]
                    placeholders = ",".join("?" for _ in chunk)
                    eligible_rows.extend(self.db.execute(
                        f"""SELECT m.*,e.vector_json FROM embeddings e
                            JOIN memories m ON m.id=e.memory_id AND m.tenant_id=e.tenant_id
                            WHERE e.tenant_id=? AND e.provider=? AND e.model=?
                              AND e.model_revision IS ? AND e.memory_id IN ({placeholders})
                              AND m.valid_from<=? AND (m.valid_to IS NULL OR m.valid_to>?)
                              {kind_clause}""",
                        [tenant_id, provider, model, model_revision, *chunk,
                         now, now, *kind_values],
                    ).fetchall())

                scored = []
                for row in eligible_rows:
                    vector = json.loads(row["vector_json"])
                    norm = math.sqrt(sum(value * value for value in vector))
                    if norm == 0 or len(vector) != len(query):
                        continue
                    cosine = sum(a * b for a, b in zip(query, vector)) / (query_norm * norm)
                    score = max(-1.0, min(1.0, cosine)) * float(row["confidence"])
                    scored.append((row, score))

                if len(scored) >= limit or candidate_count >= total:
                    break
                candidate_count = min(total, candidate_count * 2)

            top_rows = heapq.nsmallest(
                limit, scored,
                key=lambda item: (-item[1], -item[0]["created_at"], item[0]["id"]),
            )
            return [
                SearchHit(self._to_memory(row), score, ["ann_vector"])
                for row, score in top_rows
            ]

    def search_vector(self, query_embedding: Iterable[float], *,
                      tenant_id: str = "default", limit: int = 10,
                      kinds: Iterable[str] | None = None,
                      as_of: float | None = None,
                      provider: str | None = None, model: str | None = None,
                      model_revision: str | None = None) -> list[SearchHit]:
        if limit < 1:
            return []
        if model_revision is not None and (provider is None or model is None):
            raise ValueError("provider and model are required when filtering by model_revision")
        query = self._validated_vector(query_embedding)
        now = time.time() if as_of is None else as_of
        kind_values = list(kinds or [])
        kind_clause = (" AND m.kind IN (" + ",".join("?" for _ in kind_values) + ")") if kind_values else ""
        model_filters = ""
        model_params: list[Any] = []
        if provider is not None:
            model_filters += " AND e.provider=?"
            model_params.append(provider)
        if model is not None:
            model_filters += " AND e.model=?"
            model_params.append(model)
        if model_revision is not None:
            model_filters += " AND e.model_revision=?"
            model_params.append(model_revision)
        with self._lock:
            rows = self.db.execute(
                f"""SELECT m.*, e.vector_json FROM embeddings e
                    JOIN memories m ON m.id=e.memory_id AND m.tenant_id=e.tenant_id
                    WHERE e.tenant_id=? AND e.dimensions=? AND m.valid_from<=?
                    AND (m.valid_to IS NULL OR m.valid_to>?) {kind_clause} {model_filters}""",
                [tenant_id, len(query), now, now, *kind_values, *model_params],
            ).fetchall()
        query_norm = math.sqrt(sum(value * value for value in query))
        scored: list[tuple[sqlite3.Row, float]] = []
        for row in rows:
            vector = json.loads(row["vector_json"])
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0 or len(vector) != len(query):
                continue
            similarity = sum(a * b for a, b in zip(query, vector)) / (query_norm * norm)
            score = max(-1.0, min(1.0, similarity)) * float(row["confidence"])
            scored.append((row, score))
        scored.sort(key=lambda item: (-item[1], -item[0]["created_at"], item[0]["id"]))
        return [SearchHit(self._to_memory(row), score, ["exact_vector"])
                for row, score in scored[:limit]]

    def search_hybrid(
        self, query: str, query_embedding: Iterable[float], *,
        tenant_id: str = "default", limit: int = 10,
        kinds: Iterable[str] | None = None, as_of: float | None = None,
        rrf_k: int = 60, candidate_limit: int | None = None,
        provider: str | None = None, model: str | None = None,
        model_revision: str | None = None,
    ) -> list[SearchHit]:
        """Fuse lexical and exact-vector rankings with deterministic RRF.

        Both candidate generators enforce tenant, temporal and kind filters.
        Scores are reciprocal-rank-fusion scores, not calibrated probabilities.
        """
        if limit < 1:
            return []
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        candidates = candidate_limit if candidate_limit is not None else max(limit * 4, 20)
        if candidates < limit:
            raise ValueError("candidate_limit must be greater than or equal to limit")

        lexical = self.search(
            query, tenant_id=tenant_id, limit=candidates, kinds=kinds, as_of=as_of
        )
        semantic = self.search_vector(
            query_embedding, tenant_id=tenant_id, limit=candidates,
            kinds=kinds, as_of=as_of, provider=provider, model=model,
            model_revision=model_revision,
        )
        fused: dict[str, tuple[Memory, float, set[str]]] = {}
        for ranked_hits in (lexical, semantic):
            for rank, hit in enumerate(ranked_hits, 1):
                memory_id = hit.memory.id
                prior = fused.get(memory_id)
                score = 1.0 / (rrf_k + rank)
                if prior is None:
                    fused[memory_id] = (hit.memory, score, set(hit.matched_by))
                else:
                    memory, total, methods = prior
                    methods.update(hit.matched_by)
                    fused[memory_id] = (memory, total + score, methods)
        results = [
            SearchHit(memory, score, sorted(methods))
            for memory, score, methods in fused.values()
        ]
        results.sort(key=lambda hit: (-hit.score, -hit.memory.created_at, hit.memory.id))
        return results[:limit]

    def get(self, memory_id: str, *, tenant_id: str = "default") -> Memory | None:
        with self._lock:
            row = self.db.execute("SELECT * FROM memories WHERE id=? AND tenant_id=?",
                                  (memory_id, tenant_id)).fetchone()
        return self._to_memory(row) if row else None

    def evidence_for(self, memory_id: str, *, tenant_id: str = "default") -> list[dict[str, Any]]:
        with self._lock:
            rows = self.db.execute(
                """SELECT id,raw_content,sha256,previous_hash,chain_hash,created_at
                   FROM evidence WHERE memory_id=? AND tenant_id=? ORDER BY created_at,rowid""",
                (memory_id, tenant_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def verify_evidence_chain(self, *, tenant_id: str = "default") -> bool:
        with self._lock:
            rows = self.db.execute(
                "SELECT * FROM evidence WHERE tenant_id=? ORDER BY created_at,rowid", (tenant_id,)
            ).fetchall()
        previous = "0" * 64
        for row in rows:
            digest = hashlib.sha256(row["raw_content"].encode()).hexdigest()
            expected = hashlib.sha256(
                (previous + digest + row["id"] + row["memory_id"]).encode()
            ).hexdigest()
            if digest != row["sha256"] or row["previous_hash"] != previous or row["chain_hash"] != expected:
                return False
            previous = expected
        return True

    def search(self, query: str, *, tenant_id: str = "default", limit: int = 10,
               kinds: Iterable[str] | None = None, as_of: float | None = None) -> list[SearchHit]:
        if limit < 1:
            return []
        terms = list(dict.fromkeys(term.casefold() for term in query.split() if term))
        if not terms:
            return []
        now = time.time() if as_of is None else as_of
        kind_values = list(kinds or [])
        kind_clause = (" AND m.kind IN (" + ",".join("?" for _ in kind_values) + ")") if kind_values else ""
        ranked: dict[str, tuple[sqlite3.Row, float, set[str]]] = {}
        with self._lock:
            if self._fts:
                quoted = ['"' + term.replace('"', '""') + '"' for term in terms]
                for operator in (" AND ", " OR "):
                    match = operator.join(quoted)
                    sql = f"""SELECT m.* FROM memory_fts_v2 f JOIN memories m ON m.id=f.memory_id
                        WHERE memory_fts_v2 MATCH ? AND f.tenant_key=? AND m.tenant_id=?
                        AND m.valid_from<=? AND (m.valid_to IS NULL OR m.valid_to>?)
                        {kind_clause} ORDER BY f.rowid DESC LIMIT ?"""
                    params: list[Any] = [match, self._tenant_key(tenant_id), tenant_id, now, now]
                    params.extend(kind_values)
                    params.append(max(limit * 20, 100))
                    try:
                        rows = self.db.execute(sql, params).fetchall()
                    except sqlite3.OperationalError:
                        rows = []
                    for rank, row in enumerate(rows, 1):
                        ranked[row["id"]] = (row, 1.0 / (60 + rank), {"fts5_and" if operator == " AND " else "fts5_or"})
                    if rows:
                        break
            if not ranked:
                params = [tenant_id, now, now, *kind_values]
                rows = self.db.execute(
                    f"""SELECT * FROM memories m WHERE tenant_id=? AND valid_from<=?
                        AND (valid_to IS NULL OR valid_to>?) {kind_clause}""", params
                ).fetchall()
                for row in rows:
                    matches = sum(term in row["content"].casefold() for term in terms)
                    if matches:
                        ranked[row["id"]] = (row, matches / len(terms), {"substring"})
        hits = [
            SearchHit(self._to_memory(row), score * float(row["confidence"]), sorted(methods))
            for row, score, methods in ranked.values()
        ]
        hits.sort(key=lambda hit: (-hit.score, -hit.memory.created_at, hit.memory.id))
        return hits[:limit]

    def context_pack(self, query: str, *, tenant_id: str = "default",
                     token_budget: int = 1200, limit: int = 20) -> ContextPack:
        hits = self.search(query, tenant_id=tenant_id, limit=limit)
        selected: list[Memory] = []
        evidence_ids: list[str] = []
        used = 0
        for hit in hits:
            cost = max(1, math.ceil(len(hit.memory.content) / 4))
            if used + cost > token_budget:
                continue
            selected.append(hit.memory)
            used += cost
            evidence_ids.extend(item["id"] for item in self.evidence_for(hit.memory.id, tenant_id=tenant_id))
        return ContextPack(query, selected, evidence_ids, used, time.time())

    def stats(self, *, tenant_id: str = "default") -> dict[str, int]:
        with self._lock:
            return {
                "memories": self.db.execute("SELECT count(*) FROM memories WHERE tenant_id=?", (tenant_id,)).fetchone()[0],
                "evidence": self.db.execute("SELECT count(*) FROM evidence WHERE tenant_id=?", (tenant_id,)).fetchone()[0],
                "edges": self.db.execute("SELECT count(*) FROM edges WHERE tenant_id=?", (tenant_id,)).fetchone()[0],
            }

    @staticmethod
    def _to_memory(row: sqlite3.Row) -> Memory:
        return Memory(row["id"], row["tenant_id"], row["kind"], row["content"],
                      row["created_at"], row["valid_from"], row["valid_to"],
                      json.loads(row["metadata_json"]), row["source_id"], row["confidence"])
