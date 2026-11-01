from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from .models import ContextPack, Memory, SearchHit

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

    def __init__(self, path: str | Path = "infinitydb-next.sqlite3") -> None:
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.RLock()
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA temp_store=MEMORY")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self._fts = self._init_fts()

    @staticmethod
    def _tenant_key(tenant_id: str) -> str:
        return "tenant_" + hashlib.sha256(tenant_id.encode()).hexdigest()[:24]

    def _init_fts(self) -> bool:
        try:
            self.db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts_v2
                USING fts5(memory_id UNINDEXED, tenant_key, content, tokenize='unicode61')""")
            count = self.db.execute("SELECT count(*) FROM memory_fts_v2").fetchone()[0]
            if count == 0:
                rows = self.db.execute("SELECT id,tenant_id,content FROM memories").fetchall()
                self.db.executemany(
                    "INSERT INTO memory_fts_v2 VALUES(?,?,?)",
                    [(r["id"], self._tenant_key(r["tenant_id"]), r["content"]) for r in rows],
                )
            self.db.commit()
            return True
        except sqlite3.OperationalError:
            return False

    def close(self) -> None:
        with self._lock:
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
            if len({row["id"] for row in rows}) != 2:
                raise ValueError("both edge endpoints must exist in the specified tenant")
            self.db.execute("INSERT OR REPLACE INTO edges VALUES(?,?,?,?,?)",
                            (tenant_id, src_id, dst_id, relation, weight))

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
