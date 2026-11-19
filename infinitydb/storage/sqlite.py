"""The standard-library SQLite backend for the storage contract.

Nothing outside the standard library is imported, so ``dependencies = []`` and
the default ``pip install infinitydb-ai`` keep working. The tables are prefixed
(``oams_`` by default) so this backend can share one database file with the core
engine's own ``l1_atoms``/``l0_logs`` tables.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from .base import (
    GENESIS_HASH,
    CausalEdge,
    FactAtom,
    PolicyRule,
    StorageBackend,
    TurnRecord,
    _require_identifier,
    _require_positive_int,
    normalise_embedding,
    payload_hash,
)

_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_ATOM_COLUMNS = (
    "atom_id, tenant_id, user_id, session_id, source_turn_id, fact_text, "
    "subject, predicate, object_val, valid_at, invalid_at, recorded_at, "
    "confidence, stability, access_count, last_accessed_at, prev_hash, "
    "payload_hash, embedding"
)


def _json_load(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


class SqliteStorage(StorageBackend):
    """SQLite implementation of :class:`StorageBackend` (stdlib only).

    One connection is shared and guarded by a re-entrant lock, matching the core
    engine's ``check_same_thread=False`` contract: the connection is safe to hand
    to multiple threads, and writes are serialised so the L6 chain cannot fork.
    """

    def __init__(self, db_path: str = ":memory:", *, table_prefix: str = "oams_") -> None:
        if not _PREFIX_RE.match(table_prefix):
            raise ValueError("table_prefix must match [A-Za-z_][A-Za-z0-9_]*")
        self.db_path = db_path
        self._prefix = table_prefix
        self._turns = f"{table_prefix}turns"
        self._atoms = f"{table_prefix}atoms"
        self._atoms_fts = f"{table_prefix}atoms_fts"
        self._edges = f"{table_prefix}edges"
        self._personas = f"{table_prefix}personas"
        self._policies = f"{table_prefix}policies"
        self._migrations = f"{table_prefix}schema_migrations"
        self._lock = threading.RLock()
        self._ready = False
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = FULL")

    @property
    def backend_name(self) -> str:
        return "sqlite"

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if self._ready:
                return
            self._conn.executescript(self._schema_sql())
            self._conn.execute(
                f"INSERT OR IGNORE INTO {self._migrations}(version, applied_at, description) "
                "VALUES (1, ?, ?)",
                (0.0, "oams-1.0 baseline"),
            )
            self._ready = True

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("SqliteStorage.initialize() must be called first")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
            self._ready = False

    def __enter__(self) -> SqliteStorage:
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _schema_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self._migrations} (
            version     INTEGER PRIMARY KEY,
            applied_at  REAL NOT NULL,
            description TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {self._turns} (
            tenant_id   TEXT NOT NULL,
            user_id     TEXT NOT NULL,
            session_id  TEXT NOT NULL,
            turn_index  INTEGER NOT NULL,
            turn_id     TEXT NOT NULL,
            speaker     TEXT NOT NULL,
            content     TEXT NOT NULL,
            occurred_at REAL NOT NULL,
            recorded_at REAL NOT NULL,
            metadata    TEXT NOT NULL DEFAULT '{{}}',
            PRIMARY KEY (tenant_id, user_id, session_id, turn_index),
            UNIQUE (tenant_id, user_id, turn_id)
        );
        CREATE INDEX IF NOT EXISTS idx_{self._prefix}turns_occurred
            ON {self._turns} (tenant_id, user_id, occurred_at DESC);

        CREATE TABLE IF NOT EXISTS {self._atoms} (
            tenant_id        TEXT NOT NULL,
            user_id          TEXT NOT NULL,
            atom_id          TEXT NOT NULL,
            chain_seq        INTEGER NOT NULL,
            session_id       TEXT NOT NULL,
            source_turn_id   TEXT NOT NULL,
            fact_text        TEXT NOT NULL,
            subject          TEXT,
            predicate        TEXT,
            object_val       TEXT,
            valid_at         REAL NOT NULL,
            invalid_at       REAL,
            recorded_at      REAL NOT NULL,
            confidence       REAL NOT NULL DEFAULT 1.0,
            stability        REAL NOT NULL DEFAULT 1.0,
            access_count     INTEGER NOT NULL DEFAULT 0,
            last_accessed_at REAL NOT NULL,
            prev_hash        TEXT NOT NULL,
            payload_hash     TEXT NOT NULL,
            embedding        TEXT,
            PRIMARY KEY (tenant_id, user_id, atom_id),
            UNIQUE (tenant_id, user_id, chain_seq),
            UNIQUE (tenant_id, user_id, payload_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_{self._prefix}atoms_active_temporal
            ON {self._atoms} (tenant_id, user_id, valid_at DESC);
        CREATE INDEX IF NOT EXISTS idx_{self._prefix}atoms_chain_seq
            ON {self._atoms} (tenant_id, user_id, chain_seq DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS {self._atoms_fts}
            USING fts5(atom_id UNINDEXED, fact_text,
                       tokenize='unicode61 remove_diacritics 2');
        CREATE TRIGGER IF NOT EXISTS {self._prefix}atoms_fts_insert
            AFTER INSERT ON {self._atoms} BEGIN
                INSERT INTO {self._atoms_fts}(atom_id, fact_text)
                VALUES (new.atom_id, new.fact_text);
            END;
        CREATE TRIGGER IF NOT EXISTS {self._prefix}atoms_fts_delete
            AFTER DELETE ON {self._atoms} BEGIN
                DELETE FROM {self._atoms_fts} WHERE atom_id = old.atom_id;
            END;
        CREATE TRIGGER IF NOT EXISTS {self._prefix}atoms_fts_update
            AFTER UPDATE OF fact_text ON {self._atoms} BEGIN
                DELETE FROM {self._atoms_fts} WHERE atom_id = old.atom_id;
                INSERT INTO {self._atoms_fts}(atom_id, fact_text)
                VALUES (new.atom_id, new.fact_text);
            END;

        CREATE TABLE IF NOT EXISTS {self._edges} (
            tenant_id     TEXT NOT NULL,
            user_id       TEXT NOT NULL,
            edge_id       TEXT NOT NULL,
            src_atom_id   TEXT NOT NULL,
            dst_atom_id   TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            weight        REAL NOT NULL DEFAULT 1.0,
            valid_at      REAL NOT NULL,
            recorded_at   REAL NOT NULL,
            PRIMARY KEY (tenant_id, user_id, edge_id),
            FOREIGN KEY (tenant_id, user_id, src_atom_id)
                REFERENCES {self._atoms} (tenant_id, user_id, atom_id) ON DELETE CASCADE,
            FOREIGN KEY (tenant_id, user_id, dst_atom_id)
                REFERENCES {self._atoms} (tenant_id, user_id, atom_id) ON DELETE CASCADE,
            CHECK (src_atom_id <> dst_atom_id)
        );
        CREATE INDEX IF NOT EXISTS idx_{self._prefix}edges_src
            ON {self._edges} (tenant_id, user_id, src_atom_id);
        CREATE INDEX IF NOT EXISTS idx_{self._prefix}edges_dst
            ON {self._edges} (tenant_id, user_id, dst_atom_id);

        CREATE TABLE IF NOT EXISTS {self._personas} (
            tenant_id    TEXT NOT NULL,
            user_id      TEXT NOT NULL,
            persona_json TEXT NOT NULL DEFAULT '{{}}',
            updated_at   REAL NOT NULL,
            PRIMARY KEY (tenant_id, user_id)
        );

        CREATE TABLE IF NOT EXISTS {self._policies} (
            tenant_id  TEXT NOT NULL,
            policy_id  TEXT NOT NULL,
            user_id    TEXT,
            rule_type  TEXT NOT NULL,
            pattern    TEXT NOT NULL,
            action     TEXT NOT NULL,
            priority   INTEGER NOT NULL DEFAULT 100,
            enabled    INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            PRIMARY KEY (tenant_id, policy_id)
        );
        CREATE INDEX IF NOT EXISTS idx_{self._prefix}policies_lookup
            ON {self._policies} (tenant_id, enabled, priority ASC);
        """

    # -- low-level helpers -------------------------------------------------

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One serialised write transaction, so chain appends cannot interleave."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    def _rows(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, tuple(params)).fetchall())

    def _latest_hash(self, tenant_id: str, user_id: str) -> str:
        row = self._conn.execute(
            f"SELECT payload_hash FROM {self._atoms} WHERE tenant_id = ? AND user_id = ? "
            "ORDER BY chain_seq DESC LIMIT 1",
            (tenant_id, user_id),
        ).fetchone()
        return row["payload_hash"] if row else GENESIS_HASH

    @staticmethod
    def _atom_from_row(row: sqlite3.Row, **scores: float) -> FactAtom:
        atom = FactAtom(
            atom_id=row["atom_id"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            source_turn_id=row["source_turn_id"],
            fact_text=row["fact_text"],
            subject=row["subject"],
            predicate=row["predicate"],
            object_val=row["object_val"],
            valid_at=row["valid_at"],
            invalid_at=row["invalid_at"],
            recorded_at=row["recorded_at"],
            confidence=row["confidence"],
            stability=row["stability"],
            access_count=row["access_count"],
            last_accessed_at=row["last_accessed_at"],
            prev_hash=row["prev_hash"],
            payload_hash=row["payload_hash"],
            embedding=_json_load(row["embedding"], None),
        )
        atom.lexical_score = float(scores.get("lexical_score", 0.0))
        atom.dense_score = float(scores.get("dense_score", 0.0))
        return atom

    # -- L0 ----------------------------------------------------------------

    def append_turns(self, turns: Sequence[TurnRecord]) -> None:
        if not turns:
            return
        rows = []
        for turn in turns:
            tenant_id = _require_identifier(turn.tenant_id, "tenant_id")
            user_id = _require_identifier(turn.user_id, "user_id")
            rows.append(
                (
                    tenant_id,
                    user_id,
                    turn.session_id,
                    int(turn.turn_index),
                    turn.turn_id,
                    turn.speaker,
                    turn.content,
                    float(turn.occurred_at),
                    float(turn.recorded_at),
                    json.dumps(turn.metadata, sort_keys=True, ensure_ascii=False),
                )
            )
        with self._lock:
            self._require_ready()
            with self._write() as conn:
                conn.executemany(
                    f"INSERT OR IGNORE INTO {self._turns} "
                    "(tenant_id, user_id, session_id, turn_index, turn_id, speaker, "
                    "content, occurred_at, recorded_at, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )

    def get_turn_window(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        center_index: int,
        window_radius: int = 1,
    ) -> list[TurnRecord]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        radius = _require_positive_int(window_radius, "window_radius", 1000)
        if isinstance(center_index, bool) or not isinstance(center_index, int):
            raise ValueError("center_index must be an integer")
        self._require_ready()
        rows = self._rows(
            f"SELECT * FROM {self._turns} WHERE tenant_id = ? AND user_id = ? "
            "AND session_id = ? AND turn_index BETWEEN ? AND ? ORDER BY turn_index ASC",
            (
                tenant_id,
                user_id,
                session_id,
                center_index - radius,
                center_index + radius,
            ),
        )
        return [
            TurnRecord(
                turn_id=row["turn_id"],
                tenant_id=row["tenant_id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                turn_index=row["turn_index"],
                speaker=row["speaker"],
                content=row["content"],
                occurred_at=row["occurred_at"],
                recorded_at=row["recorded_at"],
                metadata=_json_load(row["metadata"], {}),
            )
            for row in rows
        ]

    # -- L1 + L6 -----------------------------------------------------------

    def get_latest_chain_hash(self, tenant_id: str, user_id: str) -> str:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        self._require_ready()
        return self._latest_hash(tenant_id, user_id)

    def insert_atoms(self, atoms: Sequence[FactAtom]) -> None:
        if not atoms:
            return
        tenant_id = _require_identifier(atoms[0].tenant_id, "tenant_id")
        user_id = _require_identifier(atoms[0].user_id, "user_id")
        for atom in atoms:
            if atom.tenant_id != tenant_id or atom.user_id != user_id:
                raise ValueError("insert_atoms() requires a single (tenant_id, user_id) scope")
            if not atom.atom_id:
                raise ValueError("atom_id must be a non-empty string")
            if not atom.fact_text:
                raise ValueError("fact_text must be a non-empty string")

        with self._lock:
            self._require_ready()
            with self._write() as conn:
                previous = self._latest_hash(tenant_id, user_id)
                row = conn.execute(
                    f"SELECT COALESCE(MAX(chain_seq) + 1, 0) AS next FROM {self._atoms} "
                    "WHERE tenant_id = ? AND user_id = ?",
                    (tenant_id, user_id),
                ).fetchone()
                seq = int(row["next"])
                for atom in atoms:
                    atom.prev_hash = previous
                    atom.payload_hash = payload_hash(previous, atom)
                    embedding = (
                        json.dumps(normalise_embedding(atom.embedding))
                        if atom.embedding is not None
                        else None
                    )
                    conn.execute(
                        f"INSERT INTO {self._atoms} ({_ATOM_COLUMNS}, chain_seq) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            atom.atom_id,
                            tenant_id,
                            user_id,
                            atom.session_id,
                            atom.source_turn_id,
                            atom.fact_text,
                            atom.subject,
                            atom.predicate,
                            atom.object_val,
                            float(atom.valid_at),
                            None if atom.invalid_at is None else float(atom.invalid_at),
                            float(atom.recorded_at),
                            float(atom.confidence),
                            float(atom.stability),
                            int(atom.access_count),
                            float(atom.last_accessed_at),
                            atom.prev_hash,
                            atom.payload_hash,
                            embedding,
                            seq,
                        ),
                    )
                    previous = atom.payload_hash
                    seq += 1

    def invalidate_atom(
        self,
        tenant_id: str,
        user_id: str,
        atom_id: str,
        invalid_at: float,
        superseded_by: str | None = None,
    ) -> None:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        with self._lock:
            self._require_ready()
            with self._write() as conn:
                updated = conn.execute(
                    f"UPDATE {self._atoms} SET invalid_at = ? "
                    "WHERE tenant_id = ? AND user_id = ? AND atom_id = ? AND invalid_at IS NULL",
                    (float(invalid_at), tenant_id, user_id, atom_id),
                ).rowcount
                if not updated or not superseded_by:
                    return
                exists = conn.execute(
                    f"SELECT 1 FROM {self._atoms} WHERE tenant_id = ? AND user_id = ? AND atom_id = ?",
                    (tenant_id, user_id, superseded_by),
                ).fetchone()
                if not exists:
                    return
                conn.execute(
                    f"INSERT OR IGNORE INTO {self._edges} "
                    "(tenant_id, user_id, edge_id, src_atom_id, dst_atom_id, "
                    "relation_type, weight, valid_at, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, 'SUPERSEDES', 1.0, ?, ?)",
                    (
                        tenant_id,
                        user_id,
                        f"supersedes:{superseded_by}:{atom_id}",
                        superseded_by,
                        atom_id,
                        float(invalid_at),
                        float(invalid_at),
                    ),
                )

    def get_atom_proof_chain(
        self, tenant_id: str, user_id: str, atom_id: str
    ) -> list[FactAtom]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        self._require_ready()
        target = self._conn.execute(
            f"SELECT chain_seq FROM {self._atoms} "
            "WHERE tenant_id = ? AND user_id = ? AND atom_id = ?",
            (tenant_id, user_id, atom_id),
        ).fetchone()
        if target is None:
            return []
        rows = self._rows(
            f"SELECT {_ATOM_COLUMNS} FROM {self._atoms} "
            "WHERE tenant_id = ? AND user_id = ? AND chain_seq <= ? ORDER BY chain_seq ASC",
            (tenant_id, user_id, target["chain_seq"]),
        )
        return [self._atom_from_row(row) for row in rows]

    # -- L2 ----------------------------------------------------------------

    def upsert_edges(self, edges: Sequence[CausalEdge]) -> None:
        if not edges:
            return
        tenant_id = _require_identifier(edges[0].tenant_id, "tenant_id")
        user_id = _require_identifier(edges[0].user_id, "user_id")
        rows = []
        for edge in edges:
            if edge.tenant_id != tenant_id or edge.user_id != user_id:
                raise ValueError("upsert_edges() requires a single (tenant_id, user_id) scope")
            rows.append(
                (
                    tenant_id,
                    user_id,
                    edge.edge_id,
                    edge.src_atom_id,
                    edge.dst_atom_id,
                    edge.relation_type,
                    float(edge.weight),
                    float(edge.valid_at),
                    float(edge.recorded_at),
                )
            )
        with self._lock:
            self._require_ready()
            with self._write() as conn:
                conn.executemany(
                    f"INSERT INTO {self._edges} "
                    "(tenant_id, user_id, edge_id, src_atom_id, dst_atom_id, "
                    "relation_type, weight, valid_at, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (tenant_id, user_id, edge_id) DO UPDATE SET "
                    "weight = max(weight, excluded.weight)",
                    rows,
                )

    def traverse_neighbors(
        self,
        tenant_id: str,
        user_id: str,
        seed_atom_ids: Sequence[str],
        max_hops: int = 1,
        limit: int = 20,
    ) -> list[tuple[FactAtom, CausalEdge]]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        hops = _require_positive_int(max_hops, "max_hops", 10)
        limit = _require_positive_int(limit, "limit", 1000)
        seeds = [atom_id for atom_id in dict.fromkeys(seed_atom_ids) if atom_id]
        self._require_ready()
        if not seeds:
            return []

        found: list[tuple[FactAtom, CausalEdge]] = []
        seen = set(seeds)
        frontier = list(seeds)
        for depth in range(1, hops + 1):
            placeholders = ",".join("?" * len(frontier))
            rows = self._rows(
                f"SELECT e.edge_id AS e_edge_id, e.src_atom_id, e.dst_atom_id, "
                "e.relation_type, e.weight, e.valid_at AS e_valid_at, "
                "e.recorded_at AS e_recorded_at, a.* "
                f"FROM {self._edges} e JOIN {self._atoms} a "
                "ON a.tenant_id = e.tenant_id AND a.user_id = e.user_id "
                "AND a.atom_id = e.dst_atom_id "
                f"WHERE e.tenant_id = ? AND e.user_id = ? AND e.src_atom_id IN ({placeholders}) "
                "ORDER BY e.weight DESC, e.valid_at DESC",
                (tenant_id, user_id, *frontier),
            )
            frontier = []
            for row in rows:
                if len(found) >= limit:
                    return found
                dst = row["dst_atom_id"]
                if dst in seen:
                    continue
                seen.add(dst)
                frontier.append(dst)
                edge = CausalEdge(
                    edge_id=row["e_edge_id"],
                    tenant_id=tenant_id,
                    user_id=user_id,
                    src_atom_id=row["src_atom_id"],
                    dst_atom_id=dst,
                    relation_type=row["relation_type"],
                    weight=row["weight"],
                    valid_at=row["e_valid_at"],
                    recorded_at=row["e_recorded_at"],
                )
                found.append((self._atom_from_row(row), edge))
            if not frontier:
                break
        return found

    # -- retrieval ---------------------------------------------------------

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = re.findall(r"[A-Za-z0-9_]+", query or "")
        return " OR ".join(f'"{term}"' for term in terms)

    def search_lexical(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int = 30,
        include_invalidated: bool = False,
    ) -> list[FactAtom]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        limit = _require_positive_int(limit, "limit", 1000)
        self._require_ready()
        match = self._fts_query(query)
        if not match:
            return []
        invalidated = "" if include_invalidated else " AND a.invalid_at IS NULL"
        rows = self._rows(
            f"SELECT a.*, bm25({self._atoms_fts}) AS rank FROM {self._atoms_fts} f "
            f"JOIN {self._atoms} a ON a.atom_id = f.atom_id AND a.tenant_id = ? AND a.user_id = ? "
            f"WHERE {self._atoms_fts} MATCH ?{invalidated} "
            "ORDER BY rank ASC LIMIT ?",
            (tenant_id, user_id, match, limit),
        )
        return [self._atom_from_row(row, lexical_score=-float(row["rank"])) for row in rows]

    def upsert_embeddings(
        self,
        tenant_id: str,
        user_id: str,
        atom_embeddings: Sequence[tuple[str, list[float]]],
    ) -> None:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        rows = [
            (json.dumps(normalise_embedding(vector)), tenant_id, user_id, atom_id)
            for atom_id, vector in atom_embeddings
        ]
        if not rows:
            return
        with self._lock:
            self._require_ready()
            with self._write() as conn:
                conn.executemany(
                    f"UPDATE {self._atoms} SET embedding = ? "
                    "WHERE tenant_id = ? AND user_id = ? AND atom_id = ?",
                    rows,
                )

    def search_dense(
        self,
        tenant_id: str,
        user_id: str,
        query_embedding: Sequence[float],
        limit: int = 30,
        include_invalidated: bool = False,
    ) -> list[FactAtom]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        limit = _require_positive_int(limit, "limit", 1000)
        query = normalise_embedding(query_embedding)
        self._require_ready()
        invalidated = "" if include_invalidated else " AND invalid_at IS NULL"
        rows = self._rows(
            f"SELECT {_ATOM_COLUMNS} FROM {self._atoms} "
            f"WHERE tenant_id = ? AND user_id = ? AND embedding IS NOT NULL{invalidated}",
            (tenant_id, user_id),
        )
        scored = []
        for row in rows:
            vector = _json_load(row["embedding"], None)
            if not vector or len(vector) != len(query):
                continue
            score = sum(a * b for a, b in zip(query, vector))
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            self._atom_from_row(row, dense_score=score)
            for score, row in scored[:limit]
        ]

    # -- L3 ----------------------------------------------------------------

    def get_persona_state(self, tenant_id: str, user_id: str) -> dict[str, Any]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        self._require_ready()
        row = self._conn.execute(
            f"SELECT persona_json FROM {self._personas} WHERE tenant_id = ? AND user_id = ?",
            (tenant_id, user_id),
        ).fetchone()
        return _json_load(row["persona_json"], {}) if row else {}

    def upsert_persona_state(
        self, tenant_id: str, user_id: str, state: dict[str, Any], updated_at: float
    ) -> None:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        if not isinstance(state, dict):
            raise ValueError("state must be a dict")
        with self._lock:
            self._require_ready()
            with self._write() as conn:
                row = conn.execute(
                    f"SELECT persona_json FROM {self._personas} "
                    "WHERE tenant_id = ? AND user_id = ?",
                    (tenant_id, user_id),
                ).fetchone()
                merged = _json_load(row["persona_json"], {}) if row else {}
                merged.update(state)
                conn.execute(
                    f"INSERT INTO {self._personas} (tenant_id, user_id, persona_json, updated_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT (tenant_id, user_id) DO UPDATE SET "
                    "persona_json = excluded.persona_json, updated_at = excluded.updated_at",
                    (tenant_id, user_id, json.dumps(merged, ensure_ascii=False), float(updated_at)),
                )

    # -- L4 ----------------------------------------------------------------

    def list_active_policies(
        self, tenant_id: str, user_id: str | None = None
    ) -> list[PolicyRule]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        self._require_ready()
        if user_id is None:
            rows = self._rows(
                f"SELECT * FROM {self._policies} WHERE tenant_id = ? AND enabled = 1 "
                "AND user_id IS NULL ORDER BY priority ASC, policy_id ASC",
                (tenant_id,),
            )
        else:
            user_id = _require_identifier(user_id, "user_id")
            rows = self._rows(
                f"SELECT * FROM {self._policies} WHERE tenant_id = ? AND enabled = 1 "
                "AND (user_id IS NULL OR user_id = ?) "
                "ORDER BY priority ASC, policy_id ASC",
                (tenant_id, user_id),
            )
        return [
            PolicyRule(
                policy_id=row["policy_id"],
                tenant_id=row["tenant_id"],
                user_id=row["user_id"],
                rule_type=row["rule_type"],
                pattern=row["pattern"],
                action=row["action"],
                priority=row["priority"],
                enabled=bool(row["enabled"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def upsert_policy(self, policy: PolicyRule) -> None:
        tenant_id = _require_identifier(policy.tenant_id, "tenant_id")
        with self._lock:
            self._require_ready()
            with self._write() as conn:
                conn.execute(
                    f"INSERT INTO {self._policies} "
                    "(tenant_id, policy_id, user_id, rule_type, pattern, action, "
                    "priority, enabled, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (tenant_id, policy_id) DO UPDATE SET "
                    "user_id = excluded.user_id, rule_type = excluded.rule_type, "
                    "pattern = excluded.pattern, action = excluded.action, "
                    "priority = excluded.priority, enabled = excluded.enabled",
                    (
                        tenant_id,
                        policy.policy_id,
                        policy.user_id,
                        policy.rule_type,
                        policy.pattern,
                        policy.action,
                        int(policy.priority),
                        1 if policy.enabled else 0,
                        float(policy.created_at),
                    ),
                )

    # -- L5 ----------------------------------------------------------------

    def reinforce_atoms(
        self,
        tenant_id: str,
        user_id: str,
        atom_ids: Sequence[str],
        accessed_at: float,
        stability_multiplier: float = 1.5,
    ) -> None:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        multiplier = float(stability_multiplier)
        if not multiplier > 0:
            raise ValueError("stability_multiplier must be positive")
        ids = [atom_id for atom_id in dict.fromkeys(atom_ids) if atom_id]
        if not ids:
            return
        with self._lock:
            self._require_ready()
            with self._write() as conn:
                conn.executemany(
                    f"UPDATE {self._atoms} SET access_count = access_count + 1, "
                    "last_accessed_at = ?, stability = stability * ? "
                    "WHERE tenant_id = ? AND user_id = ? AND atom_id = ?",
                    [(float(accessed_at), multiplier, tenant_id, user_id, atom_id) for atom_id in ids],
                )

    # -- erasure -----------------------------------------------------------

    def delete_user_data(self, tenant_id: str, user_id: str) -> dict[str, int]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        with self._lock:
            self._require_ready()
            with self._write() as conn:
                edges = conn.execute(
                    f"DELETE FROM {self._edges} WHERE tenant_id = ? AND user_id = ?",
                    (tenant_id, user_id),
                ).rowcount
                atoms = conn.execute(
                    f"DELETE FROM {self._atoms} WHERE tenant_id = ? AND user_id = ?",
                    (tenant_id, user_id),
                ).rowcount
                turns = conn.execute(
                    f"DELETE FROM {self._turns} WHERE tenant_id = ? AND user_id = ?",
                    (tenant_id, user_id),
                ).rowcount
                personas = conn.execute(
                    f"DELETE FROM {self._personas} WHERE tenant_id = ? AND user_id = ?",
                    (tenant_id, user_id),
                ).rowcount
                policies = conn.execute(
                    f"DELETE FROM {self._policies} WHERE tenant_id = ? AND user_id = ?",
                    (tenant_id, user_id),
                ).rowcount
        return {
            "turns": turns,
            "atoms": atoms,
            "edges": edges,
            "personas": personas,
            "policies": policies,
        }
