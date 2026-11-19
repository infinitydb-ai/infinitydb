"""PostgreSQL + pgvector backend for the storage contract.

Install the optional extra to use it::

    pip install "infinitydb-ai[postgres]"

The database needs the ``vector`` extension; ``initialize()`` creates it when the
connected role is allowed to, and raises an actionable error when it is not.

Two things make this backend safe under concurrency:

* every transaction binds ``app.current_tenant_id`` with ``SET LOCAL`` so the
  Row-Level Security policies (enabled and forced by ``initialize()``) can never
  be bypassed by a query that forgets a tenant predicate, and
* ``insert_atoms`` takes a per-user advisory transaction lock before reading the
  chain tip, so the L6 SHA-256 chain stays linear without serialising writes
  across different users.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by the install-error test
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'The Postgres backend needs the optional extra: pip install "infinitydb-ai[postgres]"'
    ) from exc

from .base import (
    GENESIS_HASH,
    CausalEdge,
    FactAtom,
    FeedbackWeights,
    PolicyRule,
    StorageBackend,
    TurnRecord,
    _require_identifier,
    _require_positive_int,
    feedback_weights_from_rows,
    normalise_embedding,
    payload_hash,
)

_ATOM_FIELDS = (
    "atom_id",
    "tenant_id",
    "user_id",
    "session_id",
    "source_turn_id",
    "fact_text",
    "subject",
    "predicate",
    "object_val",
    "valid_at",
    "invalid_at",
    "recorded_at",
    "confidence",
    "stability",
    "access_count",
    "last_accessed_at",
    "prev_hash",
    "payload_hash",
)
_ATOM_COLUMNS = ", ".join(_ATOM_FIELDS) + ", embedding::text AS embedding"
_ATOM_COLUMNS_A = ", ".join(f"a.{field}" for field in _ATOM_FIELDS) + (
    ", a.embedding::text AS embedding"
)
_RLS_TABLES = ("turns", "atoms", "edges", "personas", "policies", "feedback")


def _vector_literal(vector: Sequence[float]) -> str:
    """pgvector's text input format; avoids depending on the pgvector client."""
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


class PostgresStorage(StorageBackend):
    """OAMS-1.0 storage on PostgreSQL 14+ with pgvector."""

    def __init__(
        self,
        dsn: str,
        *,
        table_prefix: str = "oams_",
        embedding_dim: int = 384,
        max_pool_size: int = 4,
    ) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("dsn must be a non-empty PostgreSQL connection string")
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or not 1 <= embedding_dim <= 16000:
            raise ValueError("embedding_dim must be an integer between 1 and 16000")
        if not isinstance(max_pool_size, int) or not 1 <= max_pool_size <= 64:
            raise ValueError("max_pool_size must be an integer between 1 and 64")
        if not table_prefix.replace("_", "a").isalnum() or not table_prefix[0:1].isalpha():
            raise ValueError("table_prefix must match [A-Za-z_][A-Za-z0-9_]*")

        self.dsn = dsn
        self._prefix = table_prefix
        self.embedding_dim = embedding_dim
        self._max_pool_size = max_pool_size
        self._turns = f"{table_prefix}turns"
        self._atoms = f"{table_prefix}atoms"
        self._edges = f"{table_prefix}edges"
        self._personas = f"{table_prefix}personas"
        self._policies = f"{table_prefix}policies"
        self._feedback = f"{table_prefix}feedback"
        self._migrations = f"{table_prefix}schema_migrations"
        self._pool: ConnectionPool | None = None
        self._ready = False

    @property
    def backend_name(self) -> str:
        return "postgres"

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        if self._ready:
            return
        schema = (
            Path(__file__).with_name("schema_pg.sql")
            .read_text(encoding="utf-8")
            .replace("{p}", self._prefix)
            .replace("{dim}", str(self.embedding_dim))
        )
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            try:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except psycopg.Error as exc:
                raise RuntimeError(
                    "PostgresStorage requires the pgvector extension in this database "
                    "(CREATE EXTENSION vector). Ask the database owner to enable it."
                ) from exc
            for statement in self._statements(schema):
                conn.execute(statement)
            for table in _RLS_TABLES:
                self._enable_tenant_policy(conn, f"{self._prefix}{table}")
            conn.execute(
                f"INSERT INTO {self._migrations}(version, applied_at, description) "
                "VALUES (1, %s, %s) ON CONFLICT (version) DO NOTHING",
                (time.time(), "oams-1.0 baseline"),
            )
        self._pool = ConnectionPool(
            self.dsn,
            min_size=0,
            max_size=self._max_pool_size,
            open=False,
        )
        self._pool.open()
        self._ready = True

    @staticmethod
    def _statements(schema: str) -> list[str]:
        lines = [line for line in schema.splitlines() if not line.lstrip().startswith("--")]
        return [statement.strip() for statement in "\n".join(lines).split(";") if statement.strip()]

    @staticmethod
    def _enable_tenant_policy(conn: Any, table: str) -> None:
        """Fail closed: without a bound tenant, a non-superuser sees no rows."""
        conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        conn.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        conn.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        conn.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            "USING (tenant_id = current_setting('app.current_tenant_id', true)) "
            "WITH CHECK (tenant_id = current_setting('app.current_tenant_id', true))"
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        self._ready = False

    def __enter__(self) -> PostgresStorage:
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @contextmanager
    def _transaction(self, tenant_id: str) -> Iterator[Any]:
        if not self._ready or self._pool is None:
            raise RuntimeError("PostgresStorage.initialize() must be called first")
        with self._pool.connection() as conn, conn.transaction():
            conn.execute(
                "SELECT set_config('app.current_tenant_id', %s, true)", (tenant_id,)
            )
            with conn.cursor(row_factory=dict_row) as cur:
                yield cur

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _atom_from_row(row: dict[str, Any], **scores: float) -> FactAtom:
        embedding = row.get("embedding")
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
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
            embedding=embedding,
        )
        atom.lexical_score = float(scores.get("lexical_score", 0.0))
        atom.dense_score = float(scores.get("dense_score", 0.0))
        return atom

    def _checked_embedding(self, vector: Sequence[float]) -> list[float]:
        normalised = normalise_embedding(vector)
        if len(normalised) != self.embedding_dim:
            raise ValueError(
                f"embedding has {len(normalised)} dimensions but this backend is "
                f"configured for {self.embedding_dim}"
            )
        return normalised

    # -- L0 ----------------------------------------------------------------

    def append_turns(self, turns: Sequence[TurnRecord]) -> None:
        if not turns:
            return
        rows = []
        for turn in turns:
            rows.append(
                (
                    _require_identifier(turn.tenant_id, "tenant_id"),
                    _require_identifier(turn.user_id, "user_id"),
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
        with self._transaction(rows[0][0]) as cur:
            cur.executemany(
                f"INSERT INTO {self._turns} "
                "(tenant_id, user_id, session_id, turn_index, turn_id, speaker, "
                "content, occurred_at, recorded_at, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT DO NOTHING",
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
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"SELECT * FROM {self._turns} WHERE tenant_id = %s AND user_id = %s "
                "AND session_id = %s AND turn_index BETWEEN %s AND %s "
                "ORDER BY turn_index ASC",
                (tenant_id, user_id, session_id, center_index - radius, center_index + radius),
            )
            rows = cur.fetchall()
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
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]

    # -- L1 + L6 -----------------------------------------------------------

    def get_latest_chain_hash(self, tenant_id: str, user_id: str) -> str:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"SELECT payload_hash FROM {self._atoms} "
                "WHERE tenant_id = %s AND user_id = %s ORDER BY chain_seq DESC LIMIT 1",
                (tenant_id, user_id),
            )
            row = cur.fetchone()
        return row["payload_hash"] if row else GENESIS_HASH

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

        with self._transaction(tenant_id) as cur:
            # Serialise only this user's chain append. A different user or tenant
            # hashes to a different lock key, so writers never queue behind each other.
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"{tenant_id}:{user_id}",),
            )
            cur.execute(
                f"SELECT payload_hash, chain_seq FROM {self._atoms} "
                "WHERE tenant_id = %s AND user_id = %s ORDER BY chain_seq DESC LIMIT 1",
                (tenant_id, user_id),
            )
            tip = cur.fetchone()
            previous = tip["payload_hash"] if tip else GENESIS_HASH
            seq = (tip["chain_seq"] + 1) if tip else 0

            rows = []
            for atom in atoms:
                atom.prev_hash = previous
                atom.payload_hash = payload_hash(previous, atom)
                embedding = (
                    _vector_literal(normalise_embedding(atom.embedding))
                    if atom.embedding is not None
                    else None
                )
                rows.append(
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
                    )
                )
                previous = atom.payload_hash
                seq += 1

            cur.executemany(
                f"INSERT INTO {self._atoms} ({_ATOM_COLUMNS.rsplit(', ', 1)[0]}, "
                "embedding, chain_seq) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)",
                rows,
            )

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
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"UPDATE {self._atoms} SET invalid_at = %s "
                "WHERE tenant_id = %s AND user_id = %s AND atom_id = %s "
                "AND invalid_at IS NULL",
                (float(invalid_at), tenant_id, user_id, atom_id),
            )
            if cur.rowcount == 0 or not superseded_by:
                return
            cur.execute(
                f"INSERT INTO {self._edges} "
                "(tenant_id, user_id, edge_id, src_atom_id, dst_atom_id, "
                "relation_type, weight, valid_at, recorded_at) "
                "SELECT %s, %s, %s, %s, %s, 'SUPERSEDES', 1.0, %s, %s "
                f"WHERE EXISTS (SELECT 1 FROM {self._atoms} WHERE tenant_id = %s "
                "AND user_id = %s AND atom_id = %s) "
                "ON CONFLICT DO NOTHING",
                (
                    tenant_id,
                    user_id,
                    f"supersedes:{superseded_by}:{atom_id}",
                    superseded_by,
                    atom_id,
                    float(invalid_at),
                    float(invalid_at),
                    tenant_id,
                    user_id,
                    superseded_by,
                ),
            )

    def get_atom_proof_chain(
        self, tenant_id: str, user_id: str, atom_id: str
    ) -> list[FactAtom]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"SELECT {_ATOM_COLUMNS} FROM {self._atoms} "
                "WHERE tenant_id = %s AND user_id = %s AND chain_seq <= "
                f"(SELECT chain_seq FROM {self._atoms} "
                "WHERE tenant_id = %s AND user_id = %s AND atom_id = %s) "
                "ORDER BY chain_seq ASC",
                (tenant_id, user_id, tenant_id, user_id, atom_id),
            )
            rows = cur.fetchall()
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
        with self._transaction(tenant_id) as cur:
            cur.executemany(
                f"INSERT INTO {self._edges} "
                "(tenant_id, user_id, edge_id, src_atom_id, dst_atom_id, "
                "relation_type, weight, valid_at, recorded_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (tenant_id, user_id, edge_id) DO UPDATE SET "
                "weight = GREATEST({0}.weight, EXCLUDED.weight)".format(self._edges),
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
        if not seeds:
            return []
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"WITH RECURSIVE walk (edge_id, src_atom_id, dst_atom_id, relation_type, "
                "weight, valid_at, recorded_at, depth) AS ("
                f"  SELECT e.edge_id, e.src_atom_id, e.dst_atom_id, e.relation_type, "
                "e.weight, e.valid_at, e.recorded_at, 1 "
                f"  FROM {self._edges} e WHERE e.tenant_id = %s AND e.user_id = %s "
                "AND e.src_atom_id = ANY(%s::text[])"
                "  UNION ALL "
                f"  SELECT e.edge_id, e.src_atom_id, e.dst_atom_id, e.relation_type, "
                "e.weight, e.valid_at, e.recorded_at, w.depth + 1 "
                f"  FROM {self._edges} e JOIN walk w ON e.src_atom_id = w.dst_atom_id "
                "WHERE e.tenant_id = %s AND e.user_id = %s AND w.depth < %s"
                ") "
                f"SELECT {_ATOM_COLUMNS_A}, w.edge_id AS e_edge_id, "
                "w.src_atom_id AS e_src, w.dst_atom_id AS e_dst, "
                "w.relation_type AS e_relation, w.weight AS e_weight, "
                "w.valid_at AS e_valid_at, w.recorded_at AS e_recorded, w.depth "
                f"FROM walk w JOIN {self._atoms} a ON a.tenant_id = %s "
                "AND a.user_id = %s AND a.atom_id = w.dst_atom_id "
                "ORDER BY w.depth ASC, w.weight DESC LIMIT %s",
                (tenant_id, user_id, seeds, tenant_id, user_id, hops, tenant_id, user_id, limit),
            )
            rows = cur.fetchall()
        found: list[tuple[FactAtom, CausalEdge]] = []
        seen: set[str] = set(seeds)
        for row in rows:
            if row["atom_id"] in seen:
                continue
            seen.add(row["atom_id"])
            edge = CausalEdge(
                edge_id=row["e_edge_id"],
                tenant_id=tenant_id,
                user_id=user_id,
                src_atom_id=row["e_src"],
                dst_atom_id=row["e_dst"],
                relation_type=row["e_relation"],
                weight=row["e_weight"],
                valid_at=row["e_valid_at"],
                recorded_at=row["e_recorded"],
            )
            found.append((self._atom_from_row(row), edge))
        return found

    # -- retrieval ---------------------------------------------------------

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
        if not isinstance(query, str) or not query.strip():
            return []
        invalidated = "" if include_invalidated else " AND invalid_at IS NULL"
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"SELECT {_ATOM_COLUMNS}, ts_rank_cd(fact_tsv, q) AS lexical_score "
                f"FROM {self._atoms}, websearch_to_tsquery('english', %s) AS q "
                "WHERE tenant_id = %s AND user_id = %s AND fact_tsv @@ q"
                f"{invalidated} ORDER BY lexical_score DESC, valid_at DESC LIMIT %s",
                (query, tenant_id, user_id, limit),
            )
            rows = cur.fetchall()
        return [
            self._atom_from_row(row, lexical_score=row["lexical_score"]) for row in rows
        ]

    def upsert_embeddings(
        self,
        tenant_id: str,
        user_id: str,
        atom_embeddings: Sequence[tuple[str, list[float]]],
    ) -> None:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        rows = [
            (_vector_literal(self._checked_embedding(vector)), tenant_id, user_id, atom_id)
            for atom_id, vector in atom_embeddings
        ]
        if not rows:
            return
        with self._transaction(tenant_id) as cur:
            cur.executemany(
                f"UPDATE {self._atoms} SET embedding = %s::vector "
                "WHERE tenant_id = %s AND user_id = %s AND atom_id = %s",
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
        literal = _vector_literal(self._checked_embedding(query_embedding))
        invalidated = "" if include_invalidated else " AND invalid_at IS NULL"
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"SELECT {_ATOM_COLUMNS}, (1 - (embedding <=> %s::vector)) AS dense_score "
                f"FROM {self._atoms} WHERE tenant_id = %s AND user_id = %s "
                "AND embedding IS NOT NULL"
                f"{invalidated} ORDER BY embedding <=> %s::vector LIMIT %s",
                (literal, tenant_id, user_id, literal, limit),
            )
            rows = cur.fetchall()
        return [self._atom_from_row(row, dense_score=row["dense_score"]) for row in rows]

    # -- L3 ----------------------------------------------------------------

    def get_persona_state(self, tenant_id: str, user_id: str) -> dict[str, Any]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"SELECT persona_json FROM {self._personas} "
                "WHERE tenant_id = %s AND user_id = %s",
                (tenant_id, user_id),
            )
            row = cur.fetchone()
        return row["persona_json"] if row else {}

    def upsert_persona_state(
        self, tenant_id: str, user_id: str, state: dict[str, Any], updated_at: float
    ) -> None:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        if not isinstance(state, dict):
            raise ValueError("state must be a dict")
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"INSERT INTO {self._personas} (tenant_id, user_id, persona_json, updated_at) "
                "VALUES (%s, %s, %s::jsonb, %s) "
                "ON CONFLICT (tenant_id, user_id) DO UPDATE SET "
                f"persona_json = {self._personas}.persona_json || EXCLUDED.persona_json, "
                "updated_at = EXCLUDED.updated_at",
                (tenant_id, user_id, json.dumps(state, ensure_ascii=False), float(updated_at)),
            )

    # -- L4 ----------------------------------------------------------------

    def list_active_policies(
        self, tenant_id: str, user_id: str | None = None
    ) -> list[PolicyRule]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        if user_id is None:
            sql = (
                f"SELECT * FROM {self._policies} WHERE tenant_id = %s AND enabled "
                "AND user_id IS NULL ORDER BY priority ASC, policy_id ASC"
            )
            params: tuple[Any, ...] = (tenant_id,)
        else:
            user_id = _require_identifier(user_id, "user_id")
            sql = (
                f"SELECT * FROM {self._policies} WHERE tenant_id = %s AND enabled "
                "AND (user_id IS NULL OR user_id = %s) ORDER BY priority ASC, policy_id ASC"
            )
            params = (tenant_id, user_id)
        with self._transaction(tenant_id) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
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
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"INSERT INTO {self._policies} "
                "(tenant_id, policy_id, user_id, rule_type, pattern, action, "
                "priority, enabled, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (tenant_id, policy_id) DO UPDATE SET "
                "user_id = EXCLUDED.user_id, rule_type = EXCLUDED.rule_type, "
                "pattern = EXCLUDED.pattern, action = EXCLUDED.action, "
                "priority = EXCLUDED.priority, enabled = EXCLUDED.enabled",
                (
                    tenant_id,
                    policy.policy_id,
                    policy.user_id,
                    policy.rule_type,
                    policy.pattern,
                    policy.action,
                    int(policy.priority),
                    bool(policy.enabled),
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
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"UPDATE {self._atoms} SET access_count = access_count + 1, "
                "last_accessed_at = %s, stability = stability * %s "
                "WHERE tenant_id = %s AND user_id = %s AND atom_id = ANY(%s::text[])",
                (float(accessed_at), multiplier, tenant_id, user_id, ids),
            )

    def get_turn_windows(
        self,
        tenant_id: str,
        user_id: str,
        turn_ids: Sequence[str],
        window_radius: int = 1,
    ) -> list[TurnRecord]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        radius = _require_positive_int(window_radius, "window_radius", 1000)
        ids = [turn_id for turn_id in dict.fromkeys(turn_ids) if turn_id]
        if not ids:
            return []
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"SELECT session_id, turn_index FROM {self._turns} "
                "WHERE tenant_id = %s AND user_id = %s AND turn_id = ANY(%s::text[])",
                (tenant_id, user_id, ids),
            )
            centers = cur.fetchall()
            if not centers:
                return []
            values = ", ".join("(%s::text, %s::integer)" for _ in centers)
            params: list[Any] = []
            for row in centers:
                params.extend((row["session_id"], row["turn_index"]))
            cur.execute(
                f"WITH centers(session_id, turn_index) AS (VALUES {values}) "
                f"SELECT DISTINCT t.* FROM {self._turns} t "
                "JOIN centers c ON c.session_id = t.session_id "
                "AND t.turn_index BETWEEN c.turn_index - %s AND c.turn_index + %s "
                "WHERE t.tenant_id = %s AND t.user_id = %s "
                "ORDER BY t.session_id ASC, t.turn_index ASC",
                (*params, radius, radius, tenant_id, user_id),
            )
            rows = cur.fetchall()
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
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]

    # -- L5 feedback -------------------------------------------------------

    def record_feedback(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        atom_id: str,
        label: float,
        created_at: float,
    ) -> int:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(atom_id, str) or not atom_id.strip():
            raise ValueError("atom_id must be a non-empty string")
        value = float(label)
        if value not in (-1.0, 1.0):
            raise ValueError("label must be 1.0 (relevant) or -1.0 (irrelevant)")
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"INSERT INTO {self._feedback} "
                "(tenant_id, user_id, query, atom_id, label, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (tenant_id, user_id, query, atom_id, value, float(created_at)),
            )
            row = cur.fetchone()
        return int(row["id"])

    def feedback_weights(
        self,
        tenant_id: str,
        user_id: str,
        min_support: float = 2.0,
        max_terms: int = 256,
    ) -> FeedbackWeights:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        if not isinstance(max_terms, int) or isinstance(max_terms, bool) or max_terms < 1:
            raise ValueError("max_terms must be a positive integer")
        with self._transaction(tenant_id) as cur:
            cur.execute(
                f"SELECT f.label AS label, a.fact_text AS fact_text, f.atom_id AS atom_id "
                f"FROM {self._feedback} f JOIN {self._atoms} a "
                "ON a.tenant_id = f.tenant_id AND a.user_id = f.user_id "
                "AND a.atom_id = f.atom_id "
                "WHERE f.tenant_id = %s AND f.user_id = %s",
                (tenant_id, user_id),
            )
            rows = cur.fetchall()
        return feedback_weights_from_rows(rows, float(min_support), max_terms)

    # -- erasure -----------------------------------------------------------

    def delete_user_data(self, tenant_id: str, user_id: str) -> dict[str, int]:
        tenant_id = _require_identifier(tenant_id, "tenant_id")
        user_id = _require_identifier(user_id, "user_id")
        counts: dict[str, int] = {}
        with self._transaction(tenant_id) as cur:
            for table, key in (
                (self._edges, "edges"),
                (self._atoms, "atoms"),
                (self._turns, "turns"),
                (self._personas, "personas"),
                (self._policies, "policies"),
                (self._feedback, "feedback"),
            ):
                cur.execute(
                    f"DELETE FROM {table} WHERE tenant_id = %s AND user_id = %s",
                    (tenant_id, user_id),
                )
                counts[key] = cur.rowcount
        return counts
