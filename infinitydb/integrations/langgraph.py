"""LangGraph ``BaseStore`` adapter for InfinityDB.

Maps LangGraph's hierarchical namespaces onto the engine's tenant-isolated
OAMS-1.0 hierarchy:

* ``PutOp`` / ``aput`` -> the deterministic, zero-LLM ``InfinityDB.add()``
  (plus an exact key-value row, and a tombstone on ``value=None``);
* ``SearchOp`` / ``asearch`` -> ``InfinityDB.recall()`` with optional
  ``budget_tokens`` and any caller-supplied reranker or dense index;
* ``GetOp`` / ``ListNamespacesOp`` -> the exact catalog, with TTL expiry and
  refresh.

A namespace tuple gets a stable ``user_id`` derived from it, so two namespaces
never share ranking state. The catalog lives in the same database (or the same
PostgreSQL tenant schema) as the engine, so there is no second system to run.

    pip install "infinitydb-ai[langgraph]"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from langgraph.store.base import (
        BaseStore,
        GetOp,
        Item,
        ListNamespacesOp,
        MatchCondition,
        Op,
        PutOp,
        Result,
        SearchItem,
        SearchOp,
    )
except ImportError as exc:  # pragma: no cover - exercised by the install-error test
    raise ImportError(
        "The LangGraph integration needs langgraph-checkpoint>=2.0.\n"
        'Install it with: pip install "infinitydb-ai[langgraph]"'
    ) from exc

from infinitydb.core.engine import InfinityDB

# The catalog is deliberately small and separate from the L0-L6 tables: it is the
# authoritative key-value record, while the engine indexes a rendering of each
# item for ranked search. ``ttl_seconds`` is stored so a refresh extends an item
# by the window it was written with rather than guessing from the timestamps.
_KV_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS lg_store_items (
        namespace    TEXT NOT NULL,
        item_key     TEXT NOT NULL,
        value_json   TEXT NOT NULL,
        created_at   DOUBLE PRECISION NOT NULL,
        updated_at   DOUBLE PRECISION NOT NULL,
        expires_at   DOUBLE PRECISION,
        ttl_seconds  DOUBLE PRECISION,
        PRIMARY KEY (namespace, item_key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_lg_store_ns ON lg_store_items (namespace)",
    "CREATE INDEX IF NOT EXISTS idx_lg_store_expires ON lg_store_items (expires_at)",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _namespace_to_str(namespace: tuple[str, ...]) -> str:
    return "/".join(namespace)


def _str_to_namespace(raw: str) -> tuple[str, ...]:
    return tuple(raw.split("/")) if raw else ()


def _namespace_to_user_id(namespace: tuple[str, ...]) -> str:
    """A deterministic ``user_id`` for one namespace tuple.

    The digest keeps namespaces whose leaves collide (``("a", "b")`` and
    ``("a/b",)``) apart, and the readable leaf keeps a database inspectable.
    """
    if not namespace:
        return "lg_ns_root"
    digest = hashlib.sha256(_namespace_to_str(namespace).encode("utf-8")).hexdigest()[:16]
    leaf = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in namespace[-1]
    )[:32]
    return f"lg_{leaf}_{digest}"


def _matches_filter(value: dict[str, Any], filter_dict: dict[str, Any] | None) -> bool:
    """Evaluate LangGraph metadata filters (``$eq`` .. ``$nin``)."""
    if not filter_dict:
        return True
    for key, condition in filter_dict.items():
        actual = value.get(key)
        if isinstance(condition, dict):
            for op, expected in condition.items():
                if op == "$eq" and actual != expected:
                    return False
                elif op == "$ne" and actual == expected:
                    return False
                elif op in ("$gt", "$gte", "$lt", "$lte"):
                    if actual is None:
                        return False
                    try:
                        if op == "$gt" and not actual > expected:
                            return False
                        if op == "$gte" and not actual >= expected:
                            return False
                        if op == "$lt" and not actual < expected:
                            return False
                        if op == "$lte" and not actual <= expected:
                            return False
                    except TypeError:
                        # Comparing unlike types is a non-match, not an error that
                        # aborts the whole batch.
                        return False
                elif op == "$in" and actual not in expected:
                    return False
                elif op == "$nin" and actual in expected:
                    return False
        elif actual != condition:
            return False
    return True


def _matches_namespace_condition(
    namespace: tuple[str, ...], condition: MatchCondition
) -> bool:
    path = condition.path
    if not path:
        return True
    if len(namespace) < len(path):
        return False
    if condition.match_type == "prefix":
        window = namespace[: len(path)]
    elif condition.match_type == "suffix":
        window = namespace[-len(path) :]
    else:
        return False
    return all(p == "*" or ns == p for ns, p in zip(window, path))


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class InfinityDBStore(BaseStore):
    """A LangGraph ``BaseStore`` backed by InfinityDB.

    ``InfinityDBStore(db_path=...)`` is the dependency-free SQLite mode; one file
    is one tenant. ``InfinityDBStore(dsn=..., tenant_id=...)`` runs on
    PostgreSQL with one schema per tenant, so the same process can serve several
    tenants without the engine's unqualified table names reaching across them.

    Example:
        ```python
        from infinitydb.integrations.langgraph import InfinityDBStore

        store = InfinityDBStore(db_path="./agent_memory.db")
        store.put(("acme", "alice"), "diet", {"rule": "vegetarian"})
        hits = store.search(("acme", "alice"), query="dietary rules")
        ```
    """

    # The catalog implements expiry and refresh, so the store advertises it.
    supports_ttl = True

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        dsn: str | None = None,
        tenant_id: str = "default",
        budget_tokens: int | None = None,
        rerank: Callable[[str, list[str]], list[float]] | None = None,
        rerank_top_k: int | None = None,
        dense: Any = None,
        reinforce_on_search: bool = True,
        text_fields: list[str] | None = None,
    ) -> None:
        if db_path is None and dsn is None:
            db_path = ":memory:"
        if db_path is not None and dsn is not None:
            raise ValueError("pass either db_path (SQLite) or dsn (PostgreSQL), not both")

        self._db_path = str(db_path) if db_path is not None else None
        self._dsn = dsn
        self._tenant_id = tenant_id
        self._budget_tokens = budget_tokens
        self._rerank = rerank
        self._rerank_top_k = rerank_top_k
        self._dense = dense
        self._reinforce_on_search = reinforce_on_search
        self._text_fields = text_fields
        self._lock = threading.RLock()
        self._closed = False
        # The catalog shares the engine's connection: one file or one tenant
        # schema, no second handle, and no cross-connection write contention.
        # PostgreSQL needs %s placeholders, SQLite needs ?.
        self._ph = "%s" if dsn is not None else "?"

        self.engine = (
            InfinityDB(dsn=dsn, tenant_id=tenant_id)
            if dsn is not None
            else InfinityDB(db_path=str(db_path))
        )
        self._init_kv_catalog()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_kv_catalog(self) -> None:
        for statement in _KV_SCHEMA:
            self.engine.conn.execute(statement)
        self.engine.conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.engine.close()

    def __enter__(self) -> "InfinityDBStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # BaseStore entrypoints
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        with self._lock:
            for op in ops:
                if isinstance(op, GetOp):
                    results.append(self._handle_get(op))
                elif isinstance(op, PutOp):
                    self._handle_put(op)
                    results.append(None)
                elif isinstance(op, SearchOp):
                    results.append(self._handle_search(op))
                elif isinstance(op, ListNamespacesOp):
                    results.append(self._handle_list_namespaces(op))
                else:
                    raise TypeError(f"Unsupported BaseStore operation: {type(op)!r}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # The engine is a synchronous SQLite/psycopg pipeline; run the batch on a
        # worker thread so an async caller does not block its event loop.
        op_list = list(ops)
        return await asyncio.to_thread(self.batch, op_list)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _extract_indexable_text(
        self,
        key: str,
        value: dict[str, Any],
        index: bool | list[str] | None,
    ) -> str | None:
        if index is False:
            return None

        fields: list[str] | None = None
        if isinstance(index, list):
            fields = index
        elif self._text_fields is not None:
            fields = self._text_fields

        if fields:
            parts = [
                f"{field}: {value[field]}" for field in fields if value.get(field) is not None
            ]
        else:
            parts = [f"{name}: {val}" for name, val in value.items() if val is not None]
        if not parts:
            return None
        # The key tag is how a recalled chunk is mapped back to the catalog row.
        return f"[{key}] " + " | ".join(parts)

    def _handle_get(self, op: GetOp) -> Item | None:
        namespace = op.namespace
        ns_str = _namespace_to_str(namespace)
        now_ts = time.time()
        row = self._kv_fetch_one(ns_str, op.key)
        if row is None:
            return None

        value_json, created_at, updated_at, expires_at, ttl_seconds = row
        if expires_at is not None and expires_at <= now_ts:
            self._kv_delete(ns_str, op.key)
            return None

        if op.refresh_ttl and expires_at is not None:
            window = ttl_seconds if ttl_seconds else max(expires_at - updated_at, 60.0)
            self._kv_update_expiry(ns_str, op.key, updated_at, now_ts + window)

        return Item(
            value=json.loads(value_json),
            key=op.key,
            namespace=namespace,
            created_at=datetime.fromtimestamp(created_at, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(updated_at, tz=timezone.utc),
        )

    def _handle_put(self, op: PutOp) -> None:
        ns_str = _namespace_to_str(op.namespace)

        # LangGraph's convention: value=None deletes the item.
        if op.value is None:
            self._kv_delete(ns_str, op.key)
            return

        now_ts = time.time()
        existing = self._kv_fetch_one(ns_str, op.key)
        created_at = existing[1] if existing is not None else now_ts
        ttl_seconds = float(op.ttl) * 60.0 if op.ttl is not None else None
        expires_at = now_ts + ttl_seconds if ttl_seconds is not None else None

        self._kv_upsert(
            ns_str=ns_str,
            key=op.key,
            value_json=json.dumps(op.value, default=str, sort_keys=True),
            created_at=created_at,
            updated_at=now_ts,
            expires_at=expires_at,
            ttl_seconds=ttl_seconds,
        )

        indexable = self._extract_indexable_text(op.key, op.value, op.index)
        if indexable:
            user_id = _namespace_to_user_id(op.namespace)
            self.engine.add(
                conversation=f"memory[{op.key}]: {indexable}",
                user_id=user_id,
                session_id=f"{user_id}:{op.key}",
            )

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        prefix_str = _namespace_to_str(op.namespace_prefix)
        now_ts = time.time()
        candidates = self._kv_list_by_prefix(prefix_str, now_ts)
        if not candidates:
            return []

        filtered: list[tuple[tuple[str, ...], str, dict[str, Any], float, float]] = []
        for ns_raw, key, value_json, created_at, updated_at in candidates:
            value = json.loads(value_json)
            if _matches_filter(value, op.filter):
                filtered.append(
                    (_str_to_namespace(ns_raw), key, value, created_at, updated_at)
                )
        if not filtered:
            return []

        if not op.query or not op.query.strip():
            # A query-less search is an ordering request: newest first.
            filtered.sort(key=lambda row: row[4], reverse=True)
            page = filtered[op.offset : op.offset + op.limit]
            return [
                SearchItem(
                    namespace=namespace,
                    key=key,
                    value=value,
                    created_at=datetime.fromtimestamp(created_at, tz=timezone.utc),
                    updated_at=datetime.fromtimestamp(updated_at, tz=timezone.utc),
                    score=1.0,
                )
                for namespace, key, value, created_at, updated_at in page
            ]

        key_scores = self._recall_scores(op, filtered)

        # Items the engine did not rank still get a weak lexical score, so a
        # partial match is visible instead of silently dropped.
        query_terms = {term for term in op.query.lower().split() if len(term) > 1}
        scored: list[tuple[float, tuple[str, ...], str, dict[str, Any], float, float]] = []
        for namespace, key, value, created_at, updated_at in filtered:
            score = key_scores.get((namespace, key), 0.0)
            if score == 0.0 and query_terms:
                blob = f"{key} {json.dumps(value, default=str)}".lower()
                overlap = sum(1 for term in query_terms if term in blob)
                if overlap:
                    score = 0.01 * (overlap / len(query_terms))
            if score > 0.0:
                scored.append((score, namespace, key, value, created_at, updated_at))

        scored.sort(key=lambda row: (row[0], row[5]), reverse=True)
        page = scored[op.offset : op.offset + op.limit]
        return [
            SearchItem(
                namespace=namespace,
                key=key,
                value=value,
                created_at=datetime.fromtimestamp(created_at, tz=timezone.utc),
                updated_at=datetime.fromtimestamp(updated_at, tz=timezone.utc),
                score=round(score, 6),
            )
            for score, namespace, key, value, created_at, updated_at in page
        ]

    def _recall_scores(
        self,
        op: SearchOp,
        filtered: list[tuple[tuple[str, ...], str, dict[str, Any], float, float]],
    ) -> dict[tuple[tuple[str, ...], str], float]:
        """Rank the filtered items through the engine, one namespace at a time.

        A namespace maps to one engine ``user_id``, so scoring per namespace is
        what keeps another namespace's memories out of the result.
        """
        recall_kwargs: dict[str, Any] = {
            "query": op.query,
            "n": max(op.limit + op.offset, 10),
            "reinforce": self._reinforce_on_search,
        }
        if self._budget_tokens is not None:
            recall_kwargs["budget_tokens"] = self._budget_tokens
        if self._rerank is not None:
            recall_kwargs["rerank"] = self._rerank
        if self._rerank_top_k is not None:
            recall_kwargs["rerank_top_k"] = self._rerank_top_k
        if self._dense is not None:
            recall_kwargs["dense"] = self._dense

        scores: dict[tuple[tuple[str, ...], str], float] = {}
        for namespace in sorted({row[0] for row in filtered}):
            user_id = _namespace_to_user_id(namespace)
            result = self.engine.recall(user_id=user_id, **recall_kwargs)
            for rank, entry in enumerate(result.get("results", [])):
                fact_text = str(entry.get("fact", ""))
                raw_score = float(entry.get("score", 1.0 / (rank + 1)))
                for item_ns, key, _value, _created, _updated in filtered:
                    if item_ns == namespace and f"[{key}]" in fact_text:
                        if raw_score > scores.get((namespace, key), 0.0):
                            scores[(namespace, key)] = raw_score
        return scores

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        parsed = [_str_to_namespace(raw) for raw in self._kv_distinct_namespaces()]

        if op.match_conditions:
            parsed = [
                namespace
                for namespace in parsed
                if all(
                    _matches_namespace_condition(namespace, condition)
                    for condition in op.match_conditions
                )
            ]

        if op.max_depth is not None:
            parsed = sorted({namespace[: op.max_depth] for namespace in parsed})
        else:
            parsed = sorted(set(parsed))

        return parsed[op.offset : op.offset + op.limit]

    # ------------------------------------------------------------------
    # Catalog primitives (shared engine connection: ? or %s)
    # ------------------------------------------------------------------

    def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            cursor = self.engine.conn.execute(sql, params)
            return [tuple(row) for row in cursor.fetchall()]

    def _write(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            self.engine.conn.execute(sql, params)
            self.engine.conn.commit()

    def _kv_fetch_one(
        self, ns_str: str, key: str
    ) -> tuple[str, float, float, float | None, float | None] | None:
        rows = self._rows(
            "SELECT value_json, created_at, updated_at, expires_at, ttl_seconds "
            f"FROM lg_store_items WHERE namespace = {self._ph} AND item_key = {self._ph}",
            (ns_str, key),
        )
        return rows[0] if rows else None

    def _kv_upsert(
        self,
        ns_str: str,
        key: str,
        value_json: str,
        created_at: float,
        updated_at: float,
        expires_at: float | None,
        ttl_seconds: float | None,
    ) -> None:
        self._write(
            "INSERT INTO lg_store_items "
            "(namespace, item_key, value_json, created_at, updated_at, expires_at, ttl_seconds) "
            f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, "
            f"{self._ph}, {self._ph}) "
            "ON CONFLICT (namespace, item_key) DO UPDATE SET "
            "value_json = EXCLUDED.value_json, updated_at = EXCLUDED.updated_at, "
            "expires_at = EXCLUDED.expires_at, ttl_seconds = EXCLUDED.ttl_seconds",
            (ns_str, key, value_json, created_at, updated_at, expires_at, ttl_seconds),
        )

    def _kv_delete(self, ns_str: str, key: str) -> None:
        self._write(
            f"DELETE FROM lg_store_items WHERE namespace = {self._ph} AND item_key = {self._ph}",
            (ns_str, key),
        )

    def _kv_update_expiry(
        self, ns_str: str, key: str, updated_at: float, expires_at: float
    ) -> None:
        self._write(
            f"UPDATE lg_store_items SET updated_at = {self._ph}, expires_at = {self._ph} "
            f"WHERE namespace = {self._ph} AND item_key = {self._ph}",
            (updated_at, expires_at, ns_str, key),
        )

    def _kv_list_by_prefix(
        self, prefix_str: str, now_ts: float
    ) -> list[tuple[str, str, str, float, float]]:
        columns = "namespace, item_key, value_json, created_at, updated_at"
        live = f"(expires_at IS NULL OR expires_at > {self._ph})"
        if prefix_str:
            return self._rows(
                f"SELECT {columns} FROM lg_store_items "
                f"WHERE (namespace = {self._ph} OR namespace LIKE {self._ph} ESCAPE '\\') "
                f"AND {live}",
                (prefix_str, f"{_escape_like(prefix_str)}/%", now_ts),
            )
        return self._rows(
            f"SELECT {columns} FROM lg_store_items WHERE {live}", (now_ts,)
        )

    def _kv_distinct_namespaces(self) -> list[str]:
        return [
            row[0]
            for row in self._rows(
                "SELECT DISTINCT namespace FROM lg_store_items "
                f"WHERE expires_at IS NULL OR expires_at > {self._ph}",
                (time.time(),),
            )
        ]
