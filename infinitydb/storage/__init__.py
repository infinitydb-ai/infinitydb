"""Portable persistence for InfinityDB's L0-L6 memory hierarchy.

The core engine keeps its own dependency-free SQLite schema. This package adds the
storage layer you can point at something else:

    from infinitydb.storage import open_backend

    backend = open_backend(db_path="memory.db")          # stdlib only
    backend = open_backend(dsn="postgresql://...")       # infinitydb-ai[postgres]

    backend.append_turns([...])
    backend.insert_atoms([...])

Both backends implement the same :class:`~infinitydb.storage.base.StorageBackend`
contract and scope every operation to one ``(tenant_id, user_id)`` pair.
"""

from __future__ import annotations

from .base import (
    GENESIS_HASH,
    CausalEdge,
    FactAtom,
    PolicyRule,
    StorageBackend,
    TurnRecord,
    canonical_payload,
    normalise_embedding,
    payload_hash,
    verify_chain,
)
from .engine import open_engine_store
from .sqlite import SqliteStorage

__all__ = [
    "GENESIS_HASH",
    "CausalEdge",
    "FactAtom",
    "PolicyRule",
    "SqliteStorage",
    "StorageBackend",
    "TurnRecord",
    "canonical_payload",
    "normalise_embedding",
    "open_backend",
    "open_engine_store",
    "payload_hash",
    "verify_chain",
]


def open_backend(
    *,
    db_path: str | None = None,
    dsn: str | None = None,
    initialize: bool = True,
    **kwargs: object,
) -> StorageBackend:
    """Open a storage backend and run its idempotent migrations.

    ``db_path`` selects SQLite (the default install, no dependencies); ``dsn``
    selects PostgreSQL and imports ``psycopg`` lazily, so a missing extra fails
    with an install instruction instead of an import error at package import
    time. Passing both, or neither, is a programming error.
    """
    if db_path is not None and dsn is not None:
        raise ValueError("pass either db_path (SQLite) or dsn (PostgreSQL), not both")
    if db_path is None and dsn is None:
        raise ValueError("pass db_path (SQLite) or dsn (PostgreSQL)")

    backend: StorageBackend
    if dsn is not None:
        from .postgres import PostgresStorage

        backend = PostgresStorage(dsn, **kwargs)  # type: ignore[arg-type]
    else:
        backend = SqliteStorage(db_path or ":memory:", **kwargs)  # type: ignore[arg-type]
    if initialize:
        backend.initialize()
    return backend
