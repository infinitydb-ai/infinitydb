"""PostgreSQL-specific checks: fixed-width vectors, real indexes, real RLS.

These need the ``pgserver`` test wheel (PostgreSQL 16 + pgvector, no server setup)
and are skipped without it, so the default test run stays dependency-free.
"""

from __future__ import annotations

import tempfile
import uuid

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")
pytest.importorskip("pgserver")

import psycopg

from infinitydb.storage import FactAtom
from infinitydb.storage.postgres import PostgresStorage

PREFIX = "oams_pgtest_"


@pytest.fixture(scope="module")
def pg_dsn():
    import pgserver

    path = tempfile.mkdtemp(prefix="infinitydb-pg-")
    server = pgserver.get_server(path)
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


@pytest.fixture(scope="module")
def storage(pg_dsn):
    backend = PostgresStorage(pg_dsn, table_prefix=PREFIX, embedding_dim=4)
    backend.initialize()
    try:
        yield backend
    finally:
        backend.close()


def make_atom(tenant: str, atom_id: str, text: str, embedding=None) -> FactAtom:
    return FactAtom(
        atom_id=atom_id,
        tenant_id=tenant,
        user_id="u",
        session_id="s",
        source_turn_id=f"turn-{atom_id}",
        fact_text=text,
        valid_at=1.0,
        recorded_at=1.0,
        last_accessed_at=1.0,
        embedding=embedding,
    )


def test_embedding_dimension_is_enforced(storage):
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    storage.insert_atoms([make_atom(tenant, "a0", "four dimensional")])
    with pytest.raises(ValueError):
        storage.upsert_embeddings(tenant, "u", [("a0", [1.0, 0.0])])
    storage.upsert_embeddings(tenant, "u", [("a0", [1.0, 0.0, 0.0, 0.0])])
    with pytest.raises(ValueError):
        storage.search_dense(tenant, "u", [1.0, 0.0, 0.0])


def test_schema_has_gin_and_hnsw_indexes(storage, pg_dsn):
    with psycopg.connect(pg_dsn) as conn:
        rows = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = %s",
            (f"{PREFIX}atoms",),
        ).fetchall()
    indexdefs = " ".join(row[0] for row in rows)
    assert "USING gin (fact_tsv)" in indexdefs
    assert "USING hnsw (embedding vector_cosine_ops)" in indexdefs


def test_rls_fails_closed_and_scopes_to_the_bound_tenant(storage, pg_dsn):
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    other = f"t-{uuid.uuid4().hex[:8]}"
    storage.insert_atoms([make_atom(tenant, "mine", "tenant scoped fact")])
    storage.insert_atoms([make_atom(other, "theirs", "another tenant fact")])

    role = f"rls_probe_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute(f"CREATE ROLE {role} LOGIN")
        conn.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
        for table in ("turns", "atoms", "edges", "personas", "policies"):
            conn.execute(f"GRANT SELECT ON {PREFIX}{table} TO {role}")
        for row in conn.execute(
            "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
            "WHERE relname = ANY(%s)",
            ([f"{PREFIX}{name}" for name in ("turns", "atoms", "edges", "personas", "policies")],),
        ).fetchall():
            assert row[1] and row[2], row

    try:
        # autocommit so the block below is a real transaction, not a savepoint:
        # a SET LOCAL must not survive its transaction.
        with psycopg.connect(pg_dsn, user=role, autocommit=True) as conn:
            # No tenant bound: the policy matches nothing rather than everything.
            assert conn.execute(f"SELECT count(*) FROM {PREFIX}atoms").fetchone()[0] == 0
            with conn.transaction():
                conn.execute(
                    "SELECT set_config('app.current_tenant_id', %s, true)", (tenant,)
                )
                rows = conn.execute(f"SELECT atom_id FROM {PREFIX}atoms").fetchall()
                assert [row[0] for row in rows] == ["mine"]
            # SET LOCAL does not leak past its transaction.
            assert conn.execute(f"SELECT count(*) FROM {PREFIX}atoms").fetchone()[0] == 0
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            conn.execute(f"DROP OWNED BY {role}")
            conn.execute(f"DROP ROLE {role}")
