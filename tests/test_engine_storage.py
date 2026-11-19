"""The core engine on both persistence backends.

The default SQLite path must stay exactly as measured; the PostgreSQL path must
run the same ``add`` -> ``recall`` -> persona/proof/feedback pipeline end to end
and must not leak across tenants. PostgreSQL arms are skipped without
``psycopg`` + ``pgserver``, so a bare install stays dependency-free.
"""

from __future__ import annotations

import tempfile
import uuid

import pytest

from infinitydb import InfinityDB


def test_sqlite_default_path_is_unchanged():
    db = InfinityDB(":memory:")
    try:
        db.add("Caroline: I work at the gym in Bristol.", user_id="u", ts=1000.0)
        db.add("Caroline: I work at the hospital now.", user_id="u", ts=2000.0)
        hit = db.recall("where does Caroline work", n=3, user_id="u", reinforce=False)[
            "results"
        ]
        assert "hospital" in hit[0]["fact"].lower()
        assert db.stats()["backend"] == "SQLite-WAL"
    finally:
        db.close()


def test_sqlite_is_single_tenant():
    with pytest.raises(ValueError, match="single-tenant"):
        InfinityDB(":memory:", tenant_id="acme")


def test_only_one_persistence_target_may_be_passed():
    with pytest.raises(ValueError, match="not both"):
        InfinityDB("memory.db", dsn="postgresql://localhost/x")


pytest.importorskip("psycopg")
pytest.importorskip("pgserver")

import psycopg  # noqa: E402


@pytest.fixture(scope="module")
def pg_dsn():
    import pgserver

    server = pgserver.get_server(tempfile.mkdtemp(prefix="infinitydb-engine-"))
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


def test_postgres_runs_the_full_pipeline(pg_dsn):
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    db = InfinityDB(dsn=pg_dsn, tenant_id=tenant)
    try:
        assert db.stats()["backend"] == "PostgreSQL"
        db.add("Caroline: I work at the gym in Bristol.", user_id="u", session_id="s1", ts=1000.0)
        db.add("Caroline: I moved to Lisbon last year.", user_id="u", session_id="s1", ts=2000.0)
        db.add(
            "Caroline: I work at the hospital now.",
            user_id="u",
            session_id="s1",
            ts=3000.0,
        )

        result = db.recall("where does Caroline work", n=3, user_id="u", reinforce=False)
        assert result["results"]
        assert "hospital" in result["results"][0]["fact"].lower()

        assert db.get_persona("u")["employer"] == "I work at the hospital now"
        assert db.contradictions("u"), "the newer employer should supersede the older one"
        proof = db.get_proof(result["results"][0]["id"])
        assert proof is not None and proof["verified"] is True

        atom_id = result["results"][0]["id"]
        assert db.record_feedback("where does caroline work", atom_id, user_id="u") > 0
        summary = db.feedback_summary("u")
        assert summary["events"] == 1 and summary["relevant"] == 1
        assert db.learned_atom_weights("u")[atom_id] == 1.0

        counts = db.delete_user("u")
        assert counts["l1_atoms_deleted"] == 6
        assert db.recall("where does Caroline work", n=3, user_id="u")["results"] == []
    finally:
        db.close()


def test_postgres_dense_index_round_trips(pg_dsn):
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    db = InfinityDB(dsn=pg_dsn, tenant_id=tenant)

    def embed(texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    try:
        db.add("Caroline: I work at the hospital.", user_id="u", ts=1.0)
        db.add("Caroline: I moved to Lisbon.", user_id="u", ts=2.0)
        assert db.index_dense(embed, model="toy") == 2
        result = db.recall(
            "anything at all",
            n=3,
            user_id="u",
            reinforce=False,
            dense=(embed, "toy", 1.0),
        )
        assert result["results"]
    finally:
        db.close()


def test_postgres_state_survives_reopen(pg_dsn):
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    user = f"u-{uuid.uuid4().hex[:8]}"
    first = InfinityDB(dsn=pg_dsn, tenant_id=tenant)
    try:
        first.add("Deploys run from the build server", user_id=user, ts=10.0)
    finally:
        first.close()

    second = InfinityDB(dsn=pg_dsn, tenant_id=tenant)
    try:
        results = second.recall("where do deploys run", n=3, user_id=user)["results"]
        assert results and "build server" in results[0]["fact"]
    finally:
        second.close()


def test_postgres_tenants_are_isolated_even_with_identical_memory(pg_dsn):
    # The engine derives ids from content, so the same conversation at the same
    # event time has the same atom ids in both tenants. Separate schemas keep
    # that from colliding and keep each tenant's rows invisible to the other.
    user = f"u-{uuid.uuid4().hex[:8]}"
    mine = InfinityDB(dsn=pg_dsn, tenant_id=f"a-{uuid.uuid4().hex[:8]}")
    other = None
    try:
        mine.add("Caroline: I work at the hospital.", user_id=user, ts=1.0)
        other = InfinityDB(dsn=pg_dsn, tenant_id=f"b-{uuid.uuid4().hex[:8]}")
        other.add("Caroline: I work at the hospital.", user_id=user, ts=1.0)
        assert mine.stats()["l1_atom_facts"] == 2
        assert other.stats()["l1_atom_facts"] == 2
        assert other.get_persona(user) == {"employer": "I work at the hospital"}
    finally:
        mine.close()
        if other is not None:
            other.close()


def test_engine_tables_live_in_a_tenant_schema_not_public(pg_dsn):
    first = f"t-{uuid.uuid4().hex[:8]}"
    second = f"t-{uuid.uuid4().hex[:8]}"
    db = InfinityDB(dsn=pg_dsn, tenant_id=first)
    try:
        db.add("Caroline: I work at the hospital.", user_id="u", ts=1.0)
        schema = db._store.schema_name
    finally:
        db.close()

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        assert conn.execute(
            "SELECT to_regclass(%s)", (f"{schema}.l1_atoms",)
        ).fetchone()[0] is not None
        assert conn.execute(
            "SELECT to_regclass('public.l1_atoms')"
        ).fetchone()[0] is None

    other = InfinityDB(dsn=pg_dsn, tenant_id=second)
    try:
        other_schema = other._store.schema_name
        other.add("Caroline: I work in Lisbon.", user_id="u", ts=2.0)
        assert other_schema != schema
    finally:
        other.close()

    # A role granted only one tenant's schema cannot read the other's.
    role = f"engine_schema_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute(f"CREATE ROLE {role} LOGIN")
        conn.execute(f"GRANT USAGE ON SCHEMA {schema} TO {role}")
        conn.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO {role}")
    try:
        with psycopg.connect(pg_dsn, user=role, autocommit=True) as conn:
            conn.execute("SELECT set_config('search_path', %s, false)", (schema,))
            assert conn.execute("SELECT count(*) FROM l1_atoms").fetchone()[0] == 2
            # Without USAGE, PostgreSQL either prunes the schema from
            # search_path (UndefinedTable) or denies resolution outright; both
            # mean the other tenant's rows are unreachable.
            conn.execute("SELECT set_config('search_path', %s, false)", (other_schema,))
            with pytest.raises(
                (psycopg.errors.InsufficientPrivilege, psycopg.errors.UndefinedTable)
            ):
                conn.execute("SELECT count(*) FROM l1_atoms")
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            conn.execute(f"REVOKE ALL ON SCHEMA {schema} FROM {role}")
            conn.execute(f"DROP OWNED BY {role}")
            conn.execute(f"DROP ROLE {role}")
