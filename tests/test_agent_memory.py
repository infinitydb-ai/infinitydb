"""Test AgentMemory high-level drop-in interface."""

from infinitydb import AgentMemory


def test_agent_memory_basic_lifecycle():
    mem = AgentMemory(":memory:")
    log_id = mem.add("User hates long answers and loves python. Project code is ABC123", user_id="user_123")
    assert log_id.startswith("L0_")

    facts = mem.search("project code?", user_id="user_123", limit=3)
    assert len(facts) > 0
    assert any("ABC123" in f or "python" in f.lower() for f in facts)

    stats = mem.stats()
    assert stats["l0_raw_logs"] >= 1
    assert stats["l1_atom_facts"] >= 1

    # Tenant data removal
    deleted = mem.delete("user_123")
    assert deleted["l0_logs_deleted"] >= 1
    assert deleted["l1_atoms_deleted"] >= 1
    assert mem.search("project code?", user_id="user_123") == []


def test_agent_memory_close_releases_connection():
    mem = AgentMemory(":memory:")
    mem.add("I prefer explicit resource cleanup.", user_id="cleanup_user")
    connection = mem.db.conn
    mem.close()
    import sqlite3
    import pytest

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_agent_memory_context_manager_closes_connection():
    import sqlite3
    import pytest

    connection = None
    with AgentMemory(":memory:") as mem:
        connection = mem.db.conn
        mem.add("I prefer context-managed resources.", user_id="context_user")
        assert mem.search("context-managed", user_id="context_user")
    assert connection is not None
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_core_engine_context_manager_closes_connection():
    import sqlite3
    import pytest

    from infinitydb.core.engine import InfinityDB

    connection = None
    with InfinityDB(":memory:") as db:
        connection = db.conn
        db.add("I prefer explicit core lifecycle handling.", user_id="core_user")
        assert db.recall("lifecycle", user_id="core_user")["results"]
    assert connection is not None
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_search_matches_inflected_forms_of_the_query_term():
    """The atom index is unstemmed, so the match string has to bridge inflections.

    Regression: ``search("work")`` returned ``[]`` against a stored "works at the
    hospital in Leeds" on 1.4.0 and 1.4.1 — the first query a caller types. The
    turn-window index is porter-stemmed, the atom index is not, so both the FTS
    match atoms and the BM25 term counts now carry stem variants.
    """
    mem = AgentMemory(":memory:")
    mem.add("User works at the hospital in Leeds and prefers concise answers", user_id="u_inflect")
    mem.add("The deployment uses Postgres 16 in the eu-west region", user_id="u_inflect")

    hospital = "hospital"
    for query in ("work", "works", "working", "worked", "where does the user work?"):
        facts = mem.search(query, user_id="u_inflect", limit=3)
        assert facts, f"no hit for {query!r}"
        assert any(hospital in f for f in facts), f"wrong fact for {query!r}: {facts}"

    # A verb whose stored form is a longer noun still matches through the stem.
    for query in ("deploy", "deployed", "deploying"):
        facts = mem.search(query, user_id="u_inflect", limit=3)
        assert facts, f"no hit for {query!r}"
        assert any("Postgres" in f for f in facts), f"wrong fact for {query!r}: {facts}"

    # Expansion must not invent hits: an absent word still returns nothing.
    assert mem.search("zebra", user_id="u_inflect") == []
