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
