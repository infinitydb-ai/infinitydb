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

    # GDPR wipe
    deleted = mem.delete("user_123")
    assert deleted["gdpr_compliant"] is True
    assert mem.search("project code?", user_id="user_123") == []
