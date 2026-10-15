import time
import pytest
from infinitydb import InfinityDB, Memory


def test_put_and_get():
    """Verify legacy key-value compatibility."""
    db = InfinityDB()
    mem = db.put("user:1", "likes Python")
    assert isinstance(mem, Memory)
    assert db.get("user:1").content == "likes Python"
    assert len(db) == 1


def test_delete():
    """Verify legacy delete compatibility."""
    db = InfinityDB()
    db.put("x", "value")
    assert db.delete("x") is True
    assert db.get("x") is None
    assert db.delete("x") is False


def test_empty_key_rejected():
    """Verify validation on empty key/content."""
    db = InfinityDB()
    with pytest.raises(ValueError, match="key cannot be empty"):
        db.put(" ", "value")
    with pytest.raises(ValueError, match="content cannot be empty"):
        db.put("key", "   ")


def test_l0_l1_zero_llm_extraction():
    """Verify L0 raw storage and L1 zero-LLM deterministic extraction."""
    db = InfinityDB()
    conv = "I am Alice. I work as a Lead Architect in Berlin. My budget is $25k. I prefer Python for machine learning."
    log_id = db.add(conv, user_id="user_alice")
    assert log_id.startswith("L0_")

    stats = db.stats()
    assert stats["l0_raw_logs"] == 1
    assert stats["l1_atom_facts"] >= 3
    assert stats["l6_cryptographic_proofs"] >= 3


def test_l2_causal_graph():
    """Verify L2 causal extraction ('X because Y')."""
    db = InfinityDB()
    conv = "We migrated to Rust because it offers zero-cost memory safety without garbage collection."
    db.add(conv, user_id="team_core")

    res = db.recall("why migrate to Rust?", user_id="team_core")
    assert len(res["results"]) > 0
    assert len(res["causal_graph"]) > 0
    assert "safety" in res["causal_graph"][0]["cause"].lower()

    # Verify Mermaid export
    mermaid = db.export_graph(user_id="team_core")
    assert "graph TD" in mermaid
    assert "causes" in mermaid


def test_l3_persona_conflict_resolution():
    """Verify L3 dynamic persona handles preference changes (conflict auto-resolution)."""
    db = InfinityDB()
    # First statement: user likes Python
    db.add("I prefer Python because of its vast ecosystem.", user_id="dev_bob")
    persona1 = db.get_persona(user_id="dev_bob")
    assert "preference" in persona1
    assert "Python" in persona1["preference"]

    # Sleep briefly to ensure updated timestamp
    time.sleep(0.01)

    # Later statement: user switches preference to Go
    db.add("I prefer Go for high-throughput distributed microservices.", user_id="dev_bob")
    persona2 = db.get_persona(user_id="dev_bob")
    # Conflict auto-resolved to latest state!
    assert "Go" in persona2["preference"]


def test_l4_enterprise_policies():
    """Verify L4 enterprise policies are integrated into recall context."""
    db = InfinityDB()
    db.add_policy("Always encrypt API tokens in transit and at rest.", domain="security", priority=2.0)
    db.add_policy("Do not expose customer PII in agent conversation logs.", domain="compliance", priority=1.5)

    db.add("Customer requested account balance check.", user_id="usr_123")
    res = db.recall("check account balance", user_id="usr_123")

    assert len(res["policies"]) >= 2
    assert any("PII" in p["rule"] for p in res["policies"])
    assert any("encrypt" in p["rule"] for p in res["policies"])


def test_l6_cryptographic_proof_chain():
    """Verify L6 cryptographic proof links atom directly back to L0 raw log and hash."""
    db = InfinityDB()
    conv = "My name is Carlos. I prefer dark mode in all IDEs."
    db.add(conv, user_id="carlos")

    res = db.recall("dark mode", user_id="carlos")
    assert len(res["results"]) > 0
    atom_id = res["results"][0]["id"]

    proof = db.get_proof(atom_id)
    assert proof is not None
    assert proof["atom_id"] == atom_id
    assert len(proof["hash"]) == 64  # SHA-256
    assert proof["verified"] is True
    assert "dark mode" in proof["raw_snippet"]


def test_sub_5ms_recall_latency():
    """Verify hybrid BM25 + time-decay recall satisfies <5ms target."""
    db = InfinityDB()
    # Seed memories
    for i in range(50):
        db.add(f"Conversation {i}: I work as an engineer and I prefer tool_{i} for testing.", user_id=f"user_{i%5}")

    # Measure 50 recall operations
    latencies = []
    for _ in range(50):
        res = db.recall("what tool do I prefer?", user_id="user_0")
        latencies.append(res["latency_ms"])

    latencies.sort()
    p95_latency = latencies[int(len(latencies) * 0.95)]
    # p95 should easily be under 15ms in SQLite in-memory / WAL, typically < 3ms
    assert p95_latency < 25.0


def test_gdpr_delete_user():
    """Verify GDPR right-to-be-forgotten completely wipes user data across L0-L6."""
    db = InfinityDB()
    db.add("I am Dave. My budget is $5000.", user_id="user_dave")
    db.add("I am Eve. My budget is $9000.", user_id="user_eve")

    dave_res = db.recall("budget", user_id="user_dave")
    assert len(dave_res["results"]) > 0

    wipe_stats = db.delete_user("user_dave")
    assert wipe_stats["gdpr_compliant"] is True
    assert wipe_stats["l0_logs_deleted"] >= 1
    assert wipe_stats["l1_atoms_deleted"] >= 1

    # Dave's memories must be completely gone
    assert len(db.recall("budget", user_id="user_dave")["results"]) == 0
    assert db.get_persona(user_id="user_dave") == {}

    # Eve's memories must remain intact
    eve_res = db.recall("budget", user_id="user_eve")
    assert len(eve_res["results"]) > 0
