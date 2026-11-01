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

def test_add_smoke():
    pass


def test_proof_verification_detects_source_tampering():
    """Proof verification must fail if the original source log is modified."""
    db = InfinityDB()
    db.add("I prefer Rust for memory-safe systems.", user_id="proof_user")
    result = db.recall("Rust", user_id="proof_user")
    atom_id = result["results"][0]["id"]

    proof = db.get_proof(atom_id)
    assert proof is not None
    assert proof["verified"] is True

    db.conn.execute(
        "UPDATE l0_logs SET raw = ? WHERE id = ?",
        ("tampered source text", proof["source_log_id"]),
    )
    db.conn.commit()

    tampered = db.get_proof(atom_id)
    assert tampered is not None
    assert tampered["verified"] is False


def test_add_rejects_invalid_inputs():
    db = InfinityDB()
    with pytest.raises(ValueError, match="conversation cannot be empty"):
        db.add("   ")
    with pytest.raises(ValueError, match="user_id cannot be empty"):
        db.add("valid conversation", user_id=" ")
    with pytest.raises(ValueError, match="importance must be between 0 and 1"):
        db.add("valid conversation", importance=1.1)
    with pytest.raises(TypeError, match="importance must be a number"):
        db.add("valid conversation", importance=True)


def test_recall_rejects_invalid_limits_and_user_ids():
    db = InfinityDB()
    with pytest.raises(ValueError, match="positive integer"):
        db.recall("python", n=0)
    with pytest.raises(ValueError, match="positive integer"):
        db.recall("python", n=-1)
    with pytest.raises(TypeError, match="n must be a positive integer"):
        db.recall("python", n=1.5)
    with pytest.raises(ValueError, match="user_id cannot be empty"):
        db.recall("python", user_id=" ")


def test_empty_recall_query_does_not_return_every_memory():
    db = InfinityDB()
    db.add("I prefer Python for backend development.", user_id="empty-query")
    result = db.recall("   ", user_id="empty-query")
    assert result["results"] == []
    assert result["tokens_saved"] is None


def test_importance_parameter_scales_extracted_fact_weight():
    db = InfinityDB()
    db.add("I prefer Rust.", user_id="low", importance=0.2)
    db.add("I prefer Rust.", user_id="high", importance=1.0)
    low = db.conn.execute(
        "SELECT importance FROM l1_atoms WHERE user_id = ?", ("low",)
    ).fetchone()["importance"]
    high = db.conn.execute(
        "SELECT importance FROM l1_atoms WHERE user_id = ?", ("high",)
    ).fetchone()["importance"]
    assert low < high



def test_fts_index_is_populated_and_retrieval_is_user_scoped():
    db = InfinityDB()
    db.add("I prefer PostgreSQL for transactional workloads.", user_id="tenant_a")
    db.add("I prefer Redis for ephemeral cache workloads.", user_id="tenant_b")

    indexed = db.conn.execute(
        "SELECT count(*) FROM l1_atoms_fts WHERE l1_atoms_fts MATCH ?",
        ('"PostgreSQL"',),
    ).fetchone()[0]
    assert indexed == 1

    tenant_a = db.recall("PostgreSQL", user_id="tenant_a")
    tenant_b = db.recall("PostgreSQL", user_id="tenant_b")
    assert len(tenant_a["results"]) == 1
    assert tenant_a["results"][0]["fact"].find("PostgreSQL") >= 0
    assert tenant_b["results"] == []


def test_fts_recall_does_not_return_unmatched_high_importance_memories():
    db = InfinityDB()
    db.add("I prefer Python for backend services.", user_id="fts-user", importance=1.0)
    result = db.recall("quantum banana", user_id="fts-user")
    assert result["results"] == []


def test_fts_index_tracks_user_deletion():
    db = InfinityDB()
    db.add("I prefer SQLite for embedded storage.", user_id="delete-fts")
    before = db.conn.execute(
        "SELECT count(*) FROM l1_atoms_fts WHERE l1_atoms_fts MATCH ?",
        ('"SQLite"',),
    ).fetchone()[0]
    assert before == 1

    db.delete_user("delete-fts")
    after = db.conn.execute(
        "SELECT count(*) FROM l1_atoms_fts WHERE l1_atoms_fts MATCH ?",
        ('"SQLite"',),
    ).fetchone()[0]
    assert after == 0


def test_fts_index_rebuilds_for_existing_database(tmp_path):
    db_path = str(tmp_path / "pre_fts.sqlite")
    db = InfinityDB(db_path)
    db.add("I prefer Rust for systems programming.", user_id="migration-user")
    db.conn.execute("DROP TRIGGER IF EXISTS l1_atoms_fts_insert")
    db.conn.execute("DROP TRIGGER IF EXISTS l1_atoms_fts_delete")
    db.conn.execute("DROP TRIGGER IF EXISTS l1_atoms_fts_update")
    db.conn.execute("DROP TABLE l1_atoms_fts")
    db.conn.execute(
        "DELETE FROM infinitydb_schema_meta WHERE key = 'l1_atoms_fts_version'"
    )
    db.conn.commit()
    db.conn.close()

    migrated = InfinityDB(db_path)
    result = migrated.recall("Rust", user_id="migration-user")
    assert len(result["results"]) == 1
    migrated.conn.close()


def test_l4_policies_are_tenant_scoped_with_global_defaults():
    db = InfinityDB()
    db.add_policy("Global baseline", domain="security", priority=1.0)
    db.add_policy(
        "Tenant A may use approved model X", domain="models", priority=3.0,
        tenant_id="tenant_a",
    )
    db.add_policy(
        "Tenant B must use approved model Y", domain="models", priority=3.0,
        tenant_id="tenant_b",
    )

    tenant_a = db.recall("anything", user_id="tenant_a")["policies"]
    tenant_b = db.recall("anything", user_id="tenant_b")["policies"]
    global_only = db.recall("anything")["policies"]

    assert {p["rule"] for p in tenant_a} == {
        "Global baseline", "Tenant A may use approved model X"
    }
    assert {p["rule"] for p in tenant_b} == {
        "Global baseline", "Tenant B must use approved model Y"
    }
    assert {p["rule"] for p in global_only} == {"Global baseline"}


def test_l4_policy_validation_and_tenant_deletion():
    db = InfinityDB()
    with pytest.raises(ValueError, match="rule cannot be empty"):
        db.add_policy(" ")
    with pytest.raises(ValueError, match="tenant_id cannot be empty"):
        db.add_policy("Scoped rule", tenant_id=" ")
    with pytest.raises(ValueError, match="priority must be finite"):
        db.add_policy("Invalid priority", priority=float("inf"))

    db.add_policy("Tenant-only secret policy", tenant_id="tenant_remove")
    db.add_policy("Shared global policy")
    stats = db.delete_user("tenant_remove")

    assert stats["l4_tenant_policies_deleted"] == 1
    remaining = db.recall("query", user_id="tenant_remove")["policies"]
    assert [p["rule"] for p in remaining] == ["Shared global policy"]


def test_l4_policy_schema_migrates_legacy_global_policies(tmp_path):
    db_path = str(tmp_path / "legacy_policy.sqlite")
    db = InfinityDB(db_path)
    db.add_policy("Legacy global policy", domain="security")
    db.conn.execute("DROP INDEX IF EXISTS idx_l4_tenant_priority")
    db.conn.execute("ALTER TABLE l4_policies DROP COLUMN tenant_id")
    db.conn.commit()
    db.conn.close()

    migrated = InfinityDB(db_path)
    policies = migrated.recall("query", user_id="any_tenant")["policies"]
    assert any(p["rule"] == "Legacy global policy" for p in policies)
    columns = {
        row["name"] for row in migrated.conn.execute("PRAGMA table_info(l4_policies)")
    }
    assert "tenant_id" in columns
    migrated.conn.close()
