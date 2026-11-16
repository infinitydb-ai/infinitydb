"""The audit bundle is the one feature a buyer will attack, so attack it here."""
from __future__ import annotations

import copy
import json
import time

import pytest

from infinitydb import audit, mcp_server
from infinitydb.core.engine import InfinityDB

FACTS = [
    ("I moved to Lisbon in March 2024.", 1709251200.0),
    ("My budget for the new cluster is forty thousand euros.", 1711929600.0),
    ("I prefer morning calls and short answers.", 1714521600.0),
]


@pytest.fixture()
def db(tmp_path):
    engine = InfinityDB(str(tmp_path / "audit.db"))
    for text, ts in FACTS:
        engine.add(text, user_id="alice", ts=ts, auto_supersede=False)
    yield engine
    engine.close()


def _payload(tool_result: dict) -> dict:
    return json.loads(tool_result["content"][0]["text"])


def test_clean_bundle_verifies_and_binds_every_fact(db):
    bundle = audit.export_bundle(db, "alice")
    assert bundle["count"] == len(bundle["records"]) >= 3
    assert all(record["proof_ok"] for record in bundle["records"])
    report = audit.verify_bundle(bundle)
    assert report["ok"] and report["checked"] == bundle["count"]
    assert report["digest"] == bundle["digest"]


def test_edited_fact_is_caught(db):
    bundle = audit.export_bundle(db, "alice")
    tampered = copy.deepcopy(bundle)
    tampered["records"][0]["fact"] = "I moved to Berlin in March 2024."
    report = audit.verify_bundle(tampered)
    assert not report["ok"]
    reasons = {failure["reason"] for failure in report["failures"]}
    assert {"fact_edited", "chain_broken"} <= reasons


def test_removed_fact_is_caught(db):
    bundle = audit.export_bundle(db, "alice")
    trimmed = copy.deepcopy(bundle)
    trimmed["records"].pop()
    assert not audit.verify_bundle(trimmed)["ok"]


def test_reordered_facts_are_caught(db):
    bundle = audit.export_bundle(db, "alice")
    swapped = copy.deepcopy(bundle)
    swapped["records"][0], swapped["records"][1] = swapped["records"][1], swapped["records"][0]
    report = audit.verify_bundle(swapped)
    assert not report["ok"]
    assert any(f["reason"] == "chain_broken" for f in report["failures"])


def test_anchor_detects_a_different_bundle(db):
    first = audit.export_bundle(db, "alice")
    db.add("I also keep bees.", user_id="alice", ts=1717200000.0, auto_supersede=False)
    second = audit.export_bundle(db, "alice")

    assert audit.verify_bundle(first, anchor=first["digest"])["anchor_checked"] is True
    mismatch = audit.verify_bundle(second, anchor=first["digest"])
    assert mismatch["anchor_checked"] is False
    assert any(f["reason"] == "anchor_mismatch" for f in mismatch["failures"])


def test_export_without_text_still_binds_the_chain(db):
    bundle = audit.export_bundle(db, "alice", include_text=False)
    assert bundle["text_included"] is False
    assert all(record["fact"] is None for record in bundle["records"])
    assert audit.verify_bundle(bundle)["ok"]
    tampered = copy.deepcopy(bundle)
    tampered["records"][0]["fact_sha256"] = "0" * 64
    assert not audit.verify_bundle(tampered)["ok"]


def test_erasure_receipt_names_what_was_removed(db):
    with pytest.raises(PermissionError):
        audit.erase_with_receipt(db, "alice")

    before = audit.export_bundle(db, "alice")["digest"]
    receipt = audit.erase_with_receipt(db, "alice", confirm=True)
    assert receipt["bundle_digest_before"] == before
    assert receipt["atoms_erased"] >= 3
    assert receipt["atoms_remaining"] == 0
    assert receipt["layers"]["l1_atoms_deleted"] >= 1
    assert audit.verify_receipt(receipt)

    modified = {**receipt, "atoms_erased": 0}
    assert audit.verify_receipt(modified) is False


def test_export_is_deterministic_for_an_unchanged_store(db):
    first = audit.export_bundle(db, "alice")
    second = audit.export_bundle(db, "alice")
    assert first["digest"] == second["digest"], "identical stores must export identically"


def test_mcp_tools_round_trip(db, monkeypatch):
    monkeypatch.setattr(mcp_server, "db", db)

    exported = _payload(mcp_server.handle_call_tool("audit_export", {"user_id": "alice"}))
    assert exported["count"] >= 3

    ok = _payload(mcp_server.handle_call_tool(
        "audit_verify", {"bundle": exported, "anchor": exported["digest"]}))
    assert ok["ok"] and ok["anchor_checked"] is True

    tampered = copy.deepcopy(exported)
    tampered["records"][0]["fact"] = "I moved to Berlin."
    bad = mcp_server.handle_call_tool("audit_verify", {"bundle": tampered})
    assert bad["isError"] is True

    refused = mcp_server.handle_call_tool("erase_with_receipt", {"user_id": "alice"})
    assert refused["isError"] is True


def test_verify_stays_linear_on_a_thousand_facts(tmp_path):
    """Guard against an accidental O(n^2): verification is a single pass."""
    engine = InfinityDB(str(tmp_path / "many.db"))
    try:
        for i in range(1000):
            engine.add(f"User note {i} about project Atlas.", user_id="bulk",
                       ts=1709251200.0 + i, auto_supersede=False)
        started = time.perf_counter()
        bundle = audit.export_bundle(engine, "bulk")
        exported = time.perf_counter() - started
        started = time.perf_counter()
        assert audit.verify_bundle(bundle)["ok"]
        verified = time.perf_counter() - started
        assert bundle["count"] >= 1000
        assert exported < 10 and verified < 10, (exported, verified)
    finally:
        engine.close()
