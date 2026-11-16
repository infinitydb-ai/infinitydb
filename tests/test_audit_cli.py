"""The CLI is what a compliance team runs, so it is tested as a process.

Exit codes are part of the contract: 0 verified, 1 failed verification, 2 bad
input. A shell script has nothing else to branch on, so these tests assert them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infinitydb import audit
from infinitydb.core.engine import InfinityDB

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def store(tmp_path):
    db = InfinityDB(str(tmp_path / "mem.db"))
    for text, ts in [
        ("I moved to Lisbon in March 2024.", 1709251200.0),
        ("My budget for the new cluster is forty thousand euros.", 1711929600.0),
    ]:
        db.add(text, user_id="alice", ts=ts, auto_supersede=False)
    db.close()
    return tmp_path


def run(*args: str) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-m", "infinitydb.audit", *args],
                          capture_output=True, text=True, timeout=120, env=env, cwd=str(ROOT))


def test_export_then_verify_round_trip(store):
    bundle_path = store / "bundle.json"
    exported = run("export", "--db", str(store / "mem.db"), "--user", "alice",
                   "--out", str(bundle_path))
    assert exported.returncode == 0, exported.stderr
    assert "digest:" in exported.stdout and "chain_head:" in exported.stdout

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["count"] >= 2
    assert audit.verify_bundle(bundle)["ok"]

    verified = run("verify", str(bundle_path))
    assert verified.returncode == 0
    assert "PASS" in verified.stdout


def test_verify_fails_on_a_tampered_bundle(store):
    bundle_path = store / "bundle.json"
    run("export", "--db", str(store / "mem.db"), "--user", "alice", "--out", str(bundle_path))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["records"][0]["fact"] = "I moved to Berlin in March 2024."
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    failed = run("verify", str(bundle_path), "--json")
    assert failed.returncode == 1
    assert "fact_edited" in failed.stdout


def test_anchor_mismatch_is_reported(store):
    bundle_path = store / "bundle.json"
    run("export", "--db", str(store / "mem.db"), "--user", "alice", "--out", str(bundle_path))
    bad_anchor = run("verify", str(bundle_path), "--anchor", "0" * 64)
    assert bad_anchor.returncode == 1
    assert "anchor" in bad_anchor.stdout.lower()


def test_no_text_export_hides_facts(store):
    bundle_path = store / "bundle.json"
    assert run("export", "--db", str(store / "mem.db"), "--user", "alice",
               "--no-text", "--out", str(bundle_path)).returncode == 0
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["text_included"] is False
    assert all(record["fact"] is None for record in bundle["records"])
    assert run("verify", str(bundle_path)).returncode == 0


def test_receipt_requires_yes_and_erases_when_given(store):
    refused = run("receipt", "--db", str(store / "mem.db"), "--user", "alice",
                  "--out", str(store / "r.json"))
    assert refused.returncode == 2
    assert "without --yes" in refused.stdout

    receipt_path = store / "receipt.json"
    erased = run("receipt", "--db", str(store / "mem.db"), "--user", "alice",
                 "--yes", "--out", str(receipt_path))
    assert erased.returncode == 0, erased.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["atoms_erased"] >= 2 and receipt["atoms_remaining"] == 0

    checked = run("verify-receipt", str(receipt_path))
    assert checked.returncode == 0 and "PASS" in checked.stdout

    receipt["atoms_erased"] = 0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert run("verify-receipt", str(receipt_path)).returncode == 1


def test_unreadable_input_exits_2(store):
    missing = run("verify", str(store / "nope.json"))
    assert missing.returncode == 2
