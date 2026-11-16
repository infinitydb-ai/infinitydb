"""Migration is the only feature that can move a user in one command, so test it."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infinitydb import migrate
from infinitydb.core.engine import InfinityDB

ROOT = Path(__file__).resolve().parents[1]

MEM0_EXPORT = {
    "results": [
        {"id": "m1", "memory": "Lives in Lisbon and works as a data engineer.",
         "user_id": "alice", "created_at": "2023-03-01T09:00:00Z"},
        {"id": "m2", "memory": "Budget for the cluster is forty thousand euros.",
         "user_id": "alice", "created_at": "2023-04-01T09:00:00Z"},
        {"id": "m3", "memory": "Prefers morning calls and short answers.",
         "user_id": "alice", "created_at": "2023-05-01T09:00:00Z"},
    ]
}


@pytest.fixture()
def mem0_file(tmp_path):
    path = tmp_path / "mem0-export.json"
    path.write_text(json.dumps(MEM0_EXPORT), encoding="utf-8")
    return path


def _db(tmp_path) -> InfinityDB:
    return InfinityDB(str(tmp_path / "memory.db"))


def test_mem0_export_round_trip(tmp_path, mem0_file):
    db = _db(tmp_path)
    try:
        stats = migrate.import_records(db, migrate.read_records(mem0_file, "json", "default"))
        assert stats["imported"] == 3
        assert stats["timestamps_inferred"] == 0, "mem0 created_at must be honoured"
        result = db.recall("where does alice live", n=3, user_id="alice",
                           reinforce=False, include_policies=False)
        assert any("Lisbon" in item["fact"] for item in result["results"])
    finally:
        db.close()


def test_created_at_becomes_event_time_not_import_time(tmp_path, mem0_file):
    db = _db(tmp_path)
    try:
        records = list(migrate.read_records(mem0_file, "json", "default"))
        migrate.import_records(db, iter(records))
        ts, inferred = migrate.parse_ts("2023-03-01T09:00:00Z")
        assert not inferred
        row = db.conn.execute(
            "SELECT min(ts) FROM l1_atoms WHERE user_id = 'alice'").fetchone()[0]
        assert abs(row - ts) < 86400, "the March fact should keep its March timestamp"
    finally:
        db.close()


def test_jsonl_with_roles_and_epoch_millis(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text("\n".join([
        json.dumps({"role": "user", "content": "I moved to Porto.", "ts": 1709251200000}),
        json.dumps({"role": "assistant", "content": "Noted.", "ts": 1709251260000}),
        "",
        "not json at all",
    ]), encoding="utf-8")
    records = list(migrate.read_records(path, "jsonl", "bob"))
    assert len(records) == 2, "a malformed line is skipped, not fatal"
    assert records[0].content == "I moved to Porto.", "user role needs no prefix"
    assert records[1].content.startswith("assistant:"), "other roles are kept for context"
    assert abs(records[0].ts - 1709251200.0) < 1, "milliseconds are normalised"


def test_plain_text_one_memory_per_line(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Lives in Lisbon.\n\nWorks at a ceramics studio.\n", encoding="utf-8")
    assert len(list(migrate.read_records(path, "text", "carol"))) == 2


def test_duplicates_within_a_file_are_skipped(tmp_path):
    path = tmp_path / "dupes.jsonl"
    row = json.dumps({"content": "Lives in Lisbon.", "ts": 1709251200})
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")
    db = _db(tmp_path)
    try:
        stats = migrate.import_records(db, migrate.read_records(path, "jsonl", "dave"))
        assert stats["imported"] == 1 and stats["skipped_existing"] == 1
    finally:
        db.close()


def test_rerunning_an_import_is_idempotent(tmp_path, mem0_file):
    db = _db(tmp_path)
    try:
        first = migrate.import_records(db, migrate.read_records(mem0_file, "json", "default"))
        second = migrate.import_records(db, migrate.read_records(mem0_file, "json", "default"))
        assert first["imported"] == 3
        assert second["imported"] == 0 and second["skipped_existing"] == 3
        assert db.stats()["l0_raw_logs"] == 3, "the store must not grow on a re-run"
    finally:
        db.close()


def test_dry_run_writes_nothing(tmp_path, mem0_file):
    db = _db(tmp_path)
    try:
        stats = migrate.import_records(db, migrate.read_records(mem0_file, "json", "default"),
                                       dry_run=True)
        assert stats["imported"] == 3 and stats["dry_run"]
        assert db.stats()["l1_atom_facts"] == 0
    finally:
        db.close()


def test_missing_timestamps_are_reported_not_hidden(tmp_path):
    path = tmp_path / "bare.json"
    path.write_text(json.dumps([{"memory": "Likes tea."}]), encoding="utf-8")
    records = list(migrate.read_records(path, "json", "erin"))
    assert records[0].ts_inferred is True


def test_cli_end_to_end(tmp_path, mem0_file):
    db_path = tmp_path / "memory.db"
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(
        [sys.executable, "-m", "infinitydb.migrate", str(mem0_file),
         "--db", str(db_path), "--user", "alice"],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "imported 3 memories" in proc.stdout
    assert "atoms now in store" in proc.stdout
    assert "recall check" in proc.stdout


def test_cli_dry_run_and_missing_file(tmp_path, mem0_file):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(ROOT)}
    dry = subprocess.run(
        [sys.executable, "-m", "infinitydb.migrate", str(mem0_file),
         "--db", str(tmp_path / "x.db"), "--dry-run"],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(ROOT),
    )
    assert dry.returncode == 0 and "would import 3" in dry.stdout

    missing = subprocess.run(
        [sys.executable, "-m", "infinitydb.migrate", str(tmp_path / "nope.json")],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(ROOT),
    )
    assert missing.returncode == 2
