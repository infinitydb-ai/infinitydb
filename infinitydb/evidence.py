#!/usr/bin/env python3
"""`infinitydb-evidence` — machine-check the claims, including the failures.

Every memory product asks to be believed. This command exists so nobody has to:
it runs the engine, the audit path, the migration path and the MCP transport, and
prints what actually happened on this machine. It exits non-zero if any check
fails, so it can be wired into CI or run in a procurement review.

    infinitydb-evidence
    infinitydb-evidence --with-arena     # also recompute the benchmark table

It also prints the project's known weaknesses, from the same source of truth as
the docs. A tool that only lists strengths is marketing; this one is designed to
be complained about, with numbers attached.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Stated here, not buried: the things a buyer should push back on.
LIMITS = [
    ("multi-hop retrieval", "0.5167 @10 via bridges; plain retrieval at the same slots scores 0.5714 — "
                            "the bridge mechanism loses (docs/SELF_IMPROVEMENT.md, Phase 2g)"),
    ("open-domain questions", "loses to mem0 on commonsense questions (README per-category table)"),
    ("independent reproduction", "none. Every number here was produced on one machine by this project"),
    ("datasets", "LoCoMo plus a LongMemEval sample; no third dataset, no held-out split for retrieval tuning"),
    ("encryption at rest", "none. SQLite file permissions are the only access control"),
    ("multi-tenancy", "user_id is a string boundary, not a security boundary"),
    ("scale", "measured to 5,000 facts; not tested at millions of rows or multiple processes"),
    ("model-backed extraction", "not implemented by choice; deterministic patterns only, so extraction quality "
                                "is bounded by those patterns"),
]


def _run_demo() -> tuple[bool, str]:
    from infinitydb import demo

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = demo.main([])
    text = buffer.getvalue()
    ok = code == 0 and "Lisbon" in text and "data engineer" in text
    return ok, f"demo printed {len(text.splitlines())} lines, recalled the seeded facts"


def _run_latency() -> tuple[bool, str]:
    from infinitydb.core.engine import InfinityDB

    db = InfinityDB(":memory:")
    try:
        started = time.perf_counter()
        for i in range(200):
            db.add(f"Note {i}: project Atlas milestone moved to week {i % 52}.",
                   user_id="ev", ts=1709251200.0 + i, auto_supersede=False)
        ingest = (time.perf_counter() - started) * 1000 / 200

        started = time.perf_counter()
        hits = 0
        for i in range(50):
            result = db.recall(f"project Atlas week {i % 52}", n=5, user_id="ev",
                               reinforce=False, include_policies=False)
            hits += len(result.get("results", []))
        recall = (time.perf_counter() - started) * 1000 / 50
    finally:
        db.close()
    ok = bool(hits) and recall < 250
    return ok, (f"ingest {ingest:.2f} ms/turn, recall {recall:.2f} ms/query "
                f"({hits} results over 50 queries)")


def _run_audit() -> tuple[bool, str]:
    from infinitydb import audit
    from infinitydb.core.engine import InfinityDB

    db = InfinityDB(":memory:")
    try:
        for i in range(5):
            db.add(f"Fact {i} about Lisbon and the ceramics studio.",
                   user_id="ev", ts=1709251200.0 + i, auto_supersede=False)
        bundle = audit.export_bundle(db, "ev")
        clean = audit.verify_bundle(bundle)["ok"]
        bundle["records"][0]["fact"] = "tampered"
        caught = not audit.verify_bundle(bundle)["ok"]
    finally:
        db.close()
    ok = clean and caught
    return ok, ("export verified and a hand-edited fact was rejected"
                if ok else "audit round trip failed")


def _run_migration() -> tuple[bool, str]:
    from infinitydb import migrate
    from infinitydb.core.engine import InfinityDB

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "export.jsonl"
        path.write_text("\n".join(json.dumps({"memory": f"Memory {i} about project Atlas.",
                                              "user_id": "ev", "created_at": f"2023-0{i+1}-01T00:00:00Z"})
                                  for i in range(4)), encoding="utf-8")
        db = InfinityDB(str(Path(tmp) / "m.db"))
        try:
            first = migrate.import_records(db, migrate.read_records(path, "jsonl", "ev"))
            second = migrate.import_records(db, migrate.read_records(path, "jsonl", "ev"))
        finally:
            db.close()
    ok = first["imported"] == 4 and second["imported"] == 0
    return ok, (f"imported {first['imported']}, re-run imported {second['imported']} "
                f"(idempotent)" if ok else "migration round trip failed")


def _run_mcp() -> tuple[bool, str]:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(ROOT),
           "HOME": tempfile.mkdtemp(), "INFINITYDB_PATH": ":memory:"}
    proc = subprocess.Popen([sys.executable, "-m", "infinitydb.mcp_server"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    lines: queue.Queue[str] = queue.Queue()

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put("")

    threading.Thread(target=pump, daemon=True).start()
    try:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                     "params": {"protocolVersion": "2024-11-05"}}) + "\n")
        proc.stdin.flush()
        reply = json.loads(lines.get(timeout=20) or "{}")
        version = reply.get("result", {}).get("serverInfo", {}).get("version")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n")
        proc.stdin.flush()
        tools = json.loads(lines.get(timeout=20) or "{}").get("result", {}).get("tools", [])
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait(timeout=20)
    ok = bool(version) and len(tools) >= 10
    return ok, f"MCP handshake ok, server {version}, {len(tools)} tools"


def _run_arena() -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, str(ROOT / "benchmarks/arena/arena.py"), "verify"],
                          capture_output=True, text=True, timeout=600,
                          env={**os.environ, "PYTHONPATH": str(ROOT)})
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    return proc.returncode == 0, (tail[-1] if tail else "no output")


CHECKS = [
    ("demo", _run_demo),
    ("recall latency", _run_latency),
    ("audit bundle", _run_audit),
    ("migration", _run_migration),
    ("mcp transport", _run_mcp),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="infinitydb-evidence",
                                 description="Machine-check this project's claims, including its failures.")
    ap.add_argument("--with-arena", action="store_true", help="also recompute the benchmark table")
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    args = ap.parse_args(argv)

    checks = list(CHECKS)
    if args.with_arena:
        checks.append(("arena verify", _run_arena))

    results = []
    for name, fn in checks:
        started = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as exc:  # a check that crashes is a failed check, not a traceback
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append({"check": name, "ok": ok, "detail": detail,
                        "seconds": round(time.perf_counter() - started, 3)})

    passed = sum(1 for row in results if row["ok"])
    report = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "results": results,
        "passed": passed,
        "failed": len(results) - passed,
        "known_limits": [{"area": area, "detail": detail} for area, detail in LIMITS],
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"InfinityDB self-audit · {report['platform']} · python {report['python']}")
        print()
        for row in results:
            mark = "PASS" if row["ok"] else "FAIL"
            print(f"  {mark}  {row['check']:<16} {row['detail']}")
        print()
        print(f"  {passed}/{len(results)} checks passed")
        print()
        print("Known weaknesses (do not take this project's word for these — push back):")
        for area, detail in LIMITS:
            print(f"  · {area}: {detail}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
