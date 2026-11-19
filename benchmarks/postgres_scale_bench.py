#!/usr/bin/env python3
"""Measure the engine at scale, and prove tenants cannot see each other.

Each tenant gets its own PostgreSQL schema. This gate writes
``--total-atoms / --tenants`` facts per tenant, with a marker token unique to
that tenant in every fact, then checks four things on the real ``InfinityDB``
read path:

1. every requested atom is stored (counted through the engine, per tenant);
2. no tenant schema contains another tenant's marker;
3. no unqualified read returns another tenant's rows (``search_path``);
4. no ``recall()`` result contains another tenant's marker.

Latency is reported as p50/p95/p99 for one ``add()`` and one ``recall()`` at
this size. This is single-host, single-process evidence; it is not a
production capacity claim. ``recall`` uses ``reinforce=False`` so the
measurement does not also write.

Run it with the optional extra installed::

    pip install "infinitydb-ai[postgres]" pgserver
    python benchmarks/postgres_scale_bench.py --total-atoms 100000 --tenants 100 --skip-locomo

``pgserver`` starts a throwaway PostgreSQL + pgvector instance, so no server
setup is needed. Pass ``--dsn`` to measure a database you already have.

Gates (stated before the run, and printed on failure):

* stored atoms == requested atoms;
* zero cross-tenant markers in storage, unqualified reads and recall results;
* ``add()`` p95 <= 100 ms and ``recall()`` p95 <= 250 ms at this size.

A gate failure exits non-zero and still writes the JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
if str(REPO_ROOT / "benchmarks" / "arena") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "arena"))

from infinitydb.core.engine import InfinityDB  # noqa: E402
from infinitydb.storage.engine import tenant_schema  # noqa: E402

TOPICS = (
    "billing",
    "search",
    "cache",
    "ingest",
    "replica",
    "auth",
    "export",
    "queue",
    "vector",
    "ledger",
    "gateway",
    "scheduler",
)

ADD_P95_LIMIT_MS = 100.0
RECALL_P95_LIMIT_MS = 250.0


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile, matching benchmarks/locomo_local_compare.py."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * len(ordered) + 0.5)) - 1))
    return ordered[k]


def latency(values: list[float]) -> dict[str, float]:
    return {
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "p99": round(percentile(values, 99), 2),
        "mean": round(statistics.fmean(values), 2) if values else 0.0,
        "max": round(max(values), 2) if values else 0.0,
    }


def marker(tenant_index: int) -> str:
    return f"zt{tenant_index:03d}q"


def fact_text(tenant_index: int, atom_index: int) -> str:
    topic = TOPICS[atom_index % len(TOPICS)]
    return (
        f"Tenant {tenant_index:03d} engineer number {atom_index:05d} said the "
        f"{topic} service keeps cache batch {atom_index:05d} warm for project colima "
        f"marker {marker(tenant_index)}."
    )


def queries_for(tenant_index: int, atoms_per_tenant: int, count: int) -> list[str]:
    """Half marker queries, half generic ones, so isolation is not marker-only."""
    picked = [f"marker {marker(tenant_index)} cache batch"]
    picked.append(f"{TOPICS[tenant_index % len(TOPICS)]} service cache batch")
    picked.append("engineer project colima cache batch")
    picked.append("tenant engineer said service keeps cache")
    step = max(1, atoms_per_tenant // max(1, count))
    for offset in range(1, max(1, count - len(picked) + 1)):
        picked.append(f"cache batch {(offset * step) % atoms_per_tenant:05d} colima")
    return picked[:count]


def scan_schema(db: InfinityDB, own_marker: str, all_markers: list[str]) -> tuple[int, int, str]:
    """Return (rows read, foreign-marker rows, current schema) for one tenant."""
    schema = db.conn.execute("SELECT current_schema()").fetchone()[0]
    rows = db.conn.execute("SELECT fact FROM l1_atoms").fetchall()
    foreign = 0
    for row in rows:
        text = row[0] or ""
        if any(m in text for m in all_markers if m != own_marker):
            foreign += 1
    return len(rows), foreign, schema


def locomo_sanity(dsn: str, dataset: Path, max_questions: int) -> dict[str, Any]:
    """Ingest one LoCoMo conversation and report evidence recall at this scale host."""
    if not dataset.exists():
        return {"status": "skipped", "reason": f"dataset not found: {dataset}"}
    from backend_parity import _evidence_hits
    from locomo_local_compare import extract_sessions

    data = json.loads(dataset.read_text(encoding="utf-8"))
    sample = data[0]
    db = InfinityDB(dsn=dsn, tenant_id="locomo-sanity")
    hits = 0
    asked = 0
    recall_ms: list[float] = []
    try:
        for session_key, turns, ts in extract_sessions(sample):
            for turn in turns:
                db.add(
                    conversation=f"{turn['speaker']}: {turn['text']}",
                    user_id="locomo",
                    session_id=f"locomo_{session_key}",
                    ts=ts,
                    auto_supersede=False,
                )
        now = max((ts for _k, _t, ts in extract_sessions(sample) if ts is not None), default=None)
        for qa in sample.get("qa", []):
            if int(qa.get("category", 0)) == 5:
                continue
            question = str(qa.get("question", "")).strip()
            gold = str(qa.get("answer", "")).strip()
            if not question or not gold:
                continue
            if asked >= max_questions:
                break
            asked += 1
            start = time.perf_counter()
            result = db.recall(
                query=question,
                n=10,
                user_id="locomo",
                include_policies=False,
                now=now,
                reinforce=False,
            )
            recall_ms.append((time.perf_counter() - start) * 1000.0)
            facts = [item["fact"] for item in result.get("results", [])]
            hit = _evidence_hits(sample, qa, facts)
            if hit:
                hits += 1
        return {
            "status": "measured",
            "questions": asked,
            "evidence_recall_at_10_pct": round(100.0 * hits / asked, 2) if asked else None,
            "recall_latency_ms": latency(recall_ms),
            "note": "one conversation, not the published 87.97% four-conversation row",
        }
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-atoms", type=int, default=100_000)
    parser.add_argument("--tenants", type=int, default=100)
    parser.add_argument("--queries-per-tenant", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--dsn", type=str, default=None)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "benchmarks/.cache/locomo10.json")
    parser.add_argument("--skip-locomo", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "benchmarks/postgres-scale-results.json")
    args = parser.parse_args()

    try:
        import pgserver  # noqa: F401
        import psycopg  # noqa: F401
    except ImportError as exc:
        print(
            f"SKIP: the PostgreSQL extra is not installed ({exc}). "
            'Install with pip install "infinitydb-ai[postgres]" pgserver.'
        )
        return 0

    if args.tenants < 1 or args.total_atoms < args.tenants:
        print("FAIL: --tenants must be >= 1 and --total-atoms >= --tenants", file=sys.stderr)
        return 2
    atoms_per_tenant = args.total_atoms // args.tenants
    requested = atoms_per_tenant * args.tenants
    if requested != args.total_atoms:
        print(
            f"note: {args.total_atoms} is not divisible by {args.tenants}; "
            f"measuring {requested} atoms ({atoms_per_tenant} per tenant)"
        )

    random.seed(args.seed)
    all_markers = [marker(i) for i in range(args.tenants)]

    server = None
    if args.dsn:
        dsn = args.dsn
    else:
        workdir = tempfile.mkdtemp(prefix="infinitydb-scale-")
        server = pgserver.get_server(workdir)
        dsn = server.get_uri()

    try:
        server_version = None
        probe = InfinityDB(dsn=dsn, tenant_id="scale-probe")
        try:
            server_version = probe.conn.execute("SHOW server_version").fetchone()[0]
            expected_schema = tenant_schema("scale-probe")
            actual_schema = probe.conn.execute("SELECT current_schema()").fetchone()[0]
            if actual_schema != expected_schema:
                print(
                    f"FAIL: search_path resolved to {actual_schema!r}, expected {expected_schema!r}",
                    file=sys.stderr,
                )
                return 1
        finally:
            probe.close()

        add_ms: list[float] = []
        per_tenant_atoms: list[int] = []
        schema_mismatches: list[dict[str, Any]] = []
        unqualified_rows: list[int] = []
        ingest_start = time.perf_counter()
        for tenant_index in range(args.tenants):
            db = InfinityDB(dsn=dsn, tenant_id=f"scale-{tenant_index:03d}")
            try:
                for atom_index in range(atoms_per_tenant):
                    start = time.perf_counter()
                    db.add(
                        conversation=fact_text(tenant_index, atom_index),
                        user_id=f"user-{tenant_index:03d}",
                        session_id=f"sess-{tenant_index:03d}",
                        ts=1_700_000_000 + atom_index,
                        auto_supersede=False,
                    )
                    add_ms.append((time.perf_counter() - start) * 1000.0)
                stored = db.stats()["l1_atom_facts"]
                per_tenant_atoms.append(int(stored))
                expected = tenant_schema(f"scale-{tenant_index:03d}")
                actual = db.conn.execute("SELECT current_schema()").fetchone()[0]
                if actual != expected:
                    schema_mismatches.append(
                        {"tenant": tenant_index, "expected": expected, "actual": actual}
                    )
            finally:
                db.close()
            if (tenant_index + 1) % 10 == 0 or tenant_index + 1 == args.tenants:
                done = (tenant_index + 1) * atoms_per_tenant
                elapsed = time.perf_counter() - ingest_start
                print(
                    f"ingested {done}/{requested} atoms across {tenant_index + 1} tenants "
                    f"({elapsed:.1f}s, {done / elapsed:.0f} atoms/s)",
                    flush=True,
                )
        ingest_seconds = time.perf_counter() - ingest_start

        # Storage isolation: read every tenant's own rows through the engine
        # connection, with no schema qualification, and look for foreign markers.
        storage_foreign = 0
        for tenant_index in range(args.tenants):
            db = InfinityDB(dsn=dsn, tenant_id=f"scale-{tenant_index:03d}", initialize=False)
            try:
                rows, foreign, _schema = scan_schema(db, all_markers[tenant_index], all_markers)
                unqualified_rows.append(rows)
                storage_foreign += foreign
            finally:
                db.close()
        print(f"storage marker scan: {storage_foreign} foreign rows", flush=True)

        recall_ms: list[float] = []
        recall_foreign: list[dict[str, Any]] = []
        empty_recalls = 0
        for tenant_index in range(args.tenants):
            db = InfinityDB(dsn=dsn, tenant_id=f"scale-{tenant_index:03d}", initialize=False)
            try:
                for query in queries_for(tenant_index, atoms_per_tenant, args.queries_per_tenant):
                    start = time.perf_counter()
                    result = db.recall(
                        query=query,
                        n=args.top_k,
                        user_id=f"user-{tenant_index:03d}",
                        include_policies=False,
                        reinforce=False,
                    )
                    recall_ms.append((time.perf_counter() - start) * 1000.0)
                    items = result.get("results", [])
                    if not items:
                        empty_recalls += 1
                    for item in items:
                        text = item.get("fact", "")
                        for other in all_markers:
                            if other != all_markers[tenant_index] and other in text:
                                recall_foreign.append(
                                    {"tenant": tenant_index, "query": query, "foreign": other}
                                )
                                break
            finally:
                db.close()
            if (tenant_index + 1) % 20 == 0 or tenant_index + 1 == args.tenants:
                print(f"recall isolation checked for {tenant_index + 1}/{args.tenants} tenants", flush=True)

        stored_total = sum(per_tenant_atoms)
        report: dict[str, Any] = {
            "gate": "postgres-scale",
            "commit": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or None,
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
            },
            "postgres_version": server_version,
            "settings": {
                "total_atoms_requested": args.total_atoms,
                "tenants": args.tenants,
                "atoms_per_tenant": atoms_per_tenant,
                "queries_per_tenant": args.queries_per_tenant,
                "top_k": args.top_k,
                "seed": args.seed,
                "locomo": "skipped" if args.skip_locomo else "one conversation",
            },
            "storage": {
                "atoms_requested": requested,
                "atoms_stored": stored_total,
                "atoms_per_tenant_min": min(per_tenant_atoms),
                "atoms_per_tenant_max": max(per_tenant_atoms),
                "distinct_tenant_schemas": args.tenants,
            },
            "ingest": {
                "turns": len(add_ms),
                "total_seconds": round(ingest_seconds, 2),
                "atoms_per_second": round(len(add_ms) / ingest_seconds, 1) if ingest_seconds else None,
                "latency_ms": latency(add_ms),
            },
            "recall": {
                "queries": len(recall_ms),
                "empty_results": empty_recalls,
                "latency_ms": latency(recall_ms),
            },
            "isolation": {
                "schema_mismatches": schema_mismatches[:10],
                "storage_foreign_marker_rows": storage_foreign,
                "recall_foreign_marker_results": len(recall_foreign),
                "recall_foreign_examples": recall_foreign[:10],
                "unqualified_rows_min": min(unqualified_rows) if unqualified_rows else None,
                "unqualified_rows_max": max(unqualified_rows) if unqualified_rows else None,
            },
            "locomo_sanity": {"status": "skipped"}
            if args.skip_locomo
            else locomo_sanity(dsn, args.dataset, 40),
            "limits": [
                "One host, one process, synthetic facts; not a production capacity claim.",
                "add() is one committed transaction per turn, as the public API does.",
                "Schema isolation is verified by marker scans and search_path; it is not an RLS policy.",
            ],
            "gates": {
                "atoms_stored_equals_requested": stored_total == requested,
                "zero_cross_tenant_markers": storage_foreign == 0 and not recall_foreign,
                "add_p95_limit_ms": ADD_P95_LIMIT_MS,
                "recall_p95_limit_ms": RECALL_P95_LIMIT_MS,
                "add_p95_within_limit": percentile(add_ms, 95) <= ADD_P95_LIMIT_MS,
                "recall_p95_within_limit": percentile(recall_ms, 95) <= RECALL_P95_LIMIT_MS,
                "schema_matches_search_path": not schema_mismatches,
            },
        }
        passed = all(
            value
            for key, value in report["gates"].items()
            if isinstance(value, bool)
        )
        report["result"] = "PASS" if passed else "FAIL"
        print(json.dumps(report, indent=2))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        if not passed:
            print("FAIL: at least one scale gate did not hold (see gates).", file=sys.stderr)
            return 1
        print(
            f"PASS: {stored_total} atoms across {args.tenants} isolated schemas, "
            f"0 cross-tenant leaks; add p50/p95 {latency(add_ms)['p50']}/"
            f"{latency(add_ms)['p95']} ms, recall p50/p95 {latency(recall_ms)['p50']}/"
            f"{latency(recall_ms)['p95']} ms."
        )
        return 0
    finally:
        if server is not None:
            server.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
