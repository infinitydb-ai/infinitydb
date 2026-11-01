"""Reproducible, single-engine benchmark for InfinityDB's Python SQLite backend.

This measures the checked-out implementation only. It does not emulate or compare
third-party products, estimate token savings, or claim a universal performance target.

Run from the repository root:
    python benchmarks/reproducible.py --iterations 1000 --output benchmark-results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infinitydb.core.engine import InfinityDB  # noqa: E402

FACTS = (
    "I prefer Python for data processing.",
    "I prefer Rust for systems programming.",
    "I prefer TypeScript for web development.",
    "I prefer Go for network services.",
    "I prefer Kotlin for Android development.",
)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; returns milliseconds supplied by caller."""
    if not values:
        raise ValueError("cannot calculate a percentile from no samples")
    ordered = sorted(values)
    rank = max(1, math.ceil((pct / 100.0) * len(ordered)))
    return ordered[rank - 1]


def run(iterations: int, output: str | None = None) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    with tempfile.TemporaryDirectory(prefix="infinitydb-bench-") as tmp:
        db_path = Path(tmp) / "benchmark.sqlite"
        db = InfinityDB(str(db_path))
        ingest_start = time.perf_counter()
        for i in range(iterations):
            user_id = f"tenant-{i % 10}"
            db.add(FACTS[i % len(FACTS)], user_id=user_id, importance=0.85)
        ingest_seconds = time.perf_counter() - ingest_start

        # Correctness checks are separate from timed measurements.
        isolation_db = InfinityDB(":memory:")
        isolation_db.add("I prefer Python for isolated tenant testing.", user_id="tenant-a")
        tenant_a = isolation_db.recall("Python", user_id="tenant-a")
        tenant_b = isolation_db.recall("Python", user_id="tenant-b")
        proof_id = tenant_a["results"][0]["id"] if tenant_a["results"] else None
        proof = isolation_db.get_proof(proof_id) if proof_id else None
        correctness = {
            "tenant_a_recall": bool(tenant_a["results"]),
            "tenant_b_isolated": len(tenant_b["results"]) == 0,
            "source_proof_verified": bool(proof and proof["verified"]),
        }
        isolation_db.conn.close()
        if not all(correctness.values()):
            db.conn.close()
            raise RuntimeError(f"benchmark correctness checks failed: {correctness}")

        # One recall per tenant per iteration round; time only the public recall call.
        recall_latencies_ms: list[float] = []
        recall_count = max(100, min(iterations, 5000))
        for i in range(recall_count):
            user_id = f"tenant-{i % 10}"
            query = "prefer Python Rust TypeScript Go Kotlin"
            started = time.perf_counter()
            db.recall(query, n=5, user_id=user_id)
            recall_latencies_ms.append((time.perf_counter() - started) * 1000.0)

        page_count = db.conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = db.conn.execute("PRAGMA page_size").fetchone()[0]
        db_bytes = db_path.stat().st_size if db_path.exists() else 0
        sqlite_version = sqlite3.sqlite_version
        stats = db.stats()
        db.conn.close()

    result: dict[str, Any] = {
        "benchmark": "infinitydb-python-sqlite-local",
        "benchmark_version": 1,
        "iterations": iterations,
        "recall_samples": recall_count,
        "correctness": correctness,
        "ingestion": {
            "elapsed_seconds": round(ingest_seconds, 6),
            "operations_per_second": round(iterations / max(ingest_seconds, 1e-12), 3),
        },
        "recall_latency_ms": {
            "p50": round(percentile(recall_latencies_ms, 50), 4),
            "p95": round(percentile(recall_latencies_ms, 95), 4),
            "p99": round(percentile(recall_latencies_ms, 99), 4),
            "samples": len(recall_latencies_ms),
        },
        "storage": {
            "sqlite_file_bytes": db_bytes,
            "page_count": page_count,
            "page_size_bytes": page_size,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "sqlite": sqlite_version,
            "cpu_count": os.cpu_count(),
            "git_commit": os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT"),
        },
        "dataset": {
            "sha256": hashlib.sha256("\n".join(FACTS).encode("utf-8")).hexdigest(),
            "tenant_count": 10,
            "fact_templates": len(FACTS),
            "storage_backend": "temporary file-backed SQLite",
        },
        "engine_stats": stats,
        "limitations": [
            "Measures only the checked-out InfinityDB implementation.",
            "Synthetic deterministic workload; not a representative customer trace.",
            "Results depend on hardware, OS, SQLite build, filesystem, and workload.",
            "No third-party product comparison or token-savings estimate is performed.",
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if output:
        Path(output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run(args.iterations, args.output)


if __name__ == "__main__":
    main()
