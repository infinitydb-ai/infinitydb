"""Measure the two latency targets the project spec states, against the real engine.

Targets (as written in the spec, not as claimed by us):
  * P99 write / ingest  < 5 ms
  * P99 multi-hop read  < 10 ms

This script reports what the checked-out Python/SQLite implementation actually does.
It does not emulate competitors, does not estimate cost, and makes no claim that the
numbers transfer to another machine, corpus, or the Rust backend (which does not exist
yet).

    python benchmarks/latency_targets.py --facts 2000 --reads 400
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infinitydb.core.engine import InfinityDB  # noqa: E402

TARGET_WRITE_P99_MS = 5.0
TARGET_MULTIHOP_READ_P99_MS = 10.0

SUBJECTS = ["Ada", "Grace", "Alan", "Edsger", "Barbara", "Donald", "Margaret", "Ken"]
CITIES = ["Lisbon", "Osaka", "Toronto", "Nairobi", "Reykjavik", "Bogota"]
TEAMS = ["Atlas", "Meridian", "Nimbus", "Vertex"]
TOOLS = ["PyTorch", "DuckDB", "Zig", "Elixir", "Solidity"]


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100.0 * len(ordered))) - 1]


def facts(n: int) -> list[str]:
    out = []
    for i in range(n):
        s, c, t, tool = SUBJECTS[i % 8], CITIES[i % 6], TEAMS[i % 4], TOOLS[i % 5]
        if i % 3 == 0:
            out.append(f"{s} moved to {c} in 2021 and now leads team {t}.")
        elif i % 3 == 1:
            out.append(f"{s} works at {t} and uses {tool} for data pipelines.")
        else:
            out.append(f"{s} studied at university {i} and later joined {t} in {c}.")
    return out


def run(n_facts: int, reads: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="infinitydb-latency-") as tmp:
        db = InfinityDB(str(Path(tmp) / "latency.sqlite"))
        corpus = facts(n_facts)
        writes: list[float] = []
        for i, text in enumerate(corpus):
            started = time.perf_counter()
            db.add(text, user_id=f"tenant-{i % 10}", importance=0.8)
            writes.append((time.perf_counter() - started) * 1000.0)

        # Two-hop style question: subject + attribute that lives in another fact.
        queries = []
        for i in range(reads):
            s = SUBJECTS[i % 8]
            t = TEAMS[i % 4]
            queries.append(f"Where does the person leading team {t} live?" if i % 2 else
                           f"Which city is associated with {s} who works at {t}?")

        plain: list[float] = []
        deep: list[float] = []
        for q in queries:
            started = time.perf_counter()
            db.recall(q, n=10, user_id=f"tenant-{len(plain) % 10}")
            plain.append((time.perf_counter() - started) * 1000.0)

            started = time.perf_counter()
            db.recall(q, n=10, user_id=f"tenant-{len(deep) % 10}", assembly_depth=1000,
                      max_chunk_turns=12, budget_tokens=1400)
            deep.append((time.perf_counter() - started) * 1000.0)

        stats = db.stats()
        db.conn.close()

    def block(series: list[float]) -> dict:
        return {
            "p50": round(pct(series, 50), 3),
            "p95": round(pct(series, 95), 3),
            "p99": round(pct(series, 99), 3),
            "max": round(max(series), 3),
            "mean": round(statistics.fmean(series), 3),
            "samples": len(series),
        }

    write = block(writes)
    plain_r = block(plain)
    deep_r = block(deep)
    return {
        "benchmark": "infinitydb-latency-targets",
        "engine": "python-sqlite",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "corpus": {"facts": n_facts, "tenants": 10, "memory_count": stats.get("memories")},
        "targets": {"write_p99_ms": TARGET_WRITE_P99_MS,
                    "multihop_read_p99_ms": TARGET_MULTIHOP_READ_P99_MS},
        "write_ms": write,
        "read_plain_ms": plain_r,
        "read_multihop_ms": deep_r,
        "verdict": {
            "write_p99_meets_target": write["p99"] < TARGET_WRITE_P99_MS,
            "multihop_p99_meets_target": deep_r["p99"] < TARGET_MULTIHOP_READ_P99_MS,
            "plain_p99_meets_target": plain_r["p99"] < TARGET_MULTIHOP_READ_P99_MS,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=int, default=2000)
    ap.add_argument("--reads", type=int, default=400)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    result = run(args.facts, args.reads)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
