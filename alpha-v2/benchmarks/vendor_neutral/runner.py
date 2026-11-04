"""Reproducible side-by-side memory benchmark runner."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

DIMENSIONS = 128


def vector_for(text: str) -> list[float]:
    """Stable dependency-free feature hash for the InfinityDB baseline adapter."""
    import hashlib
    values = [0.0] * DIMENSIONS
    for token in text.lower().split() or [text.lower()]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, len(digest), 4):
            slot = int.from_bytes(digest[offset:offset + 2], "big") % DIMENSIONS
            values[slot] += 1.0 if digest[offset + 2] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def load_factory(spec: str) -> Callable[[], Any]:
    if ":" not in spec:
        raise ValueError("adapter must use module:factory syntax")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{spec} is not callable")
    return function


def run(adapter: Any, records: int, queries: int, tenants: int, top_k: int) -> dict[str, Any]:
    ingest_seconds: list[float] = []
    search_seconds: list[float] = []
    expected: list[tuple[str, str, str, list[float]]] = []
    tenant_by_id = {f"memory-{index:08d}": f"tenant-{index % tenants}" for index in range(records)}
    tracemalloc.start()
    started = time.perf_counter()
    try:
        for index in range(records):
            tenant = f"tenant-{index % tenants}"
            memory_id = f"memory-{index:08d}"
            content = f"benchmark record {index} tenant {tenant} unique token-{index}"
            vector = vector_for(content)
            before = time.perf_counter()
            adapter.add(memory_id, tenant, content, vector)
            ingest_seconds.append(time.perf_counter() - before)
            if index < queries:
                expected.append((tenant, memory_id, content, vector))
        ingest_total = time.perf_counter() - started

        hits = 0
        returned = 0
        cross_tenant_leaks = 0
        for tenant, expected_id, content, vector in expected:
            before = time.perf_counter()
            result = list(adapter.search(tenant, content, vector, top_k) or [])
            search_seconds.append(time.perf_counter() - before)
            returned += len(result)
            hits += int(expected_id in result)
            # Adapters return benchmark-local IDs only for results owned by
            # this tenant. The harness records any foreign IDs as a leak.
            cross_tenant_leaks += sum(
                1 for memory_id in result
                if memory_id in tenant_by_id and tenant_by_id[memory_id] != tenant
            )
        query_total = sum(search_seconds)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return {
        "schema_version": 2,
        "workload": {"records": records, "queries": len(expected), "tenants": tenants, "top_k": top_k, "vector_dimensions": DIMENSIONS},
        "ingest": {
            "elapsed_seconds": ingest_total,
            "records_per_second": records / ingest_total if ingest_total else 0,
            "latency_ms_p50": percentile(ingest_seconds, .50) * 1000,
            "latency_ms_p95": percentile(ingest_seconds, .95) * 1000,
        },
        "query": {
            "elapsed_seconds": query_total,
            "queries_per_second": len(expected) / query_total if query_total else 0,
            "latency_ms_p50": percentile(search_seconds, .50) * 1000,
            "latency_ms_p95": percentile(search_seconds, .95) * 1000,
            "recall_at_k": hits / len(expected) if expected else 0,
            "expected_recall_hits": hits,
            "returned_candidates": returned,
            "cross_tenant_id_leaks": cross_tenant_leaks,
        },
        "python_tracemalloc_peak_bytes": peak_bytes,
        "adapter": adapter.metadata(),
        "notes": [
            "Same text corpus and query strings are used for every adapter.",
            "Competitor adapters use each product's configured embedding and retrieval pipeline; InfinityDB baseline uses the supplied deterministic feature-hash vectors.",
            "This is a systems smoke benchmark, not a semantic-quality leaderboard.",
            "tracemalloc excludes native allocations and remote service memory; measure process RSS and cloud cost separately.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter", action="append", required=True,
        help="Python module:factory; repeat for side-by-side comparison",
    )
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--tenants", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.records, args.queries, args.tenants, args.top_k) < 1 or args.queries > args.records:
        parser.error("records, queries, tenants and top-k must be positive; queries <= records")

    comparisons: dict[str, Any] = {}
    for spec in args.adapter:
        factory = load_factory(spec)
        adapter = factory()
        try:
            comparisons[spec] = run(adapter, args.records, args.queries, args.tenants, args.top_k)
        finally:
            adapter.close()

    report = {
        "benchmark": "InfinityDB vendor-neutral adapter comparison",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "comparisons": comparisons,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + chr(10), encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
