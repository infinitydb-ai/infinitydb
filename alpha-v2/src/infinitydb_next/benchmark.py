from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

from .store import MemoryStore

TOPICS = [
    "sqlite indexing retrieval", "temporal agent memory",
    "python deployment pipeline", "graph relationship evidence",
    "token budget context compiler", "tenant isolation security",
    "long term persona memory", "agent skill execution",
]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * p)))]


def latency_summary(samples: list[float]) -> dict[str, float]:
    return {
        "p50": round(percentile(samples, .50), 3),
        "p95": round(percentile(samples, .95), 3),
        "p99": round(percentile(samples, .99), 3),
        "mean": round(statistics.mean(samples), 3),
    }


def run(
    records: int = 10000, queries: int = 200, batch_size: int = 1000,
    vector_limit: int = 10,
) -> dict:
    if min(records, queries, batch_size, vector_limit) < 1:
        raise ValueError("records, queries, batch_size and vector_limit must be positive")
    with tempfile.TemporaryDirectory() as directory:
        store = MemoryStore(Path(directory) / "bench.sqlite")
        started = time.perf_counter()
        for offset in range(0, records, batch_size):
            stop = min(records, offset + batch_size)
            batch = [
                {"content": f"Record {i}: {TOPICS[i % len(TOPICS)]}. Task {i}."}
                for i in range(offset, stop)
            ]
            ids = store.add_memories(batch)
            for i, memory_id in enumerate(ids, start=offset):
                angle = (i % 1000) * (2.0 * math.pi / 1000)
                store.set_embedding(
                    memory_id, [math.cos(angle), math.sin(angle)],
                    provider="benchmark", model="synthetic-2d",
                )
        ingest_seconds = time.perf_counter() - started

        for i in range(min(20, queries)):
            store.search(TOPICS[i % len(TOPICS)], limit=10)
            store.search_vector([1.0, 0.0], limit=vector_limit)

        lexical_samples: list[float] = []
        vector_samples: list[float] = []
        for i in range(queries):
            query_started = time.perf_counter()
            store.search(TOPICS[i % len(TOPICS)], limit=10)
            lexical_samples.append((time.perf_counter() - query_started) * 1000)

            query_started = time.perf_counter()
            store.search_vector(
                [1.0, 0.0], limit=vector_limit,
                provider="benchmark", model="synthetic-2d",
            )
            vector_samples.append((time.perf_counter() - query_started) * 1000)

        result = {
            "records": records,
            "queries": queries,
            "vector_limit": vector_limit,
            "backend": "sqlite+fts5" if store._fts else "sqlite-substring-fallback",
            "ingestion": {
                "seconds": round(ingest_seconds, 4),
                "records_per_second": round(records / ingest_seconds, 2),
                "batch_size": batch_size,
            },
            "latency_ms": {
                "lexical": latency_summary(lexical_samples),
                "exact_vector_top_k": latency_summary(vector_samples),
            },
            "methodology": (
                "single-process local SQLite; synthetic 2D vectors; warm-cache queries; "
                "exact vector scan with bounded top-K selection; not a competitor comparison"
            ),
        }
        store.close()
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--vector-limit", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(
        run(args.records, args.queries, args.batch_size, args.vector_limit),
        indent=2,
    )
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
