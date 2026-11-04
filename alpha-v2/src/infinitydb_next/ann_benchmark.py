from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import tempfile
import time
from pathlib import Path

from .store import MemoryStore


def _unit_vector(rng: random.Random, dimensions: int) -> list[float]:
    values = [rng.gauss(0.0, 1.0) for _ in range(dimensions)]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "p50": round(_percentile(samples, 0.50), 3),
        "p95": round(_percentile(samples, 0.95), 3),
        "p99": round(_percentile(samples, 0.99), 3),
        "mean": round(statistics.mean(samples), 3),
    }


def run(
    records: int = 3000,
    queries: int = 100,
    dimensions: int = 64,
    limit: int = 10,
    seed: int = 42,
) -> dict[str, object]:
    if min(records, queries, dimensions, limit) < 1:
        raise ValueError("records, queries, dimensions and limit must be positive")
    if limit > records:
        raise ValueError("limit cannot exceed records")

    try:
        import hnswlib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install HNSW support with pip install -e '.[ann]'") from exc

    rng = random.Random(seed)
    vectors = [_unit_vector(rng, dimensions) for _ in range(records)]
    query_vectors = [vectors[(index * 37) % records] for index in range(queries)]

    with tempfile.TemporaryDirectory() as directory:
        store = MemoryStore(Path(directory) / "ann-benchmark.sqlite")
        started = time.perf_counter()
        ids = store.add_memories([
            {"content": f"Synthetic benchmark memory {index}", "tenant_id": "bench"}
            for index in range(records)
        ])
        for memory_id, vector in zip(ids, vectors):
            store.set_embedding(
                memory_id, vector, tenant_id="bench",
                provider="synthetic", model=f"random-{dimensions}d",
            )
        ingestion_ms = (time.perf_counter() - started) * 1000

        exact_ms: list[float] = []
        ann_ms: list[float] = []
        recalls: list[float] = []
        build_and_first_query_ms = 0.0

        for index, query in enumerate(query_vectors):
            started = time.perf_counter()
            exact = store.search_vector(
                query, tenant_id="bench", provider="synthetic",
                model=f"random-{dimensions}d", limit=limit,
            )
            exact_ms.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            approximate = store.search_vector_ann(
                query, tenant_id="bench", provider="synthetic",
                model=f"random-{dimensions}d", limit=limit,
            )
            elapsed = (time.perf_counter() - started) * 1000
            if index == 0:
                build_and_first_query_ms = round(elapsed, 3)
            else:
                ann_ms.append(elapsed)
            exact_ids = {hit.memory.id for hit in exact}
            approximate_ids = {hit.memory.id for hit in approximate}
            recalls.append(
                len(exact_ids & approximate_ids) / max(1, len(exact_ids))
            )

        result: dict[str, object] = {
            "records": records,
            "queries": queries,
            "dimensions": dimensions,
            "k": limit,
            "seed": seed,
            "tenant_count": 1,
            "embedding_model": f"synthetic/random-{dimensions}d",
            "ingestion_ms": round(ingestion_ms, 3),
            "ann_index_build_plus_first_query_ms": build_and_first_query_ms,
            "exact_query_latency_ms": _summary(exact_ms),
            "ann_warm_query_latency_ms": _summary(ann_ms) if ann_ms else {},
            "mean_recall_at_k": round(statistics.mean(recalls), 5),
            "recall_at_k_samples": len(recalls),
            "methodology": (
                "single-process local SQLite and HNSW; deterministic synthetic unit vectors; "
                "exact SQLite cosine search is the reference; first ANN query includes lazy "
                "index construction; subsequent ANN queries are warm; not a competitor comparison"
            ),
        }
        store.close()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare InfinityDB exact and HNSW retrieval.")
    parser.add_argument("--records", type=int, default=3000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(
        run(args.records, args.queries, args.dimensions, args.limit, args.seed),
        indent=2,
    )
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
