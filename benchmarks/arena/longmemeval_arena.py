"""Comparable retrieval-only LongMemEval-S arena for local memory systems.

Runs each question as an isolated memory bank, matching the dataset's protocol.
No reader LLM or hosted API is called. Outputs are append-only JSONL and resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCHMARKS))

from arena import longmemeval  # noqa: E402
from arena.locomo import Session, Turn  # noqa: E402
from arena.score import answer_in_context  # noqa: E402
from arena.systems.infinitydb_system import InfinityDBSystem  # noqa: E402
from arena.systems.mem0_system import Mem0System  # noqa: E402
from arena.systems.base import turn_line  # noqa: E402

SYSTEMS = {"infinitydb", "mem0", "hybrid"}


def get_system_class(name: str):
    if name == "infinitydb":
        return InfinityDBSystem
    if name == "mem0":
        return Mem0System
    if name == "hybrid":
        # numpy/FastEmbed are optional and are installed only in the embedding venv.
        from arena.systems.hybrid_system import HybridSystem
        return HybridSystem
    raise ValueError(f"unsupported system: {name}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_entry(entry: longmemeval.Entry) -> tuple[list[Session], dict[str, set[str]]]:
    """Convert LongMemEval's sessions to the arena's shared adapter contract."""
    sessions: list[Session] = []
    line_to_sessions: dict[str, set[str]] = defaultdict(set)
    for index, (session_id, raw_turns, ts) in enumerate(
        zip(entry.session_ids, entry.sessions, entry.session_ts)
    ):
        turns: list[Turn] = []
        for turn_index, raw in enumerate(raw_turns):
            role = str(raw.get("role", "user")).lower()
            speaker = "Assistant" if role == "assistant" else "User"
            content = str(raw.get("content", ""))
            item = Turn(
                speaker=speaker,
                text=content,
                dia_id=f"{session_id}:{turn_index}",
                ts=ts,
            )
            turns.append(item)
            line_to_sessions[turn_line(item)].add(session_id)
        sessions.append(Session(index=index, ts=ts, turns=turns))
    return sessions, line_to_sessions


def resolve_session_ids(
    retrieved: list[str], line_to_sessions: dict[str, set[str]]
) -> list[str]:
    """Map adapter-returned verbatim turns back to their source sessions."""
    found: set[str] = set()
    for chunk in retrieved:
        for line, session_ids in line_to_sessions.items():
            if line and line in chunk:
                found.update(session_ids)
    return sorted(found)


def run_system(
    system_name: str,
    entries: list[longmemeval.Entry],
    dataset_hash: str,
    output_dir: Path,
    top_k: int,
    limit: int,
    force: bool,
) -> dict[str, Any]:
    adapter_cls = get_system_class(system_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{system_name}-{dataset_hash[:12]}-k{top_k}.jsonl"
    if force and out_path.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = output_dir / f"{system_name}-{dataset_hash[:12]}-k{top_k}-rerun-{stamp}.jsonl"
    done: set[str] = set()
    if out_path.exists() and not force:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("type") == "result" and row.get("dataset_sha256") == dataset_hash:
                    done.add(row["question_id"])
    pending = [entry for entry in entries if entry.question_id not in done]
    if limit > 0:
        pending = pending[:limit]

    print(
        f"\n[{system_name}] {len(pending)} pending / {len(entries)} questions; "
        f"top_k={top_k}; output={out_path}",
        flush=True,
    )
    started_all = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as handle:
        for number, entry in enumerate(pending, start=1):
            sessions, line_map = prepare_entry(entry)
            with tempfile.TemporaryDirectory(
                prefix=f"lme-{system_name}-"
            ) as workdir:
                adapter = adapter_cls(workdir, entry.question_id)
                try:
                    adapter.reset()
                    ingest_started = time.perf_counter()
                    for session in sessions:
                        adapter.ingest(session)
                    ingest_seconds = time.perf_counter() - ingest_started

                    query_started = time.perf_counter()
                    retrieved = adapter.search(entry.question, top_k)
                    query_ms = (time.perf_counter() - query_started) * 1000.0
                finally:
                    adapter.close()

            retrieved_sessions = resolve_session_ids(retrieved, line_map)
            answerable = bool(entry.answer_session_ids) and not entry.question_id.endswith("_abs")
            hit = bool(set(retrieved_sessions) & set(entry.answer_session_ids))
            row = {
                "type": "result",
                "question_id": entry.question_id,
                "question_type": entry.question_type,
                "system": system_name,
                "dataset_sha256": dataset_hash,
                "top_k": top_k,
                "answerable": answerable,
                "recall_at_k": (1.0 if hit else 0.0) if answerable else None,
                "hit_any_all_questions": hit,
                "abstention_empty_retrieval": (
                    not retrieved if not answerable else None
                ),
                "answer_in_context": answer_in_context(retrieved, entry.answer),
                "retrieved_session_ids": retrieved_sessions,
                "answer_session_ids": entry.answer_session_ids,
                "ingest_seconds": round(ingest_seconds, 4),
                "latency_ms": round(query_ms, 3),
                "context_words": sum(len(chunk.split()) for chunk in retrieved),
                "retrieved_count": len(retrieved),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if number % 10 == 0 or number == len(pending):
                print(
                    f"  {number}/{len(pending)} processed "
                    f"({time.perf_counter() - started_all:.1f}s)",
                    flush=True,
                )

    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [
        row for row in rows
        if row.get("type") == "result" and row.get("dataset_sha256") == dataset_hash
    ]
    answerable_rows = [row for row in rows if row["answerable"]]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)
    summary = {
        "system": system_name,
        "dataset": "LongMemEval-S cleaned",
        "dataset_sha256": dataset_hash,
        "top_k": top_k,
        "questions_completed": len(rows),
        "answerable_questions": len(answerable_rows),
        "session_recall_at_k_answerable": (
            statistics.fmean(row["recall_at_k"] for row in answerable_rows)
            if answerable_rows else None
        ),
        "hit_any_rate_all_questions": (
            statistics.fmean(float(row["hit_any_all_questions"]) for row in rows)
            if rows else None
        ),
        "answer_in_context_rate": (
            statistics.fmean(float(row["answer_in_context"]) for row in rows)
            if rows else None
        ),
        "abstention_empty_retrieval_rate": (
            statistics.fmean(
                float(row["abstention_empty_retrieval"])
                for row in rows if row["abstention_empty_retrieval"] is not None
            )
            if any(row["abstention_empty_retrieval"] is not None for row in rows)
            else None
        ),
        "query_latency_ms_p50": (
            statistics.median(row["latency_ms"] for row in rows) if rows else None
        ),
        "query_latency_ms_p95": (
            sorted(row["latency_ms"] for row in rows)[
                max(0, math.ceil(0.95 * len(rows)) - 1)
            ] if rows else None
        ),
        "ingest_seconds_median_per_question": (
            statistics.median(row["ingest_seconds"] for row in rows) if rows else None
        ),
        "context_words_median": (
            statistics.median(row["context_words"] for row in rows) if rows else None
        ),
        "by_question_type": {
            name: {
                "n": len(group),
                "answerable_n": sum(1 for row in group if row["answerable"]),
                "session_recall_at_k_answerable": (
                    statistics.fmean(
                        row["recall_at_k"] for row in group if row["answerable"]
                    )
                    if any(row["answerable"] for row in group) else None
                ),
                "answer_in_context_rate": statistics.fmean(
                    float(row["answer_in_context"]) for row in group
                ),
            }
            for name, group in sorted(by_type.items())
        },
        "raw_results": str(out_path),
    }
    summary_path = output_dir / f"{system_name}-{dataset_hash[:12]}-k{top_k}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="LongMemEval-S cleaned JSON path")
    parser.add_argument(
        "--systems", default="infinitydb,mem0,hybrid",
        help="Comma-separated systems; each system is run sequentially",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Run only N pending questions per system")
    parser.add_argument("--output-dir", default=str(ROOT / "benchmarks" / "arena" / "results" / "longmemeval"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    path = Path(args.path)
    if not path.is_file():
        parser.error(f"dataset does not exist: {path}")
    dataset_hash = sha256_file(path)
    entries = longmemeval.load(path)
    selected = [name.strip() for name in args.systems.split(",") if name.strip()]
    unknown = sorted(set(selected) - SYSTEMS)
    if unknown:
        parser.error(f"unsupported systems: {', '.join(unknown)}; choose from {', '.join(SYSTEMS)}")
    if not selected:
        parser.error("select at least one system")
    print(
        f"LongMemEval-S: {len(entries)} questions; SHA-256 {dataset_hash}; "
        f"systems={selected}",
        flush=True,
    )
    summaries = []
    for name in selected:
        summaries.append(
            run_system(
                name, entries, dataset_hash, Path(args.output_dir),
                args.top_k, args.limit, args.force,
            )
        )
    comparison = {
        "benchmark": "InfinityDB Benchmark Arena v2 — LongMemEval-S",
        "dataset_sha256": dataset_hash,
        "dataset_questions": len(entries),
        "systems": summaries,
        "metric_note": (
            "Retrieval-only. Session recall is computed over answerable questions; "
            "abstention questions are reported separately. No reader LLM is used."
        ),
    }
    comparison_path = Path(args.output_dir) / f"comparison-{dataset_hash[:12]}-k{args.top_k}.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"\nComparison written: {comparison_path}")


if __name__ == "__main__":
    main()
