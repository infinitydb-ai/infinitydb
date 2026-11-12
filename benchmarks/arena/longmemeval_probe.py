"""LongMemEval retrieval probe: session-level recall@k, no reader LLM.

Each question ships its own ~54-session haystack (~550 turns, ~120k tokens), so
every question gets a fresh database: the friendliest setting, identical to how
the arena runs LoCoMo, which keeps the number comparable to published claims.

A hit means at least one returned memory belongs to an annotated answer session.

    python arena/longmemeval_probe.py --path data/longmemeval_s.json --max-seconds 240
    python arena/longmemeval_probe.py --limit 50 --top-k 5

Rows are appended to ``results/longmemeval/<system>.jsonl``; re-running resumes
where it stopped, so the full 500 questions can be covered in a few short calls.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import longmemeval  # noqa: E402

from infinitydb.core.engine import InfinityDB  # noqa: E402


def answer_session_of(db: InfinityDB, source_id: str) -> str:
    row = db.conn.execute(
        "SELECT session_id FROM l0_logs WHERE id = ?", (source_id,)
    ).fetchone()
    return row["session_id"] if row else ""


def run_one(entry: longmemeval.Entry, top_k: int) -> dict:
    db = InfinityDB()
    try:
        started = time.perf_counter()
        for session_id, ts, role, content in entry.turns():
            speaker = "User" if role == "user" else "Assistant"
            db.add(
                f"{speaker}: {content}",
                user_id="lme",
                session_id=session_id,
                ts=ts or None,
                auto_supersede=False,
            )
        ingest = time.perf_counter() - started

        started = time.perf_counter()
        result = db.recall(
            entry.question,
            n=top_k,
            user_id="lme",
            now=entry.question_ts or None,
            reinforce=False,
            include_policies=False,
        )
        latency = (time.perf_counter() - started) * 1000.0
        sessions = [
            answer_session_of(db, item["source_id"]) for item in result["results"]
        ]
    finally:
        db.close()

    return {
        "question_id": entry.question_id,
        "question_type": entry.question_type,
        "recall_at_k": 1.0
        if set(sessions) & set(entry.answer_session_ids)
        else 0.0,
        "sessions_returned": sessions,
        "answer_sessions": entry.answer_session_ids,
        "ingest_seconds": round(ingest, 3),
        "latency_ms": round(latency, 2),
        "context_words": sum(
            len(item["fact"].split()) for item in result["results"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(longmemeval.DATA_PATH))
    parser.add_argument("--system", default="infinitydb")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N new questions.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Stratified subsample: N questions per question_type (evenly spaced).",
    )
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--types", nargs="*", default=None)
    args = parser.parse_args()

    entries = longmemeval.load(args.path)
    if args.types:
        entries = [e for e in entries if e.question_type in args.types]
    if args.sample:
        by_type: dict[str, list[longmemeval.Entry]] = defaultdict(list)
        for entry in entries:
            by_type[entry.question_type].append(entry)
        picked: list[longmemeval.Entry] = []
        for name in sorted(by_type):
            group = by_type[name]
            step = max(1, len(group) // args.sample)
            picked.extend(group[::step][: args.sample])
        entries = picked

    out_dir = HERE / "results" / "longmemeval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.system}-k{args.top_k}.jsonl"
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["question_id"])

    pending = [e for e in entries if e.question_id not in done]
    started = time.time()
    processed = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for entry in pending:
            row = run_one(entry, args.top_k)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            processed += 1
            if args.limit and processed >= args.limit:
                break
            if args.max_seconds and time.time() - started > args.max_seconds:
                break

    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row["recall_at_k"])
    overall = statistics.fmean(r["recall_at_k"] for r in rows) if rows else 0.0
    print(f"processed {processed} new question(s); {len(rows)}/{len(entries)} total")
    print(f"R@{args.top_k} overall {overall:.4f}")
    for name, values in sorted(by_type.items()):
        print(f"  {name:<26} {statistics.fmean(values):.4f}  (n={len(values)})")
    if rows:
        print(
            f"ingest median {statistics.median(r['ingest_seconds'] for r in rows):.1f}s, "
            f"recall p50 {statistics.median(r['latency_ms'] for r in rows):.1f}ms, "
            f"context words median {statistics.median(r['context_words'] for r in rows):.0f}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
