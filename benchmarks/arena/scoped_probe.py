"""Scoped-recall probe: does one database stay as accurate as ten databases?

Every arena row gives each conversation its own database, which is the friendliest
possible setting and not how a real agent runs: one store accumulates many chats,
projects and people. This probe measures that directly with the same scorer the
arena uses.

Three conditions over the same questions, `top_k=10`, no reader LLM:

    isolated  one database per conversation (reproduces the arena number)
    pooled    one database holding all conversations, unscoped recall
    scoped    the same pooled database, recalled with session_id=<conversation>

The gap between `pooled` and `isolated` is what cross-conversation contamination
costs. The gap between `scoped` and `pooled` is what scoping buys back.

Usage:
    python arena/scoped_probe.py
    python arena/scoped_probe.py --convs conv-26 conv-30
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import locomo  # noqa: E402
from score import answer_in_context, evidence_hit  # noqa: E402

from infinitydb.core.engine import InfinityDB  # noqa: E402

DEFAULT_CONVS = [
    "conv-26",
    "conv-30",
    "conv-41",
    "conv-42",
    "conv-43",
    "conv-44",
    "conv-47",
    "conv-48",
    "conv-49",
    "conv-50",
]


def render(item: dict) -> str:
    day = datetime.fromtimestamp(item["ts"]).strftime("%d %B %Y")
    return f"[{day}] {item['fact']}"


def build(convs: list[locomo.Conversation], path: Path | None) -> InfinityDB:
    """Ingest conversations, optionally all into one store, keyed by session."""
    if path is None:
        db = InfinityDB()
    else:
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(path) + suffix).unlink()
            except FileNotFoundError:
                pass
        db = InfinityDB(str(path))
    for conv in convs:
        for session in conv.sessions:
            for turn in session.turns:
                db.add(
                    f"{turn.speaker}: {turn.text}",
                    user_id="pooled" if path is not None else conv.sample_id,
                    session_id=conv.sample_id if path is not None else None,
                    ts=turn.ts,
                    auto_supersede=False,
                )
    return db


def evaluate(
    db: InfinityDB,
    conv: locomo.Conversation,
    *,
    user_id: str,
    session_id: str | None,
) -> dict:
    evidence_text = conv.turn_text_by_dia()
    now = max((t.ts for t in conv.all_turns()), default=0.0)
    recalls: list[float] = []
    in_ctx: list[float] = []
    latencies: list[float] = []
    tokens: list[int] = []
    for qa in conv.qa:
        started = time.perf_counter()
        result = db.recall(
            qa.question,
            n=10,
            user_id=user_id,
            now=now,
            reinforce=False,
            include_policies=False,
            session_id=session_id,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        retrieved = [render(item) for item in result["results"]]
        tokens.append(sum(len(chunk.split()) for chunk in retrieved))
        gold_evidence = [evidence_text.get(d, "") for d in qa.evidence]
        gold_evidence = [text for text in gold_evidence if text]
        # Matches score.py: adversarial questions are scored separately, never
        # folded into the recall / answer-in-context means.
        if qa.is_adversarial:
            continue
        if gold_evidence:
            recalls.append(evidence_hit(retrieved, gold_evidence))
        in_ctx.append(1.0 if answer_in_context(retrieved, qa.answer) else 0.0)
    return {
        "recall": statistics.fmean(recalls) if recalls else 0.0,
        "in_ctx": statistics.fmean(in_ctx) if in_ctx else 0.0,
        "p50_ms": statistics.median(latencies) if latencies else 0.0,
        "words": statistics.median(tokens) if tokens else 0,
        "n": len(conv.qa),
    }


def mean(rows: list[dict]) -> dict:
    total = sum(r["n"] for r in rows) or 1
    return {
        "recall": sum(r["recall"] * r["n"] for r in rows) / total,
        "in_ctx": sum(r["in_ctx"] * r["n"] for r in rows) / total,
        "p50_ms": statistics.median([r["p50_ms"] for r in rows]),
        "words": statistics.median([r["words"] for r in rows]),
        "n": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convs", nargs="*", default=DEFAULT_CONVS)
    parser.add_argument("--out", default=str(HERE / "results" / "scores" / "scoped_probe.json"))
    args = parser.parse_args()

    conversations = locomo.load()
    selected = [locomo.by_id(conversations, cid) for cid in args.convs]

    print("ingesting…", flush=True)
    started = time.perf_counter()
    isolated_db = None  # built per conversation below
    pooled_db = build(selected, HERE / "results" / "scoped_probe.db")
    print(f"pooled ingest {time.perf_counter() - started:.1f}s", flush=True)

    isolated_rows: list[dict] = []
    for conv in selected:
        db = build([conv], None)
        try:
            measured = evaluate(db, conv, user_id=conv.sample_id, session_id=None)
        finally:
            db.close()
        isolated_rows.append(measured)
        print(f"  isolated {conv.sample_id}: recall {measured['recall']:.4f}", flush=True)

    pooled_rows = [evaluate(pooled_db, conv, user_id="pooled", session_id=None) for conv in selected]
    scoped_rows = [
        evaluate(pooled_db, conv, user_id="pooled", session_id=conv.sample_id)
        for conv in selected
    ]

    summary = {
        "isolated": mean(isolated_rows),
        "pooled": mean(pooled_rows),
        "scoped": mean(scoped_rows),
        "conversations": args.convs,
        "top_k": 10,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"{'condition':<10} {'evRec@10':>9} {'inCtx':>7} {'words':>6} {'p50':>8}")
    for name in ("isolated", "pooled", "scoped"):
        row = summary[name]
        print(
            f"{name:<10} {row['recall']:>9.4f} {row['in_ctx']:>7.4f} "
            f"{row['words']:>6} {row['p50_ms']:>7.1f}ms"
        )
    lost = summary["isolated"]["recall"] - summary["pooled"]["recall"]
    recovered = summary["scoped"]["recall"] - summary["pooled"]["recall"]
    share = (recovered / lost) if lost > 0 else 0.0
    print(
        f"\ncontamination cost {lost:+.4f} recall; scoping recovers {recovered:+.4f} "
        f"({share:.0%} of the loss)"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
