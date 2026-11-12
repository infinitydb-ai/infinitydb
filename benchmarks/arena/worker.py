"""Run one memory system over one LoCoMo conversation and record raw results.

The worker is deliberately self-contained so it can be launched with a different
interpreter (one venv per pinned competitor) while the arena stays in charge of
configuration, caching and scoring::

    python benchmarks/arena/worker.py --system infinitydb --conv conv-26 \
        --out results/infinitydb/conv-26.jsonl --workdir .arena/infinitydb/conv-26

Output is JSON lines: one ``meta`` row, one ``session`` row per ingested session,
one ``qa`` row per question, and a final ``done`` row. Raw rows are kept so any
scoring can be recomputed later without re-running the systems.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parent
ROOT = ARENA_DIR.parents[1]
for p in (str(ROOT), str(ARENA_DIR.parents[0])):
    if p not in sys.path:
        sys.path.insert(0, p)

from arena import locomo  # noqa: E402

SYSTEMS = ("infinitydb", "mem0", "hybrid", "graphiti", "tencentdb")


def build_system(name: str, workdir: str, conversation_id: str, options: dict):
    if name == "infinitydb":
        from arena.systems.infinitydb_system import InfinityDBSystem
        return InfinityDBSystem(workdir, conversation_id, **options)
    if name == "mem0":
        from arena.systems.mem0_system import Mem0System
        return Mem0System(workdir, conversation_id, **options)
    if name == "hybrid":
        from arena.systems.hybrid_system import HybridSystem
        return HybridSystem(workdir, conversation_id, **options)
    if name in ("graphiti", "tencentdb"):
        # Both are cloud/graph services: Graphiti (Zep's open-source engine)
        # needs an LLM key plus a graph database, and TencentDB's memory service
        # needs Tencent Cloud credentials. Neither can be run offline, so the
        # adapter is deliberately absent rather than silently emulated.
        raise SystemExit(
            f"the {name} adapter is not implemented: it needs credentials or a "
            "graph database this harness cannot invent. Run "
            "`python3 benchmarks/arena/competitor_check.py` for the exact missing "
            "prerequisite, and see docs/COMPETITOR_BENCHMARK_PLAYBOOK.md. "
            "No adapter ships before it has produced a real row."
        )
    raise SystemExit(f"unknown system {name!r}; choose from {SYSTEMS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--conv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--dataset", default=str(locomo.DATA_PATH))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-qa", type=int, default=0, help="0 = all questions")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(r.get("type") == "done" for r in rows):
            print(f"[{args.system}/{args.conv}] already complete: {out}")
            return

    conversation = locomo.by_id(locomo.load(args.dataset), args.conv)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    qa_items = conversation.qa[: args.max_qa] if args.max_qa else conversation.qa

    system = build_system(args.system, str(workdir), conversation.sample_id, {})
    system.reset()
    with out.open("w", encoding="utf-8") as fh:
        def emit(row: dict) -> None:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

        emit({
            "type": "meta",
            "system": args.system,
            "conv": conversation.sample_id,
            "sessions": len(conversation.sessions),
            "turns": conversation.turn_count,
            "qa": len(qa_items),
            "top_k": args.top_k,
            "python": sys.version.split()[0],
            "started": time.time(),
        })

        ingest_started = time.perf_counter()
        for session in conversation.sessions:
            t0 = time.perf_counter()
            system.ingest(session)
            emit({
                "type": "session",
                "session": session.index,
                "turns": len(session.turns),
                "seconds": round(time.perf_counter() - t0, 3),
            })
        emit({
            "type": "ingest_done",
            "seconds": round(time.perf_counter() - ingest_started, 3),
        })
        print(f"[{args.system}/{args.conv}] ingest {time.perf_counter() - ingest_started:.1f}s", flush=True)

        for n, qa in enumerate(qa_items, 1):
            t0 = time.perf_counter()
            try:
                retrieved = system.search(qa.question, args.top_k)
                error = None
            except Exception as exc:  # keep going, record the failure
                retrieved, error = [], f"{type(exc).__name__}: {exc}"
            ms = (time.perf_counter() - t0) * 1000.0
            emit({
                "type": "qa",
                "qa_id": qa.qa_id,
                "question": qa.question,
                "answer": qa.answer,
                "category": qa.category,
                "category_name": qa.category_name,
                "evidence": qa.evidence,
                "retrieved": retrieved,
                "ms": round(ms, 2),
                "error": error,
            })
            if n % 25 == 0:
                print(f"[{args.system}/{args.conv}] {n}/{len(qa_items)} queries", flush=True)

        emit({"type": "done", "finished": time.time()})
    system.close()
    print(f"[{args.system}/{args.conv}] wrote {out}")


if __name__ == "__main__":
    main()
