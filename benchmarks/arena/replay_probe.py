"""Re-run recall over already-ingested arena databases, in seconds.

Every other probe re-ingests and re-runs a system. This one reuses the SQLite
files the arena already built (``.arena/infinitydb/conv-*/conv-*.db``) and scores
recall with the arena's own scorer, so a ranking change can be measured before
it is worth a full run::

    python3 benchmarks/arena/replay_probe.py
    python3 benchmarks/arena/replay_probe.py --convs conv-26,conv-41 --dump /tmp/rows.json

It reports the same evidence-recall metric as ``score.py`` plus a per-category
breakdown, and it is the harness the two rejected ranking experiments in
``docs/COMPETITIVE_TEARDOWN.md`` were measured with.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parent
ROOT = ARENA_DIR.parents[1]
for p in (str(ROOT), str(ARENA_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from arena import locomo  # noqa: E402
from arena.score import evidence_hit  # noqa: E402
from infinitydb.core.engine import InfinityDB  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rebuild-edges", action="store_true", help="backfill typed edges from stored turns")
    ap.add_argument("--convs", default="", help="comma-separated conversation ids")
    ap.add_argument("--dense", default="", help="fastembed model name; enables lexical+dense RRF")
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--dump", default="", help="write per-question scores as JSON")
    args = ap.parse_args()

    convs = args.convs.split(",") if args.convs else sorted(
        p.name for p in (ROOT / ".arena" / "infinitydb").iterdir() if p.is_dir()
    )
    dataset = {c.sample_id: c for c in locomo.load()}
    buckets: dict[str, list[float]] = {}
    all_scores: list[float] = []
    rows: list[dict] = []
    for conv_id in convs:
        db_path = ROOT / ".arena" / "infinitydb" / conv_id / f"{conv_id}.db"
        if not db_path.exists():
            print(f"missing {db_path}: run the arena worker first", file=sys.stderr)
            continue
        conversation = dataset[conv_id]
        evidence_text = conversation.turn_text_by_dia()
        now = max((t.ts for t in conversation.all_turns()), default=0.0)
        db = InfinityDB(str(db_path))
        try:
            dense = None
            if args.dense:
                from fastembed import TextEmbedding
                _model = TextEmbedding(model_name=args.dense, cache_dir="/tmp/fastembed_cache")
                def embedder(texts, _m=_model):
                    return [list(map(float, v)) for v in _m.embed(list(texts))]
                db.index_dense(embedder, model=args.dense)
                dense = (embedder, args.dense, args.dense_weight)
            if args.rebuild_edges:
                db.rebuild_edges()
            for qa in conversation.qa:
                out = db.recall(
                    qa.question,
                    n=args.top_k,
                    user_id=conv_id,
                    now=now,
                    reinforce=False,
                    include_policies=False,
                    dense=dense,
                )
                retrieved = [item["fact"] for item in out["results"]]
                ev_texts = [evidence_text[e] for e in qa.evidence if e in evidence_text]
                score = evidence_hit(retrieved, ev_texts)
                if math.isnan(score):
                    continue
                rows.append({"conv": conv_id, "q": qa.question, "cat": qa.category_name, "s": score})
                buckets.setdefault(qa.category_name, []).append(score)
                all_scores.append(score)
        finally:
            db.close()

    if args.dump:
        Path(args.dump).write_text(json.dumps(rows), encoding="utf-8")
    if not all_scores:
        print("nothing scored", file=sys.stderr)
        return
    label = args.dense or "off"
    print(f"dense={label} n={len(all_scores)} overall={sum(all_scores)/len(all_scores):.4f}")
    for name, values in sorted(buckets.items()):
        print(f"  {name:28s} {sum(values)/len(values):.4f}  (n={len(values)})")


if __name__ == "__main__":
    main()
