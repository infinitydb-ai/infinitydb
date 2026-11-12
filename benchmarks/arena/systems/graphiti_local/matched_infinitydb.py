"""Same slice, same questions, InfinityDB side of the matched comparison.

Ingests exactly the sessions and answers exactly the questions that the
graphiti_local harness uses, so the two rows differ only in the memory system.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from arena import locomo  # noqa: E402

def field(obj, name):
    """Read a LoCoMo field whether it is a dataclass or a mapping."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)

from infinitydb.core.engine import InfinityDB  # noqa: E402

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

CONV, SESSIONS, QUESTIONS = "conv-26", 1, 12

conv = locomo.by_id(locomo.load(), CONV)
sessions = conv.sessions[:SESSIONS]
qa = list(conv.qa)[:QUESTIONS]

db = InfinityDB(".arena/graphiti/matched_infinitydb.db")
now = 0.0
start = time.time()
for session in sessions:
    for turn in session.turns:
        db.add(f"{turn.speaker}: {turn.text}", user_id=CONV, session_id=CONV,
               ts=turn.ts, auto_supersede=False)
        now = max(now, turn.ts)
ingest = time.time() - start
print(f"infinitydb ingested {sum(len(s.turns) for s in sessions)} turns in {ingest:.2f}s")

out = Path("benchmarks/arena/results/infinitydb-slice/conv-26.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "type": "meta", "system": "infinitydb", "conv": CONV,
        "sessions": len(sessions), "turns": sum(len(s.turns) for s in sessions),
        "qa": len(qa), "top_k": 10, "partial": True,
    }) + "\n")
    for index, question in enumerate(qa, 1):
        begin = time.time()
        result = db.recall(field(question, "question"), n=10, user_id=CONV, session_id=CONV,
                           now=now, reinforce=False)
        handle.write(json.dumps({
            "type": "qa", "qa_id": f"{CONV}-q{index:03d}",
            "question": field(question, "question"), "answer": field(question, "answer"),
            "category": field(question, "category"),
                "category_name": CATEGORY_NAMES.get(int(field(question, "category") or 0), "unknown"),
                "evidence": field(question, "evidence"),
            "retrieved": [item["fact"] for item in result["results"]],
            "ms": round((time.time() - begin) * 1000, 2), "error": None,
        }) + "\n")
    handle.write(json.dumps({"type": "done"}) + "\n")
print(f"wrote {out}")
