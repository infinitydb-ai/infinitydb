#!/usr/bin/env python3
"""Phase 2 scaffold: measure bridge selection against the multi-hop baseline.

The baseline to beat is a published number: **multi-hop evidence recall@10 =
0.4939** on the LoCoMo arena (`dense=off`, all 1,986 questions, `top_k=10`).

This harness measures one conversation at a time so it stays runnable in a
minute, and reports:

* `baseline` — recall@10 with the default path, no bridge
* `scoped`  — recall@10 with the selector's bridge objects passed as
  `entities=[...]`, which restricts and re-statistics the candidate set
* how many multi-hop questions produced any bridge at all

It is a scaffold, not a result: with the default `NullBridge` the two arms are
identical by construction, and that is the control that proves the harness
measures the selector rather than an unrelated change. Nothing here is wired
into the engine's default path.

Run:
    python benchmarks/arena/bridge_probe.py --convs conv-26
    python benchmarks/arena/bridge_probe.py --selector predicate --convs conv-26 conv-30
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parent
ROOT = ARENA_DIR.parents[1]
for path in (str(ROOT), str(ARENA_DIR.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from arena import locomo, score  # noqa: E402
from arena.systems.base import turn_line  # noqa: E402
from infinitydb.bridge import AllBridge, Bridge, NullBridge, PredicateBridge  # noqa: E402
from infinitydb.core.engine import InfinityDB  # noqa: E402

SELECTORS = {"null": NullBridge, "all": AllBridge, "predicate": PredicateBridge}
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def ingest(convo) -> tuple[InfinityDB, float]:
    """Same configuration as the arena adapter: one user per conversation,
    per-turn writes with real event times, supersession off for two speakers."""
    db = InfinityDB(":memory:")
    now = 0.0
    for session in convo.sessions:
        for turn in session.turns:
            db.add(turn_line(turn), user_id=convo.sample_id, ts=turn.ts, auto_supersede=False)
            now = max(now, turn.ts)
    return db, now


def entities_in_question(db: InfinityDB, question: str, user_id: str) -> list[str]:
    """Capitalised words in the question that are actually edge subjects here."""
    # "Caroline's" must resolve to the entity "Caroline".
    words = {re.sub(r"'s$", "", w) for w in WORD.findall(question) if w[:1].isupper()}
    if not words:
        return []
    subjects = {row["subject"] for row in db.conn.execute(
        "SELECT DISTINCT subject FROM l1_entity_edges WHERE user_id = ?", (user_id,))}
    return sorted(w for w in words if w.lower() in subjects)


def candidates_for(db: InfinityDB, entities: list[str], user_id: str, hops: int) -> list[Bridge]:
    out: list[Bridge] = []
    seen: set[tuple[str, str, str]] = set()
    for entity in entities:
        for relation in db.relations(entity, user_id=user_id, hops=hops):
            key = (relation["subject"], relation["predicate"], relation["object"])
            if key in seen:
                continue
            seen.add(key)
            out.append(Bridge(
                subject=relation["subject"],
                predicate=relation["predicate"],
                object=relation["object"],
                atom_id=relation.get("atom_id", ""),
                hops=relation.get("hop", 1),
            ))
    return out


def render(result: dict) -> list[str]:
    """Same rendering as the arena adapter: the event date, then the fact.

    Without the date a "when" question is decided by formatting rather than
    retrieval, and the number stops being comparable to the published table.
    """
    out = []
    for item in result.get("results", []):
        day = datetime.fromtimestamp(item["ts"]).strftime("%d %B %Y")
        out.append(f"[{day}] {item['fact']}")
    return out


def witness_facts(db: InfinityDB, bridges: list[Bridge], top_k: int) -> list[str]:
    """The turns that asserted each edge, rendered like the adapter does.

    This is the second hypothesis: a bridge is useful because a sentence exists
    that says it, so quote that sentence rather than scoping retrieval by the
    object string (which turned out to halve multi-hop recall).
    """
    out: list[str] = []
    seen: set[str] = set()
    for bridge in bridges:
        if not bridge.atom_id or bridge.atom_id in seen:
            continue
        seen.add(bridge.atom_id)
        row = db.conn.execute(
            "SELECT fact, ts FROM l1_atoms WHERE id = ?", (bridge.atom_id,)
        ).fetchone()
        if row is None:
            continue
        day = datetime.fromtimestamp(row["ts"]).strftime("%d %B %Y")
        out.append(f"[{day}] {row['fact']}")
        if len(out) >= top_k - 1:
            break
    return out


def recall_hit(result: dict, ev_texts: list[str]) -> float:
    """Fraction of evidence turns found, matching score.py rather than a
    binary any-hit: multi-hop questions carry several evidence turns, and
    crediting a partial answer as a full one inflates the baseline."""
    facts = render(result)
    if not ev_texts or not facts:
        return 0.0
    value = score.evidence_hit(facts, ev_texts)
    return 0.0 if value != value else value  # nan -> 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", nargs="+", default=[],
                    help="conversation ids (default: every conversation in the dataset)")
    ap.add_argument("--selector", default="null", choices=sorted(SELECTORS))
    ap.add_argument("--mode", default="scope", choices=("scope", "witness"),
                    help="scope: pass bridge objects as entity filters. "
                         "witness: quote the turns that asserted the edges instead.")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max-bridges", type=int, default=0,
                    help="override the selector's cap; each quoted witness displaces one recall slot")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    selector = SELECTORS[args.selector]()
    if args.max_bridges and hasattr(selector, "max_bridges"):
        selector.max_bridges = args.max_bridges
    convo_map = {c.sample_id: c for c in locomo.load()}
    report: dict[str, object] = {
        "selector": selector.name,
        "mode": args.mode,
        "hops": args.hops,
        "top_k": args.top_k,
        "published_multi_hop_baseline": 0.4939,
        "conversations": {},
    }

    all_base: list[float] = []
    all_scoped: list[float] = []
    # Metrics restricted to questions the selector actually fired on, because an
    # overall delta diluted by hundreds of untouched questions says nothing
    # about whether the mechanism works where it applies.
    fired_base: list[float] = []
    fired_scoped: list[float] = []

    for conv_id in (args.convs or sorted(convo_map)):
        convo = convo_map[conv_id]
        db, now = ingest(convo)
        questions = [qa for qa in convo.qa if qa.category_name == "multi-hop"]
        by_dia = convo.turn_text_by_dia()
        base_hits: list[float] = []
        scoped_hits: list[float] = []
        bridge_count = 0
        for qa in questions:
            ev = [by_dia[d] for d in qa.evidence if d in by_dia]
            if not ev:
                continue
            baseline = db.recall(qa.question, n=args.top_k, user_id=conv_id, now=now,
                                 reinforce=False, include_policies=False)
            base_hits.append(recall_hit(baseline, ev))

            entities = entities_in_question(db, qa.question, conv_id)
            bridges = selector.select(qa.question, candidates_for(db, entities, conv_id, args.hops))
            if not bridges:
                scoped_hits.append(recall_hit(baseline, ev))
                continue
            bridge_count += 1
            if args.mode == "witness":
                witnesses = witness_facts(db, bridges, args.top_k)
                keep = max(1, args.top_k - len(witnesses))
                base_result = db.recall(qa.question, n=keep, user_id=conv_id, now=now,
                                        reinforce=False, include_policies=False)
                facts = render(base_result)[:keep] + witnesses
                value = score.evidence_hit(facts, ev)
                scoped_hits.append(0.0 if value != value else value)
            else:
                scoped = db.recall(qa.question, n=args.top_k, user_id=conv_id, now=now,
                                   reinforce=False, include_policies=False,
                                   entities=[b.object for b in bridges])
                scoped_hits.append(recall_hit(scoped, ev))
            fired_base.append(base_hits[-1])
            fired_scoped.append(scoped_hits[-1])

        base = round(sum(base_hits) / len(base_hits), 4) if base_hits else float("nan")
        scoped = round(sum(scoped_hits) / len(scoped_hits), 4) if scoped_hits else float("nan")
        print(f"{conv_id}: multi-hop n={len(base_hits)} baseline={base} scoped={scoped} "
              f"questions_with_bridges={bridge_count}")
        report["conversations"][conv_id] = {
            "questions": len(base_hits),
            "baseline_recall@10": base,
            "scoped_recall@10": scoped,
            "questions_with_bridges": bridge_count,
        }
        all_base.extend(base_hits)
        all_scoped.extend(scoped_hits)
        db.close()

    overall_base = round(sum(all_base) / len(all_base), 4) if all_base else float("nan")
    overall_scoped = round(sum(all_scoped) / len(all_scoped), 4) if all_scoped else float("nan")
    report["overall"] = {
        "questions": len(all_base),
        "baseline_recall@10": overall_base,
        "scoped_recall@10": overall_scoped,
        "delta": round(overall_scoped - overall_base, 4),
    }
    fired_summary = {
        "questions": len(fired_base),
        "baseline_recall@10": round(sum(fired_base) / len(fired_base), 4) if fired_base else None,
        "arm_recall@10": round(sum(fired_scoped) / len(fired_scoped), 4) if fired_scoped else None,
    }
    if fired_base:
        fired_summary["delta"] = round(
            (sum(fired_scoped) - sum(fired_base)) / len(fired_base), 4)
    report["fired_subset"] = fired_summary
    print(f"FIRED SUBSET n={fired_summary['questions']} "
          f"baseline={fired_summary['baseline_recall@10']} arm={fired_summary['arm_recall@10']} "
          f"delta={fired_summary.get('delta')}")
    print(f"\nOVERALL multi-hop n={len(all_base)} baseline={overall_base} scoped={overall_scoped} "
          f"delta={report['overall']['delta']:+.4f}")
    print(f"published multi-hop baseline (all conversations): 0.4939")

    out = Path(args.out) if args.out else ARENA_DIR / "results" / f"bridge_probe_{selector.name}_{args.mode}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
