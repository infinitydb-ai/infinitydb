#!/usr/bin/env python3
"""Phase 1 of docs/SELF_IMPROVEMENT.md: does feedback reweighting help?

Protocol, fixed before running:

* Split **by conversation** (never by question): fit on even-indexed
  conversations, evaluate on odd-indexed ones. A question-level split would leak
  near-duplicate context between halves.
* Labels are simulated two ways, and both are reported because they cost
  different things:
  - ``oracle`` — a retrieved memory is positive when it contains an annotated
    evidence turn. Requires gold evidence, so it is an upper bound only.
  - ``answer`` — a retrieved memory is positive when it contains the gold answer
    string. Closer to "the response could have been produced from this", but it
    still needs the gold answer, so it is not deployable either. A deployable
    label needs a reader or a user thumb; neither exists offline, and this probe
    does not pretend otherwise.
* Hyperparameters (``boost``, ``min_support``) are chosen on the fit half only.
* Metrics are reported at k=1, 3 and 10. **Reordering cannot change recall@10**
  — it is the same ten memories in a different order — so a reranker can only
  move @1 and @3. Reporting @10 would be reporting a constant.

Run: python benchmarks/arena/feedback_probe.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parent
ROOT = ARENA_DIR.parents[1]
for path in (str(ROOT), str(ARENA_DIR.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from arena import locomo, score  # noqa: E402
from infinitydb.core.engine import InfinityDB  # noqa: E402

STEM = InfinityDB._stem


def load_questions(results: Path, system: str) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for path in sorted((results / system).glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        qa = [row for row in rows if row.get("type") == "qa" and row.get("retrieved")]
        for row in qa:
            row["conv"] = path.stem
        if qa:
            runs[path.stem] = qa
    return runs


def evidence_texts(convo, dia_ids: list[str]) -> list[str]:
    by_dia = convo.turn_text_by_dia()
    return [by_dia[d] for d in dia_ids if d in by_dia]


def label_memory(memory: str, row: dict, ev_texts: list[str], scheme: str) -> bool:
    if scheme == "oracle":
        return bool(ev_texts) and score.evidence_hit([memory], ev_texts) > 0.0
    if scheme == "answer":
        return score.answer_in_context([memory], row["answer"])
    raise ValueError(scheme)


def learn(labelled: list[tuple[str, bool]], min_support: float) -> tuple[dict[str, float], dict[str, float]]:
    """Term prior and per-memory prior from labelled memories."""
    term_votes: Counter[str] = Counter()
    atom_votes: Counter[str] = Counter()
    for memory, positive in labelled:
        weight = 1.0 if positive else -0.5
        atom_votes[memory] += weight
        for term in {STEM(t) for t in re.findall(r"\w+", memory.lower()) if len(t) > 2}:
            term_votes[term] += weight
    terms = {t: v for t, v in term_votes.items() if v >= min_support}
    atoms = {a: v for a, v in atom_votes.items() if v >= min_support}
    if terms:
        ceiling = max(terms.values())
        terms = {t: v / ceiling for t, v in terms.items()}
    if atoms:
        ceiling = max(atoms.values())
        atoms = {a: v / ceiling for a, v in atoms.items()}
    return terms, atoms


def rerank(
    retrieved: list[str],
    base: list[float],
    terms: dict[str, float],
    atoms: dict[str, float],
    boost: float,
) -> list[str]:
    scored = []
    for rank, (memory, prior_score) in enumerate(zip(retrieved, base)):
        stemmed = {STEM(t) for t in re.findall(r"\w+", memory.lower())}
        term_prior = sum(weight for term, weight in terms.items() if term in stemmed)
        value = prior_score + boost * (atoms.get(memory, 0.0) + min(term_prior, 3.0))
        scored.append((value, -rank, memory))
    scored.sort(reverse=True)
    return [memory for _, _, memory in scored]


def recall_at(retrieved: list[str], ev_texts: list[str], k: int) -> float:
    if not ev_texts:
        return float("nan")
    return 1.0 if score.evidence_hit(retrieved[:k], ev_texts) > 0.0 else 0.0


def evaluate(rows: list[dict], conv_map, terms, atoms, boost: float) -> dict[str, float]:
    out = {f"recall@{k}": [] for k in (1, 3, 10)}
    for row in rows:
        ev = evidence_texts(conv_map[row["conv"]], row.get("evidence") or [])
        if not ev:
            continue
        base = [1.0 / (i + 1) for i in range(len(row["retrieved"]))]
        ranked = rerank(row["retrieved"], base, terms, atoms, boost)
        for k in (1, 3, 10):
            out[f"recall@{k}"].append(recall_at(ranked, ev, k))
    return {key: round(sum(v) / len(v), 4) if v else float("nan") for key, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ARENA_DIR / "results"))
    ap.add_argument("--system", default="infinitydb")
    ap.add_argument("--out", default=str(ARENA_DIR / "results" / "feedback_probe.json"))
    ap.add_argument("--split", default="conversation", choices=("conversation", "within"),
                    help="conversation: fit on whole conversations. within: fit on the first "
                         "60%% of each conversation's questions, evaluate on the rest (same "
                         "speaker vocabulary, closer to one user over time).")
    args = ap.parse_args()

    conv_map = {c.sample_id: c for c in locomo.load()}
    runs = load_questions(Path(args.results), args.system)
    convs = sorted(runs)
    if args.split == "conversation":
        fit_convs, held_convs = convs[::2], convs[1::2]
        fit_rows = lambda: [r for c in fit_convs for r in runs[c]]
        held_rows = lambda: [r for c in held_convs for r in runs[c]]
    else:
        fit_convs, held_convs = convs, convs
        def fit_rows():
            return [r for c in convs for r in runs[c][: int(len(runs[c]) * 0.6)]]

        def held_rows():
            return [r for c in convs for r in runs[c][int(len(runs[c]) * 0.6):]]
    print(f"split: {args.split}")
    print(f"fit conversations:  {fit_convs}")
    print(f"held-out conversations: {held_convs}\n")

    results: dict[str, object] = {
        "protocol": f"{args.split} split, hyperparameters chosen on fit half only",
        "system": args.system,
        "fit_conversations": fit_convs,
        "heldout_conversations": held_convs,
        "labels": {},
    }

    for scheme in ("oracle", "answer"):
        labelled: list[tuple[str, bool]] = []
        for row in fit_rows():
            ev = evidence_texts(conv_map[row["conv"]], row.get("evidence") or [])
            for memory in row["retrieved"]:
                labelled.append((memory, label_memory(memory, row, ev, scheme)))
        positive = sum(1 for _, ok in labelled if ok)
        print(f"[{scheme}] fit labels: {len(labelled)} judgements, {positive} positive "
              f"({positive / len(labelled):.1%})")

        best = None
        for min_support in (1.0, 2.0):
            terms, atoms = learn(labelled, min_support)
            for boost in (0.1, 0.25, 0.5, 1.0):
                fit = evaluate(fit_rows(), conv_map, terms, atoms, boost)
                candidate = (fit["recall@1"], fit["recall@3"], min_support, boost, terms, atoms)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        fit_r1, fit_r3, min_support, boost, terms, atoms = best
        print(f"[{scheme}] chosen on fit: min_support={min_support} boost={boost} "
              f"(fit recall@1={fit_r1:.4f} @3={fit_r3:.4f}, {len(terms)} terms)")

        baseline = evaluate(held_rows(), conv_map, {}, {}, 0.0)
        learned = evaluate(held_rows(), conv_map, terms, atoms, boost)
        print(f"[{scheme}] held-out baseline: {baseline}")
        print(f"[{scheme}] held-out learned:  {learned}\n")
        results["labels"][scheme] = {
            "fit_labels": len(labelled),
            "fit_positive": positive,
            "hyperparameters": {"min_support": min_support, "boost": boost},
            "fit": {"recall@1": fit_r1, "recall@3": fit_r3},
            "heldout_baseline": baseline,
            "heldout_learned": learned,
            "delta": {k: round(learned[k] - baseline[k], 4) for k in baseline},
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
