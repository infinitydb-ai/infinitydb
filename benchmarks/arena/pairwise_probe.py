#!/usr/bin/env python3
"""Phase 1b: a pairwise ranker over features that already exist.

Phase 1 (see docs/SELF_IMPROVEMENT.md) ruled out a pointwise term prior: it
improved the fit half and lost on held-out data. This is the follow-up that
phase 1 specified — a pairwise perceptron over interpretable features, trained
on *lists* (one question's candidates) rather than per-memory labels.

Same protocol, deliberately unchanged so the two results are comparable:

* split by conversation (or within conversation with ``--split within``),
* labels simulated as ``oracle`` (contains annotated evidence) and ``answer``
  (contains the gold answer string) — both still need gold data, so neither is
  deployable; that is stated rather than hidden,
* features are computed from the question and the memory alone. The candidate's
  own rank is one feature, so a ranker that cannot beat the existing order will
  learn to preserve it,
* metrics at k=1, 3 and 10; @10 is set-invariant and reported only to show it.

Pure Python on purpose: this is a probe, not a dependency. Nothing here is used
by the engine and no engine default changes.

Run: python benchmarks/arena/pairwise_probe.py [--split within]
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

from arena import locomo, feedback_probe as fb  # noqa: E402

FEATURES = ["bias", "rank", "overlap", "date", "place", "who", "recency", "length"]
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
DATE_RE = re.compile(r"\[(\d{1,2}) ([A-Z][a-z]{2}),? (\d{4})\]")
WORD_RE = re.compile(r"\w+")
STOP = {"the", "and", "was", "were", "did", "does", "what", "when", "where", "who",
        "whom", "which", "how", "why", "that", "this", "with", "for", "from", "they",
        "she", "her", "his", "him", "has", "have", "had", "not", "you", "your", "been"}


def terms(text: str) -> set[str]:
    return {fb.STEM(t) for t in WORD_RE.findall(text.lower())
            if len(t) > 2 and t not in STOP}


def parse_date(text: str) -> float | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    day, month, year = match.groups()
    try:
        return datetime(int(year), MONTHS[month], int(day)).timestamp()
    except (KeyError, ValueError):
        return None


def question_type(question: str) -> str:
    q = question.lower()
    if q.startswith(("when", "what year", "what date", "how long")):
        return "date"
    if "where" in q:
        return "place"
    if q.startswith(("who", "whose")):
        return "who"
    return "other"


def features(question: str, memory: str, rank: int, span: tuple[float, float] | None) -> list[float]:
    q_terms = terms(question)
    m_terms = terms(memory)
    overlap = len(q_terms & m_terms) / len(q_terms) if q_terms else 0.0
    kind = question_type(question)
    has_date = 1.0 if DATE_RE.search(memory) else 0.0
    has_place = 1.0 if re.search(r"\bin [A-Z][a-z]+", memory) else 0.0
    has_name = 1.0 if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", memory) else 0.0
    parsed = parse_date(memory)
    if parsed is not None and span and span[1] > span[0]:
        recency = max(0.0, min(1.0, (parsed - span[0]) / (span[1] - span[0])))
    else:
        recency = 0.0
    length = min(1.0, len(m_terms) / 60.0)
    return [
        1.0,
        1.0 / (rank + 1),
        overlap,
        has_date if kind == "date" else 0.0,
        has_place if kind == "place" else 0.0,
        has_name if kind == "who" else 0.0,
        recency,
        length,
    ]


def spans(runs: dict[str, list[dict]]) -> dict[str, tuple[float, float]]:
    out = {}
    for conv, rows in runs.items():
        stamps = [d for row in rows for memory in row["retrieved"]
                  if (d := parse_date(memory)) is not None]
        if stamps:
            out[conv] = (min(stamps), max(stamps))
    return out


def labels_for(row: dict, conv_map, scheme: str) -> list[bool]:
    ev = fb.evidence_texts(conv_map[row["conv"]], row.get("evidence") or [])
    return [fb.label_memory(memory, row, ev, scheme) for memory in row["retrieved"]]


def _independent_score(question: str, memory: str, span) -> float:
    """A deliberately different ranking: overlap and recency, with no rank feature.

    Its disagreement with the lexical order is what the agreement label uses.
    """
    return (
        2.0 * features(question, memory, 0, span)[2]
        + 0.5 * features(question, memory, 0, span)[6]
        + 0.3 * features(question, memory, 0, span)[4]
    )


def agreement_labels(row: dict, span) -> list[bool | None]:
    """Labels available in a real deployment: two rankers agreeing, no gold used.

    Positive when a memory is in the top 3 of both the existing ranking and an
    independent overlap/recency ranking; negative when it is in the bottom half
    of both. Everything else is abstained (``None``), because a weak label
    should say nothing rather than guess. Nothing here reads the answer or the
    evidence ids.
    """
    memories = row["retrieved"]
    n = len(memories)
    base_top = set(range(min(3, n)))
    independent = sorted(range(n), key=lambda i: _independent_score(row["question"], memories[i], span),
                         reverse=True)[: min(3, n)]
    independent_top = set(independent)
    labels: list[bool | None] = []
    for index in range(n):
        if index in base_top and index in independent_top:
            labels.append(True)
        elif index not in base_top and index not in independent_top and index >= n // 2:
            labels.append(False)
        else:
            labels.append(None)
    return labels


def train(rows: list[dict], conv_map, span_map, scheme: str, epochs: int, lr: float) -> list[float]:
    weights = [0.0] * len(FEATURES)
    for _ in range(epochs):
        for row in rows:
            if scheme == "agreement":
                labels = agreement_labels(row, span_map.get(row["conv"]))
            else:
                labels = labels_for(row, conv_map, scheme)
            if not any(label for label in labels) or all(label for label in labels):
                continue
            feats = [features(row["question"], memory, rank, span_map.get(row["conv"]))
                     for rank, memory in enumerate(row["retrieved"])]
            positives = [i for i, ok in enumerate(labels) if ok]
            negatives = [i for i, ok in enumerate(labels) if ok is False][:5]
            if not positives or not negatives:
                continue
            for pos in positives:
                for neg in negatives:
                    margin = sum(w * (a - b) for w, a, b in zip(weights, feats[pos], feats[neg]))
                    if margin < 1.0:
                        for j, (a, b) in enumerate(zip(feats[pos], feats[neg])):
                            weights[j] += lr * (a - b)
    return weights


def evaluate(rows: list[dict], conv_map, span_map, weights: list[float] | None) -> dict[str, float]:
    scores = {f"recall@{k}": [] for k in (1, 3, 10)}
    for row in rows:
        ev = fb.evidence_texts(conv_map[row["conv"]], row.get("evidence") or [])
        if not ev:
            continue
        if weights is None:
            ranked = row["retrieved"]
        else:
            scored = []
            for rank, memory in enumerate(row["retrieved"]):
                value = sum(w * f for w, f in zip(
                    weights, features(row["question"], memory, rank, span_map.get(row["conv"]))))
                scored.append((value, -rank, memory))
            scored.sort(reverse=True)
            ranked = [memory for _, _, memory in scored]
        for k in (1, 3, 10):
            scores[f"recall@{k}"].append(fb.recall_at(ranked, ev, k))
    return {key: round(sum(v) / len(v), 4) if v else float("nan") for key, v in scores.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ARENA_DIR / "results"))
    ap.add_argument("--system", default="infinitydb")
    ap.add_argument("--split", default="conversation", choices=("conversation", "within"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--reverse", action="store_true",
                    help="swap the conversation halves, to check the result is not one lucky split")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    conv_map = {c.sample_id: c for c in locomo.load()}
    runs = fb.load_questions(Path(args.results), args.system)
    span_map = spans(runs)
    convs = sorted(runs)
    if args.split == "conversation":
        fit_convs, held_convs = (convs[1::2], convs[::2]) if args.reverse else (convs[::2], convs[1::2])
        fit_rows = [r for c in fit_convs for r in runs[c]]
        held_rows = [r for c in held_convs for r in runs[c]]
    else:
        fit_rows = [r for c in convs for r in runs[c][: int(len(runs[c]) * 0.6)]]
        held_rows = [r for c in convs for r in runs[c][int(len(runs[c]) * 0.6):]]
        fit_convs = held_convs = convs
    print(f"split: {args.split}")
    print(f"fit conversations:      {fit_convs}")
    print(f"held-out conversations: {held_convs}\n")

    report: dict[str, object] = {
        "protocol": f"{args.split} split; perceptron trained on the fit half only",
        "features": FEATURES,
        "epochs": args.epochs,
        "labels": {},
    }

    baseline = evaluate(held_rows, conv_map, span_map, None)
    print(f"baseline (existing ranking) held-out: {baseline}\n")
    report["heldout_baseline"] = baseline

    for scheme in ("oracle", "answer", "agreement"):
        best = None
        for epochs in (4, args.epochs, 16):
            weights = train(fit_rows, conv_map, span_map, scheme, epochs, 0.05)
            fit = evaluate(fit_rows, conv_map, span_map, weights)
            if best is None or (fit["recall@1"], fit["recall@3"]) > (best[1]["recall@1"], best[1]["recall@3"]):
                best = (weights, fit, epochs)
        weights, fit, epochs = best
        held = evaluate(held_rows, conv_map, span_map, weights)
        print(f"[{scheme}] epochs={epochs} fit={fit}")
        print(f"[{scheme}] held-out={held}")
        print(f"[{scheme}] weights: " + ", ".join(
            f"{name}={value:+.3f}" for name, value in zip(FEATURES, weights)))
        delta = {k: round(held[k] - baseline[k], 4) for k in baseline}
        print(f"[{scheme}] delta vs baseline: {delta}\n")
        report["labels"][scheme] = {
            "deployable_labels": scheme == "agreement",
            "epochs": epochs,
            "fit": fit,
            "heldout": held,
            "delta": delta,
            "weights": {name: round(value, 4) for name, value in zip(FEATURES, weights)},
        }

    out = Path(args.out) if args.out else ARENA_DIR / "results" / f"pairwise_probe_{args.split}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
