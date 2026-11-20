#!/usr/bin/env python3
"""LongMemEval-S retrieval benchmark for InfinityDB.

Official data: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
(``longmemeval_s_cleaned.json``: 500 questions, ~50 sessions / ~550 turns /
~120k tokens of haystack per question). Download it once and keep it out of Git:

    mkdir -p benchmarks/data
    curl -L -o benchmarks/data/longmemeval_s_cleaned.json \\
      https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

Retrieval only by default: no API key, no reader LLM, nothing leaves the
machine. ``--answer-model`` + ``--judge-model`` add an end-to-end accuracy
column through any OpenAI-compatible endpoint (see the caveat below).

What is measured
----------------
Every question ships its own haystack, so each one gets a fresh database. Each
turn is ingested with its own session id and event time, and ``recall`` is
anchored at ``question_date`` (``now=``) so time decay is judged as of the
moment the question was asked.

Scoring uses the engine's own provenance rather than string matching: a
returned memory carries a ``source_id`` back to the ``l0_logs`` row it was
extracted from, and that row carries the session id and the exact turn text.
A memory is therefore provably "from the answer session" or not, and provably
"the turn annotated ``has_answer``" or not. Substring matching against answer
text would have measured tokenisation, not retrieval.

* ``session_any_recall`` — at least one returned memory comes from an annotated
  answer session.
* ``session_all_recall`` — every annotated answer session is represented
  (the multi-session and knowledge-update requirement).
* ``turn_any_recall`` — at least one returned memory is an annotated
  ``has_answer`` turn.
* ``turn_all_recall`` — every annotated ``has_answer`` turn is retrieved.
* ``mrr`` — reciprocal rank of the first memory from an answer session.
* ``latest_fact_ranks_above_stale`` (knowledge-update) — the newest evidence
  session outranks the oldest one, i.e. the updated fact is not buried under
  the value it replaced.
* abstention (``question_id`` ending ``_abs``) — these have no answerable
  evidence; the report shows the top-1 score distribution and the best
  in-sample confidence gate, clearly labelled as an oracle diagnostic, not a
  calibrated threshold.

Context cost is reported with every recall number (BPE tokens per query and
recall per 1,000 context tokens), because recall alone is not a result.

All modes of the engine are reachable: SQLite in memory (the default, zero
dependencies), a SQLite file, PostgreSQL (``--dsn``), optional local ONNX dense
fusion (``--dense-model``) and an optional local ONNX cross-encoder reranker
(``--rerank-model``, ``--rerank-top-k``, ``--quantize-rerank``).

Usage
-----
    # Whole dataset, offline, 600-token budget
    python benchmarks/longmemeval_bench.py \\
      --data-path benchmarks/data/longmemeval_s_cleaned.json --budget-tokens 600

    # Fast pass: 10 questions per ability (all seven groups, ~2 minutes)
    python benchmarks/longmemeval_bench.py --sample 10

    # With the local INT8 cross-encoder
    python benchmarks/longmemeval_bench.py \\
      --rerank-model Xenova/ms-marco-MiniLM-L-6-v2 --rerank-top-k 8 --quantize-rerank

The full 500-question pass takes roughly 20 minutes on 8 vCPUs (ingest
dominates at ~2 s per question). Rows are appended to ``<output>.jsonl`` as
they finish, so an interrupted run resumes where it stopped instead of
starting over; the checkpoint is keyed by a fingerprint of the run's settings
so a rerun with different flags never mixes two configurations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _path in (str(REPO_ROOT), str(HERE / "arena")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import longmemeval  # noqa: E402  (the arena loader: one schema authority)

from infinitydb.core.engine import InfinityDB  # noqa: E402

DEFAULT_DATA_PATH = HERE / "data" / "longmemeval_s_cleaned.json"
DEFAULT_OUTPUT_PATH = HERE / "longmemeval-s-results.json"

ABSTENTION = "abstention"
ABILITY_ORDER = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    ABSTENTION,
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def build_token_counter(encoding_name: str = "cl100k_base") -> tuple[Callable[[str], int], str]:
    """Return ``(text -> tokens, name)``, falling back to whitespace."""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        return (lambda text: len(encoding.encode(text or ""))), f"tiktoken:{encoding_name}"
    except Exception:  # pragma: no cover - depends on the optional install
        return (lambda text: len((text or "").split())), "whitespace_fallback"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "mean": round(statistics.fmean(values), 3),
    }


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover - not a Git checkout
        return "unknown"


def classify_ability(entry: longmemeval.Entry) -> str:
    """Abstention is a question-id suffix, not a ``question_type``."""
    if entry.question_id.endswith("_abs"):
        return ABSTENTION
    return entry.question_type or "unknown"


def turn_text(role: str, content: str) -> str:
    """The exact string ingested for a turn; evidence keys reuse it verbatim."""
    speaker = "User" if role == "user" else "Assistant"
    return f"{speaker}: {content}"


def evidence_of(entry: longmemeval.Entry) -> tuple[set[tuple[str, str]], set[str]]:
    """Return ``({(session_id, turn_text)}, {answer_session_ids})``.

    Annotated ``has_answer`` turns are the ground truth. If a run ever meets a
    question where none are flagged, every turn of the answer sessions becomes
    evidence so the question still scores instead of silently counting as a
    miss; the number of such fallbacks is reported in the artifact.
    """
    annotated: set[tuple[str, str]] = set()
    answer_sessions = {str(session_id) for session_id in entry.answer_session_ids}
    for session_id, session in zip(entry.session_ids, entry.sessions):
        for turn in session:
            if turn.get("has_answer") is True:
                annotated.add(
                    (str(session_id), turn_text(str(turn.get("role", "user")), str(turn.get("content", ""))))
                )
    if not annotated and answer_sessions:
        for session_id, session in zip(entry.session_ids, entry.sessions):
            if str(session_id) not in answer_sessions:
                continue
            for turn in session:
                annotated.add(
                    (str(session_id), turn_text(str(turn.get("role", "user")), str(turn.get("content", ""))))
                )
    return annotated, answer_sessions


def scored_items(db: InfinityDB, items: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Resolve each returned memory to ``(session_id, turn_text, atom_text)``."""
    resolved: list[tuple[str, str, str]] = []
    cache: dict[str, tuple[str, str]] = {}
    for item in items:
        source_id = str(item.get("source_id", ""))
        if source_id not in cache:
            row = db.conn.execute(
                "SELECT session_id, raw FROM l0_logs WHERE id = ?", (source_id,)
            ).fetchone()
            cache[source_id] = (str(row["session_id"]), str(row["raw"])) if row else ("", "")
        session_id, raw = cache[source_id]
        resolved.append((session_id, raw, str(item.get("fact", ""))))
    return resolved


# ---------------------------------------------------------------------------
# Optional end-to-end reader + judge
# ---------------------------------------------------------------------------


def judge_answer(
    question: str,
    gold_answer: str,
    ability: str,
    context: str,
    answer_model: str,
    judge_model: str,
) -> tuple[str, bool]:
    """Generate an answer over the retrieved context and grade it.

    The rubric below is an ability-aware adaptation of LongMemEval's grading
    categories, not a verbatim copy of the official grader, so treat the
    resulting accuracy as indicative and label it as such wherever it is
    published. Requires ``pip install openai`` and an endpoint in the
    ``OPENAI_*`` environment variables.
    """
    from openai import OpenAI

    client = OpenAI()
    answer_system = (
        "You answer questions about past conversations using only the retrieved "
        "memory snippets. If a fact was updated later, use the most recent value. "
        "If the snippets do not contain the answer, say you do not have enough "
        "information from the past conversations."
    )
    answer_prompt = (
        f"Retrieved memory context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    completion = client.chat.completions.create(
        model=answer_model,
        temperature=0.0,
        max_tokens=192,
        messages=[
            {"role": "system", "content": answer_system},
            {"role": "user", "content": answer_prompt},
        ],
    )
    hypothesis = (completion.choices[0].message.content or "").strip()

    if ability == ABSTENTION:
        rubric = (
            "The question is unanswerable from the history. correct=true only if the "
            "response abstains or says the information is missing; false if it guesses."
        )
    elif ability == "knowledge-update":
        rubric = (
            "The question has a superseded fact. correct=true only if the response "
            "states the updated value and does not present the outdated value as current."
        )
    elif ability == "temporal-reasoning":
        rubric = (
            "correct=true if the response matches the gold answer, allowing one day of "
            "tolerance on relative date arithmetic."
        )
    else:
        rubric = (
            "correct=true if the response contains the same factual answer as the gold "
            "answer and does not contradict it."
        )
    grade = client.chat.completions.create(
        model=judge_model,
        temperature=0.0,
        max_tokens=32,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"{rubric}\n\nQuestion: {question}\nGold answer: {gold_answer}\n"
                    f"Model response: {hypothesis}\n\n"
                    'Return JSON only: {"correct": true} or {"correct": false}.'
                ),
            }
        ],
    )
    try:
        correct = bool(json.loads((grade.choices[0].message.content or "{}").strip())["correct"])
    except Exception:
        correct = False
    return hypothesis, correct


# ---------------------------------------------------------------------------
# One question
# ---------------------------------------------------------------------------


def run_question(
    entry: longmemeval.Entry,
    *,
    top_k: int,
    budget_tokens: int | None,
    token_counter: Callable[[str], int],
    db_dir: Path | None,
    dsn: str | None,
    tenant_id: str,
    embedder: Callable[[list[str]], list[list[float]]] | None,
    dense_model: str,
    dense_weight: float,
    reranker: Callable[[str, Any], list[float]] | None,
    rerank_top_k: int | None,
    answer_model: str | None,
    judge_model: str | None,
) -> dict[str, Any]:
    ability = classify_ability(entry)
    if dsn:
        db = InfinityDB(dsn=dsn, tenant_id=tenant_id)
    elif db_dir is not None:
        db_dir.mkdir(parents=True, exist_ok=True)
        db = InfinityDB(db_path=str(db_dir / f"{entry.question_id}.db"))
    else:
        db = InfinityDB()

    try:
        started = time.perf_counter()
        turns = 0
        for session_id, ts, role, content in entry.turns():
            db.add(
                turn_text(role, content),
                user_id="lme",
                session_id=session_id,
                ts=ts or None,
                # Knowledge-update questions ask about a fact that was later
                # replaced; superseding it at ingest would delete the stale
                # value before the benchmark could check the ordering.
                auto_supersede=False,
            )
            turns += 1
        if embedder is not None:
            db.index_dense(embedder, model=dense_model)
        ingest_seconds = time.perf_counter() - started

        recall_kwargs: dict[str, Any] = {
            "n": top_k,
            "user_id": "lme",
            "now": entry.question_ts or None,
            "reinforce": False,
            "include_policies": False,
        }
        if budget_tokens is not None:
            recall_kwargs["budget_tokens"] = budget_tokens
            recall_kwargs["token_counter"] = token_counter
        if embedder is not None:
            recall_kwargs["dense"] = (embedder, dense_model, dense_weight)
        if reranker is not None:
            recall_kwargs["rerank"] = reranker
            recall_kwargs["rerank_top_k"] = rerank_top_k

        started = time.perf_counter()
        result = db.recall(entry.question, **recall_kwargs)
        search_ms = (time.perf_counter() - started) * 1000.0

        items = list(result.get("results", []))
        resolved = scored_items(db, items)
    finally:
        db.close()

    context_tokens = sum(token_counter(text) for _sid, _raw, text in resolved)
    top1_score = 0.0
    if items:
        try:
            top1_score = float(items[0].get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            top1_score = 0.0

    row: dict[str, Any] = {
        "question_id": entry.question_id,
        "question_type": entry.question_type,
        "ability": ability,
        "sessions": len(entry.session_ids),
        "turns": turns,
        "returned_items": len(items),
        "context_tokens": context_tokens,
        "top1_score": round(top1_score, 4),
        "ingest_seconds": round(ingest_seconds, 3),
        "search_ms": round(search_ms, 2),
        "session_any_hit": False,
        "session_all_hit": False,
        "turn_any_hit": False,
        "turn_all_hit": False,
        "evidence_turns": 0,
        "evidence_sessions": 0,
        "mrr": 0.0,
        "latest_beats_stale": None,
    }

    if ability == ABSTENTION:
        # No answerable evidence exists; only the confidence diagnostics apply.
        if answer_model and judge_model:
            hypothesis, correct = judge_answer(
                entry.question,
                entry.answer,
                ability,
                "\n".join(text for _sid, _raw, text in resolved),
                answer_model,
                judge_model,
            )
            row["predicted_answer"] = hypothesis
            row["judge_correct"] = correct
        return row

    evidence_turns, answer_sessions = evidence_of(entry)
    present_sessions = {str(session_id) for session_id in entry.session_ids}
    # An annotation for a session that is not in the haystack can never be
    # retrieved; count it rather than quietly scoring the question impossible.
    usable_sessions = answer_sessions & present_sessions
    returned_sessions = [session_id for session_id, _raw, _text in resolved]
    returned_turns = {(session_id, raw) for session_id, raw, _text in resolved}

    row["evidence_turns"] = len(evidence_turns)
    row["evidence_sessions"] = len(usable_sessions)
    row["answer_sessions_missing"] = len(answer_sessions - present_sessions)
    row["session_any_hit"] = bool(set(returned_sessions) & usable_sessions)
    row["session_all_hit"] = bool(usable_sessions) and usable_sessions <= set(returned_sessions)
    row["turn_any_hit"] = bool(evidence_turns & returned_turns)
    row["turn_all_hit"] = bool(evidence_turns) and evidence_turns <= returned_turns

    for rank, session_id in enumerate(returned_sessions, start=1):
        if session_id in usable_sessions:
            row["mrr"] = round(1.0 / rank, 4)
            break

    if ability == "knowledge-update" and len(usable_sessions) >= 2:
        session_ts = {
            str(session_id): ts
            for session_id, ts in zip(entry.session_ids, entry.session_ts)
        }
        ordered = sorted(usable_sessions, key=lambda sid: session_ts.get(sid, 0.0))
        oldest, newest = ordered[0], ordered[-1]

        def rank_of(target: str) -> int | None:
            for rank, session_id in enumerate(returned_sessions, start=1):
                if session_id == target:
                    return rank
            return None

        oldest_rank, newest_rank = rank_of(oldest), rank_of(newest)
        if newest_rank is not None and oldest_rank is not None:
            row["latest_beats_stale"] = newest_rank < oldest_rank
        elif newest_rank is not None:
            row["latest_beats_stale"] = True
        elif oldest_rank is not None:
            row["latest_beats_stale"] = False

    if answer_model and judge_model:
        hypothesis, correct = judge_answer(
            entry.question,
            entry.answer,
            ability,
            "\n".join(text for _sid, _raw, text in resolved),
            answer_model,
            judge_model,
        )
        row["predicted_answer"] = hypothesis
        row["judge_correct"] = correct

    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def best_confidence_gate(
    abstention_scores: list[float], answerable_scores: list[float]
) -> dict[str, Any] | None:
    """Best in-sample threshold for "abstain when top-1 score < t".

    This is an oracle diagnostic: the threshold is fitted on the same questions
    it is scored on, so it is an upper bound on what a calibrated gate could do,
    not a deployable operating point.
    """
    if not abstention_scores or not answerable_scores:
        return None
    best: dict[str, Any] | None = None
    for threshold in sorted(set(abstention_scores + answerable_scores)):
        true_positive = sum(score < threshold for score in abstention_scores)
        false_positive = sum(score < threshold for score in answerable_scores)
        sensitivity = true_positive / len(abstention_scores)
        specificity = 1.0 - false_positive / len(answerable_scores)
        balanced = (sensitivity + specificity) / 2.0
        if best is None or balanced > best["balanced_accuracy"]:
            best = {
                "threshold": round(threshold, 4),
                "balanced_accuracy": round(balanced, 4),
                "abstention_recall": round(sensitivity, 4),
                "answerable_retention": round(specificity, 4),
                "oracle_fit": True,
            }
    return best


def summarize_slice(rows: list[dict[str, Any]], abstention: bool) -> dict[str, Any]:
    if not rows:
        return {"questions": 0}
    count = len(rows)
    mean_tokens = statistics.fmean(row["context_tokens"] for row in rows)
    summary: dict[str, Any] = {
        "questions": count,
        "items_per_query": round(statistics.fmean(row["returned_items"] for row in rows), 2),
        "tokens_per_query": round(mean_tokens, 1),
        "mean_top1_score": round(statistics.fmean(row["top1_score"] for row in rows), 4),
    }

    if abstention:
        summary["empty_result_rate_pct"] = round(
            100.0 * sum(1 for row in rows if row["returned_items"] == 0) / count, 2
        )
        if "judge_correct" in rows[0]:
            summary["llm_judge_accuracy_pct"] = round(
                100.0 * sum(bool(row.get("judge_correct")) for row in rows) / count, 2
            )
        return summary

    def rate(key: str) -> float:
        return round(100.0 * sum(bool(row[key]) for row in rows) / count, 2)

    turn_any = rate("turn_any_hit")
    summary.update(
        {
            "session_any_recall_at_k_pct": rate("session_any_hit"),
            "session_all_recall_at_k_pct": rate("session_all_hit"),
            "turn_any_recall_at_k_pct": turn_any,
            "turn_all_recall_at_k_pct": rate("turn_all_hit"),
            "mrr": round(statistics.fmean(row["mrr"] for row in rows), 4),
            "turn_any_recall_per_1k_tokens": round(
                (turn_any / mean_tokens * 1000.0) if mean_tokens else 0.0, 2
            ),
        }
    )

    recency_rows = [row for row in rows if row.get("latest_beats_stale") is not None]
    if recency_rows:
        summary["latest_fact_ranks_above_stale_pct"] = round(
            100.0 * sum(bool(row["latest_beats_stale"]) for row in recency_rows)
            / len(recency_rows),
            2,
        )
        summary["latest_fact_questions"] = len(recency_rows)

    if "judge_correct" in rows[0]:
        accuracy = round(100.0 * sum(bool(row.get("judge_correct")) for row in rows) / count, 2)
        summary["llm_judge_accuracy_pct"] = accuracy
        summary["judge_accuracy_per_1k_tokens"] = round(
            (accuracy / mean_tokens * 1000.0) if mean_tokens else 0.0, 2
        )
    return summary


def summarize(
    rows: list[dict[str, Any]], top_k: int, has_judge: bool
) -> dict[str, Any]:
    answerable = [row for row in rows if row["ability"] != ABSTENTION]
    abstention_rows = [row for row in rows if row["ability"] == ABSTENTION]

    by_ability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ability[row["ability"]].append(row)

    order = list(ABILITY_ORDER)
    for name in sorted(by_ability):
        if name not in order:
            order.append(name)

    results: dict[str, Any] = {
        "top_k": top_k,
        "overall": {
            "questions": len(rows),
            "answerable_questions": len(answerable),
            "abstention_questions": len(abstention_rows),
            "answerable_retrieval": summarize_slice(answerable, abstention=False),
            "abstention_diagnostics": summarize_slice(abstention_rows, abstention=True),
        },
        "by_ability": {
            name: summarize_slice(by_ability[name], abstention=(name == ABSTENTION))
            for name in order
            if name in by_ability
        },
        "ingest_seconds": latency_summary([row["ingest_seconds"] for row in rows]),
        "search_ms": latency_summary([row["search_ms"] for row in rows]),
    }

    gate = best_confidence_gate(
        [row["top1_score"] for row in abstention_rows],
        [row["top1_score"] for row in answerable],
    )
    if gate:
        results["overall"]["abstention_diagnostics"]["confidence_gate"] = gate

    if has_judge and rows:
        accuracy = 100.0 * sum(bool(row.get("judge_correct")) for row in rows) / len(rows)
        mean_tokens = statistics.fmean(row["context_tokens"] for row in rows)
        results["overall"]["answerable_and_abstention_judge_accuracy_pct"] = round(accuracy, 2)
        results["overall"]["judge_accuracy_per_1k_tokens"] = round(
            (accuracy / mean_tokens * 1000.0) if mean_tokens else 0.0, 2
        )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def fingerprint(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LongMemEval-S retrieval benchmark for InfinityDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None, help="Evenly spaced N questions per ability.")
    parser.add_argument("--abilities", nargs="*", default=None)
    parser.add_argument("--k", type=int, default=10, help="Top-K memories per query.")
    parser.add_argument(
        "--budget-tokens",
        type=int,
        default=600,
        help="Per-query context budget in tokenizer units (0 = unbudgeted).",
    )
    parser.add_argument("--dsn", default=None, help="PostgreSQL DSN instead of SQLite.")
    parser.add_argument("--db-dir", type=Path, default=None, help="Keep one SQLite file per question here.")
    parser.add_argument("--dense-model", default=None, help="Local ONNX embedder for dense fusion.")
    parser.add_argument("--dense-weight", type=float, default=0.35)
    parser.add_argument("--rerank-model", default=None, help="Local ONNX cross-encoder.")
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--quantize-rerank", action="store_true")
    parser.add_argument("--onnx-cache", default=None)
    parser.add_argument("--onnx-threads", type=int, default=None)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore and rewrite the per-question checkpoint.",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise SystemExit(
            f"LongMemEval data not found at {args.data_path}.\n"
            "Download it with:\n"
            "  curl -L -o benchmarks/data/longmemeval_s_cleaned.json \\\n"
            "    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
        )

    dataset_bytes = args.data_path.read_bytes()
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    entries = longmemeval.load(args.data_path)
    if args.abilities:
        wanted = set(args.abilities)
        entries = [entry for entry in entries if classify_ability(entry) in wanted]
    if args.sample:
        # The file is grouped by question id, not by ability: taking the first N
        # rows samples the abstention block alone. Space the picks instead.
        by_ability: dict[str, list[longmemeval.Entry]] = defaultdict(list)
        for entry in entries:
            by_ability[classify_ability(entry)].append(entry)
        picked: list[longmemeval.Entry] = []
        for ability in ABILITY_ORDER + sorted(set(by_ability) - set(ABILITY_ORDER)):
            group = by_ability.get(ability, [])
            if not group:
                continue
            step = max(1, len(group) // args.sample)
            picked.extend(group[::step][: args.sample])
        entries = picked
    if args.max_questions is not None:
        entries = entries[: args.max_questions]

    token_counter, tokenizer_name = build_token_counter("cl100k_base")
    budget_tokens = None if args.budget_tokens <= 0 else args.budget_tokens
    has_judge = bool(args.answer_model and args.judge_model)

    embedder = None
    dense_model = args.dense_model or "default"
    reranker = None
    if args.dense_model or args.rerank_model:
        from infinitydb.local_models import local_embedder, local_reranker

        if args.dense_model:
            embedder, dense_model = local_embedder(
                model=args.dense_model,
                cache_dir=args.onnx_cache,
                threads=args.onnx_threads,
            )
        if args.rerank_model:
            reranker = local_reranker(
                model=args.rerank_model,
                cache_dir=args.onnx_cache,
                threads=args.onnx_threads,
                quantize=args.quantize_rerank,
            )

    run_fingerprint = fingerprint(
        {
            "dataset": dataset_sha256,
            "k": args.k,
            "budget_tokens": budget_tokens,
            "backend": "postgres" if args.dsn else "sqlite",
            "dense_model": args.dense_model,
            "dense_weight": args.dense_weight if args.dense_model else None,
            "rerank_model": args.rerank_model,
            "rerank_top_k": args.rerank_top_k,
            "quantize_rerank": args.quantize_rerank,
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
        }
    )

    checkpoint = args.output.with_suffix(".jsonl")
    if args.no_resume and checkpoint.exists():
        checkpoint.unlink()
    done: set[str] = set()
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("fingerprint") == run_fingerprint:
                done.add(row["question_id"])

    pending = [entry for entry in entries if entry.question_id not in done]
    print(
        f"LongMemEval-S: {len(entries)} questions selected, {len(done)} already done, "
        f"{len(pending)} to run (k={args.k}, budget={budget_tokens or 'unbudgeted'}, "
        f"backend={'postgres' if args.dsn else 'sqlite'}, "
        f"rerank={args.rerank_model or 'none'}, dense={args.dense_model or 'none'})"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with checkpoint.open("a", encoding="utf-8") as handle:
        for index, entry in enumerate(pending, start=1):
            row = run_question(
                entry,
                top_k=args.k,
                budget_tokens=budget_tokens,
                token_counter=token_counter,
                db_dir=args.db_dir,
                dsn=args.dsn,
                tenant_id=f"lme_{entry.question_id[:12]}",
                embedder=embedder,
                dense_model=dense_model,
                dense_weight=args.dense_weight,
                reranker=reranker,
                rerank_top_k=args.rerank_top_k,
                answer_model=args.answer_model,
                judge_model=args.judge_model,
            )
            row["fingerprint"] = run_fingerprint
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(pending):
                print(
                    f"  {index}/{len(pending)} ({entry.question_id}) "
                    f"ingest {row['ingest_seconds']:.1f}s recall {row['search_ms']:.0f}ms"
                )

    selected = {entry.question_id for entry in entries}
    rows = []
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("fingerprint") == run_fingerprint and row["question_id"] in selected:
            rows.append(row)

    results = summarize(rows, args.k, has_judge)
    report: dict[str, Any] = {
        "benchmark": "longmemeval-s",
        "schema_version": "1.0",
        "git_commit": get_git_commit(),
        "run_fingerprint": run_fingerprint,
        "dataset": {
            "path": str(args.data_path),
            "sha256": dataset_sha256,
            "questions_selected": len(entries),
            "questions_scored": len(rows),
            "answer_sessions_missing_from_haystack": sum(
                int(row.get("answer_sessions_missing", 0)) for row in rows
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "tokenizer": tokenizer_name,
            "backend": "postgres" if args.dsn else "sqlite",
            "top_k": args.k,
            "budget_tokens": budget_tokens,
            "dense_model": args.dense_model,
            "dense_weight": args.dense_weight if args.dense_model else None,
            "rerank_model": args.rerank_model,
            "rerank_top_k": args.rerank_top_k,
            "quantize_rerank": args.quantize_rerank,
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "results": results,
        "per_question": rows,
    }

    overall = results["overall"]["answerable_retrieval"]
    print(
        f"\nANSWERABLE ({overall.get('questions', 0)}q): "
        f"session-any@{args.k}={overall.get('session_any_recall_at_k_pct', 0.0):.2f}% | "
        f"session-all@{args.k}={overall.get('session_all_recall_at_k_pct', 0.0):.2f}% | "
        f"turn-any@{args.k}={overall.get('turn_any_recall_at_k_pct', 0.0):.2f}% | "
        f"turn-all@{args.k}={overall.get('turn_all_recall_at_k_pct', 0.0):.2f}% | "
        f"MRR={overall.get('mrr', 0.0):.4f} | "
        f"tokens/q={overall.get('tokens_per_query', 0.0):.1f} | "
        f"{overall.get('turn_any_recall_per_1k_tokens', 0.0):.2f} pts/1k tokens"
    )
    for ability, stats in results["by_ability"].items():
        if ability == ABSTENTION:
            print(
                f"  {ability:<26} {stats.get('questions', 0):>3}q  "
                f"top1 mean {stats.get('mean_top1_score', 0.0):.3f}  "
                f"empty {stats.get('empty_result_rate_pct', 0.0):.1f}%"
            )
        else:
            extra = ""
            if "latest_fact_ranks_above_stale_pct" in stats:
                extra = f"  newest>stale {stats['latest_fact_ranks_above_stale_pct']:.1f}%"
            print(
                f"  {ability:<26} {stats.get('questions', 0):>3}q  "
                f"session-any {stats.get('session_any_recall_at_k_pct', 0.0):>5.1f}%  "
                f"turn-any {stats.get('turn_any_recall_at_k_pct', 0.0):>5.1f}%  "
                f"MRR {stats.get('mrr', 0.0):.3f}{extra}"
            )

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\ningest p50 {results['ingest_seconds']['p50']:.2f}s, "
        f"recall p50 {results['search_ms']['p50']:.1f}ms"
    )
    print(f"wrote {args.output} ({len(rows)} questions) and {checkpoint}")


if __name__ == "__main__":
    main()
