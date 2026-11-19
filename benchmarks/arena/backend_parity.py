#!/usr/bin/env python3
"""Measure the same recall pipeline on SQLite and PostgreSQL, question by question.

The engine has one ranker: extraction, window scoring, chunk assembly, budget
packing and provenance are Python and are identical on both backends. What
differs is candidate narrowing (FTS5 ``bm25`` vs ``tsvector`` ``ts_rank_cd``).
This gate measures whether that difference changes what a caller receives.

Run it with the optional extra installed::

    pip install "infinitydb-ai[postgres]" pgserver
    python benchmarks/arena/backend_parity.py --max-convos 1 --max-questions 60

``pgserver`` starts a throwaway PostgreSQL + pgvector instance, so the gate needs
no running database. Pass ``--dsn`` to measure a server you already have.

Gate: the run fails if PostgreSQL's aggregate evidence recall is below SQLite's.
Per-backend recall, the rate of *identical* evidence decisions, per-question
gains/losses, exact returned-memory identity and latency are printed as
measurements. Porter and Snowball stemming are not identical and the two
databases order borderline candidates differently, so per-question flips are
expected; they must be reported, not hidden.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from infinitydb.core.engine import InfinityDB  # noqa: E402
from locomo_local_compare import (  # noqa: E402
    DEFAULT_DATASET,
    count_tokens,
    extract_sessions,
    normalize_answer,
    percentile,
)


def _load(path: Path, max_convos: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"dataset not found: {path}\n"
            "The vendored copy is benchmarks/.cache/locomo10.json; run any arena "
            "command once to fetch it."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[:max_convos]


class Runner:
    """One engine instance, scoped to a single backend."""

    def __init__(self, label: str, db: InfinityDB) -> None:
        self.label = label
        self.db = db
        self.now: float | None = None
        self.ingest_ms: list[float] = []
        self.recall_ms: list[float] = []

    def ingest(self, sample: dict[str, Any], user_id: str) -> None:
        for session_key, turns, ts in extract_sessions(sample):
            start = time.perf_counter()
            for turn in turns:
                self.db.add(
                    conversation=f"{turn['speaker']}: {turn['text']}",
                    user_id=user_id,
                    session_id=f"{user_id}_{session_key}",
                    ts=ts,
                    auto_supersede=False,
                )
                if ts is not None:
                    self.now = max(self.now or ts, ts)
            self.ingest_ms.append((time.perf_counter() - start) * 1000.0)

    def recall(self, query: str, user_id: str, top_k: int, budget: int | None):
        start = time.perf_counter()
        result = self.db.recall(
            query=query,
            n=top_k,
            user_id=user_id,
            include_policies=False,
            now=self.now,
            reinforce=False,
            budget_tokens=budget,
            token_counter=count_tokens if budget is not None else None,
        )
        self.recall_ms.append((time.perf_counter() - start) * 1000.0)
        return result.get("results", [])


def _evidence_hits(
    sample: dict[str, Any], qa: dict[str, Any], facts: list[str]
) -> bool | None:
    evidence_text: dict[str, str] = {}
    for _key, turns, _ts in extract_sessions(sample):
        for turn in turns:
            identifier = turn.get("dia_id")
            if identifier:
                evidence_text[str(identifier)] = normalize_answer(
                    f"{turn['speaker']}: {turn['text']}"
                )
    ids: list[str] = []
    for raw in qa.get("evidence") or []:
        ids.extend(part for part in str(raw).replace(";", " ").replace(",", " ").split() if part)
    targets = [evidence_text[identifier] for identifier in ids if identifier in evidence_text]
    if not targets:
        return None
    blob = normalize_answer("\n".join(f"- {fact}" for fact in facts))
    return any(target in blob for target in targets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-convos", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=60, help="per conversation")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--budget-tokens", type=int, default=None)
    parser.add_argument("--dsn", type=str, default=None, help="use an existing PostgreSQL")
    parser.add_argument("--tenant-id", type=str, default="parity")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        import pgserver  # noqa: F401
        import psycopg  # noqa: F401
    except ImportError as exc:
        print(
            f"SKIP: the PostgreSQL extra is not installed ({exc}). "
            'Install with pip install "infinitydb-ai[postgres]" pgserver.'
        )
        return 0

    server = None
    if args.dsn:
        dsn = args.dsn
    else:
        import pgserver

        workdir = tempfile.mkdtemp(prefix="infinitydb-parity-")
        server = pgserver.get_server(workdir)
        dsn = server.get_uri()

    try:
        dataset = _load(args.dataset, args.max_convos)
        sqlite = Runner("sqlite", InfinityDB(db_path=":memory:"))
        postgres = Runner("postgres", InfinityDB(dsn=dsn, tenant_id=args.tenant_id))

        questions = 0
        evidence_equal = 0
        evidence_total = 0
        identical_payloads = 0
        hits: dict[str, int] = {"sqlite": 0, "postgres": 0}
        regressions: list[dict[str, Any]] = []
        differing: list[dict[str, Any]] = []

        for index, sample in enumerate(dataset):
            user = f"parity_{index}"
            sqlite.ingest(sample, user)
            postgres.ingest(sample, user)

            local_questions = 0
            for qa in sample.get("qa", []):
                if int(qa.get("category", 0)) == 5:
                    continue
                if local_questions >= args.max_questions:
                    break
                question = str(qa.get("question", "")).strip()
                gold = str(qa.get("answer", "")).strip()
                if not question or not gold:
                    continue
                local_questions += 1
                questions += 1

                a = sqlite.recall(question, user, args.top_k, args.budget_tokens)
                b = postgres.recall(question, user, args.top_k, args.budget_tokens)
                facts_a = [item["fact"] for item in a]
                facts_b = [item["fact"] for item in b]
                ids_a = [item["id"] for item in a]
                ids_b = [item["id"] for item in b]
                if facts_a == facts_b and ids_a == ids_b:
                    identical_payloads += 1

                hit_a = _evidence_hits(sample, qa, facts_a)
                hit_b = _evidence_hits(sample, qa, facts_b)
                if hit_a is not None and hit_b is not None:
                    evidence_total += 1
                    hits["sqlite"] += int(hit_a)
                    hits["postgres"] += int(hit_b)
                    if hit_a == hit_b:
                        evidence_equal += 1
                    else:
                        entry = {
                            "conversation": index,
                            "question": question,
                            "sqlite_hit": hit_a,
                            "postgres_hit": hit_b,
                        }
                        differing.append(entry)
                        if hit_a and not hit_b:
                            # Postgres losing evidence SQLite found is the only
                            # direction that would be a real portability defect.
                            regressions.append(entry)

        def recall_evidence(runner: Runner) -> dict[str, float]:
            return {
                "p50": round(percentile(runner.recall_ms, 50), 2),
                "p95": round(percentile(runner.recall_ms, 95), 2),
                "p99": round(percentile(runner.recall_ms, 99), 2),
            }

        try:
            dataset_label = str(args.dataset.resolve().relative_to(REPO_ROOT))
        except ValueError:
            dataset_label = str(args.dataset)
        report = {
            "dataset": dataset_label,
            "settings": {
                "max_convos": args.max_convos,
                "max_questions_per_convo": args.max_questions,
                "top_k": args.top_k,
                "budget_tokens": args.budget_tokens,
            },
            "questions": questions,
            "identical_payloads": identical_payloads,
            "identical_payload_rate_pct": round(
                100.0 * identical_payloads / questions, 2
            )
            if questions
            else None,
            "evidence_recall_at_k_pct": {
                "sqlite": round(100.0 * hits["sqlite"] / evidence_total, 2)
                if evidence_total
                else None,
                "postgres": round(100.0 * hits["postgres"] / evidence_total, 2)
                if evidence_total
                else None,
            },
            "evidence_regressions": regressions,
            "evidence_decisions_equal": evidence_equal,
            "evidence_questions": evidence_total,
            "evidence_decision_parity_pct": round(100.0 * evidence_equal / evidence_total, 2)
            if evidence_total
            else None,
            "differing": differing[:20],
            "latency_ms": {
                "sqlite": recall_evidence(sqlite),
                "postgres": recall_evidence(postgres),
            },
            "ingest_p50_ms": {
                "sqlite": round(statistics.median(sqlite.ingest_ms), 2)
                if sqlite.ingest_ms
                else None,
                "postgres": round(statistics.median(postgres.ingest_ms), 2)
                if postgres.ingest_ms
                else None,
            },
        }

        sqlite.db.close()
        postgres.db.close()
        print(json.dumps(report, indent=2))
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        if evidence_total == 0:
            print("FAIL: no question carried usable evidence ids", file=sys.stderr)
            return 1
        sqlite_recall = 100.0 * hits["sqlite"] / evidence_total
        postgres_recall = 100.0 * hits["postgres"] / evidence_total
        gains = len(differing) - len(regressions)
        print(
            f"evidence recall@k: SQLite {sqlite_recall:.2f}% vs PostgreSQL "
            f"{postgres_recall:.2f}%; per-question agreement "
            f"{evidence_equal}/{evidence_total}; PostgreSQL gained {gains}, "
            f"lost {len(regressions)}; identical returned memories "
            f"{identical_payloads}/{questions}."
        )
        if postgres_recall < sqlite_recall:
            print(
                "FAIL: PostgreSQL's evidence recall is below SQLite's on this run",
                file=sys.stderr,
            )
            return 1
        print("PASS: the PostgreSQL backend does not lower evidence recall.")
        return 0
    finally:
        if server is not None:
            server.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
