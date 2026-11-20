"""Consolidated Markdown scorecard for ``benchmarks/reproduce.sh``.

Reads the JSON a reproduction run just wrote and the committed artifacts it is
being compared against, prints one table and writes the same data as JSON for
upload as a CI artifact. Nothing here recomputes a metric: it only reads the
harnesses' own output, so a difference in the scorecard means a difference in
what was measured, not in how it was summarised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    file = Path(path)
    if not file.is_file():
        return None
    return json.loads(file.read_text(encoding="utf-8"))


def longmemeval_metrics(blob: dict[str, Any] | None) -> dict[str, Any]:
    if not blob:
        return {}
    overall = blob.get("results", {}).get("overall", {})
    answerable = overall.get("answerable_retrieval", {})
    abstention = overall.get("abstention_diagnostics", {})
    gate = abstention.get("calibrated_gate") or {}
    return {
        "questions": answerable.get("questions"),
        "session_any_recall_at_k_pct": answerable.get("session_any_recall_at_k_pct"),
        "turn_any_recall_at_k_pct": answerable.get("turn_any_recall_at_k_pct"),
        "mrr": answerable.get("mrr"),
        "tokens_per_query": answerable.get("tokens_per_query"),
        "abstention_recall_pct": gate.get("true_abstention_rate_pct"),
        "false_abstention_pct": gate.get("false_abstention_rate_pct"),
        "abstention_threshold": gate.get("threshold"),
    }


def locomo_metrics(blob: dict[str, Any] | None, engine: str) -> dict[str, Any]:
    if not blob:
        return {}
    for row in blob.get("summary", []):
        if row.get("engine") == engine:
            return {
                "engine": engine,
                "queries": row.get("queries_evaluated"),
                "evidence_recall_at_k_pct": row.get("accuracy", {}).get(
                    "evidence_recall_at_k_pct"
                ),
                "tokens_per_query": row.get("tokens", {}).get("mean_retrieved_tokens"),
            }
    return {}


def row(
    name: str,
    current: Any,
    baseline: Any,
    unit: str,
    higher_is_better: bool = True,
) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.2f}{unit}"
        return f"{value}{unit}"

    if current is None or baseline is None:
        delta = "n/a"
    else:
        diff = float(current) - float(baseline)
        sign = "+" if diff >= 0 else ""
        delta = f"{sign}{diff:.2f}"
        if higher_is_better and diff < 0:
            delta += " (worse)"
    return f"| {name} | {fmt(current)} | {fmt(baseline)} | {delta} |"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lme", required=True)
    parser.add_argument("--baseline-lme", default=None)
    parser.add_argument("--locomo", default=None)
    parser.add_argument("--baseline-locomo", default=None)
    parser.add_argument("--locomo-engine", default="infinitydb")
    parser.add_argument("--locomo-skip", action="store_true")
    parser.add_argument("--suite", default="full")
    parser.add_argument("--elapsed", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    current = {
        "longmemeval": longmemeval_metrics(load(args.lme)),
        "locomo": (
            {}
            if args.locomo_skip
            else locomo_metrics(load(args.locomo), args.locomo_engine)
        ),
    }
    # The committed LoCoMo baseline is chosen to match the configuration.
    baseline_locomo_path = args.baseline_locomo
    if baseline_locomo_path is None and args.locomo_engine == "infinitydb_budget_420":
        baseline_locomo_path = "benchmarks/locomo-equal-token-results.json"
    baseline = {
        "longmemeval": longmemeval_metrics(load(args.baseline_lme)),
        "locomo": (
            {}
            if args.locomo_skip
            else locomo_metrics(load(baseline_locomo_path), args.locomo_engine)
        ),
    }

    full = args.suite == "full"
    env = load(str(Path(args.json).parent / "environment.json")) if args.json else None
    lines = ["## InfinityDB benchmark scorecard", ""]
    if env:
        lines.append(
            f"commit `{str(env.get('commit'))[:12]}` · Python {env.get('python')} · "
            f"{env.get('cpus')} CPUs · suite **{env.get('suite')}** · "
            f"backend **{env.get('backend')}**"
        )
    if args.elapsed:
        lines.append(f"elapsed {args.elapsed}s")
    lines.append("")

    lme = current["longmemeval"]
    if lme:
        lines += [
            f"### LongMemEval-S ({lme.get('questions')} answerable questions)",
            "",
            "| metric | reproduced | committed baseline | delta |",
            "| --- | --- | --- | --- |",
        ]
        if full:
            lines.append(
                row(
                    "session-any recall@10",
                    lme["session_any_recall_at_k_pct"],
                    baseline["longmemeval"].get("session_any_recall_at_k_pct"),
                    "%",
                )
            )
            lines.append(
                row(
                    "turn-any recall@10",
                    lme["turn_any_recall_at_k_pct"],
                    baseline["longmemeval"].get("turn_any_recall_at_k_pct"),
                    "%",
                )
            )
            lines.append(row("MRR", lme["mrr"], baseline["longmemeval"].get("mrr"), ""))
            lines.append(
                row(
                    "abstention recall",
                    lme["abstention_recall_pct"],
                    baseline["longmemeval"].get("abstention_recall_pct"),
                    "%",
                )
            )
            lines.append(
                row(
                    "false abstention",
                    lme["false_abstention_pct"],
                    baseline["longmemeval"].get("false_abstention_pct"),
                    "%",
                    higher_is_better=False,
                )
            )
        lines.append(
            row(
                "tokens/query",
                lme["tokens_per_query"],
                baseline["longmemeval"].get("tokens_per_query"),
                "",
                higher_is_better=False,
            )
        )
        lines.append("")

    locomo = current["locomo"]
    if args.locomo_skip:
        lines += ["### LoCoMo", "", "Skipped: the LoCoMo harness has no PostgreSQL mode.", ""]
    elif locomo:
        lines += [
            f"### LoCoMo ({locomo.get('queries')} questions, `{locomo.get('engine')}`)",
            "",
            "| metric | reproduced | committed baseline | delta |",
            "| --- | --- | --- | --- |",
        ]
        lines.append(
            row(
                "evidence recall@10",
                locomo["evidence_recall_at_k_pct"],
                baseline["locomo"].get("evidence_recall_at_k_pct"),
                "%",
            )
        )
        lines.append(
            row(
                "tokens/query",
                locomo["tokens_per_query"],
                baseline["locomo"].get("tokens_per_query"),
                "",
                higher_is_better=False,
            )
        )
        lines.append("")

    if not full:
        lines += [
            "> Smoke run: question counts differ from the committed baselines, so the",
            "> delta column is not a comparison. Run without `--smoke` for the real one.",
            "",
        ]

    report = "\n".join(lines)
    print(report)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"suite": args.suite, "elapsed_seconds": args.elapsed,
                 "current": current, "baseline": baseline, "markdown": report},
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
