"""Unit tests for the benchmark scorer; no dataset or model downloads required."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from arena.score import answer_in_context, evidence_hit, f1, percentile  # noqa: E402


def test_exact_answer_f1() -> None:
    assert f1("Paris", "Paris") == 1.0


def test_month_name_normalization() -> None:
    assert f1("May 2023", "5 2023") == 1.0


def test_answer_in_context_requires_all_content_tokens() -> None:
    assert answer_in_context(["The trip was to Paris in May."], "Paris May")
    assert not answer_in_context(["The trip was to Paris."], "Paris May")


def test_evidence_recall_exact_turn() -> None:
    turn = "Speaker A: I visited Paris last summer."
    assert evidence_hit([turn], [turn]) == 1.0


def test_evidence_recall_empty_annotations_is_nan() -> None:
    import math
    assert math.isnan(evidence_hit(["some memory"], []))


def test_nearest_rank_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0
