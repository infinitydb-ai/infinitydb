"""Recency tie-break ordering inside a relevance band.

The tie-break exists to fix knowledge-update ordering without the damage a
blanket recency multiplier causes (see ``_RECENCY_TIE_BAND`` in the engine).
These tests pin the two properties that make it safe: a clearly more relevant
chunk still wins, and only near-ties are re-ordered.
"""

from __future__ import annotations

from infinitydb.core.engine import _recency_band_order


def _chunk(fact: str, ts: float, raw: float, ce: float | None = None) -> dict[str, object]:
    item: dict[str, object] = {"fact": fact, "ts": ts, "_raw": raw}
    if ce is not None:
        item["_ce"] = ce
    return item


def test_near_tied_chunks_prefer_newest() -> None:
    older = _chunk("older", ts=1_000.0, raw=10.0)
    newer = _chunk("newer", ts=2_000.0, raw=9.5)

    ordered = _recency_band_order([older, newer])

    assert [item["fact"] for item in ordered] == ["newer", "older"]


def test_clearly_more_relevant_chunk_keeps_its_rank() -> None:
    older = _chunk("older", ts=1_000.0, raw=10.0)
    newer = _chunk("newer", ts=2_000.0, raw=1.0)

    ordered = _recency_band_order([older, newer])

    assert [item["fact"] for item in ordered] == ["older", "newer"]


def test_band_is_measured_against_the_run_leader() -> None:
    # 4.4 is outside the band of 10.0, so it starts a new group; 2.2 is inside
    # the band of 4.4 and is therefore re-ordered against it.
    a = _chunk("a", ts=1_000.0, raw=10.0)
    b = _chunk("b", ts=3_000.0, raw=4.4)
    c = _chunk("c", ts=2_000.0, raw=2.2)

    ordered = _recency_band_order([a, b, c])

    assert [item["fact"] for item in ordered] == ["a", "b", "c"]


def test_band_of_one_only_reorders_exact_ties() -> None:
    older = _chunk("older", ts=1_000.0, raw=10.0)
    exact_tie = _chunk("tie", ts=2_000.0, raw=10.0)
    just_below = _chunk("below", ts=3_000.0, raw=9.9)

    ordered = _recency_band_order([older, exact_tie, just_below], band=1.0)

    assert [item["fact"] for item in ordered] == ["tie", "older", "below"]


def test_cross_encoder_logits_use_an_additive_window() -> None:
    older = _chunk("older", ts=1_000.0, raw=9.0, ce=4.0)
    near = _chunk("near", ts=2_000.0, raw=9.0, ce=3.6)
    far = _chunk("far", ts=3_000.0, raw=9.0, ce=-4.0)

    ordered = _recency_band_order([older, near, far])

    assert [item["fact"] for item in ordered] == ["near", "older", "far"]


def test_chunks_from_different_rankers_never_share_a_band() -> None:
    # The reranked head and the lexical tail are not comparable scores.
    reranked = _chunk("reranked", ts=1_000.0, raw=10.0, ce=4.0)
    lexical = _chunk("lexical", ts=9_000.0, raw=9.9)

    ordered = _recency_band_order([reranked, lexical])

    assert [item["fact"] for item in ordered] == ["reranked", "lexical"]


def test_single_item_and_empty_inputs_are_returned_unchanged() -> None:
    only = _chunk("only", ts=1_000.0, raw=1.0)
    assert _recency_band_order([]) == []
    assert _recency_band_order([only]) == [only]
