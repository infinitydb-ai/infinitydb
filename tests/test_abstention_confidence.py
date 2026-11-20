from __future__ import annotations

from infinitydb.core.confidence import (
    calibrate_reranker_logit,
    compute_corpus_idf,
    compute_lexical_confidence,
    score_candidates_confidence,
)
from infinitydb.core.engine import InfinityDB


def test_lexical_confidence_distinguishes_answerable_vs_abstention() -> None:
    corpus = [
        "User bought a Breville Bambino espresso machine in March.",
        "User lives in Bengaluru and works on distributed databases.",
        "User adopted a golden retriever named Bruno.",
        "User prefers dark roast coffee beans from Chikmagalur.",
    ]
    idf = compute_corpus_idf(corpus)

    # Answerable query: all high-IDF terms ("breville", "espresso", "machine") match
    ans_conf = compute_lexical_confidence(
        query="What espresso machine did I buy?",
        candidate_text=corpus[0],
        corpus_idf=idf,
        n_docs=len(corpus),
    )
    assert ans_conf >= 0.70

    # Unanswerable (_abs) query: shares generic word ("bought") but missing anchor ("telescope")
    abs_conf = compute_lexical_confidence(
        query="What brand of telescope did I buy?",
        candidate_text=corpus[0],
        corpus_idf=idf,
        n_docs=len(corpus),
    )
    assert abs_conf < 0.30
    assert ans_conf > abs_conf + 0.40


def test_sigmoid_reranker_calibration_monotonic_and_bounded() -> None:
    low = calibrate_reranker_logit(-5.0)
    mid = calibrate_reranker_logit(-1.5)
    high = calibrate_reranker_logit(3.0)

    assert 0.0 <= low < 0.20
    assert abs(mid - 0.50) < 1e-5
    assert 0.85 < high <= 1.0


def test_score_candidates_confidence_filters_when_min_score_set() -> None:
    corpus = [
        "User flew to Tokyo for a conference in October.",
        "User takes Spanish lessons every Tuesday evening.",
    ]
    # Unanswerable query about scuba diving certification
    candidates = [
        {"atom_id": "a1", "fact": corpus[0], "score": 4.2},
        {"atom_id": "a2", "fact": corpus[1], "score": 3.1},
    ]

    # Default mode: preserves results for backward compatibility, sets abstain_recommended=True
    kept, meta = score_candidates_confidence(
        query="When did I get my scuba diving certification?",
        candidates=[dict(c) for c in candidates],
        corpus_documents=corpus,
        has_reranker=False,
    )
    assert len(kept) == 2
    assert meta["abstain_recommended"] is True
    assert meta["confidence"] < meta["confidence_threshold"]

    # Gated mode (`abstain_on_low_confidence=True` or explicit `min_score`): returns empty list
    filtered, meta_gated = score_candidates_confidence(
        query="When did I get my scuba diving certification?",
        candidates=[dict(c) for c in candidates],
        corpus_documents=corpus,
        has_reranker=False,
        abstain_on_low_confidence=True,
    )
    assert filtered == []
    assert meta_gated["abstain_recommended"] is True


def test_engine_recall_abstention_end_to_end(tmp_path) -> None:
    db = InfinityDB(db_path=tmp_path / "abstain_test.db")
    try:
        db.add("I booked a flight to Kyoto for November 12.", user_id="u_abs", session_id="s1")
        db.add("My favorite programming language is Rust.", user_id="u_abs", session_id="s2")

        # Answerable query should not recommend abstention
        ok_res = db.recall(
            query="Where did I book a flight to?",
            n=5,
            user_id="u_abs",
            abstain_on_low_confidence=True,
            reinforce=False,
        )
        assert ok_res["abstain_recommended"] is False
        assert len(ok_res["results"]) >= 1
        assert "Kyoto" in ok_res["results"][0]["fact"]

        # Unanswerable query should recommend abstention and return [] when gated
        abs_res = db.recall(
            query="What is my cat's veterinarian's phone number?",
            n=5,
            user_id="u_abs",
            abstain_on_low_confidence=True,
            reinforce=False,
        )
        assert abs_res["abstain_recommended"] is True
        assert abs_res["results"] == []
    finally:
        db.close()
