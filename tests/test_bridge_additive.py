"""The additive bridge API is the one mechanism that improved multi-hop.

Phase 2f: witnesses appended after the retrieved head gained +0.0228, while the
same witnesses displacing retrieved turns lost 0.0254. These tests hold the API
to that shape: recall results are preserved, witnesses are extras, never swaps.
"""
from __future__ import annotations

from infinitydb import InfinityDB
from infinitydb.bridge import (NullBridge, PredicateBridge, SubjectBridge, candidates,
                               entities_in_question, recall_with_bridges, witness_texts)

DIALOGUE = [
    ("Caroline: I relocated to Lisbon in March 2024 and I work at a ceramics studio.",
     1709251200.0),
    ("Caroline: My budget for the new cluster is forty thousand euros.", 1711929600.0),
    ("Tom: I work at a data engineering consultancy.", 1714521600.0),
]


def _db() -> InfinityDB:
    db = InfinityDB(":memory:")
    for text, ts in DIALOGUE:
        db.add(text, user_id="u", ts=ts, auto_supersede=False)
    return db


def test_entities_and_candidates_come_from_named_speakers():
    db = _db()
    try:
        named = [e.lower() for e in entities_in_question(db, "Where does Caroline live?", "u")]
        assert "caroline" in named
        assert candidates(db, "Where does Caroline live?", "u"), "speaker edges must be reachable"
        # A question naming nobody yields no candidates rather than all edges.
        assert candidates(db, "Where does the user live?", "u") == []
    finally:
        db.close()


def test_witnesses_are_added_on_top_not_swapped_in():
    db = _db()
    try:
        plain = db.recall("Where does Caroline live?", n=3, user_id="u", reinforce=False)
        bridged = recall_with_bridges(db, "Where does Caroline live?", user_id="u", n=3,
                                      reinforce=False)
        assert bridged["bridge_witnesses"], "a named speaker should produce witnesses"
        assert len(bridged["results"]) == len(plain["results"]) + len(bridged["bridge_witnesses"])
        # The retrieved head is untouched; witnesses only extend the tail.
        assert [r["fact"] for r in bridged["results"][:len(plain["results"])]] == \
               [r["fact"] for r in plain["results"]]
        assert all(r.get("source") == "bridge" for r in bridged["results"][-len(bridged["bridge_witnesses"]):])
    finally:
        db.close()


def test_selector_choice_is_respected():
    db = _db()
    try:
        nothing = recall_with_bridges(db, "Where does Caroline live?", user_id="u", n=2,
                                      selector=NullBridge(), reinforce=False)
        assert nothing["bridge_witnesses"] == []
        assert nothing["bridges"] == []

        strict = recall_with_bridges(db, "Where does Caroline live?", user_id="u", n=2,
                                     selector=PredicateBridge(max_bridges=2), reinforce=False)
        assert isinstance(strict["bridge_witnesses"], list)
    finally:
        db.close()


def test_witness_limit_and_deduplication():
    db = _db()
    try:
        picked = candidates(db, "Where does Caroline live?", "u")
        texts = witness_texts(db, picked, limit=1)
        assert len(texts) == 1
        # The same atom twice must not produce the same sentence twice.
        assert len(witness_texts(db, list(picked) * 2, limit=50)) == len(
            witness_texts(db, picked, limit=50))
    finally:
        db.close()


def test_subject_selector_default_caps_witnesses():
    db = _db()
    try:
        result = recall_with_bridges(db, "What did Caroline mention about budget?",
                                     user_id="u", n=2, max_bridges=1, reinforce=False)
        assert len(result["bridge_witnesses"]) <= 1
        assert result["selector"] == SubjectBridge().name
    finally:
        db.close()
