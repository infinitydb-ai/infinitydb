"""Bridge selectors: the veto interface for multi-hop experiments.

These are unit tests of the selectors only. The multi-hop measurement lives in
`benchmarks/arena/bridge_probe.py`, which reproduces the published 0.4939
baseline with the null selector before any comparison is trusted.
"""

from infinitydb.bridge import AllBridge, Bridge, NullBridge, PredicateBridge


def _edges():
    return [
        Bridge("caroline", "works_at", "the gym", "L1_a"),
        Bridge("caroline", "lives_in", "Bristol", "L1_b"),
        Bridge("melanie", "studied_at", "Oxford", "L1_c"),
        Bridge("caroline", "likes", "pottery", "L1_d"),
    ]


def test_null_bridge_accepts_nothing():
    assert NullBridge().select("where does Caroline work?", _edges()) == []


def test_all_bridge_dedupes_and_caps():
    duplicates = _edges() + [Bridge("melanie", "works_at", "the gym", "L1_e")]
    chosen = AllBridge(max_bridges=2).select("anything", duplicates)
    assert len(chosen) == 2
    assert chosen[0].object == "the gym"


def test_predicate_bridge_matches_the_question_aspect():
    chosen = PredicateBridge().select("Where does Caroline live?", _edges())
    assert [b.predicate for b in chosen] == ["lives_in"]
    assert chosen[0].object == "Bristol"


def test_predicate_bridge_accepts_nothing_without_an_aspect():
    assert PredicateBridge().select("What did Caroline research?", _edges()) == []


def test_predicate_bridge_rejects_an_object_the_question_already_mentions():
    assert PredicateBridge().select("Is Caroline based in Bristol?", _edges()) == []


def test_predicate_bridge_caps_bridges():
    edges = [Bridge("caroline", "works_at", f"company{i}", f"L1_{i}") for i in range(6)]
    assert len(PredicateBridge(max_bridges=3).select("where does Caroline work?", edges)) == 3
