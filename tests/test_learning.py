"""Feedback loop: recording outcomes and letting them reorder recall.

These tests cover the mechanism only. No retrieval-quality claim is made here:
the benchmark in `benchmarks/arena` still runs with `learned_boost=0`, so every
published number is unaffected by anything in this file.
"""

from infinitydb.core.engine import InfinityDB


def _ids(db, user_id="u"):
    return {r["fact"]: r["id"] for r in db.conn.execute(
        "SELECT id, fact FROM l1_atoms WHERE user_id = ?", (user_id,)
    )}


def test_feedback_is_recorded_and_summarised():
    db = InfinityDB(":memory:")
    db.add("Deploys run from the build server", user_id="u")
    atom = _ids(db)["Deploys run from the build server"]

    db.record_feedback("how do deploys run", atom, user_id="u", relevant=True)
    db.record_feedback("how do deploys run", atom, user_id="u", relevant=False)

    summary = db.feedback_summary("u")
    assert summary["events"] == 2
    assert summary["relevant"] == 1
    assert summary["irrelevant"] == 1
    assert summary["queries"] == 1
    assert summary["precision"] == 0.5


def test_feedback_requires_a_real_query_and_atom():
    db = InfinityDB(":memory:")
    for query, atom in (("", "L1_x"), ("q", " ")):
        try:
            db.record_feedback(query, atom, user_id="u")
        except ValueError:
            continue
        raise AssertionError("empty query or atom id must be rejected")


def test_learned_weights_need_repeated_support():
    db = InfinityDB(":memory:")
    db.add("The staging cluster lives in Frankfurt", user_id="u")
    atom = _ids(db)["The staging cluster lives in Frankfurt"]

    db.record_feedback("where is staging", atom, user_id="u")
    # A term association needs two acceptances (min_support defaults to 2)...
    assert db.learned_term_weights("u") == {}
    # ...but a label naming this exact atom is direct evidence, so one is enough.
    assert db.learned_atom_weights("u")[atom] == 1.0

    db.record_feedback("where is staging", atom, user_id="u")
    terms = db.learned_term_weights("u")
    assert "frankfurt" in terms
    assert 0 < terms["frankfurt"] <= 1.0
    assert db.learned_atom_weights("u")[atom] == 1.0


def test_feedback_reorders_recall_only_when_boost_is_requested():
    db = InfinityDB(":memory:")
    db.add("Release notes live in the wiki", user_id="u", ts=1000.0)
    db.add("Release notes live in the repo", user_id="u", ts=2000.0)
    ids = _ids(db)
    older = ids["Release notes live in the wiki"]

    baseline = [r["fact"] for r in db.recall("release notes", user_id="u", n=2, now=2000.0)["results"]]
    assert baseline[0] == "Release notes live in the repo"

    for _ in range(2):
        db.record_feedback("release notes", older, user_id="u")

    boosted = [
        r["fact"]
        for r in db.recall("release notes", user_id="u", n=2, now=2000.0, learned_boost=1.0)["results"]
    ]
    assert boosted[0] == "Release notes live in the wiki"

    # The default path is untouched: no boost, no reordering.
    again = [r["fact"] for r in db.recall("release notes", user_id="u", n=2, now=2000.0)["results"]]
    assert again == baseline


def test_feedback_cannot_rescue_a_fact_with_no_shared_term():
    db = InfinityDB(":memory:")
    db.add("The coffee machine is on floor three", user_id="u")
    atom = _ids(db)["The coffee machine is on floor three"]
    for _ in range(3):
        db.record_feedback("deployment region", atom, user_id="u")

    results = db.recall("deployment region", user_id="u", n=3, learned_boost=1.0)["results"]
    assert results == []


def test_clear_feedback_keeps_memories():
    db = InfinityDB(":memory:")
    db.add("Backups are encrypted at rest", user_id="u")
    atom = _ids(db)["Backups are encrypted at rest"]
    db.record_feedback("are backups encrypted", atom, user_id="u")

    assert db.clear_feedback("u") == 1
    assert db.feedback_summary("u")["events"] == 0
    assert atom in _ids(db).values()
