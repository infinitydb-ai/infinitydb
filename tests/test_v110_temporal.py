"""v1.1.0: event time, automatic supersession, tighter extraction, recall speed."""
import time

import pytest

from infinitydb import InfinityDB

DAY = 86400.0
T0 = 1_700_000_000.0


def facts(res):
    return [r["fact"] for r in res["results"]]


def test_newer_single_valued_fact_supersedes_older():
    db = InfinityDB()
    db.add("I live in Paris.", user_id="a", ts=T0)
    db.add("I live in Berlin now.", user_id="a", ts=T0 + 10 * DAY)
    cur = facts(db.recall("where do I live", user_id="a", now=T0 + 11 * DAY))
    assert cur == ["I live in Berlin now"]


def test_as_of_returns_the_fact_that_was_current_then():
    db = InfinityDB()
    db.add("I live in Paris.", user_id="a", ts=T0)
    db.add("I live in Berlin now.", user_id="a", ts=T0 + 10 * DAY)
    past = facts(db.recall("where do I live", user_id="a", as_of=T0 + 5 * DAY))
    assert past == ["I live in Paris"]
    later = facts(db.recall("where do I live", user_id="a", as_of=T0 + 20 * DAY))
    assert later == ["I live in Berlin now"]


def test_late_arriving_older_fact_does_not_become_current():
    db = InfinityDB()
    db.add("I live in Berlin now.", user_id="a", ts=T0 + 10 * DAY)
    db.add("I live in Paris.", user_id="a", ts=T0)            # history imported afterwards
    assert facts(db.recall("where do I live", user_id="a", now=T0 + 11 * DAY)) == ["I live in Berlin now"]
    assert facts(db.recall("where do I live", user_id="a", as_of=T0 + 5 * DAY)) == ["I live in Paris"]


def test_multi_valued_preferences_are_not_superseded():
    db = InfinityDB()
    db.add("I like Python.", user_id="a", ts=T0)
    db.add("I like Rust.", user_id="a", ts=T0 + DAY)
    assert set(facts(db.recall("what do I like", user_id="a", now=T0 + 2 * DAY))) == {"I like Python", "I like Rust"}


def test_auto_supersede_can_be_disabled():
    db = InfinityDB()
    db.add("I live in Paris.", user_id="a", ts=T0, auto_supersede=False)
    db.add("I live in Berlin.", user_id="a", ts=T0 + DAY, auto_supersede=False)
    assert len(facts(db.recall("where do I live", user_id="a", now=T0 + 2 * DAY))) == 2


def test_users_do_not_supersede_each_other():
    db = InfinityDB()
    db.add("I live in Paris.", user_id="a", ts=T0)
    db.add("I live in Berlin.", user_id="b", ts=T0 + DAY)
    assert facts(db.recall("where do I live", user_id="a", now=T0 + 2 * DAY)) == ["I live in Paris"]


def test_very_old_facts_still_rank_newest_first():
    """Decay underflow used to make every old fact score 0 and tie in insertion order."""
    db = InfinityDB()
    db.add("I like Python.", user_id="a", ts=T0, auto_supersede=False)
    db.add("I like Rust.", user_id="a", ts=T0 + 400 * DAY, auto_supersede=False)
    top = facts(db.recall("what do I like", user_id="a", now=T0 + 3000 * DAY))
    assert top[0] == "I like Rust"


@pytest.mark.parametrize("text,category", [
    ("I'm so tired today.", "identity"),
    ("I am a bit confused about this.", "role"),
    ("I am a fan of jazz.", "role"),
])
def test_chatter_is_not_stored_as_identity_or_role(text, category):
    db = InfinityDB()
    db.add(text, user_id="u")
    cats = {r["category"] for r in db.conn.execute("SELECT category FROM l1_atoms")}
    assert category not in cats


@pytest.mark.parametrize("text,category", [
    ("I'm Maria.", "identity"),
    ("My name is Dr Meyer.", "identity"),
    ("I am a software engineer.", "role"),
    ("I moved to Berlin last week.", "location"),
    ("I work at Stripe.", "employer"),
    ("I joined Acme Corp.", "employer"),
    ("My sister lives in Oslo.", "relation"),
])
def test_useful_facts_are_extracted(text, category):
    db = InfinityDB()
    db.add(text, user_id="u")
    cats = {r["category"] for r in db.conn.execute("SELECT category FROM l1_atoms")}
    assert category in cats


def test_moved_to_supersedes_lives_in():
    db = InfinityDB()
    db.add("I live in Paris.", user_id="a", ts=T0)
    db.add("I moved to Madrid.", user_id="a", ts=T0 + 30 * DAY)
    assert facts(db.recall("where do I live", user_id="a", now=T0 + 31 * DAY)) == ["I moved to Madrid"]


def test_proofs_verify_with_event_time_and_detect_tampering():
    db = InfinityDB()
    db.add("I live in Paris.", user_id="a", ts=T0)
    atom = db.conn.execute("SELECT id FROM l1_atoms").fetchone()[0]
    assert db.get_proof(atom)["verified"] is True
    db.conn.execute("UPDATE l1_atoms SET fact='I live in Mars' WHERE id=?", (atom,))
    assert db.get_proof(atom)["verified"] is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "yesterday"])
def test_invalid_event_time_rejected(bad):
    with pytest.raises(ValueError):
        InfinityDB().add("I live in Paris.", ts=bad)


def test_recall_with_rare_term_stays_fast_on_thousands_of_facts():
    db = InfinityDB()
    for i in range(3000):
        db.add(f"I prefer widget{i} for project number {i}.", user_id="u")
    t = time.perf_counter()
    res = db.recall("which one is widget1234", user_id="u", reinforce=False)
    elapsed = (time.perf_counter() - t) * 1000
    assert any("widget1234" in f for f in facts(res))
    assert elapsed < 250          # was ~hundreds of ms because stop-words matched every row


def test_update_tracking_benchmark_is_deterministic_and_correct():
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "update_tracking.py"
    spec = importlib.util.spec_from_file_location("update_tracking", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    a = mod.run(60, 300, 5, True); b = mod.run(60, 300, 5, True)
    assert a["current_answer_first"] == b["current_answer_first"] == 1.0
    assert a["outdated_fact_in_top3"] == 0.0
