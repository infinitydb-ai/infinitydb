"""Typed relation edges, contradiction detection, graph queries, extraction."""

from __future__ import annotations

import pytest

from infinitydb.core.engine import InfinityDB
from infinitydb.extraction import parse_facts, openai_extractor, ollama_extractor


@pytest.fixture()
def db():
    database = InfinityDB(":memory:")
    yield database
    database.close()


def test_typed_edges_are_written_and_queryable(db):
    db.add("Caroline: I work at the gym.", user_id="u", session_id="s1", ts=1000.0)
    db.add("Caroline: I moved to Bristol.", user_id="u", session_id="s1", ts=1001.0)

    edges = db.conn.execute(
        "SELECT subject, predicate, object FROM l1_entity_edges ORDER BY ts"
    ).fetchall()
    assert [(row["subject"], row["predicate"]) for row in edges] == [
        ("caroline", "works_at"),
        ("caroline", "lives_in"),
    ]

    direct = db.relations("Caroline", user_id="u", hops=1)
    assert {path["predicate"] for path in direct} == {"works_at", "lives_in"}
    assert all(path["hop"] == 1 for path in direct)


def test_relations_follow_two_hops(db):
    db.add("Caroline: I work at the gym.", user_id="u", ts=1000.0)
    db.add("Tom: I work at the gym.", user_id="u", ts=1001.0)
    db.add("Tom: I moved to Leeds.", user_id="u", ts=1002.0)

    paths = db.relations("caroline", user_id="u", hops=2)
    two_hop = [p for p in paths if p["hop"] == 2]
    assert two_hop, "the second hop must follow the object as a subject"
    assert {p["subject"] for p in two_hop} == {"tom"}
    assert db.relations("nobody", user_id="u") == []


def test_contradicting_edges_supersede_and_are_auditable(db):
    db.add("Caroline: I work at the gym.", user_id="u", session_id="s1", ts=1000.0)
    db.add("Caroline: I work at the hospital.", user_id="u", session_id="s2", ts=2000.0)

    conflicts = db.contradictions("u")
    assert conflicts, "a changed employer must be recorded as a contradiction"
    assert any(c["category"] == "employer" for c in conflicts)
    assert "gym" in conflicts[0]["old_fact"].lower() or "gym" in conflicts[1]["old_fact"].lower()

    # The superseded turn stops being returned, but history stays queryable.
    current = db.recall("where does Caroline work", n=5, user_id="u", reinforce=False)
    facts = " ".join(item["fact"] for item in current["results"]).lower()
    assert "hospital" in facts
    assert "gym" not in facts
    historical = db.recall(
        "where does Caroline work", n=5, user_id="u", reinforce=False, as_of=1500.0
    )
    assert "gym" in " ".join(item["fact"] for item in historical["results"]).lower()


def test_opposite_polarity_is_a_contradiction(db):
    db.add("User: I love jazz.", user_id="v", ts=10.0)
    db.add("User: I hate jazz.", user_id="v", ts=20.0)
    conflicts = db.contradictions("v")
    assert conflicts and "jazz" in conflicts[0]["old_fact"].lower()


def test_supersession_can_be_disabled_for_multi_party_stores(db):
    db.add("Caroline: I work at the gym.", user_id="u", ts=1000.0, auto_supersede=False)
    db.add("Caroline: I work at the hospital.", user_id="u", ts=2000.0, auto_supersede=False)
    assert db.contradictions("u") == []


def test_persona_summary_is_compact_and_ordered(db):
    db.add("User: I am Maria and I live in Porto.", user_id="w", ts=30.0)
    summary = db.persona_summary("w")
    assert summary.startswith("Profile — ")
    assert "identity: I am Maria" in summary
    assert "location: I live in Porto" in summary
    assert len(summary) < 200
    assert db.persona_summary("nobody") == ""


def test_rebuild_edges_backfills_existing_turns(db):
    db.add("Caroline: I work at the gym.", user_id="u", ts=1000.0)
    db.conn.execute("DELETE FROM l1_entity_edges")
    db.conn.commit()
    assert db.relations("caroline", user_id="u") == []
    assert db.rebuild_edges() >= 1
    assert db.relations("caroline", user_id="u")


def test_parse_facts_handles_model_reply_shapes():
    fenced = '```json\n{"facts": [{"fact": "Caroline works at a gym", "category": "fact", "importance": 0.9}]}\n```'
    assert parse_facts(fenced)[0]["fact"] == "Caroline works at a gym"

    bare = '[{"fact": "Likes jazz", "category": "preference", "importance": 9}]'
    parsed = parse_facts(bare)[0]
    assert parsed["category"] == "preference"
    assert parsed["importance"] == 1.0, "importance is clamped, not rejected"

    assert parse_facts('[{"fact": "x", "category": "Not Snake"}]')[0]["category"] == "fact"
    assert parse_facts('{"facts": []}') == []

    for bad in ("", "no json here"):
        with pytest.raises(ValueError):
            parse_facts(bad)


def test_extractors_are_callables_and_do_not_need_a_key_locally():
    assert callable(ollama_extractor())
    with pytest.raises(RuntimeError):
        openai_extractor(api_key_env="INFINITYDB_TEST_MISSING_KEY")
