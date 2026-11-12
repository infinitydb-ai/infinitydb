"""Relative time expressions must resolve against the event timestamp."""
from datetime import datetime, timedelta, timezone

from infinitydb.core.engine import InfinityDB, normalise_relative_dates

TS = 1683522000.0  # 2023-05-08 UTC


def _day(ts: float):
    return datetime.fromtimestamp(ts, timezone.utc).date()


def test_last_year_becomes_a_year():
    assert normalise_relative_dates("I painted a sunrise last year.", TS) == (
        f"I painted a sunrise {_day(TS).year - 1}."
    )


def test_next_year():
    assert normalise_relative_dates("we move next year", TS) == f"we move {_day(TS).year + 1}"


def test_yesterday_and_today():
    assert normalise_relative_dates("I called yesterday.", TS) == (
        f"I called {(_day(TS) - timedelta(days=1)).isoformat()}."
    )
    assert normalise_relative_dates("I called today.", TS) == f"I called {_day(TS).isoformat()}."


def test_written_numbers_and_units():
    assert normalise_relative_dates("three weeks ago I moved", TS) == (
        f"{(_day(TS) - timedelta(weeks=3)).isoformat()} I moved"
    )
    assert normalise_relative_dates("2 years ago", TS) == str(_day(TS).year - 2)
    assert normalise_relative_dates("a few days ago", TS) == (
        f"{(_day(TS) - timedelta(days=3)).isoformat()}"
    )


def test_duration_phrases_are_left_alone():
    # These are spans, not points in time; rewriting them would be a bug.
    for phrase in ("the last year", "past year", "over the last year", "last summer"):
        text = f"I worked hard {phrase}"
        assert normalise_relative_dates(text, TS) == text


def test_text_without_phrases_is_identical():
    text = "Ana: I work at the hospital in Bristol."
    assert normalise_relative_dates(text, TS) == text


def test_deterministic():
    a = normalise_relative_dates("we met last week", TS)
    b = normalise_relative_dates("we met last week", TS)
    assert a == b and a != "we met last week"


def test_engine_normalises_but_keeps_raw_log(tmp_path):
    db = InfinityDB(str(tmp_path / "t.db"))
    db.add("Ana: I painted a sunrise last year.", user_id="u", ts=TS, normalise_dates=True)
    facts = [row[0] for row in db.conn.execute("SELECT fact FROM l1_atoms")]
    assert any(str(_day(TS).year - 1) in f for f in facts)
    assert all("last year" not in f for f in facts)
    raw_row = db.conn.execute("SELECT * FROM l0_logs").fetchone()
    assert any("last year" in str(value) for value in raw_row), (
        "the raw log must keep the original wording"
    )
