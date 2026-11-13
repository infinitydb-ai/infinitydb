"""Unit tests for the shared LongMemEval adapter bridge."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from arena.longmemeval import Entry
from arena.longmemeval_arena import prepare_entry, resolve_session_ids


def test_longmemeval_sessions_map_to_shared_adapter_contract() -> None:
    entry = Entry(
        question_id="q1",
        question_type="single-session-user",
        question="Where did I go?",
        answer="Paris",
        question_ts=100.0,
        session_ids=["s1", "s2"],
        sessions=[
            [{"role": "user", "content": "I went to Paris."}],
            [{"role": "assistant", "content": "You visited Paris."}],
        ],
        session_ts=[10.0, 20.0],
        answer_session_ids=["s1"],
    )
    sessions, line_map = prepare_entry(entry)
    assert len(sessions) == 2
    assert sessions[0].turns[0].speaker == "User"
    assert sessions[1].turns[0].speaker == "Assistant"
    retrieved = ["[08 May 2023] User: I went to Paris."]
    assert resolve_session_ids(retrieved, line_map) == ["s1"]
