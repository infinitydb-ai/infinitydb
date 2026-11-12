"""Integration tests for the MCP tool surface and transcript capture."""

from __future__ import annotations

import json

import pytest

from infinitydb import mcp_server
from infinitydb.capture import capture
from infinitydb.core.engine import InfinityDB


@pytest.fixture()
def server(tmp_path, monkeypatch):
    db = InfinityDB(str(tmp_path / "mcp.db"))
    monkeypatch.setattr(mcp_server, "db", db)
    yield db
    db.close()


def _payload(response: dict) -> dict:
    return json.loads(response["content"][0]["text"])


def test_mcp_tools_round_trip(server):
    added = mcp_server.handle_call_tool(
        "add_memory",
        {"content": "User: I moved to Lisbon and prefer morning calls.", "user_id": "u1", "session_id": "s1"},
    )
    assert _payload(added)["status"] == "ok"

    recalled = _payload(
        mcp_server.handle_call_tool(
            "recall_memory", {"query": "where did I move", "user_id": "u1", "session_id": "s1"}
        )
    )
    assert "Lisbon" in recalled["memories"]
    assert recalled["count"] >= 1
    atom_id = recalled["memories"].split("(")[1].split(")")[0]

    persona = json.dumps(
        _payload(mcp_server.handle_call_tool("get_persona", {"user_id": "u1"}))
    )
    assert "Lisbon" in persona, "a moved-to fact must reach the persona profile"
    assert _payload(mcp_server.handle_call_tool("verify_proof", {"atom_id": atom_id}))
    assert _payload(mcp_server.handle_call_tool("stats", {}))["l1_atom_facts"] >= 1
    assert _payload(
        mcp_server.handle_call_tool("add_policy", {"rule": "Never share keys.", "domain": "security"})
    )["status"] == "ok"

    refused = mcp_server.handle_call_tool("forget", {"user_id": "u1"})
    assert refused["isError"] is True
    erased = _payload(mcp_server.handle_call_tool("forget", {"user_id": "u1", "confirm": True}))
    assert erased, "forget returns per-layer counts"

    assert mcp_server.handle_call_tool("nope", {})["isError"] is True


def test_capture_is_incremental(tmp_path):
    transcript = tmp_path / "session.jsonl"
    rows = [
        {
            "type": "user",
            "message": {"role": "user", "content": "I moved to Lisbon in May."},
            "timestamp": "2026-05-08T09:00:00Z",
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Noted."}]},
            "timestamp": "2026-05-08T09:00:05Z",
        },
        {"type": "summary", "summary": "ignored"},
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    db_path = str(tmp_path / "memory.db")
    state = tmp_path / "state.json"

    first = capture(transcript, db_path, "u", "s", state)
    second = capture(transcript, db_path, "u", "s", state)
    assert first["turns"] == 2
    assert second["turns"] == 0, "a re-run must not duplicate turns"

    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "user",
                    "message": {"role": "user", "content": "I work at the clinic."},
                    "timestamp": "2026-05-09T10:00:00Z",
                }
            )
            + "\n"
        )
    third = capture(transcript, db_path, "u", "s", state)
    assert third["turns"] == 1

    db = InfinityDB(db_path)
    try:
        facts = " ".join(
            item["fact"] for item in db.recall("clinic", n=3, user_id="u", reinforce=False)["results"]
        )
    finally:
        db.close()
    assert "clinic" in facts.lower()
