"""Dependency-free contract tests for optional framework adapters."""
from __future__ import annotations
from typing import Any
import pytest

from integrations._common import recall_payload, recall_turns, store_turn
from integrations.google_adk import create_google_adk_tools
from integrations.langgraph import create_langgraph_memory_nodes
from integrations.openai_agents import build_openai_tool_functions


class FakeEngine:
    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []
        self.last_recall: dict[str, Any] = {}

    def add(self, **kwargs: Any) -> str:
        self.writes.append(kwargs)
        return f"log-{len(self.writes)}"

    def recall(self, **kwargs: Any) -> dict[str, Any]:
        self.last_recall = kwargs
        return {
            "results": [{"id": "fact-1", "fact": "Prefers Python",
                         "ts": 123.0, "score": 0.9}],
            "policies": [{"rule": "Do not expose secrets"}],
        }


class FakeMemory:
    def __init__(self) -> None:
        self.db = FakeEngine()


def test_shared_helpers_scope_and_validate() -> None:
    memory = FakeMemory()
    assert store_turn(memory, "hello", user_id="u1", session_id="s1") == "log-1"
    assert memory.db.writes[0]["user_id"] == "u1"
    assert memory.db.writes[0]["session_id"] == "s1"
    result = recall_turns(memory, "what do I prefer?", user_id="u1", limit=3)
    assert result["results"][0]["fact"] == "Prefers Python"
    assert memory.db.last_recall["reinforce"] is False
    assert recall_payload(result)["count"] == 1
    with pytest.raises(ValueError):
        store_turn(memory, " ", user_id="u1")
    with pytest.raises(ValueError):
        recall_turns(memory, "query", user_id="u1", limit=0)


def test_openai_tool_functions_store_and_recall() -> None:
    memory = FakeMemory()
    save, recall = build_openai_tool_functions(memory, user_id="u2", session_id="thread")
    assert save("Remember Python")["status"] == "stored"
    assert memory.db.writes[0]["user_id"] == "u2"
    assert recall("preferred language")["memories"][0]["fact"] == "Prefers Python"
    assert memory.db.last_recall["session_id"] == "thread"


def test_google_adk_tools_are_typed_callables() -> None:
    memory = FakeMemory()
    save, recall = create_google_adk_tools(memory, user_id="u3")
    assert save("Remember Go")["status"] == "stored"
    assert recall("preferred language")["count"] == 1


def test_langgraph_nodes_inject_and_persist_memory() -> None:
    memory = FakeMemory()
    recall_node, persist_node = create_langgraph_memory_nodes(memory, user_id="u4")
    state = {"messages": [{"role": "user", "content": "Which language do I prefer?"}]}
    update = recall_node(state, {"configurable": {"thread_id": "t-1"}})
    assert "Prefers Python" in update["messages"][0]["content"]
    persist_node(
        {"messages": [{"role": "user", "content": "I prefer Python"},
                      {"role": "assistant", "content": "I will remember that."}]},
        {"configurable": {"thread_id": "t-1"}},
    )
    assert memory.db.writes[-1]["session_id"] == "t-1"
    assert "Assistant:" in memory.db.writes[-1]["conversation"]
