"""OpenAI Agents SDK tools backed by InfinityDB.

Install the optional framework package with: pip install openai-agents
The SDK import is deferred until create_openai_tools() is called.
"""
from __future__ import annotations
from typing import Any
from integrations._common import recall_payload, recall_turns, store_turn


def build_openai_tool_functions(memory: Any, *, user_id: str = "default",
                                session_id: str | None = None,
                                limit: int = 5) -> tuple[Any, Any]:
    """Build plain typed callables suitable for OpenAI Agents function tools."""
    def save_memory(content: str, event_session_id: str = "") -> dict[str, str]:
        """Save a durable fact, decision, or conversation detail.

        Args:
            content: Information to retain in long-term memory.
            event_session_id: Optional session override for this event.
        """
        log_id = store_turn(memory, content, user_id=user_id,
                            session_id=event_session_id or session_id)
        return {"status": "stored", "log_id": log_id}

    def recall_memory(query: str, event_session_id: str = "") -> dict[str, Any]:
        """Retrieve relevant dated memories and applicable policies.

        Args:
            query: Natural-language question about previously stored information.
            event_session_id: Optional session scope; empty searches user scope.
        """
        result = recall_turns(memory, query, user_id=user_id,
                              session_id=event_session_id or session_id,
                              limit=limit)
        return recall_payload(result)
    return save_memory, recall_memory


def create_openai_tools(memory: Any, *, user_id: str = "default",
                        session_id: str | None = None,
                        limit: int = 5) -> list[Any]:
    """Return FunctionTools that can be passed to Agent(tools=[...])."""
    try:
        from agents import function_tool
    except ImportError as exc:
        raise ImportError(
            "Install openai-agents to use create_openai_tools()."
        ) from exc
    save_memory, recall_memory = build_openai_tool_functions(
        memory, user_id=user_id, session_id=session_id, limit=limit
    )
    return [function_tool(save_memory), function_tool(recall_memory)]


__all__ = ["build_openai_tool_functions", "create_openai_tools"]
