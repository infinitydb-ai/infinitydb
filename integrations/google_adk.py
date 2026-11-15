"""Google Agent Development Kit tools backed by InfinityDB.

Install the optional framework package with: pip install google-adk
ADK accepts ordinary typed Python functions in an Agent's tools list.
"""
from __future__ import annotations
from typing import Any
from integrations._common import recall_payload, recall_turns, store_turn


def create_google_adk_tools(memory: Any, *, user_id: str = "default",
                            session_id: str | None = None,
                            limit: int = 5) -> list[Any]:
    """Return typed functions ready for google.adk.agents.Agent(tools=...)."""
    def infinitydb_save_memory(content: str, event_session_id: str = "") -> dict[str, str]:
        """Persist a useful fact, decision, or completed task.

        Args:
            content: Text to store in the user's long-term memory.
            event_session_id: Optional session override for this event.
        """
        log_id = store_turn(memory, content, user_id=user_id,
                            session_id=event_session_id or session_id)
        return {"status": "stored", "log_id": log_id}

    def infinitydb_recall_memory(query: str, event_session_id: str = "") -> dict[str, Any]:
        """Recall relevant memories and governance policies.

        Args:
            query: Natural-language description of what the agent needs.
            event_session_id: Optional session scope; empty searches user scope.
        """
        result = recall_turns(memory, query, user_id=user_id,
                              session_id=event_session_id or session_id,
                              limit=limit)
        return recall_payload(result)
    return [infinitydb_save_memory, infinitydb_recall_memory]


__all__ = ["create_google_adk_tools"]
