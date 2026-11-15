"""Shared helpers for optional agent-framework adapters."""
from __future__ import annotations
from typing import Any


def engine_for(memory: Any) -> Any:
    """Accept AgentMemory or the lower-level InfinityDB engine."""
    return getattr(memory, "db", memory)


def store_turn(memory: Any, content: str, *, user_id: str,
               session_id: str | None = None, importance: float = 0.9) -> str:
    """Persist a turn through the stable InfinityDB engine API."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty string")
    return engine_for(memory).add(
        conversation=content.strip(), user_id=user_id,
        session_id=session_id, importance=importance,
    )


def recall_turns(memory: Any, query: str, *, user_id: str,
                 session_id: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Run scoped recall without mutating reinforcement counters."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    return engine_for(memory).recall(
        query=query.strip(), n=limit, user_id=user_id,
        session_id=session_id, reinforce=False,
    )


def recall_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe payload suitable for model tool results."""
    memories = [
        {"id": str(item.get("id", "")), "fact": str(item.get("fact", "")),
         "timestamp": item.get("ts"), "score": item.get("score")}
        for item in result.get("results", [])
    ]
    policies = [str(item.get("rule", item)) for item in result.get("policies", [])]
    return {"memories": memories, "policies": policies, "count": len(memories)}
