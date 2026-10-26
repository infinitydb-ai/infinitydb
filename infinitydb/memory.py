"""InfinityDB High-Level Memory Interface.

Provides a drop-in 3-line API for autonomous AI agents:
- Zero external LLM token extraction costs ($0)
- Sub-millisecond local SQLite WAL recall (<1ms)
- Causal episode tracking and conflict auto-resolution
- Cryptographic proof verification
"""

from __future__ import annotations

from typing import Any

from infinitydb.core.engine import InfinityDB, Memory as ItemMemory


class AgentMemory:
    """High-level Agent Memory engine with sub-millisecond retrieval."""

    def __init__(
        self,
        db_path: str = ":memory:",
        auto_embed: bool = True,
        auto_extract: bool = True,
        **kwargs: Any,
    ) -> None:
        self.db = InfinityDB(db_path=db_path)
        self.auto_embed = auto_embed
        self.auto_extract = auto_extract

    def add(
        self,
        content: str,
        user_id: str = "default",
        session_id: str | None = None,
        importance: float = 0.9,
        **kwargs: Any,
    ) -> str:
        """Store conversational transcript or observation and extract atomic facts."""
        return self.db.add(
            conversation=content,
            user_id=user_id,
            session_id=session_id,
            importance=importance,
        )

    def search(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
        **kwargs: Any,
    ) -> list[str]:
        """Recall relevant compressed facts using hybrid BM25 and temporal decay scoring."""
        res = self.db.recall(query=query, n=limit, user_id=user_id)
        results = res.get("results", [])
        return [item["fact"] for item in results]

    def recall(
        self,
        query: str,
        n: int = 5,
        user_id: str | None = None,
        include_policies: bool = True,
    ) -> dict[str, Any]:
        """Full hybrid recall returning facts, causal graph, and enterprise policies."""
        return self.db.recall(
            query=query,
            n=n,
            user_id=user_id,
            include_policies=include_policies,
        )

    def get_all(self, user_id: str = "default", limit: int = 100) -> list[str]:
        """Retrieve all active persona facts for a user."""
        persona = self.db.get_persona(user_id=user_id)
        return [f"{k}: {v}" for k, v in persona.items()]

    def delete(self, user_id: str) -> dict[str, int]:
        """GDPR compliant cryptographic wipe of all memories for a user."""
        return self.db.delete_user(user_id=user_id)

    def stats(self) -> dict[str, Any]:
        """Return statistics across L0-L6 memory layers."""
        return self.db.stats()


# Alias for intuitive drop-in usage
Memory = AgentMemory

__all__ = ["AgentMemory", "Memory", "ItemMemory"]
