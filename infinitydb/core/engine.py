"""Minimal local memory engine for the initial InfinityDB scaffold."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Memory:
    key: str
    content: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class InfinityDB:
    """In-memory key-based store; persistence is not implemented yet."""

    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}

    def put(self, key: str, content: str) -> Memory:
        if not key.strip():
            raise ValueError("key cannot be empty")
        if not content.strip():
            raise ValueError("content cannot be empty")
        memory = Memory(key=key, content=content)
        self._memories[key] = memory
        return memory

    def get(self, key: str) -> Memory | None:
        return self._memories.get(key)

    def delete(self, key: str) -> bool:
        return self._memories.pop(key, None) is not None

    def __len__(self) -> int:
        return len(self._memories)
