"""Adapter contract shared by every memory system in the arena.

Each system is driven by `worker.py` in its own virtualenv so pinned dependency
sets cannot interfere with each other. An adapter only has to:

    reset()                       drop any previous state for this conversation
    ingest(session)               store one session of dialogue turns
    search(question, top_k)       return the memory strings a reader would see

Everything about latency, LLM calls and token use is measured outside the
adapter by the arena (see ``cost_proxy.py`` and ``worker.py``).
"""
from __future__ import annotations

from typing import Any, Protocol

from ..locomo import Session


class MemorySystem(Protocol):
    name: str

    def reset(self) -> None: ...

    def ingest(self, session: Session) -> None: ...

    def search(self, question: str, top_k: int) -> list[str]: ...

    def close(self) -> None: ...


def turn_line(turn) -> str:
    """The textual form of a turn handed to a memory system."""
    return f"{turn.speaker}: {turn.text}"


def session_text(session: Session) -> str:
    return "\n".join(turn_line(t) for t in session.turns)


class BaseSystem:
    """Small helper base class: subclasses implement ``_reset``/``_ingest``/``_search``."""

    name = "base"

    def __init__(self, workdir: str, conversation_id: str, **options: Any) -> None:
        self.workdir = workdir
        self.conversation_id = conversation_id
        self.options = options

    def reset(self) -> None:
        self._reset()

    def ingest(self, session: Session) -> None:
        self._ingest(session)

    def search(self, question: str, top_k: int = 10) -> list[str]:
        return self._search(question, top_k)

    def close(self) -> None:
        self._close()

    def _reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _ingest(self, session: Session) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _search(self, question: str, top_k: int) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def _close(self) -> None:  # pragma: no cover - interface
        pass
