"""mem0 OSS adapter: raw-turn memories, local embeddings, no LLM extraction.

This is mem0's documented ``infer=False`` mode: each dialogue turn is stored as
one memory exactly as written, embedded locally with FastEmbed (mem0's own
default FastEmbed model, ``thenlper/gte-large``), and retrieved by cosine
similarity from an on-disk Qdrant collection. No LLM is called at write or read
time, which is what makes the comparison against InfinityDB's zero-LLM path
fair — and what makes it possible to run without an API key.

An Ollama LLM entry is present because mem0's config schema expects one; its
base URL points at a closed port and nothing in the run may call it. The arena
runs mem0 in its own virtualenv (see ``benchmarks/arena/requirements-mem0.txt``)
so its pinned dependencies cannot disturb InfinityDB's.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime

from ..locomo import Session
from .base import BaseSystem, turn_line

EMBED_MODEL = os.environ.get("ARENA_MEM0_EMBED_MODEL", "thenlper/gte-large")
EMBED_DIMS = int(os.environ.get("ARENA_MEM0_EMBED_DIMS", "1024"))


class Mem0System(BaseSystem):
    name = "mem0"

    def _reset(self) -> None:
        path = os.path.join(self.workdir, "qdrant")
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
        from mem0 import Memory

        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "unused-infer-false",
                    "ollama_base_url": "http://127.0.0.1:1",
                    "temperature": 0,
                },
            },
            "embedder": {
                "provider": "fastembed",
                "config": {"model": EMBED_MODEL},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": path,
                    "collection_name": "arena",
                    "embedding_model_dims": EMBED_DIMS,
                    "on_disk": True,
                },
            },
            "history_db_path": os.path.join(self.workdir, "history.db"),
        }
        self.memory = Memory.from_config(config)
        self.user_id = self.conversation_id
        self._now = 0.0

    def _ingest(self, session: Session) -> None:
        for turn in session.turns:
            # mem0 OSS rejects the ``timestamp`` argument, so the event date goes
            # into the memory text. InfinityDB receives the same date through
            # ``ts``; both systems therefore see identical information.
            day = datetime.fromtimestamp(turn.ts).strftime("%Y-%m-%d")
            self.memory.add(
                [{"role": "user", "content": f"[{day}] {turn_line(turn)}"}],
                user_id=self.user_id,
                infer=False,
                metadata={"speaker": turn.speaker, "dia_id": turn.dia_id},
            )
            self._now = max(self._now, turn.ts)

    def _search(self, question: str, top_k: int) -> list[str]:
        # mem0's kwarg is ``top_k``; passing ``limit`` is silently ignored and
        # mem0 then returns its default 20 memories.
        result = self.memory.search(question, filters={"user_id": self.user_id}, top_k=top_k)
        return [item["memory"] for item in result.get("results", [])]

    def _close(self) -> None:
        pass
