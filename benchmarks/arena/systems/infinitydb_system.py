"""InfinityDB adapter: zero-LLM ingestion, SQLite-backed recall.

Configuration choices, kept identical across every run:

* one ``user_id`` per conversation (the memory belongs to the chat, not to a
  single speaker), matching how the other systems are scoped;
* ``auto_supersede=False``: single-valued supersession assumes one subject per
  user, and a two-speaker transcript has two. Leaving it on silently deletes a
  speaker's earlier fact, which no other system here does.
* each turn is added with its real event time, and recall is evaluated at the
  conversation's final timestamp (the moment the question is asked).
* returned memories are rendered with their event date, because InfinityDB
  stores an event timestamp per atom and a reader needs it for "when" questions.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

from ..locomo import Session
from .base import BaseSystem, turn_line

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from infinitydb.core.engine import InfinityDB  # noqa: E402

# Optional cross-encoder pass (e.g. "Xenova/ms-marco-MiniLM-L-6-v2"). The
# engine core stays dependency-free; the adapter is what plugs a model in.
_RERANK_MODEL = os.environ.get("ARENA_INFINITYDB_RERANK") or ""
_RERANK_CACHE = os.environ.get("ARENA_RERANK_CACHE", "/tmp/fastembed_cache")
_reranker = None

# Optional dense fusion (e.g. "sentence-transformers/all-MiniLM-L6-v2").
# Off unless asked for: the engine ships no model and the published default row
# must stay dependency-free.
_DENSE_MODEL = os.environ.get("ARENA_INFINITYDB_DENSE") or ""
_DENSE_WEIGHT = float(os.environ.get("ARENA_INFINITYDB_DENSE_WEIGHT", "0.3"))
_DENSE_CACHE = None


def _dense() -> tuple | None:
    global _DENSE_CACHE
    if not _DENSE_MODEL:
        return None
    if _DENSE_CACHE is None:
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=_DENSE_MODEL, cache_dir=_RERANK_CACHE)

        def embedder(texts: list[str]) -> list[list[float]]:
            return [list(map(float, v)) for v in model.embed(list(texts))]

        _DENSE_CACHE = (embedder, _DENSE_MODEL)
    return _DENSE_CACHE


def _rerank():
    global _reranker
    if not _RERANK_MODEL:
        return None
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        model = TextCrossEncoder(model_name=_RERANK_MODEL, cache_dir=_RERANK_CACHE)

        def score(query: str, passages: list[str]) -> list[float]:
            return [float(s) for s in model.rerank(query, list(passages))]

        _reranker = score
    return _reranker


class InfinityDBSystem(BaseSystem):
    name = "infinitydb"

    def _reset(self) -> None:
        path = os.path.join(self.workdir, f"{self.conversation_id}.db")
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(path + suffix)
            except FileNotFoundError:
                pass
        self.db = InfinityDB(path)
        self.user_id = self.conversation_id
        self._now = 0.0

    def _ingest(self, session: Session) -> None:
        for turn in session.turns:
            self.db.add(
                turn_line(turn),
                user_id=self.user_id,
                ts=turn.ts,
                auto_supersede=False,
            )
            self._now = max(self._now, turn.ts)

    def _search(self, question: str, top_k: int) -> list[str]:
        dense = _dense()
        if dense is not None:
            # Incremental: only windows without a vector for this model are embedded.
            self.db.index_dense(dense[0], model=dense[1])
        result = self.db.recall(
            question,
            n=top_k,
            user_id=self.user_id,
            now=self._now,
            reinforce=False,
            include_policies=False,
            # The ablation row in the README is produced by setting
            # ARENA_INFINITYDB_ATOMS=1, which scores atoms without their turn
            # window.
            turn_windows=not os.environ.get("ARENA_INFINITYDB_ATOMS"),
            # Equal-context row: cap the returned words at mem0's measured
            # median context size.
            budget_tokens=(
                int(os.environ["ARENA_INFINITYDB_BUDGET"])
                if os.environ.get("ARENA_INFINITYDB_BUDGET")
                else None
            ),
            rerank=_rerank(),
            dense=(dense[0], dense[1], _DENSE_WEIGHT) if dense is not None else None,
        )
        rendered = []
        for item in result["results"]:
            # A reader needs the event date in the same form a person would say
            # it ("8 May 2023"), not an ISO stamp, to answer "when" questions.
            day = datetime.fromtimestamp(item["ts"]).strftime("%d %B %Y")
            rendered.append(f"[{day}] {item['fact']}")
        return rendered

    def _close(self) -> None:
        self.db.close()
