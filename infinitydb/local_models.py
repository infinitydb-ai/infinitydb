"""Optional local ONNX models, for callers who opt into a semantic signal.

The engine itself ships no model and no runtime dependency: ``recall(...)``
accepts any callable embedder or reranker, and this module is one convenient
way to build them. It is the only place that imports ``fastembed``, and nothing
imports it automatically, so ``pip install infinitydb-ai`` still installs zero
third-party packages.

    pip install "infinitydb-ai[onnx]"

    from infinitydb import InfinityDB
    from infinitydb.local_models import local_embedder, local_reranker

    db = InfinityDB("memory.db")
    embed, model = local_embedder()              # BAAI/bge-small-en-v1.5
    db.index_dense(embed, model=model)           # embed each turn window once
    hits = db.recall("where did the user move", user_id="alice",
                     dense=(embed, model, 0.3))

    cross_encoder = local_reranker()             # ms-marco MiniLM cross-encoder
    hits = db.recall("where did the user move", user_id="alice",
                     rerank=cross_encoder)

Both models run on CPU through ONNX Runtime, need no API key and make no
network call after the one-time model download. Neither is on by default,
because both cost latency and the measured trade-off is published in
``benchmarks/README.md``.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Iterable, Sequence

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

_INSTALL_HINT = 'pip install "infinitydb-ai[onnx]"'


def _load(module: str, attribute: str) -> Any:
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised by test stub
        raise ImportError(
            f"{attribute} needs the optional local-model extra: {_INSTALL_HINT}"
        ) from exc
    target = getattr(mod, attribute, None)
    if target is None:  # pragma: no cover - depends on the fastembed version
        raise ImportError(
            f"{module}.{attribute} is missing; upgrade the extra: {_INSTALL_HINT}"
        )
    return target


def local_embedder(
    model: str = DEFAULT_EMBED_MODEL,
    cache_dir: str | None = None,
    threads: int | None = None,
) -> tuple[Callable[[list[str]], list[list[float]]], str]:
    """Return ``(embed, model_name)`` for use with ``index_dense`` / ``recall``.

    ``embed`` takes a list of strings and returns one float vector per string,
    which is exactly what ``InfinityDB.index_dense`` and ``recall(dense=...)``
    expect. The model name is returned with it because the index is keyed by
    model: switching models means re-embedding, not mixing vector spaces.
    """
    TextEmbedding = _load("fastembed", "TextEmbedding")
    engine = TextEmbedding(model_name=model, cache_dir=cache_dir, threads=threads)

    def embed(texts: list[str]) -> list[list[float]]:
        return [[float(value) for value in vector] for vector in engine.embed(list(texts))]

    return embed, model


def local_reranker(
    model: str = DEFAULT_RERANK_MODEL,
    cache_dir: str | None = None,
    threads: int | None = None,
) -> Callable[[str, Sequence[str]], list[float]]:
    """Return ``rerank(query, passages) -> scores`` for ``recall(rerank=...)``.

    Higher is better; one score per passage, in input order. The scores are the
    cross-encoder's logits, not probabilities — only their order matters to the
    engine.
    """
    TextCrossEncoder = _load("fastembed.rerank.cross_encoder", "TextCrossEncoder")
    engine = TextCrossEncoder(model_name=model, cache_dir=cache_dir, threads=threads)

    def rerank(query: str, passages: Sequence[str]) -> list[float]:
        documents = [str(passage) for passage in passages]
        if not documents:
            return []
        return [float(score) for score in engine.rerank(query, documents)]

    return rerank


def local_dense_index(
    db: Any,
    model: str = DEFAULT_EMBED_MODEL,
    cache_dir: str | None = None,
    threads: int | None = None,
    **scope: Any,
) -> tuple[Callable[[list[str]], list[list[float]]], str]:
    """Build an embedder and index every turn window in ``db`` with it.

    Convenience for the common case; pass ``user_id`` / ``session_id`` through
    ``**scope`` to index only that scope. Returns the ``(embed, model)`` pair to
    hand to ``recall(dense=(embed, model, weight))``.
    """
    embed, name = local_embedder(model=model, cache_dir=cache_dir, threads=threads)
    db.index_dense(embed, model=name, **scope)
    return embed, name


__all__ = [
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_RERANK_MODEL",
    "local_dense_index",
    "local_embedder",
    "local_reranker",
]
