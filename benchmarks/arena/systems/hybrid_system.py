"""Hybrid RAG baseline: BM25 + local dense embeddings fused with RRF.

This is the strongest free, offline "memory" stack that does not call an LLM:
every turn is stored verbatim, indexed lexically (BM25, porter-stemmed) and
semantically (FastEmbed ``thenlper/gte-large``, the same local model mem0 uses
by default), and the two rankings are fused with reciprocal rank fusion
(k=60). No API key, no network at query time and no model provider.

It exists so InfinityDB is not only compared with mem0's single-mode cosine
retrieval. Hybrid lexical+dense retrieval is the baseline a well-built RAG
pipeline reaches, and it is a harder bar than plain vector search.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from datetime import datetime

import numpy as np

from ..locomo import Session
from .base import BaseSystem, turn_line

EMBED_MODEL = os.environ.get("ARENA_HYBRID_EMBED_MODEL", "thenlper/gte-large")
CACHE_DIR = os.environ.get("ARENA_HYBRID_EMBED_CACHE", "/tmp/fastembed_cache")
CANDIDATES = int(os.environ.get("ARENA_HYBRID_CANDIDATES", "50"))
RRF_K = 60.0
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Same conservative suffix stripping the lexical index uses."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith(("sses", "shes", "ches", "xes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in TOKEN_RE.findall(text.lower())]


class HybridSystem(BaseSystem):
    name = "hybrid"

    def _reset(self) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=CACHE_DIR)
        self.texts: list[str] = []
        self.tokens: list[Counter] = []
        self.doc_len: list[int] = []
        self.df: Counter = Counter()
        self.vectors: list[np.ndarray] = []
        self._now = 0.0

    def _ingest(self, session: Session) -> None:
        batch: list[str] = []
        for turn in session.turns:
            day = datetime.fromtimestamp(turn.ts).strftime("%Y-%m-%d")
            text = f"[{day}] {turn_line(turn)}"
            toks = _tokens(text)
            self.texts.append(text)
            self.tokens.append(Counter(toks))
            self.doc_len.append(len(toks))
            self.df.update(set(toks))
            batch.append(text)
            self._now = max(self._now, turn.ts)
        for vec in self.model.passage_embed(batch):
            vec = np.asarray(vec, dtype=np.float32)
            self.vectors.append(vec / (np.linalg.norm(vec) or 1.0))

    def _bm25(self, query: str, top_n: int) -> list[int]:
        n_docs = len(self.texts)
        if not n_docs:
            return []
        avg_len = sum(self.doc_len) / n_docs or 1.0
        q_terms = [t for t in _tokens(query) if t in self.df]
        if not q_terms:
            return []
        k1, b = 1.2, 0.75
        idf = {
            t: math.log(1.0 + (n_docs - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for t in q_terms
        }
        scores = np.zeros(n_docs, dtype=np.float32)
        for i, counts in enumerate(self.tokens):
            dl = self.doc_len[i] or 1
            s = 0.0
            for t in q_terms:
                tf = counts.get(t, 0)
                if tf:
                    s += idf[t] * (tf * (k1 + 1.0)) / (
                        tf + k1 * (1.0 - b + b * dl / avg_len)
                    )
            scores[i] = s
        return np.argsort(-scores)[:top_n].tolist()

    def _dense(self, query: str, top_n: int) -> list[int]:
        if not self.vectors:
            return []
        q = next(iter(self.model.query_embed([query])))
        q = np.asarray(q, dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        mat = np.vstack(self.vectors)
        return np.argsort(-(mat @ q))[:top_n].tolist()

    def _search(self, question: str, top_k: int) -> list[str]:
        fused: dict[int, float] = {}
        for ranking in (self._bm25(question, CANDIDATES), self._dense(question, CANDIDATES)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [self.texts[idx] for idx, _ in best]
