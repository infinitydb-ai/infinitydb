#!/usr/bin/env python3
"""100% local, zero-API-key LoCoMo benchmark.

Compares InfinityDB against optionally-installed local baselines:

* ``mem0_nollm``     Mem0 OSS with ``infer=False``: no LLM, FastEmbed + local Qdrant
* ``mem0_local``     Mem0 with an Ollama LLM + Ollama embedder + local Qdrant
* ``cognee_local``   Cognee with Ollama LLM + embeddings
* ``tencentdb_local`` a TencentDB-style flat vector baseline (local pgvector DSN,
  or a SQLite dense-vector emulation when no database is running)

InfinityDB itself needs no service, no model and no key: ``--engines infinitydb``
runs on a bare Python install. The other adapters are only imported when
selected, so their dependencies never affect the default path.

Measures, per engine:

1. ingestion latency per session (P50 / P95 / P99 ms)
2. retrieval latency per query (P50 / P95 / P99 ms)
3. retrieved items and context tokens vs. the full-conversation baseline
   (token savings)
4. context token-overlap F1, and optionally an Ollama answer + judge score
5. for ``infinitydb``, the same row at each ``--budget-tokens`` value, so recall
   can be compared with another system at a matched token cost

The F1 column is overlap between retrieved text and the gold answer, not reader
accuracy. Treat it as a relative signal only; reader accuracy needs the judge.

Usage::

    # InfinityDB alone: no Ollama, no keys, no extra packages
    python benchmarks/locomo_local_compare.py --engines infinitydb --max-convos 5

    # Full local comparison (requires Ollama + mem0ai + cognee)
    python benchmarks/locomo_local_compare.py \\
        --engines infinitydb mem0_local cognee_local tencentdb_local \\
        --max-convos 5 --use-local-judge

Results are written to ``benchmarks/locomo-local-results.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import statistics
import string
import sys
import tempfile
import time
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infinitydb.core.engine import InfinityDB  # noqa: E402

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
# The arena already vendors the dataset, so a normal run needs no network at all.
VENDORED_DATASET = REPO_ROOT / "benchmarks" / "arena" / "data" / "locomo10.json"
DEFAULT_DATASET = VENDORED_DATASET if VENDORED_DATASET.exists() else (
    REPO_ROOT / "benchmarks" / ".cache" / "locomo10.json"
)


# ---------------------------------------------------------------------------
# Tokenization and scoring
# ---------------------------------------------------------------------------

_TOKEN_COUNTER = "chars/4 estimate"


def _detect_token_counter() -> None:
    global _TOKEN_COUNTER
    try:
        import tiktoken  # noqa: F401

        _TOKEN_COUNTER = "tiktoken/cl100k_base"
    except ImportError:
        _TOKEN_COUNTER = "chars/4 estimate"


_detect_token_counter()


def count_tokens(text: str) -> int:
    """Token count for the context columns.

    Uses tiktoken's cl100k_base when installed, otherwise a chars/4 estimate.
    The counter actually used is recorded in the report's ``environment`` block,
    because an estimate must not be presented as a tokenizer measurement.
    """
    if _TOKEN_COUNTER.startswith("tiktoken"):
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    return max(1, math.ceil(len(text) / 4))


def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((p / 100.0) * len(ordered)) - 1))
    return round(ordered[idx], 4)


def local_ollama_judge(
    question: str,
    retrieved_context: str,
    ground_truth: str,
    ollama_url: str = "http://localhost:11434",
    model: str = "llama3.2:3b",
) -> tuple[str, float]:
    """Answer and grade locally through Ollama. Returns ("", 0.0) on failure."""

    def _chat(system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "stream": False,
                "options": {"temperature": 0.0},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{ollama_url.rstrip('/')}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "").strip()

    try:
        prediction = _chat(
            "Answer the question concisely using ONLY the provided memory context. "
            "If the answer is missing, reply 'Insufficient information'.",
            f"Memory Context:\n{retrieved_context}\n\nQuestion: {question}",
        )
        verdict = _chat(
            "Compare the predicted answer with the gold answer for the question. "
            "Reply ONLY 'CORRECT' or 'WRONG'.",
            f"Question: {question}\nGold Answer: {ground_truth}\nPredicted Answer: {prediction}",
        ).upper()
    except Exception as exc:  # a judge outage must not abort a measured run
        print(f"    ! local judge unavailable: {exc}", file=sys.stderr)
        return "", 0.0
    score = 1.0 if "CORRECT" in verdict and "WRONG" not in verdict else 0.0
    return prediction, score


# ---------------------------------------------------------------------------
# Engine adapters
# ---------------------------------------------------------------------------


class MemoryAdapter(ABC):
    name: str

    @abstractmethod
    def ingest_session(
        self,
        user_id: str,
        session_id: str,
        turns: list[dict[str, str]],
        ts: float | None = None,
    ) -> float:
        """Ingest one conversation session; return elapsed milliseconds.

        ``ts`` is when the session happened. Systems that store event time use
        it; the others ignore it.
        """

    @abstractmethod
    def retrieve(self, user_id: str, query: str, limit: int = 5) -> tuple[list[str], float]:
        """Return (retrieved memory strings, elapsed milliseconds)."""

    def close(self) -> None:
        pass


class InfinityDBAdapter(MemoryAdapter):
    """Zero-dependency InfinityDB adapter: one SQLite file, no model, no key."""

    name = "infinitydb"

    def __init__(self, db_path: str, budget_tokens: int | None = None) -> None:
        self.db = InfinityDB(db_path=db_path)
        self._now: float | None = None
        # ``recall(budget_tokens=...)`` packs the highest-ranked memories into a
        # word budget, so the same engine at different budgets is a different
        # row on the recall/token curve. Label it as such.
        self.budget_tokens = budget_tokens
        if budget_tokens is not None:
            self.name = f"infinitydb_budget_{budget_tokens}"

    def ingest_session(
        self,
        user_id: str,
        session_id: str,
        turns: list[dict[str, str]],
        ts: float | None = None,
    ) -> float:
        # One ``add`` per turn, as in the published arena adapter: turn-window
        # assembly scores each turn with its neighbours, so a whole transcript
        # added as a single memory merges unrelated topics into one chunk and
        # wrecks precision.
        t0 = time.perf_counter()
        for turn in turns:
            self.db.add(
                conversation=f"{turn['speaker']}: {turn['text']}",
                user_id=user_id,
                session_id=session_id,
                ts=ts,
                auto_supersede=False,
            )
            if ts is not None:
                self._now = max(self._now or ts, ts)
        return (time.perf_counter() - t0) * 1000.0

    def retrieve(self, user_id: str, query: str, limit: int = 5) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        res = self.db.recall(
            query=query,
            n=limit,
            user_id=user_id,
            include_policies=False,
            # Evaluate at the moment the question is asked, so L5 decay scores
            # event time instead of today's wall clock.
            now=self._now,
            # Measuring must not mutate stored memories.
            reinforce=False,
            budget_tokens=self.budget_tokens,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        facts = []
        for item in res.get("results", []):
            if "fact" not in item:
                continue
            # A reader needs the event date to answer "when" questions, which is
            # how the published arena rows render memories.
            day = datetime.fromtimestamp(item["ts"]).strftime("%d %B %Y") if item.get("ts") else ""
            facts.append(f"[{day}] {item['fact']}" if day else item["fact"])
        return facts, elapsed_ms

    def close(self) -> None:
        self.db.close()


class Mem0LocalAdapter(MemoryAdapter):
    """Mem0 with an Ollama LLM + Ollama embedder + local Qdrant."""

    name = "mem0_local"

    def __init__(self, storage_dir: str, llm_model: str, embed_model: str, ollama_url: str) -> None:
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise RuntimeError("Install mem0ai (`pip install mem0ai`) to run mem0_local.") from exc

        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": llm_model,
                    "temperature": 0.0,
                    "max_tokens": 1500,
                    "ollama_base_url": ollama_url,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": embed_model,
                    "ollama_base_url": ollama_url,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "locomo_mem0_local",
                    "path": storage_dir,
                },
            },
        }
        self.client = Memory.from_config(config)

    def ingest_session(
        self,
        user_id: str,
        session_id: str,
        turns: list[dict[str, str]],
        ts: float | None = None,
    ) -> float:
        messages = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"{t['speaker']}: {t['text']}",
            }
            for i, t in enumerate(turns)
        ]
        t0 = time.perf_counter()
        self.client.add(messages, user_id=user_id)
        return (time.perf_counter() - t0) * 1000.0

    def retrieve(self, user_id: str, query: str, limit: int = 5) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        res = self.client.search(query=query, user_id=user_id, limit=limit)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        items = res.get("results", res) if isinstance(res, dict) else res
        facts = [m["memory"] for m in items or [] if isinstance(m, dict) and "memory" in m]
        return facts, elapsed_ms


class Mem0NoLlmAdapter(MemoryAdapter):
    """Mem0 OSS in its documented ``infer=False`` mode: no LLM anywhere.

    Each turn is stored as written, embedded locally with FastEmbed, and
    retrieved by cosine similarity from an on-disk Qdrant collection. This is
    the only Mem0 configuration that is comparable to a zero-LLM write path;
    ``mem0_local`` (Ollama extraction) is a different product behaviour and is
    measured separately.

    mem0 is imported lazily, so its dependencies stay optional. Install the
    arena's pinned set (``benchmarks/arena/requirements-mem0.txt``) in a venv
    that also has this repository on ``PYTHONPATH``.
    """

    name = "mem0_nollm"
    embed_model = os.environ.get("MEM0_EMBED_MODEL", "thenlper/gte-large")
    embed_dims = int(os.environ.get("MEM0_EMBED_DIMS", "1024"))

    def __init__(self, workdir: str) -> None:
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise RuntimeError(
                "Install mem0's arena pins (`pip install -r benchmarks/arena/requirements-mem0.txt`) "
                "in a venv to run mem0_nollm."
            ) from exc

        qdrant_path = os.path.join(workdir, "qdrant")
        os.makedirs(qdrant_path, exist_ok=True)
        config = {
            # mem0's schema requires an llm block even when infer=False; this
            # points at a closed port so an accidental call fails loudly.
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "unused-infer-false",
                    "ollama_base_url": "http://127.0.0.1:1",
                    "temperature": 0,
                },
            },
            "embedder": {"provider": "fastembed", "config": {"model": self.embed_model}},
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": qdrant_path,
                    "collection_name": "locomo_local",
                    "embedding_model_dims": self.embed_dims,
                    "on_disk": True,
                },
            },
            "history_db_path": os.path.join(workdir, "mem0_history.db"),
        }
        self.memory = Memory.from_config(config)

    def ingest_session(
        self,
        user_id: str,
        session_id: str,
        turns: list[dict[str, str]],
        ts: float | None = None,
    ) -> float:
        # mem0 OSS rejects a ``timestamp`` argument, so the event date goes into
        # the memory text, the same way InfinityDB's adapter renders it.
        day = datetime.fromtimestamp(ts).strftime("%d %B %Y") if ts is not None else ""
        t0 = time.perf_counter()
        for turn in turns:
            content = f"{turn['speaker']}: {turn['text']}"
            self.memory.add(
                [{"role": "user", "content": f"[{day}] {content}" if day else content}],
                user_id=user_id,
                infer=False,
                metadata={"speaker": turn["speaker"]},
            )
        return (time.perf_counter() - t0) * 1000.0

    def retrieve(self, user_id: str, query: str, limit: int = 5) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        # mem0's kwarg is ``top_k``; ``limit`` is silently ignored and mem0 then
        # returns its own default number of memories.
        result = self.memory.search(query, filters={"user_id": user_id}, top_k=limit)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return [item["memory"] for item in result.get("results", [])], elapsed_ms


class CogneeLocalAdapter(MemoryAdapter):
    """Cognee graph+vector memory, fully local through Ollama."""

    name = "cognee_local"

    def __init__(self, llm_model: str, embed_model: str, ollama_url: str) -> None:
        try:
            import cognee
        except ImportError as exc:
            raise RuntimeError("Install cognee (`pip install cognee`) to run cognee_local.") from exc

        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["LLM_MODEL"] = llm_model
        os.environ["LLM_ENDPOINT"] = f"{ollama_url.rstrip('/')}/v1"
        os.environ["LLM_API_KEY"] = "ollama"
        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        os.environ["EMBEDDING_MODEL"] = embed_model
        os.environ["EMBEDDING_ENDPOINT"] = f"{ollama_url.rstrip('/')}/api/embeddings"
        os.environ["EMBEDDING_DIMENSIONS"] = "768"
        self.cognee = cognee
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.cognee.prune.prune_data())
        self.loop.run_until_complete(self.cognee.prune.prune_system(metadata=True))

    def ingest_session(
        self,
        user_id: str,
        session_id: str,
        turns: list[dict[str, str]],
        ts: float | None = None,
    ) -> float:
        transcript = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
        t0 = time.perf_counter()
        self.loop.run_until_complete(self.cognee.add(transcript, dataset_name=user_id))
        self.loop.run_until_complete(self.cognee.cognify(datasets=[user_id]))
        return (time.perf_counter() - t0) * 1000.0

    def retrieve(self, user_id: str, query: str, limit: int = 5) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        results = self.loop.run_until_complete(self.cognee.search(query_text=query, top_k=limit))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        facts = [str(r) for r in (results or [])[:limit]]
        return facts, elapsed_ms

    def close(self) -> None:
        self.loop.close()


class TencentDBLocalVectorAdapter(MemoryAdapter):
    """Flat-vector baseline in the shape of a TencentDB vector table.

    Uses a pgvector DSN when ``TENCENTDB_DSN`` is set (PostgreSQL is the
    MySQL-compatible route TencentDB exposes), otherwise a local SQLite dense
    store that ranks by cosine similarity. The SQLite path is an emulation of
    flat vector retrieval, not the TencentDB product; the report records which
    backend was used.
    """

    name = "tencentdb_local"

    def __init__(self, db_path: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers (`pip install sentence-transformers`) for tencentdb_local."
            ) from exc

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.pg_dsn = os.environ.get("TENCENTDB_DSN")
        self.backend = "pgvector" if self.pg_dsn else "sqlite-flat-vector-emulation"
        if self.pg_dsn:
            import psycopg

            self.conn = psycopg.connect(self.pg_dsn, autocommit=True)
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tencentdb_memories (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector(384)
                    );
                    """
                )
        else:
            self.conn = sqlite3.connect(db_path)
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tencentdb_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                );
                """
            )
            self.conn.commit()

    def ingest_session(
        self,
        user_id: str,
        session_id: str,
        turns: list[dict[str, str]],
        ts: float | None = None,
    ) -> float:
        chunks = [f"{t['speaker']}: {t['text']}" for t in turns if t.get("text")]
        if not chunks:
            return 0.0
        t0 = time.perf_counter()
        vectors = self.encoder.encode(chunks, normalize_embeddings=True).tolist()
        if self.pg_dsn:
            with self.conn.cursor() as cur:
                for chunk, vec in zip(chunks, vectors):
                    cur.execute(
                        "INSERT INTO tencentdb_memories "
                        "(user_id, session_id, content, embedding) VALUES (%s, %s, %s, %s)",
                        (user_id, session_id, chunk, str(vec)),
                    )
        else:
            self.conn.executemany(
                "INSERT INTO tencentdb_memories "
                "(user_id, session_id, content, embedding_json) VALUES (?, ?, ?, ?)",
                [(user_id, session_id, c, json.dumps(v)) for c, v in zip(chunks, vectors)],
            )
            self.conn.commit()
        return (time.perf_counter() - t0) * 1000.0

    def retrieve(self, user_id: str, query: str, limit: int = 5) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        q_vec = self.encoder.encode([query], normalize_embeddings=True)[0].tolist()
        if self.pg_dsn:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content FROM tencentdb_memories
                    WHERE user_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (user_id, str(q_vec), limit),
                )
                facts = [row[0] for row in cur.fetchall()]
        else:
            rows = self.conn.execute(
                "SELECT content, embedding_json FROM tencentdb_memories WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            scored = [
                (sum(a * b for a, b in zip(q_vec, json.loads(emb))), content)
                for content, emb in rows
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            facts = [c for _, c in scored[:limit]]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return facts, elapsed_ms

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_locomo_dataset(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading LoCoMo dataset from {LOCOMO_URL} ...")
        urllib.request.urlretrieve(LOCOMO_URL, path)
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


_SESSION_TIME_RE = re.compile(r"^\s*(\d{1,2}:\d{2}\s*[ap]m)\s+on\s+(\d{1,2}\s+\w+,\s*\d{4})\s*$")


def parse_session_time(raw: str) -> float | None:
    """Parse LoCoMo's ``1:56 pm on 8 May, 2023`` into a Unix timestamp."""
    m = _SESSION_TIME_RE.match(raw or "")
    if not m:
        return None
    clock = m.group(1).upper().replace(" ", "")
    date = m.group(2).replace(" ", "")
    try:
        return datetime.strptime(f"{clock} {date}", "%I:%M%p %d%B,%Y").timestamp()
    except ValueError:
        return None


def extract_sessions(
    sample: dict[str, Any],
) -> list[tuple[str, list[dict[str, str]], float | None]]:
    conv = sample.get("conversation", {})
    keys = sorted(
        (k for k in conv if re.fullmatch(r"session_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )
    out = []
    for k in keys:
        turns = conv.get(k)
        if isinstance(turns, list) and turns:
            out.append((k, turns, parse_session_time(conv.get(f"{k}_date_time", ""))))
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class EngineMetrics:
    engine: str
    conversations_evaluated: int = 0
    sessions_ingested: int = 0
    queries_evaluated: int = 0
    ingest_latencies_ms: list[float] = field(default_factory=list)
    recall_latencies_ms: list[float] = field(default_factory=list)
    retrieved_tokens_per_query: list[int] = field(default_factory=list)
    returned_items_per_query: list[int] = field(default_factory=list)
    full_context_tokens_per_query: list[int] = field(default_factory=list)
    retrieval_f1_scores: list[float] = field(default_factory=list)
    retrieval_containment: list[float] = field(default_factory=list)
    evidence_hits: list[float] = field(default_factory=list)
    qa_f1_scores: list[float] = field(default_factory=list)
    llm_judge_scores: list[float] = field(default_factory=list)
    per_question_logs: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        avg_ret = statistics.mean(self.retrieved_tokens_per_query) if self.retrieved_tokens_per_query else 0.0
        avg_full = statistics.mean(self.full_context_tokens_per_query) if self.full_context_tokens_per_query else 1.0
        savings_pct = max(0.0, (1.0 - (avg_ret / avg_full)) * 100.0) if avg_full else 0.0
        out: dict[str, Any] = {
            "engine": self.engine,
            "conversations_evaluated": self.conversations_evaluated,
            "sessions_ingested": self.sessions_ingested,
            "queries_evaluated": self.queries_evaluated,
            "ingestion_latency_ms": {
                "p50": percentile(self.ingest_latencies_ms, 50),
                "p95": percentile(self.ingest_latencies_ms, 95),
                "p99": percentile(self.ingest_latencies_ms, 99),
                "mean": round(statistics.mean(self.ingest_latencies_ms), 4) if self.ingest_latencies_ms else 0.0,
            },
            "retrieval_latency_ms": {
                "p50": percentile(self.recall_latencies_ms, 50),
                "p95": percentile(self.recall_latencies_ms, 95),
                "p99": percentile(self.recall_latencies_ms, 99),
                "mean": round(statistics.mean(self.recall_latencies_ms), 4) if self.recall_latencies_ms else 0.0,
            },
            "tokens": {
                "mean_returned_items": round(
                    statistics.mean(self.returned_items_per_query), 2
                )
                if self.returned_items_per_query
                else 0.0,
                "mean_full_context_tokens": round(avg_full, 1),
                "mean_retrieved_tokens": round(avg_ret, 1),
                "token_savings_pct": round(savings_pct, 2),
                "api_cost_usd": 0.0,
            },
            "accuracy": {
                # Token overlap between retrieved text and the gold answer.
                "context_token_overlap_f1_pct": round(
                    statistics.mean(self.retrieval_f1_scores) * 100.0, 2
                )
                if self.retrieval_f1_scores
                else 0.0,
                # Gold answer string found anywhere in the retrieved context.
                "answer_contained_pct": round(
                    statistics.mean(self.retrieval_containment) * 100.0, 2
                )
                if self.retrieval_containment
                else 0.0,
                # Does the retrieved context contain the dialogue turn that
                # justifies the answer? This is LoCoMo's evidence recall, the
                # metric the arena reports, so a row here can be checked
                # against a published arena row for the same system.
                "evidence_recall_at_k_pct": round(
                    statistics.mean(self.evidence_hits) * 100.0, 2
                )
                if self.evidence_hits
                else None,
            },
        }
        if self.qa_f1_scores:
            out["accuracy"]["qa_answer_f1_pct"] = round(statistics.mean(self.qa_f1_scores) * 100.0, 2)
        if self.llm_judge_scores:
            out["accuracy"]["local_llm_judge_accuracy_pct"] = round(
                statistics.mean(self.llm_judge_scores) * 100.0, 2
            )
        return out


def evaluate_engine(
    adapter: MemoryAdapter,
    dataset: list[dict[str, Any]],
    max_convos: int,
    max_qa_per_convo: int,
    top_k: int,
    use_local_judge: bool,
    ollama_url: str,
    judge_model: str,
) -> EngineMetrics:
    metrics = EngineMetrics(engine=adapter.name)
    salt = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]

    for idx, sample in enumerate(dataset[:max_convos]):
        user_id = f"locomo_{adapter.name}_{idx}_{salt}"
        sessions = extract_sessions(sample)

        full_lines: list[str] = []
        for sk, turns, session_ts in sessions:
            elapsed_ms = adapter.ingest_session(
                user_id=user_id,
                session_id=f"{user_id}_{sk}",
                turns=turns,
                ts=session_ts,
            )
            metrics.ingest_latencies_ms.append(elapsed_ms)
            metrics.sessions_ingested += 1
            full_lines.extend(f"{t['speaker']}: {t['text']}" for t in turns)

        full_tokens = count_tokens("\n".join(full_lines))
        # Evidence ids ("D8:6") are turn ids; matching them to retrieved text
        # needs the turn's own words, since chunks are returned as strings.
        evidence_text: dict[str, str] = {}
        for _sk, turns, _ts in sessions:
            for turn in turns:
                dia = turn.get("dia_id")
                if dia:
                    evidence_text[str(dia)] = normalize_answer(
                        f"{turn['speaker']}: {turn['text']}"
                    )
        evaluated = 0

        for qa in sample.get("qa", []):
            if int(qa.get("category", 0)) == 5 or evaluated >= max_qa_per_convo:
                continue  # category 5 is adversarial: unanswerable by design
            question = str(qa.get("question", "")).strip()
            gold = str(qa.get("answer", "")).strip()
            if not question or not gold:
                continue

            facts, recall_ms = adapter.retrieve(user_id=user_id, query=question, limit=top_k)
            blob = "\n".join(f"- {f}" for f in facts)
            ret_tokens = count_tokens(blob) if blob else 0
            ctx_f1 = max([token_f1(f, gold) for f in facts] + [token_f1(blob, gold)], default=0.0)
            # Token-overlap F1 against a short gold answer is harsh on long
            # context; containment answers a different, cruder question: is the
            # gold string present anywhere in what was retrieved at all?
            gold_norm = normalize_answer(gold)
            blob_norm = normalize_answer(blob)
            contained = bool(gold_norm) and gold_norm in blob_norm

            evidence_ids: list[str] = []
            for raw_ev in qa.get("evidence") or []:
                evidence_ids.extend(p for p in re.split(r"[;,\s]+", str(raw_ev)) if p)
            targets = [evidence_text[e] for e in evidence_ids if e in evidence_text]
            evidence_hit = any(t in blob_norm for t in targets) if targets else None

            metrics.recall_latencies_ms.append(recall_ms)
            metrics.retrieved_tokens_per_query.append(ret_tokens)
            # Token budgets buy recall by returning fewer, larger snippets, so
            # the item count has to be published beside the token count.
            metrics.returned_items_per_query.append(len(facts))
            metrics.full_context_tokens_per_query.append(full_tokens)
            metrics.retrieval_f1_scores.append(ctx_f1)
            metrics.retrieval_containment.append(1.0 if contained else 0.0)
            if evidence_hit is not None:
                metrics.evidence_hits.append(1.0 if evidence_hit else 0.0)
            metrics.queries_evaluated += 1
            evaluated += 1

            entry: dict[str, Any] = {
                "conversation_idx": idx,
                "question": question,
                "gold_answer": gold,
                "recall_latency_ms": round(recall_ms, 4),
                "retrieved_tokens": ret_tokens,
                "context_f1": round(ctx_f1, 4),
                "answer_contained": contained,
                "evidence_hit": evidence_hit,
            }

            if use_local_judge:
                pred, judge_score = local_ollama_judge(
                    question=question,
                    retrieved_context=blob,
                    ground_truth=gold,
                    ollama_url=ollama_url,
                    model=judge_model,
                )
                metrics.qa_f1_scores.append(token_f1(pred, gold))
                metrics.llm_judge_scores.append(judge_score)
                entry["predicted_answer"] = pred
                entry["local_judge_score"] = judge_score

            metrics.per_question_logs.append(entry)
        metrics.conversations_evaluated += 1

    return metrics


def build_adapter(
    name: str,
    tmpdir: str,
    args: argparse.Namespace,
    budget_tokens: int | None = None,
) -> MemoryAdapter:
    if name == "infinitydb":
        suffix = f"_{budget_tokens}" if budget_tokens is not None else ""
        return InfinityDBAdapter(os.path.join(tmpdir, f"infinity{suffix}.db"), budget_tokens)
    if name == "mem0_local":
        return Mem0LocalAdapter(
            storage_dir=os.path.join(tmpdir, "mem0_qdrant"),
            llm_model=args.llm_model,
            embed_model=args.embed_model,
            ollama_url=args.ollama_url,
        )
    if name == "mem0_nollm":
        return Mem0NoLlmAdapter(os.path.join(tmpdir, "mem0_nollm"))
    if name == "cognee_local":
        return CogneeLocalAdapter(
            llm_model=args.llm_model,
            embed_model=args.embed_model,
            ollama_url=args.ollama_url,
        )
    if name == "tencentdb_local":
        return TencentDBLocalVectorAdapter(os.path.join(tmpdir, "tencentdb_local.db"))
    raise ValueError(f"Unknown engine: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="100% local, zero-API-key LoCoMo benchmark.")
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["infinitydb"],
        choices=["infinitydb", "mem0_local", "mem0_nollm", "cognee_local", "tencentdb_local"],
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-convos", type=int, default=3)
    parser.add_argument("--max-qa", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--llm-model", type=str, default="llama3.2:3b")
    parser.add_argument("--embed-model", type=str, default="nomic-embed-text")
    parser.add_argument(
        "--budget-tokens",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Run InfinityDB with recall(budget_tokens=N) for each N. The engine enforces "
            "this as a word budget, so measured tiktoken tokens per question are reported "
            "and must be compared against another system's measured tokens, not against N."
        ),
    )
    parser.add_argument("--use-local-judge", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "locomo-local-results.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort instead of recording an engine whose optional dependency is missing.",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Write only the per-engine summaries, not every per-question record.",
    )
    args = parser.parse_args()

    dataset, sha256 = load_locomo_dataset(args.dataset)
    summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    logs: dict[str, list[dict[str, Any]]] = {}

    with tempfile.TemporaryDirectory(prefix="infinitydb_local_bench_") as tmpdir:
        # A budget sweep only makes sense for the engine that has a budget, so
        # expand the InfinityDB row and leave the other engines at one run each.
        runs: list[tuple[str, int | None]] = []
        for name in args.engines:
            if name == "infinitydb" and args.budget_tokens:
                # Keep the unbudgeted row so the curve has a measured endpoint.
                runs.append((name, None))
                runs.extend((name, budget) for budget in args.budget_tokens)
            else:
                runs.append((name, None))

        for name, budget in runs:
            label = name if budget is None else f"{name} (budget_tokens={budget})"
            print(f"\n>>> Running local benchmark for: {label}")
            try:
                adapter = build_adapter(name, tmpdir, args, budget)
            except Exception as exc:
                if args.strict:
                    raise
                print(f"    ! skipped {label}: {exc}", file=sys.stderr)
                skipped.append({"engine": label, "reason": str(exc)})
                continue

            try:
                res = evaluate_engine(
                    adapter=adapter,
                    dataset=dataset,
                    max_convos=args.max_convos,
                    max_qa_per_convo=args.max_qa,
                    top_k=args.top_k,
                    use_local_judge=args.use_local_judge,
                    ollama_url=args.ollama_url,
                    judge_model=args.llm_model,
                )
                summary = res.summary()
                # Baselines that can run on more than one store say which one ran.
                backend = getattr(adapter, "backend", None)
                if backend:
                    summary["backend"] = backend
                if budget is not None:
                    summary["recall_budget_tokens"] = budget
                summaries.append(summary)
                logs[adapter.name] = res.per_question_logs
                print(json.dumps(summary, indent=2))
            finally:
                adapter.close()

    report = {
        "benchmark": "locomo-local",
        "dataset": str(args.dataset.relative_to(REPO_ROOT)) if args.dataset.is_relative_to(REPO_ROOT) else str(args.dataset),
        "dataset_sha256": sha256,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "token_counter": _TOKEN_COUNTER,
            "local_llm_model": args.llm_model if args.use_local_judge else None,
            "local_embed_model": args.embed_model,
        },
        "settings": {
            "max_convos": args.max_convos,
            "max_qa_per_convo": args.max_qa,
            "top_k": args.top_k,
            "category_5_excluded": True,
            "recall_budget_tokens": args.budget_tokens,
        },
        "notes": [
            "evidence_recall_at_k_pct is the only accuracy column comparable to the LoCoMo arena rows.",
            "answer_contained_pct and context_token_overlap_f1_pct are string tests against the gold answer; they punish long retrieved context and are not reader accuracy.",
            "Ingestion latency is per session, with turns added one at a time.",
            "Category 5 (adversarial) questions are excluded.",
            "tencentdb_local is a flat dense-vector baseline in a TencentDB-shaped table, not the TencentDB product.",
        ],
        "summary": summaries,
        "skipped_engines": skipped,
        "per_example_logs": {} if args.no_logs else logs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved local benchmark report to: {args.output}")


if __name__ == "__main__":
    main()
