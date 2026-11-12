"""Run Graphiti locally on a LoCoMo slice: Ollama LLM + nomic embeddings + embedded Kuzu.

No Docker, no Neo4j, no API key. Emits arena-format JSONL so score.py can grade it
with the same evidence-recall code used for every other row.

    python run.py --conv conv-26 --sessions 1 --questions 5 --probe
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from arena import locomo  # noqa: E402

def field(obj, name):
    """Read a LoCoMo field whether it is a dataclass or a mapping."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

API_KEY = (
    os.environ.get("ARENA_GRAPHITI_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or "ollama"
)
LLM_BASE = (
    os.environ.get("ARENA_GRAPHITI_BASE")
    or os.environ.get("OPENAI_LLM_BASE")
    or "http://127.0.0.1:11434/v1"
)
EMBED_BASE = os.environ.get("ARENA_GRAPHITI_EMBED_BASE") or "http://127.0.0.1:11434/v1"
LLM_MODEL = os.environ.get("ARENA_GRAPHITI_LLM", "llama3.2:3b")
EMBED_MODEL = os.environ.get("ARENA_GRAPHITI_EMBED", "nomic-embed-text")
EMBED_DIM = int(os.environ.get("ARENA_GRAPHITI_EMBED_DIM", "768"))


from graphiti_core.cross_encoder.client import CrossEncoderClient


class PassThroughReranker(CrossEncoderClient):
    """No cross-encoder: Graphiti's default one needs an OpenAI key and a GPU.

    Returning the passages unchanged (equal scores) keeps Graphiti's own fusion
    ordering intact, and the row is labelled "rerank off" because of it.
    """

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(passage, 1.0) for passage in passages]


def build_graph(db_path: str):
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    from openai import AsyncOpenAI

    # The SDK default is a 600 s request timeout, which a CPU-only
    # extraction blow through. Give it an hour.
    llm_client = AsyncOpenAI(api_key=API_KEY, base_url=LLM_BASE, timeout=3600.0, max_retries=0)
    llm = OpenAIGenericClient(
        client=llm_client,
        config=LLMConfig(
            api_key=API_KEY,
            model=LLM_MODEL,
            small_model=LLM_MODEL,
            base_url=LLM_BASE,
            temperature=0.0,
            max_tokens=4096,
        ),
        # Ollama honours json mode; json_schema needs server-side schema support.
        structured_output_mode=os.environ.get("ARENA_GRAPHITI_STRUCT", "json_schema"),
    )
    embed_client = AsyncOpenAI(api_key=API_KEY, base_url=EMBED_BASE, timeout=600.0, max_retries=0)
    embedder = OpenAIEmbedder(
        client=embed_client,
        config=OpenAIEmbedderConfig(
            api_key=API_KEY,
            base_url=EMBED_BASE,
            embedding_model=EMBED_MODEL,
            embedding_dim=EMBED_DIM,
        )
    )
    driver = KuzuDriver(db=db_path)
    return Graphiti(
        graph_driver=driver,
        llm_client=llm,
        embedder=embedder,
        cross_encoder=PassThroughReranker(),
    )


def session_text(session) -> str:
    lines = []
    for turn in session.turns:
        stamp = datetime.fromtimestamp(turn.ts, timezone.utc).strftime("%Y-%m-%d")
        lines.append(f"{stamp} {turn.speaker}: {turn.text}")
    return "\n".join(lines)


async def run(args) -> None:
    from graphiti_core.nodes import EpisodeType

    conv = locomo.by_id(locomo.load(), args.conv)
    sessions = conv.sessions[: args.sessions]
    qa = list(conv.qa)[: args.questions]
    print(f"conv={args.conv} sessions={len(sessions)} turns="
          f"{sum(len(s.turns) for s in sessions)} questions={len(qa)}")
    print(f"llm={LLM_MODEL} embed={EMBED_MODEL}({EMBED_DIM}) driver=kuzu at {args.db}")

    graph = build_graph(args.db)
    t0 = time.time()
    await graph.build_indices_and_constraints()
    print(f"indices ready in {time.time() - t0:.1f}s")

    chunk = args.episode_turns or 0
    episodes = []
    for session in (sessions if not args.skip_ingest else []):
        turns = list(session.turns)
        if chunk:
            episodes.extend(turns[i:i + chunk] for i in range(0, len(turns), chunk))
        else:
            episodes.append(turns)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    for index, turns in enumerate(episodes, 1):
        start = time.time()
        body = "\n".join(
            f"{datetime.fromtimestamp(turn.ts, timezone.utc).strftime('%Y-%m-%d')} "
            f"{turn.speaker}: {turn.text}"
            for turn in turns
        )
        await graph.add_episode(
            name=f"{args.conv} ep{index}",
            episode_body=body,
            source=EpisodeType.text,
            source_description="LoCoMo conversation turns",
            reference_time=datetime.fromtimestamp(turns[0].ts, timezone.utc),
        )
        print(f"  episode {index}/{len(episodes)}: {len(turns)} turns in "
              f"{time.time() - start:.1f}s (total {time.time() - t0:.1f}s)", flush=True)

    if args.probe:
        question = qa[0]
        start = time.time()
        results = await graph.search(field(question, "question"), num_results=10)
        print(f"\nprobe search {time.time() - start:.1f}s -> {len(results)} results")
        for item in results[:3]:
            print("  -", str(getattr(item, "fact", item))[:110])
        print(f"gold: {question['answer']!r}")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "type": "meta", "system": "graphiti-local", "conv": args.conv,
            "sessions": len(sessions), "turns": sum(len(s.turns) for s in sessions),
            "qa": len(qa), "top_k": 10, "partial": True,
            "llm": LLM_MODEL, "embed": EMBED_MODEL, "driver": "kuzu",
            "episode_turns": args.episode_turns or "whole-session",
        }) + "\n")
        for index, question in enumerate(qa, 1):
            start = time.time()
            try:
                results = await graph.search(field(question, "question"), num_results=10)
                retrieved = [str(getattr(r, "fact", r)) for r in results]
                error = None
            except Exception as exc:  # keep going; the row records the failure
                retrieved, error = [], f"{type(exc).__name__}: {exc}"
            handle.write(json.dumps({
                "type": "qa", "qa_id": field(question, "qa_id") or f"{args.conv}-q{index:03d}",
                "question": field(question, "question"), "answer": field(question, "answer"),
                "category": field(question, "category"),
                "category_name": CATEGORY_NAMES.get(int(field(question, "category") or 0), "unknown"),
                "evidence": field(question, "evidence"),
                "retrieved": retrieved, "ms": round((time.time() - start) * 1000, 2),
                "error": error,
            }) + "\n")
            handle.flush()
            print(f"  q{index}/{len(qa)} {len(retrieved)} hits {time.time() - start:.1f}s", flush=True)
        handle.write(json.dumps({"type": "done"}) + "\n")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--questions", type=int, default=5)
    ap.add_argument("--db", default=".arena/graphiti/kuzu")
    ap.add_argument("--out", default="benchmarks/arena/results/graphiti-local/conv-26.jsonl")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--skip-ingest", action="store_true", help="reuse an existing kuzu db")
    ap.add_argument("--episode-turns", type=int, default=0,
                    help="chunk each session into episodes of N turns (0 = one episode per session)")
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="cap the number of episodes ingested (0 = all)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
