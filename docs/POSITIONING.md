# Launch and positioning

Internal brief for announcements, investor conversations and acquisition
discussions. Every claim here must be traceable to an artifact in this
repository. If a number is not in an artifact, it does not go in a pitch.

## What is defensible today

Source: [`benchmarks/arena`](../benchmarks/arena/README.md), 10 LoCoMo
conversations, 1,986 questions, one host, 2026-11-10.

| Claim | Evidence |
| --- | --- |
| Retrieval quality leads every competitor we could run offline, without an LLM | evidence recall@10 0.780 vs mem0 0.678 vs a free BM25+dense hybrid 0.676; answer-in-context 0.545 vs 0.386 vs 0.389 |
| It is not bought with prompt size | at `budget_tokens=320`, recall leads (0.696 vs 0.678) and answer-in-context is +24% (0.479 vs 0.386) |
| Retrieval survives a crowded store | pooled 10-conversation store: 0.733 unscoped → **0.780** with `session_id`, identical to ten separate databases |
| Ingestion is orders of magnitude cheaper | 7.5 s vs 544.3 s for the same 10 conversations (no LLM extraction, no embedding API) |
| Retrieval latency is an order of magnitude lower | p50 10.7 ms vs 152.1 ms, p95 14.1 ms vs 191.7 ms |
| Semantic fusion is available without an LLM | `dense=` with a local MiniLM: 0.787 recall, 0.549 in-context, 8% fewer context tokens, 29 ms p50 |
| It beats the one competitor that runs without a key | TencentDB Agent Memory 1.0.3, keyword/FTS retriever, same 1,977 questions: 0.5254 vs 0.780 |
| It holds up on the dataset MemPalace publishes against | LongMemEval R@5 0.933 (180-question stratified sample) vs their claimed 96.6% on the full 500 |
| The memory is auditable | SHA-256 proof per fact, verified against the raw source range; `get_proof()` fails on tampering |
| It runs with zero third-party dependencies | `pyproject.toml` has an empty dependency list; SQLite only |

Those five points — cheap ingest, fast local recall, explicit token budget,
provenance, no vendor dependency — are the product story. Lead with them, not
with adjectives.

## What is not established

- **No Zep, TencentDB or Letta comparison.** They require an LLM key plus a
  graph database, or cloud credentials. Do not put them in a slide until
  `benchmarks/arena` has run them and the raw rows are committed. Their shipped
  code has been read, not run: see
  [the competitive teardown](COMPETITIVE_TEARDOWN.md).
- **No answer-accuracy number.** `qa_f1` and `gold_in_answer` are 0 in every
  committed run because no reader model was available. "Answer accuracy" claims
  would be unsupported.
- **One dataset, one host, no held-out split, no independent reproduction.**
  Treat 0.780 as a tuned LoCoMo observation, not an industry benchmark.
- **No valuation, revenue or market-share evidence.** A $200M valuation is a
  company outcome (distribution, contracts, retention), not something a
  benchmark or a repository can establish.
- **No users, stars, revenue or design partners.** None of that exists in this
  repository, so none of it belongs in a pitch.

## Claims to avoid

The repository has already retracted 89.7% PersonaMem accuracy, "4.2 ms vs Zep
142 ms", "81.4% token saving" and a "World Leaderboard" claim
([audit](BENCHMARK_INTEGRITY_AUDIT.md)). Repeating any of them, or saying
"world's #1 memory engine" without a Zep and TencentDB run, would repeat that
mistake and is the fastest way to lose technical credibility on launch day.

Defensible phrasing: *"On the public LoCoMo retrieval benchmark, InfinityDB
recalls more evidence than mem0's no-LLM mode and than a free BM25+dense hybrid,
with ~72× cheaper ingestion and ~14× lower query latency, keeps that lead with
an explicit token budget, and keeps it inside a store holding ten conversations."*

## GitHub launch surface

Stars follow distribution, not code volume, but the repository has to be ready
before distribution:

1. README first screen: one-sentence promise, install line, a runnable example,
   and the measured table above with the link to its methodology. *(present)*
2. A reproducible `pip install` + five-line quickstart that works offline.
3. An honest comparison page: what beats us (mem0 on open-domain questions) and
   why, plus the competitor teardown of what we adopted and what we have not.
4. `benchmarks/arena/run.sh` so a sceptic can reproduce the table themselves.
5. Launch channels: Show HN (technical audience, will check the claims), the
   mem0/Zep/Letta issue trackers ("we ran your package, here are the raw rows"),
   r/LocalLLaMA, and MCP/agent directories (the repo already ships an MCP
   server).
6. A short post that leads with the methodology and the equal-context result,
   not with a ranking.

## Investor and acquisition conversations

What a technical diligence team will ask, and where the answer lives:

| Question | Where |
| --- | --- |
| Can it be reproduced? | `benchmarks/arena/run.sh`, raw JSONL in `results/` |
| Does it need an LLM or a vector DB? | No: zero dependencies, SQLite only |
| What is the cost per 1,000 memories? | ingest 11 s for 10 conversations vs 544 s for mem0 on the same host |
| How does it compare with Zep/TencentDB? | not measured — say so, and offer to run it with access |
| Is the memory auditable/private? | per-fact SHA-256 proof, local-first, no external calls |
| What is the moat? | deterministic extraction + turn-window retrieval + provenance, none of which needs a model provider |

Acquisition interest usually follows a working integration inside someone's
stack, not a benchmark. The concrete next step is a design-partner deployment
(tool logs, support transcripts, codebase memory) with the retrieval numbers
measured on that workload and the raw rows kept.
