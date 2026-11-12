# Competitor benchmark playbook (verified November 2026)

This is what it actually takes to run Zep and TencentDB Agent Memory against
InfinityDB, and what to measure when you do. It exists because a lot of the
advice circulating online describes a setup that no longer exists, or measures
something that has nothing to do with memory quality.

Read this before spending money or a weekend on a comparison.

## First: what these products are now

**Zep Community Edition is retired.** Zep stopped maintaining and releasing it
in April 2025 and moved its open-source effort to [Graphiti](https://github.com/getzep/graphiti)
(Apache-2.0), the temporal knowledge-graph framework that powers Zep's managed
product. Source: [Zep's own announcement](https://blog.getzep.com/announcing-a-new-direction-for-zeps-open-source-strategy/).
Consequences:

- Docker images published for the old self-hosted Zep server are unmaintained.
  A `docker-compose.yml` that starts `zep-server` / `zep-nlp` from 2023-era
  tags is not a benchmark of anything Zep ships today, and if it starts at all
  it is benchmarking abandoned code.
- The self-hostable engine is **Graphiti**. You supply the graph database
  (Neo4j, FalkorDB, or Amazon Neptune — the Kuzu driver is deprecated), an LLM
  for entity/edge extraction, and an embedder. Graphiti is a framework, not a
  service: ingestion, tenancy and serving are yours to build.
- **Zep Cloud** is the managed product and is measurable with an API key.

**TencentDB Agent Memory is not TencentDB for MySQL.** They are unrelated
products that share a brand. [TencentDB-Agent-Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory)
is a Node-based memory plugin (MIT) with a local `SQLite + sqlite-vec` backend,
a BM25+vector RRF retriever, and an optional Docker/Hermes gateway. It requires
Node ≥ 22 and an **OpenAI-compatible LLM key** (`MODEL_API_KEY` for the Docker
path, `TDAI_LLM_*` for the gateway path). Its published claims are agent-task
deltas (WideSearch, SWE-bench, PersonaMem), not retrieval recall on LoCoMo.

**Sysbench against MySQL is not a memory benchmark.** It measures OLTP
throughput of a relational database. It produces no recall number, no context
quality number, and no memory-system latency — and TencentDB Agent Memory does
not run on MySQL anyway. Presenting QPS from `mysql:8.0` as a competitor
comparison is the kind of row that fails technical diligence in the first
meeting.

## What a memory comparison has to measure

The arena already defines this; competitors plug into the same harness.

| Metric | Why it matters | Produced by |
| :--- | :--- | :--- |
| Evidence recall@10 | did retrieval surface the turns that answer the question | `arena/score.py` |
| Answer-in-context | is the gold answer present in what the model is handed | `arena/score.py` |
| Ingest seconds / conversation | what building the memory costs, per conversation | `arena/worker.py` |
| Recall p50 / p95 | whether memory blocks the generation loop | `arena/worker.py` |
| Median context tokens | whether the win is bought with prompt size | `arena/worker.py` |
| Token budget at equal context | the same comparison at equal cost | `ARENA_INFINITYDB_BUDGET` |

Accuracy first, then cost and latency at equal accuracy. A latency table with no
recall column is not a comparison, it is a microbenchmark.

## Graphiti (the real Zep engine), self-hosted

```bash
# 1. Graph database — FalkorDB is the lightest option
docker run -d --name falkordb -p 6379:6379 falkordb/falkordb:latest

# 2. Python client
python3 -m venv .arena/venvs/graphiti
.arena/venvs/graphiti/bin/pip install graphiti-core

# 3. Credentials (the blocking part)
export OPENAI_API_KEY=...            # extraction + embeddings; Graphiti is LLM-heavy
export FALKORDB_HOST=127.0.0.1
export FALKORDB_PORT=6379
```

You can point Graphiti at any OpenAI-compatible endpoint (including a local
Ollama) with a custom `LLMClient`, which is the only way to run it without a
paid key. Expect extraction quality to drop, and say so in the results — a
weak-model Graphiti row is not a Graphiti row.

Ingest cost warning: Graphiti calls an LLM per episode (and several times per
episode for edge resolution). Ten LoCoMo conversations is a real bill and takes
minutes to hours, not the 7.5 seconds InfinityDB needs. That asymmetry is a
finding, not a nuisance — report it.

## TencentDB Agent Memory, local

```bash
node --version                        # must be >= 22
mkdir -p ~/.memory-tencentdb && cd ~/.memory-tencentdb
npm install @tencentdb-agent-memory/memory-tencentdb@latest --omit=dev

# LLM for extraction (any OpenAI-compatible endpoint)
export TDAI_LLM_API_KEY=...
export TDAI_LLM_BASE_URL=https://api.openai.com/v1
export TDAI_LLM_MODEL=gpt-4o
```

Its interface is an agent plugin/gateway (`tdai_memory_search`,
`tdai_conversation_search`), not a `recall(query)` library call. Driving the
arena means either running the Hermes gateway on `:8420` and calling its tools,
or importing the plugin's core module directly. Both are a day of adapter work
once a key exists — and neither can be written honestly against a service that
has never returned a single row here.

### Measured: keyword mode runs today

Their retriever's FTS branch needs no key, so it has already been run — version
1.0.3, `recall.strategy=keyword`, one record per turn, 1,977 LoCoMo questions:
**evidence recall@10 = 0.5254** (multi-hop 0.189, single-hop 0.593). The LLM
extraction and embedding branch remain unmeasured and require `TDAI_LLM_API_KEY`.

Harness and reproduction steps: `benchmarks/arena/systems/tencentdb_keyword/`.

## Zep Cloud, if you want the managed number

```bash
export ZEP_API_KEY=...                # Zep Cloud console -> Project Settings -> API Keys
```

Cloud latency necessarily includes network round-trips from your machine. Label
the row "Zep Cloud, from <region>" or don't publish it.

## What can be measured today, with no keys

Already in the tables in `README.md` and `arena/README.md`:

- **mem0 2.2.1** with `infer=False` — a real, key-free long-term memory system.
- **BM25 + FastEmbed dense hybrid** — a free, strong, non-memory-system baseline.

Those two rows are the honest floor of the claim "leads the systems we can run".
Everything else needs credentials, and this repository will not invent them.

## The rule

An adapter ships only when it has produced a real row against a real endpoint.
Untested adapters are how benchmark tables become marketing: they silently
measure the wrong thing, and the first competent reviewer who reads the adapter
finds it. If a competitor cannot be run, the table says "not run" and the reason
— which is exactly what `arena/worker.py` does for `graphiti` and `tencentdb`.

## Checking readiness

```bash
python3 benchmarks/arena/competitor_check.py           # human-readable
python3 benchmarks/arena/competitor_check.py --json    # machine-readable
python3 benchmarks/arena/competitor_check.py --ping    # also probe endpoints
```

It reports each competitor as ready / blocked with the exact missing variable,
and exits non-zero when nothing is ready. Run it before, not after, you spend
the evening.

## Local attempt, 2026-11-11 (Graphiti)

Harness committed at `benchmarks/arena/systems/graphiti_local/`: graphiti-core
0.31.2 + embedded Kuzu + Ollama (`llama3.2:3b` for extraction, `nomic-embed-text`
for embeddings). The graph builds; the extraction step does not finish inside
Ollama's request timeout on 8 CPU cores (HTTP 500 at 10m0s for one 18-turn
session). Zero episodes, zero rows. This says nothing about Graphiti's retrieval
quality — only that a CPU-only host cannot pay its write-time LLM bill. With a GPU
or an API key the same command produces the row.
