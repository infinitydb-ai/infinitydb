# LoCoMo memory arena

A head-to-head harness that runs **actual memory systems** on the public
[LoCoMo](https://github.com/snap-research/locomo) long-conversation dataset:
same turns, same questions, same `top_k`, same scoring code. The point is to
replace the "architecture emulator" comparisons that the repository used to
publish with runs of the real packages, pinned by version.

This directory is a measurement tool, not a certification. Read
[Limitations](#limitations) before quoting any number from it.

## What is measured

Each adapter is driven by [`worker.py`](worker.py) in its own interpreter, so a
competitor's dependency tree cannot affect InfinityDB (which has no third-party
dependencies). The adapter contract is three calls — `reset()`,
`ingest(session)`, `search(question, top_k)` — in
[`systems/base.py`](systems/base.py).

For every conversation the worker records raw JSON lines (one row per session
ingest and per question, including the exact retrieved memories), and
[`score.py`](score.py) computes:

| Metric | Definition |
| --- | --- |
| `evidence_recall@10` | Fraction of the dataset's annotated evidence turns that appear in the 10 returned memories (a turn counts when ≥60% of its content words are in one memory, or ≥90% are spread across all ten). |
| `answer_in_context` | Fraction of questions whose gold answer tokens all appear in the 10 returned memories. A retrieval proxy, not an answer. |
| `context_tokens_median` | Median word count handed to a reader. Retrieval quality alone is not comparable if one system ships 3× the context. |
| `latency_ms` p50/p95 | Wall-clock `search()` time, measured by the worker outside the adapter. |
| `ingest_seconds` | Total time to store every turn of every session. |
| `qa_f1`, `gold_in_answer` | End-to-end answer quality from a local reader model (Ollama). **Not run here** — see Limitations. |

Raw rows are kept, so every score can be recomputed without re-running a
system. `scores/summary.json` plus one file per conversation hold the summary
metrics; the per-question retrieved memories stay in `results/<system>/`.

## Systems

| System | What actually ran | Configuration |
| --- | --- | --- |
| `infinitydb` | This repository, `infinitydb/core/engine.py` | Zero-LLM deterministic extraction; SQLite WAL; each dialogue turn is scored inside its one-turn neighbour window (porter-stemmed FTS5) and adjacent hits are merged into contiguous chunks; rare terms from the best three chunks are fed back into the query (`expansion=0.15`, light pseudo-relevance feedback). One `user_id` per conversation, `auto_supersede=False` (two speakers share a user id, so single-valued supersession would delete one speaker's facts). |
| `mem0` | `mem0ai==2.2.1` (OSS), `fastembed==0.9.1`, `qdrant-client==1.20.0`, `onnxruntime==1.31.1` | `infer=False`: one memory per turn, embedded locally with mem0's default FastEmbed model (`thenlper/gte-large`), stored in on-disk Qdrant, retrieved by cosine similarity. No LLM at write or read time. |
| `hybrid` | `fastembed==0.9.1` (BM25 + `thenlper/gte-large`), local only | A strong free baseline, not a product: porter-stemmed BM25 over the same verbatim turns, concatenated with dense cosine similarities via reciprocal rank fusion (k=60), one memory per turn. No LLM anywhere; it isolates how much a "BM25 + embeddings" stack gets from the same text. |
| `infinitydb-dense` | same engine + `fastembed==0.9.1` (`sentence-transformers/all-MiniLM-L6-v2`) | The default configuration plus semantic fusion: every turn window is embedded once into `l2_vectors`, and the dense ranking is fused with the lexical ranking by reciprocal rank fusion at weight 0.3. Still no LLM. Set `ARENA_INFINITYDB_DENSE=<model>` and `ARENA_INFINITYDB_DENSE_WEIGHT=0.3` to reproduce. |
| `tencentdb-keyword` | `@tencentdb-agent-memory/memory-tencentdb@1.0.3` (MIT), Node 26, `tsx` | Their own `executeMemorySearch` in `recall.strategy=keyword` mode: one L1 record per verbatim turn written through their store API, FTS5/BM25 retrieval, top 10. The LLM extraction pass and the embedding/vector branch were **not** run (both need a model key), so this row measures their retriever, not their full pipeline. Harness: `benchmarks/arena/systems/tencentdb_keyword/`. |
| `graphiti` (Zep) | **not run** | Zep's open-source engine needs an LLM provider key plus Neo4j/FalkorDB; Zep Cloud needs `ZEP_API_KEY`. Neither is available in this environment, so no adapter is wired up and no comparison is claimed. |
| `tencentdb` | **not run** | Tencent Cloud's memory service needs Tencent Cloud credentials. Not available here; not emulated. |

Both systems that ran store one memory per dialogue turn and receive the same
event timestamp per session; the reader-facing date is rendered the same way
(`8 May 2023`) so temporal questions are not decided by date formatting.
mem0's adapter puts the date into the memory text because `Memory.add()` rejects
a timestamp argument in this release; InfinityDB receives the timestamp as
`ts` and renders it at read time.

## Results

Measured 2026-11-10, one process per system, CPU-only Linux VM (8 vCPU, 15 GB
RAM, Python 3.12.3), all 10 LoCoMo conversations, 1,986 questions (1,540
answerable + 446 adversarial), `top_k=10`, no reader LLM.

| System | evidence_recall@10 | answer_in_context | context tokens (p50) | ingest (s) | search p50 (ms) | search p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| **InfinityDB** (turn windows + PRF) | **0.7795** | **0.5454** | 711 | **7.55** | **10.66** | **14.13** |
| **InfinityDB + dense fusion** (MiniLM, RRF 0.3) | 0.7867 | 0.5487 | 652 | 8.02 | 29.04 | 35.26 |
| **InfinityDB**, `budget_tokens=320` | 0.6958 | 0.4792 | 325 | 8.37 | 10.80 | 14.57 |
| mem0 2.2.1 (`infer=False`) | 0.6777 | 0.3864 | **321** | 544.29 | 152.05 | 191.65 |
| TencentDB Agent Memory 1.0.3 (keyword/FTS, no LLM) | 0.5254 | — | — | — | — | — |
| hybrid BM25 + dense (RRF, free baseline) | 0.6757 | 0.3890 | **314** | 582.17 | 151.87 | 191.85 |
| InfinityDB, atom-only ablation | 0.4978 | 0.3545 | 329 | 7.91 | 5.05 | 7.66 |
| InfinityDB + cross-encoder rerank | 0.7852 | 0.5500 | 710 | 7.98 | 139.91 | 214.15 |

Per category (evidence recall / answer-in-context):

| Category | n | InfinityDB | mem0 | hybrid |
| --- | --- | --- | --- | --- |
| multi-hop | 282 | **0.494** / **0.227** | 0.471 / 0.213 | 0.430 / 0.202 |
| open-domain | 96 | 0.354 / 0.073 | **0.396** / 0.063 | 0.372 / **0.073** |
| single-hop | 841 | **0.913** / **0.740** | 0.752 / 0.599 | 0.766 / 0.606 |
| temporal | 321 | **0.798** / **0.458** | 0.744 / 0.078 | 0.737 / 0.078 |

### Reading these numbers

- InfinityDB recalls more annotated evidence (+15.0% relative) and puts the gold
  answer in context considerably more often (+41% relative), while ingesting
  the same conversations ~72× faster and answering ~14× faster on this host.
- It uses **~2.2× more context** by default (710 vs 321 words) because a merged
  turn chunk is longer than a single mem0 memory. That is why the budget row
  exists: see below.
- The hybrid row matters as a control: BM25 plus gte-large embeddings, fused
  with RRF, lands within noise of mem0 (0.6757 vs 0.6777) and far below
  InfinityDB, despite the same verbatim text. Embedding a bag of turn text is
  not what is missing here; ranking turns in their conversational context is.
- The ablation row isolates the retrieval change. Scoring each atom on its own
  text gives 0.4978 recall; almost the entire gain comes from turn-window
  retrieval, stemming, chunk merging and query expansion, not from the harness.
- The cross-encoder row is an optional plug-in (`recall(..., rerank=fn)` with
  `Xenova/ms-marco-MiniLM-L-6-v2`). It buys +0.7% relative recall for **13× the
  latency**, so it is shipped as an option and is **not** the default.
- mem0 still wins **open-domain** (commonsense) questions. InfinityDB wins
  single-hop, temporal and multi-hop, and temporal answers additionally benefit
  from rendering the event date in words (`8 May 2023`) rather than an ISO stamp.
- No ranking against Zep or TencentDB is claimed, because neither product was
  run. Published Zep/mem0 LoCoMo figures usually use an LLM to extract memories
  and/or answer, which is a different (and much more expensive) operating point.

### Equal-context comparison (`budget_tokens`)

`recall(..., budget_tokens=N)` packs the highest-ranked memories until `N` words
are reached, so a caller can hold prompt cost constant. At mem0's measured
context size:

| Budget | evidence_recall@10 | answer_in_context | actual tokens |
| --- | --- | --- | --- |
| 200 words | 0.6041 | 0.4182 | 201 |
| 250 words | 0.6603 | 0.4526 | 252 |
| 320 words | **0.6958** | **0.4792** | 325 |
| 400 words | 0.7238 | 0.5019 | 407 |
| no budget | 0.7795 | 0.5454 | 711 |
| mem0 (for reference) | 0.6777 | 0.3864 | 321 |

At the same ~320-word context, InfinityDB leads mem0's evidence recall
(0.696 vs 0.678) and puts the gold answer in context 24% more often. It reaches
mem0's answer-in-context rate at ~200 words, i.e. with ~37% less context. These
are retrieval proxies, not answers, and the token counts are word counts, not
model-specific tokens.

### Scoped recall under contamination (`scoped_probe.py`)

Every row above gives each conversation its own database. A real agent
accumulates conversations, projects and people in one store. This probe pools
all ten LoCoMo conversations into a single database and compares three
conditions on the same questions:

| Condition | evidence_recall@10 | answer_in_context | context words | p50 |
| --- | --- | --- | --- | --- |
| `isolated` (one DB per conversation) | 0.7799 | 0.5456 | 695 | 10.5 ms |
| `pooled`, unscoped | 0.7331 | 0.5217 | 634 | 33.0 ms |
| `pooled`, `session_id=<conversation>` | **0.7799** | **0.5450** | 694 | 39.9 ms |

Cross-conversation contamination costs 4.7 points of recall. Scoping recovers
all of it, because document frequencies and average length are computed inside
the scope, not against the whole store. Without in-scope corpus statistics the
same filter recovers only 68% of the loss (0.7626), which is what scoping
naively does.

Reproduce: `python benchmarks/arena/scoped_probe.py` (writes
`scores/scoped_probe.json`).

### LongMemEval (`longmemeval_probe.py`)

MemPalace publishes retrieval numbers on
[LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval), not LoCoMo,
so LoCoMo scores cannot be compared to theirs. This probe runs the same dataset
in the same retrieval-only setting. A question is a hit when any returned memory
belongs to an annotated answer session.

| Setting | R@5 |
| --- | --- |
| InfinityDB, full (stratified 180-question sample) | **0.9333** |
| — `knowledge-update` (n=30) | 1.0000 |
| — `multi-session` (n=30) | 1.0000 |
| — `single-session-user` (n=30) | 1.0000 |
| — `single-session-assistant` (n=30) | 0.9667 |
| — `temporal-reasoning` (n=30) | 0.9667 |
| — `single-session-preference` (n=30) | 0.6667 |
| MemPalace (their published claim, full 500) | 96.6% R@5 |

Each question carries its own ~54-session, ~550-turn haystack and gets a fresh
database; ingest median 2.0 s, recall p50 15.2 ms, median 534 context words.
**This is a 180-question stratified subsample of 500, not the full set**, so it
is close to the published claim rather than a win over it. The weak type is
preference questions, which want the resolved L3 profile that recall does not
merge into the returned context.

The 278 MB dataset is not committed; download
`longmemeval_s` into `arena/data/` and run
`python benchmarks/arena/longmemeval_probe.py --sample 30`. Rows are appended
per question, so an interrupted run resumes.

## Competitors that need credentials

Zep Community Edition was retired in April 2025; the self-hostable engine is
Graphiti (needs a graph database plus an LLM), and TencentDB Agent Memory needs
Node >= 22 plus an OpenAI-compatible key. Neither can run here without those, so
no adapter is shipped rather than an adapter that measures the wrong thing.

```bash
python3 benchmarks/arena/competitor_check.py           # what is ready, what is missing
```

Full setup steps, costs and the metrics a memory comparison must report:
[`docs/COMPETITOR_BENCHMARK_PLAYBOOK.md`](../../docs/COMPETITOR_BENCHMARK_PLAYBOOK.md).

## Fast iteration

`replay_probe.py` reuses the databases the arena already ingested and scores
them with the same evidence-recall function as `score.py`, so a ranking change
can be measured in seconds instead of a full 10-conversation run:

```bash
python3 benchmarks/arena/replay_probe.py
```

Any change still has to be confirmed with a full `worker.py` + `score.py` run
before its numbers go in the tables above.

## Reproduce

```bash
# every registered system, all conversations, then score (retrieval only)
bash benchmarks/arena/run.sh

# just one system
bash benchmarks/arena/run.sh infinitydb

# ablation: atom-level recall without turn windows
ARENA_INFINITYDB_ATOMS=1 python3 benchmarks/arena/worker.py \
  --system infinitydb --conv conv-26 \
  --out benchmarks/arena/results/infinitydb-atoms/conv-26.jsonl \
  --workdir .arena/infinitydb-atoms/conv-26 --force

# equal-context row: cap returned words
ARENA_INFINITYDB_BUDGET=320 python3 benchmarks/arena/worker.py \
  --system infinitydb --conv conv-26 \
  --out benchmarks/arena/results/infinitydb-budget320/conv-26.jsonl \
  --workdir .arena/infinitydb-budget320/conv-26 --force

# optional cross-encoder rerank (needs fastembed; see requirements-hybrid.txt)
ARENA_INFINITYDB_RERANK=Xenova/ms-marco-MiniLM-L-6-v2 python3 benchmarks/arena/worker.py \
  --system infinitydb --conv conv-26 \
  --out benchmarks/arena/results/infinitydb-rerank/conv-26.jsonl \
  --workdir .arena/infinitydb-rerank/conv-26 --force
```

End-to-end QA additionally needs a reader model served by Ollama (default
`llama3.1:8b` at `http://127.0.0.1:11434`); without it `score.py` must be run
with `--no-reader` and the `qa_f1`/`gold_in_answer` columns stay at 0.

Work directories under `.arena/` are disposable and git-ignored; the JSONL rows
in `results/` are committed as evidence.

## Windows PowerShell runner (Benchmark Arena v2)

The original `run.sh` remains the Unix entry point. Native Windows users can
run the same existing worker and scorer through `run.ps1`; it does not replace
or rewrite any adapter.

From the repository root in PowerShell:

```powershell
# Optional: allow this script in the current PowerShell process only
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Fast local validation: dataset, scorer and real InfinityDB adapter
.\benchmarks\arena\run.ps1 -Mode smoke

# Full LoCoMo run: all 10 conversations, InfinityDB + mem0 OSS + BM25/dense
# Installs the pinned mem0/hybrid environment into .arena\venvs\mem0.
.\benchmarks\arena\run.ps1 -InstallDependencies

# Optional: enable a local Ollama reader for end-to-end QA metrics
.\benchmarks\arena\run.ps1 -WithReader

# Optional: raise the synthetic InfinityDB-only scale workload to 1M writes
.\benchmarks\arena\run.ps1 -StressIterations 1000000
```

The Windows runner performs the following without Bash or WSL:

1. Checks Python 3.10+, verifies the LoCoMo dataset and records its SHA-256.
2. Compiles the arena Python modules and runs the selected real adapters over
   all ten LoCoMo conversations. Completed JSONL files are reused unless
   `-Force` is supplied.
3. Scores retrieval with the existing scorer. Reader-model evaluation is off
   by default; enable it only when Ollama and the configured model are ready.
4. Runs the existing pooled-vs-isolated scoped-recall probe and a separate
   synthetic InfinityDB scale test (100K writes by default). The synthetic
   result is **not** a competitor score.
5. Runs the full LongMemEval probe if
   `benchmarks/arena/data/longmemeval_s.json` is present. The 278 MB dataset
   is not committed; without it the runner records that this probe was skipped.
6. Writes a Windows run manifest under `benchmarks/arena/results/scores/`
   containing the dataset hash, OS, Python version, selected systems and probe
   status. Raw per-question JSONL remains under `results/<system>/`.

The no-credential default compares InfinityDB, mem0 OSS in `infer=False`
mode, and the local BM25 + FastEmbed hybrid control. These are the systems
currently supported by the shared Python worker. TencentDB Agent Memory's
keyword-mode result remains a separately measured Node harness; its full
extraction pipeline needs a model endpoint. Graphiti is open source but its
normal pipeline needs a graph backend and an LLM; it is not silently emulated
or included in the no-credential leaderboard. See
[the competitor playbook](../../docs/COMPETITOR_BENCHMARK_PLAYBOOK.md).

For machines with 8 GB RAM, keep the default 100K synthetic workload initially.
The 1M option is deliberately opt-in because memory and runtime depend on the
host. FastEmbed downloads its embedding model on the first mem0/hybrid run.

## Limitations

1. **Retrieval proxies, not answers.** `answer_in_context` says the gold tokens
   are present, not that a model would answer correctly. The reader-based
   columns are empty in this run because no local LLM was available.
2. **One dataset.** LoCoMo is ten long two-person conversations with
   evidence-annotated questions. It does not cover tool logs, code, documents,
   multi-user tenancy or update/forget behaviour. Ranking on it is not a general
   leadership claim.
3. **Heuristics tuned on this dataset.** The neighbour window, stemming, BM25
   parameters and chunk-merge cap were chosen by measuring LoCoMo. They are
   standard IR techniques, but the exact values are tuned, and a held-out set
   would give lower numbers. No held-out split was used.
4. **One machine, one run.** Both systems are deterministic given the same
   inputs, so repeats are not expected to move quality, but latency is
   workload- and host-specific. Ingest and search speed compare the two systems
   on this host only.
5. **Adapter choices are part of the result.** mem0 runs in its documented
   no-LLM `infer=False` mode; a mem0 deployment with LLM extraction would score
   differently and cost far more. Documenting the configuration is not the same
   as showing it is optimal.
6. **Not independent.** The harness, the adapters and the tuning were written by
   the InfinityDB authors. No third party has reproduced these results.

## Adding a competitor

Design notes for the systems that were read but not run (Graphiti/Zep, Letta,
Memobase, Cognee), and for the changes this harness caused, are in
[`docs/COMPETITIVE_TEARDOWN.md`](../../docs/COMPETITIVE_TEARDOWN.md).

Implement `systems/<name>.py` against the three-method contract, register it in
`worker.py`, add a pinned `requirements-<name>.txt`, and run it in its own
virtualenv. Do not emulate a product and label it with the product's name; the
repository already paid for that mistake (see
[the benchmark integrity audit](../../docs/BENCHMARK_INTEGRITY_AUDIT.md)).

### Optional: end-to-end answers with a local reader (Ollama)

Retrieval metrics need no LLM. To also measure whether an answer is actually
correct, `score.py` runs one fixed local reader over the retrieved context:

```bash
ollama pull llama3.2:3b                      # ~2 GB, CPU is fine
ARENA_READER_MODEL=llama3.2:3b python3 arena/score.py \
    --systems infinitydb,infinitydb-dense --limit 10 --readers 4 \
    --out arena/results/scores-reader-sample
```

`--limit N` scores the first N questions per conversation — a deterministic
sample. It exists because a 3B model on CPU prefills a 700-token context at
roughly 45 tokens/s (~16 s per question), so all 1,986 questions take hours.
Answers are cached per `(model, question, context)` in
`results/scores-reader-sample/<system>-<conv>-answers.jsonl`, so a stopped run
resumes where it left off, and `python3 .arena/reader_stats.py <system>` reports
F1 over whatever is cached so far. Point `ARENA_OLLAMA` at another host if the
server is not on loopback.

### Graphiti (Zep): attempted locally — no row yet, and here is why

`systems/graphiti_local/` runs **graphiti-core 0.31.2** against a local Ollama
server with the **embedded Kuzu** driver: no Docker, no Java, no API key. It emits
arena-format JSONL so `score.py` grades it with the same code as every other row.
That harness works; the row does not exist yet:

| attempt | setup | result |
|---|---|---|
| 2026-11-11 | `llama3.2:3b` Q4, 8 CPU cores, **one** 18-turn session | extraction `POST /v1/chat/completions` returned **HTTP 500 at 10m0s**, retried and failed again at 7m30s; 0 episodes completed, 0 rows |

The extraction prompt is multi-thousand tokens and the model has to emit a large
JSON object; on CPU that blows past Ollama's request timeout before a single
episode lands. Everything else is confirmed working: `json_schema` response format
(`json_object` made the 3B echo the schema back instead of an instance), Kuzu
mounts and builds indices, and `nomic-embed-text` serves 768-dim embeddings
through the OpenAI-compatible endpoint. The default OpenAI reranker cannot be used
without a key, so the harness installs a pass-through reranker and the row is
labelled as rerank-off.

To get the row, give it a GPU or a hosted model:

```bash
ARENA_GRAPHITI_LLM=qwen2.5:14b python benchmarks/arena/systems/graphiti_local/run.py \
    --conv conv-26 --sessions 19 --questions 199 --db .arena/graphiti/kuzu_conv26 \
    --out benchmarks/arena/results/graphiti-local/conv-26.jsonl
python benchmarks/arena/score.py --systems graphiti-local --no-reader
```

Kuzu is deprecated upstream (the driver prints a warning); use Neo4j or FalkorDB
for anything beyond local testing. Second local attempt, 2026-11-12 (retry with a longer client timeout): the 10-minute
cutoff turned out to be the OpenAI SDK's default request timeout, not an Ollama
limit, so the client now gets 3600 s. Individual extraction calls then succeed
(observed 30.9 s, 49.9 s, 56.6 s), but building the node/edge graph for a single
18-turn session still did not finish within 15 minutes of continuous work on 8 CPU
cores, and a mid-session call exceeded 3 minutes before being cancelled. So the
blocker is throughput, not a hard error: one conversation would take hours, and ten
would take days. A GPU or a hosted model is required for this row.

Third local attempt, 2026-11-12 (reduced pipeline: 2-turn episodes so each
extraction call is small). The chunking path is implemented and runs
(`--episode-turns N`, `--max-episodes N`), but ingestion fails with a hard driver
error, not slowness:

```
Error executing Kuzu query: Binder exception: Table RelatesToNode_ doesn't have
an index with name edge_name_and_fact
```

Graphiti 0.31.2 expects a Kuzu **full-text** index created with
`CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name','fact'])`;
that call has to run after the edge table exists, so on a fresh database it fails
during `build_indices_and_constraints`, and Kuzu rejects `CREATE INDEX IF NOT
EXISTS` outright. Kuzu is deprecated upstream and prints a warning on import, so
this is an unsupported-backend bug rather than a graph problem. The supported
backends are Neo4j and FalkorDB, both of which need a server process — unavailable
here (no Docker daemon, no Java).

Net: **no Graphiti row is obtainable in this environment at any episode size.**
The harness is committed so a host with Neo4j/FalkorDB (or a GPU plus a working
backend) can produce it with one command.

Until that row exists, **no comparison against
Graphiti or Zep is claimed** — this is a missing row, not a win.
