# InfinityDB Benchmark Suite

The supported benchmark entry point is the deterministic, file-backed, single-engine harness:

```bash
python -m pip install -e ".[dev]"
python benchmarks/reproducible.py --iterations 1000 --output benchmark-results.json
```

The harness measures the checked-out Python InfinityDB implementation only. It records ingestion throughput, recall p50/p95/p99, SQLite storage footprint after close, environment metadata, dataset SHA-256, and basic tenant-isolation/provenance correctness checks.

The committed JSON files from earlier benchmark runs are historical snapshots, not release certification. They must not be interpreted as current results or competitor comparisons. The historical scripts `benchmark.py` and `run_benchmark.py` are compatibility entry points that now invoke the reproducible single-engine harness.

## Competitor comparisons: the LoCoMo arena

`benchmarks/arena/` runs real memory systems on the public LoCoMo dataset — InfinityDB, mem0 2.2.1 OSS (`infer=False`), an optional cross-encoder rerank row, and a free BM25 + dense `hybrid` control — with identical turns, questions, `top_k` and scoring, and keeps every raw retrieved-memory row:

```bash
bash benchmarks/arena/run.sh          # both systems, then score
```

`arena/scoped_probe.py` adds the contamination experiment (one store, ten
conversations, scoped vs unscoped), and `arena/longmemeval_probe.py` runs
LongMemEval in the same retrieval-only setting — the dataset MemPalace publishes
against, so at least one number exists on their metric. Read
[its README](arena/README.md) for the metric definitions, per-category results, the atom-only ablation and the equal-context table, and the limitations (retrieval proxies without a reader LLM, one dataset, tuned heuristics, one host, no independent reproduction). Systems that need credentials (Zep/Graphiti, TencentDB) are deliberately absent rather than emulated; the local harness's
`tencentdb_local` row is a flat-vector baseline in a TencentDB-shaped table, not the product.

## Fully local comparison harness

`benchmarks/locomo_local_compare.py` runs the same public LoCoMo file with every
component local: no API keys and no hosted service. InfinityDB needs nothing
beyond Python; the optional baseline adapters (Mem0 over local Ollama, Cognee
over local Ollama, and a flat dense-vector baseline in a TencentDB-shaped table)
are imported only when selected — including `mem0_nollm`, Mem0's `infer=False`
mode, which is the one competitor row measurable with no LLM at all. An adapter
whose dependency is missing is recorded under `skipped_engines` rather than
aborting the run.

```bash
PYTHONPATH=$PWD python benchmarks/locomo_local_compare.py \
    --engines infinitydb --max-convos 10 --max-qa 500 --top-k 10 --no-logs
```

It reads the copy of `locomo10.json` already vendored for the arena, so a run
needs no network. Per engine it records ingestion latency per session, retrieval
latency per query (p50/p95/p99), retrieved tokens against the full-conversation
baseline, and three accuracy signals. Only `evidence_recall_at_k_pct` is
comparable to the arena's rows: it asks whether the retrieved context contains
the dialogue turn LoCoMo cites as justification. `answer_contained_pct` and
`context_token_overlap_f1_pct` are string tests against the gold answer — they
punish long retrieved context and are not reader accuracy, so they must not be
quoted as one.

Measured run (Linux 6.1.102 x86_64, 8 vCPU, Python 3.12.3, `tiktoken/cl100k_base`,
[k=10, all ten conversations, category 5 excluded](../locomo-local-results.json)):

| metric | value |
| --- | --- |
| questions evaluated | 1,540 (272 sessions ingested) |
| evidence recall@10 | **87.1%** |
| retrieval latency | p50 21.8 ms, p95 31.5 ms, p99 54.9 ms |
| ingestion latency (per ~20-turn session) | p50 29.3 ms, p95 54.7 ms, p99 69.6 ms |
| retrieved tokens vs full context | 1,643 vs 18,468 (91.1% fewer) |
| gold-answer containment / token-overlap F1 | 35.97% / 5.61% (proxies only) |

The arena reports 0.8480 evidence recall at the same `k` over all 1,986
questions, including the category-5 adversarial ones this harness drops, so the
two are consistent but not identical populations. Latency here is higher than
the arena's 13.11 ms p50 because one store holds all ten conversations and
`top_k` is 10, where the arena gives each conversation its own database.

### Measured comparison: zero-LLM write paths

Both rows cover the same four conversations, 99 sessions and 584 questions at
`k=10` (Mem0's CPU embedding of all ten conversations takes about eight minutes,
past this host's per-command limit). One run each:

| metric | InfinityDB | Mem0 OSS `infer=False` |
| --- | --- | --- |
| ingestion p50 / p99 per session | **32.9 / 74.7 ms** | 1,494 / 2,961 ms |
| retrieval p50 / p99 | **15.7 / 48.2 ms** | 147.6 / 228.2 ms |
| evidence recall@10 | **87.97%** | 74.40% |
| retrieved tokens per question | 1,748 | **419** |
| gold-answer containment | **33.56%** | 29.28% |
| token-overlap F1 | 5.48% | **14.44%** |

This is a trade, not a sweep. Mem0's `infer=False` mode stores each turn as
written and embeds it with FastEmbed's `thenlper/gte-large` on CPU, so its write
path is 45x slower here even with **no LLM at all**; a Mem0 configured with an
Ollama extractor is a different, much slower mode that this host cannot run.
InfinityDB's advantage is offset by context: ten returned slots mean ten merged
turn chunks (1,748 tokens) against ten single turns (419), and at that size Mem0's
short, precise contexts score better on answer-level F1. The equal-token control
(the InfinityDB row with `budget_tokens` capped to ~420) has **not** been run in
this harness, so no claim is made that the recall lead survives at matched
context; the arena's equal-context table is the closest measurement we have.

The Ollama-backed rows (`mem0_local`, `cognee_local`) were not measured here:
this host has no Ollama and no model weights, so a row for them would be
fiction. The arena's Mem0 numbers come from its own pinned virtualenv
(`arena/requirements-mem0.txt`); those are the ones to quote for the shipped
engine. The `tencentdb_local` adapter is a flat dense-vector
baseline, not the TencentDB product, and the report names its backend
(`pgvector` or `sqlite-flat-vector-emulation`).

## Reporting results

Record the exact commit SHA, Python and SQLite versions, operating system, CPU/RAM, dataset revision/hash, workload size, and command used alongside every shared result. Preserve raw output, including failed runs. Do not combine values from different workloads or present a single run as a universal performance guarantee.

## Limitations

- Synthetic deterministic workload; not a representative customer trace.
- Measures only the checked-out InfinityDB Python implementation.
- No third-party product comparison or token-savings estimate. Competitor comparisons are produced separately by the [LoCoMo memory arena](arena/README.md), which runs each actual product with its own pinned dependencies.
- Results depend on hardware, OS, SQLite build, filesystem, and workload.
- Correctness checks are smoke checks, not a complete retrieval-quality evaluation.

## Requirements for a fair external comparison

Before publishing competitor rankings, run each actual product using pinned implementations and documented configurations. Use the same hardware, dataset revision, question set, scoring rules, warm-up policy, concurrency, and repetition count. Report retrieval quality alongside latency, including p50, p95, p99, throughput, ingestion cost, storage footprint, and failure counts. Publish raw per-example outputs and scripts so others can reproduce the result.

Until those requirements are met, use this harness for InfinityDB regression testing only—not as evidence of market leadership.

## Latency targets

`benchmarks/latency_targets.py` measures the two numbers the project brief sets as
targets — P99 write under 5 ms and multi-hop read under 10 ms — against the real
engine, on a 2,000-fact / 10-tenant synthetic corpus, and writes
`benchmarks/benchmark_latency_targets.json`:

```
python benchmarks/latency_targets.py --facts 2000 --reads 300
```

As measured in this workspace: P99 write **3.53-3.77 ms** (target met), median and P95
read **5.5 ms / 6.1-7.4 ms** (target met), P99 read **8.20 / 10.76 / 12.77 ms across
three identical runs** (target *not* reliably met — the tail is host scheduling noise,
not a code path). See `docs/SPEC_CONFORMANCE.md` for the full table and the LoCoMo
corpus numbers (p50 13.11 ms), which are above the target because assembly traverses
far more text per question.

## LongMemEval

`arena/longmemeval_probe.py` produces session-level recall@k on LongMemEval; it
resumes where it stopped, so a full run is several short calls:

```
python benchmarks/arena/longmemeval_probe.py --path benchmarks/arena/data/longmemeval_s.json
```

180 of 500 questions are covered in `arena/results/longmemeval/infinitydb-k5.jsonl`:
recall@5 **0.9333**, median context 534 words, median latency 15.16 ms.
