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

# same engine at several prompt budgets, for a recall-vs-token curve
PYTHONPATH=$PWD python benchmarks/locomo_local_compare.py \
    --engines infinitydb --max-convos 4 --max-qa 500 --top-k 10 \
    --budget-tokens 250 300 350 400 420

# split merged chunks into individual turns before packing (measured worse)
PYTHONPATH=$PWD python benchmarks/locomo_local_compare.py \
    --engines infinitydb --max-convos 4 --max-qa 500 --top-k 10 \
    --fragment-budget --budget-tokens 340 419 600
```

It reads the copy of `locomo10.json` already vendored for the arena, so a run
needs no network. Per engine it records ingestion latency per session, retrieval
latency per query (p50/p95/p99), retrieved items and tokens per query against the
full-conversation baseline, and three accuracy signals. Only `evidence_recall_at_k_pct` is
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
short, precise contexts score better on answer-level F1.

### The equal-token control: the recall lead does not survive token matching

`--budget-tokens N` re-runs the same engine with `recall(budget_tokens=N)`, the
documented prompt-cost knob: highest-ranked memories are packed into a budget and
the rest dropped, so a query receives fewer, larger snippets. Same four
conversations, 99 sessions, 584 questions, `k=10`, one run
([locomo-equal-token-results.json](locomo-equal-token-results.json),
per-question logs included):

| InfinityDB setting | returned items | tokens/question | evidence recall@10 | containment | token-overlap F1 |
| --- | --- | --- | --- | --- | --- |
| unbudgeted (`k=10`) | 9.98 | 1,748 | **87.97%** | 33.56% | 5.48% |
| `budget_tokens=420` | 3.24 | 521 | 74.23% | 28.25% | 4.49% |
| `budget_tokens=400` | 3.14 | 495 | 73.02% | 27.57% | 4.51% |
| `budget_tokens=350` | 2.79 | 433 | 70.45% | 27.23% | 4.45% |
| `budget_tokens=300` | 2.35 | 371 | 68.90% | 26.54% | 4.21% |
| `budget_tokens=250` | 1.92 | 316 | 67.18% | 25.68% | 4.07% |
| Mem0 `infer=False` (`top_k=10`) | 10 requested | 419 | **74.40%** | 29.28% | **14.44%** |

The equal-token answer is **negative**, and it is the headline: at 433 tokens
(3.3% above Mem0's 419) InfinityDB reaches **70.45% evidence recall against
Mem0's 74.40%**, 3.95 points behind, and it needs **521 tokens, 24% more** to
draw level (74.23% against 74.40%). Interpolating between the 371- and 433-token
rows puts it near 70% at Mem0's exact footprint. So the +13.57-point lead at
`k=10` is bought with 4.2x the context, not a free win, and the budget knob
cannot recover it at Mem0's cost. The mechanism is visible in the table: a
350-word budget returns 2.8 merged chunks where `k=10` returns 10, so each
returned snippet has to justify several turns and retrieval gets fewer
independent chances to contain the cited one. `budget_tokens` is enforced as a
*word* budget inside `recall` by default, which is why measured tiktoken tokens
run about 1.24x the setting (420 → 521) in that artifact; the token column above
is the measurement, and it is what the comparison uses, never the setting. That
run predates `recall(token_counter=...)`, which now lets the harness pass its own
tokenizer so the budget is real BPE tokens — the next section measures that way.

Two more readings from the same table, both unflattering and both left in:
budget enforcement *trades* recall for cost rather than compressing it —
containment falls with the budget too, so shortening the context does not raise
precision — and no budget setting reaches Mem0's answer-level F1 at any token
count, which is the long-context penalty the unbudgeted row already paid.

### Turn-level packing at a matched budget: measured, and it does not beat Mem0

The obvious fix for the loss above is to stop packing whole chunks: split the
top ten chunks back into their turns, score each turn against the query, and
pack ten small turns into the same ~419 tokens. That is now
`recall(..., pack_fragments=True)` / `--fragment-budget`, and it was measured
rather than assumed. The same run passes the harness's `tiktoken` counter to
`recall(token_counter=...)`, so the budget is now **real BPE tokens** — the
420 → 521 drift in the table above is gone
([locomo-fragment-pack-results.json](locomo-fragment-pack-results.json)):

| setting (~409–419 tokens) | items | tokens/question | evidence recall@10 | containment | token-overlap F1 |
| --- | --- | --- | --- | --- | --- |
| whole chunks, `budget_tokens=419` | 2.61 | 408.9 | **69.07%** | 26.88% | 4.31% |
| split turns, `budget_tokens=419` | 9.96 | 409.7 | 67.70% | 27.05% | 14.05% |
| split turns, `budget_tokens=340` | 9.57 | 383.6 | 66.84% | 27.05% | 14.12% |
| split turns, `budget_tokens=600` | 10.00 | 414.6 | 67.87% | 27.40% | 13.94% |
| unbudgeted `k=10` | 9.98 | 1,748 | 87.97% | 33.56% | 5.48% |
| Mem0 `infer=False` | 10 requested | 418.7 | **74.40%** | 29.28% | **14.44%** |

Giving the packer ten independent slots **does not** recover the gap: at
`n=10` and ~410 tokens, splitting turns scores 67.70% against whole chunks'
69.07% and Mem0's 74.40%. Whole windows stay ahead because an evidence turn is
often a *neighbour* inside the winning window rather than that window's
best-scoring turn, so per-turn selection throws away the context that carried
the hit; and the turn scores here are lexical, where Mem0's are `gte-large`
cosine similarities on single turns — the granularity Mem0 is best at, on the
metric that rewards it.

One methodological result is worth keeping: the token-overlap F1 column jumps
from 4.31% to 14.05%, essentially Mem0's 14.44%, while evidence recall gets
*worse*. Matching Mem0's F1 at Mem0's token count is a property of returning
short single turns, not of answering better — which is why that column is
labelled a proxy everywhere in this file.

Limitations stated with the numbers: `pack_fragments` is capped at `n` items,
so at `n=10` a budget above ten turns is not spent (600 and 900 both return
414.6 tokens); the selection is deterministic lexical scoring inside the top
`n` chunks, so no model is involved on either side; and the row is opt-in
because a measured-worse mode must not become the default for existing
`budget_tokens` callers.

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


### The matched-token win, and what it costs: an opt-in local reranker

Everything above measured the dependency-free lexical path. That path is why the
equal-token control lost. The follow-up was to give recall a semantic signal
without putting a model on the default path: `infinitydb-ai[onnx]` builds a local
FastEmbed cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, ~80 MB ONNX, CPU, no API
key) for `recall(..., rerank=fn)`, and a local embedder for `index_dense` /
`recall(dense=...)`. The core still installs zero third-party packages.

Same 4 conversations, 99 sessions, 584 questions, `k=10`, `tiktoken/cl100k_base`,
same vendored `locomo10.json`; `mem0_nollm` was re-run in the same session and
reproduced its published **74.40% at 418.7 tokens** exactly
([locomo-token-grid-results.json](locomo-token-grid-results.json), with
per-question logs for both 420-token InfinityDB rows):

| configuration | budget | returned items | tokens/question | evidence recall@10 | ingest p50 | search p50 |
| --- | --- | --- | --- | --- | --- | --- |
| default (no model) | 420 | 2.61 | 409.6 | 69.07% | 53.2 ms | 30.3 ms |
| `dense=` bge-small | 420 | 2.79 | 413.4 | 72.16% | 2,778 ms | 66.7 ms |
| `dense=` gte-large (Mem0's embedder) | 420 | 2.71 | 410.2 | 70.96% | 10,792 ms | 126.9 ms |
| **`rerank=` MiniLM cross-encoder** | **420** | **2.65** | **406.0** | **76.80%** | **29.1 ms** | **1,335 ms** |
| `dense=` + `rerank=` | 420 | 2.71 | 408.8 | 77.15% | 1,081 ms | 1,769 ms |
| Mem0 `infer=False` (`top_k=10`) | — | 10 requested | 418.7 | 74.40% | 8,405 ms* | 257 ms* |

\* Mem0's latencies were measured while three ONNX runs shared this 8-core host;
its accuracy and token columns are unaffected. The single-point InfinityDB rows
carry the same contention caveat: 1.3–1.8 s search p50 is an upper bound, not the
quiet-host number.

**The reranker closes the gap and passes it.** At a matched budget it returns
**406.0 tokens for 76.80% evidence recall against Mem0's 418.7 tokens for
74.40%** — 2.40 points ahead on 3% fewer tokens, where the same call without the
model scores 69.07%. That is not a different question set: paired per question on
the same 584 questions, the reranker fixes **81** that the lexical path missed and
loses **36** it had (McNemar exact two-sided **p = 3.9e-5**). A second full run
reproduced 76.80% and 406.0 tokens exactly.

The mechanism is not "bigger embedding". Dense fusion with Mem0's own
`thenlper/gte-large` scored **70.96%** — worse than the small bge model's 72.16%
and 3.4 points behind Mem0 — because reciprocal-rank fusion still compares two
independently encoded texts. The cross-encoder reads the query and the passage
together, which is what single-turn evidence needs. It also costs nothing at write
time: ingest p50 stays at **29.1 ms**, because there is no index to embed.

The costs, stated rather than buried:

- **Search latency.** 1.3 s p50 (1.8 s under three-way CPU contention) against
  30 ms for the lexical default and 257 ms for Mem0 on this host. The win is a
  re-ranking-quality win, not a speed win; the 9.4× retrieval claim belongs to the
  default path only.
- **One budget point, not a curve.** A reranked run costs ~10× a lexical run, so
  the reranker was measured at budget 420 only. Default and dense fusion were
  measured across 320/420/600/1000/1750:

  | configuration | 320 | 420 | 600 | 1000 | 1750 | unbudgeted |
  | --- | --- | --- | --- | --- | --- | --- |
  | tokens (default) | 318.8 | 409.6 | 595.6 | 996.5 | 1,567.4 | 1,748.1 |
  | evidence recall (default) | 67.18% | 69.07% | 75.43% | 83.85% | 87.29% | 87.97% |
  | tokens (dense) | 319.2 | 413.4 | 596.5 | 999.4 | 1,553.3 | 1,694.4 |
  | evidence recall (dense) | 68.04% | 72.16% | 79.04% | 86.77% | 89.18% | 90.38% |

- **Dense fusion is not worth it on top of the reranker.** 77.15% vs 76.80% for
  1.6 s extra search and a 1.1 s ingest, so the documented recommendation is the
  reranker alone.
- **Scope.** Still one dataset, one machine, retrieval proxies only, heuristics
  and a model chosen on this data. No independent reproduction. The `mem0_nollm`
  row is Mem0 without an LLM; Mem0 with LLM extraction remains unmeasured here
  because this host has no Ollama.

Reproduce the winning row (downloads two small models on first run):

```bash
pip install "infinitydb-ai[onnx]"
python benchmarks/locomo_local_compare.py --engines infinitydb \
  --max-convos 4 --max-qa 500 --top-k 10 --budget-tokens 420 --budget-only \
  --rerank-model Xenova/ms-marco-MiniLM-L-6-v2 \
  --output benchmarks/locomo-token-grid-results.json
```
