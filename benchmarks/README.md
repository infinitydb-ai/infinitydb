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

### The full 500 questions, and what the reranker buys there

`longmemeval_bench.py` is the question-level harness. It ingests every haystack
turn of one question into a fresh store, asks `recall(k=10)`, and scores what came
back against the turns and sessions the dataset marks with `has_answer`. Hits are
resolved through `l0_logs` provenance, not by comparing returned text to evidence
strings, so a hit means the memory was actually retrieved. It checkpoints per
question, so an interrupted run resumes.

```
curl -L -o benchmarks/data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
python benchmarks/longmemeval_bench.py --data-path benchmarks/data/longmemeval_s_cleaned.json \
  --output benchmarks/longmemeval-s-lexical.json
python benchmarks/longmemeval_bench.py --data-path benchmarks/data/longmemeval_s_cleaned.json \
  --rerank-model Xenova/ms-marco-MiniLM-L-6-v2 --rerank-top-k 8 --quantize-rerank \
  --output benchmarks/longmemeval-s-results.json
```

All 500 questions, `k=10`, `budget_tokens=600`, `tiktoken/cl100k_base`, in-memory
SQLite per question, `longmemeval_s_cleaned.json` sha256 `d6f21ea9…`, commit
`0f950ae`. 470 answerable questions and 30 abstention questions; **no reader and no
judge** — every number below is retrieval.

| configuration | session-any@10 | session-all@10 | turn-any@10 | turn-all@10 | MRR | tokens/q | search p50 | ingest p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default lexical | 91.91% | 70.85% | 48.30% | 18.72% | 0.8789 | 563.5 | 13.3 ms | 2.11 s |
| `rerank_top_k=8`, INT8 MiniLM | **95.74%** | **73.62%** | **51.91%** | **20.43%** | **0.9424** | 562.2 | 182.0 ms | 2.15 s |

Paired per question on the same 500 questions, the reranked run gains 18 and loses
0 on session-any (exact McNemar **p = 8e-6**), gains 22 and loses 5 on turn-any
(p = 0.0015), gains 24 and loses 11 on session-all (p = 0.041) and gains 8 and
loses 0 on turn-all (p = 0.0078). The reranker buys 3.83 points of session-any for
**13.7× the search latency** and no measurable token cost, which is the same
trade-off the LoCoMo reranker rows show.

| ability | n | session-any lexical → rerank | turn-any lexical → rerank |
| --- | --- | --- | --- |
| single-session-user | 64 | 98.4% → 100.0% | 28.1% → 28.1% |
| single-session-assistant | 56 | 92.9% → 96.4% | 8.9% → 8.9% |
| single-session-preference | 30 | 56.7% → 73.3% | 20.0% → 30.0% |
| multi-session | 121 | 95.0% → 96.7% | 65.3% → 70.2% |
| temporal-reasoning | 127 | 89.8% → 95.3% | 63.0% → 67.7% |
| knowledge-update | 72 | 98.6% → 100.0% | 54.2% → 56.9% |

**The knowledge-update ordering result was negative in these artifacts.** On the 71
questions where ordering applies, the newest valid version outranks the superseded
one **46.5%** of the time lexically and **59.7%** with the reranker, although the
right session is retrieved 98.6–100% of the time. The run deliberately uses
`auto_supersede=False` with `now=question_date` so this is measured rather than
hidden. It is the number these two artifacts carry; the engine has since gained a
recency tie-break that moves it to **77.5%** without changing any other retrieval
number, described under "Knowledge-update ordering: a recency tie-break" below.

Recorded beside the table, because each one bounds the claim:

- **Abstention is not measured as accuracy.** The 30 unanswerable questions have no
  grounding turn, so the harness reports the top-1 score distribution instead:
  mean **27.67** lexical and **20.09** reranked, with **0%** of queries returning
  nothing. The engine does now expose a calibrated confidence and gate, measured
  separately under "Calibrated abstention" below; it is not a usable
  "I don't know" either.
- **turn-all@10 of 18.72% is a property of `k=10` against ~550 turns**, not a
  failure of retrieval: 300 of the questions have more than one evidence session
  and one slot cannot hold them all.
- **Ingest dominates** at ~2.1 s per question for ~550 turns, unchanged by the
  reranker because nothing is embedded at write time.
- One host, 8 cores, one run per configuration; the artifacts carry the dataset
  sha256, run fingerprint and per-question rows (`longmemeval-s-lexical.json`,
  `longmemeval-s-results.json`).


### Knowledge-update ordering: a recency tie-break, not a decay

The measurement above says the right session is almost always retrieved and only
the order is wrong: **95% of the ordering failures return both the newest and the
superseded session**. So the fix is ordering, not retrieval.

`recall()` now re-orders the memories it is already returning: when two adjacent
memories are within `_RECENCY_TIE_BAND` (0.45) of each other's score, the newer one
(`ts`) goes first. Four details make it safe:

- **It runs after budget packing.** Ordering earlier changes which memories survive
  `budget_tokens`, which is exactly what the recall columns measure. The first
  version of this change ordered before packing and cost 3.4 points of session-all
  and 5.8 points of turn-any on multi-session questions.
- **It is a band, not a multiplier.** A blanket recency decay was measured earlier
  and demoted the older sessions that long-horizon questions ask about.
- **Historical-intent queries skip it.** `used to`, `previously`, `originally`,
  `before`, `old`, `first time` go through un-reordered: those questions want the
  superseded value.
- **With a reranker the band is additive** (`_RECENCY_CE_DELTA = 0.5` over the
  cross-encoder logit), because logits are not ratio-scaled.

Full 500 questions, `k=10`, `budget_tokens=600`, lexical, one run per row:

| band | newest > stale | session-any | session-all | turn-any | turn-all | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| off (published artifact) | 46.5% | 91.91% | 70.85% | 48.30% | 18.72% | 0.8789 |
| **0.45 (shipped)** | **77.5%** | 91.91% | 70.85% | 48.30% | 18.72% | 0.8709 |
| 0.25 | 87.3% | 91.91% | 70.85% | 48.30% | 18.72% | 0.8446 |

Recall is identical by construction; the price is MRR, because the evidence for a
temporal question is often not the newest memory in the list. 0.25 buys another 9.8
points of ordering for four times the MRR cost (−0.034 against −0.008), which is why
0.45 ships. For scale, an earlier variant that sorted candidates newest-first
*before* packing reached 98.3% newest>stale and lost 12 points of session-any — the
band exists to keep that trade off the table.

LoCoMo is unchanged by the tie-break in both arms measured here: 87.97% evidence
recall@10 unbudgeted and 75.60% at the 420-token reranked budget. Both are set-based,
which is why ordering cannot move them.


### Calibrated abstention: implemented, measured, and negative

`recall()` now always returns `confidence` (0–1), `abstain_recommended` and
`confidence_threshold`, and `min_score=` / `abstain_on_low_confidence=True` filter
on them. The signal is IDF-weighted coverage of the query's content terms in the
best candidate, penalised when the query's rarest term is missing
(`infinitydb/core/confidence.py`, pure stdlib); with a reranker it blends
0.8 × the temperature-scaled cross-encoder logit with 0.2 × lexical coverage. The
`_abs` questions have no grounding turn, so the harness scores a gate by whether
it abstains on them and how much answerable retrieval it gives up.

The hypothesis was that this separates answerable from unanswerable questions
where the raw top-1 BM25 score does not (mean 27.67, with an *oracle-fit* threshold
at 86.7% abstention recall for 45.7% answerable retention). It does not:

| run | threshold | abstention recall | false abstention | balanced accuracy | best grid point |
| --- | --- | --- | --- | --- | --- |
| lexical | 0.42 | 43.3% | 48.5% | 0.474 | 70.0% / 66.0% @ 0.60 |
| rerank top-8 INT8 | 0.45 | 46.7% | 32.8% | 0.569 | 66.7% / 43.6% @ 0.60 |

All 500 questions (30 `_abs`, 470 answerable), `k=10`, `budget_tokens=600`
([lexical](longmemeval-s-calibrated-results.json),
[reranked](longmemeval-s-calibrated-rerank-results.json)). The lexical gate is at
chance; the reranked blend is better but nowhere near the 80% true-abstention /
<3% false-abstention target this was built for. No threshold on a 0.05 grid comes
close, because the confidence distributions overlap: mean top confidence is 0.50
for answerable vs 0.52 for `_abs` lexically, and 0.59 vs 0.48 reranked. Retrieval
is unchanged by construction in both runs (91.91% / 70.85% / 48.30% / 18.72% and
95.74% / 73.62% / 51.91% / 20.43% session-any/all, turn-any/all; MRR 0.8709 and
0.9417), because the gate is off by default.

Two things are recorded with it. Some `_abs` questions are confidently answered
(three at 1.0 lexically) because the question names an entity the store does
discuss while the *attribute* is missing, so term coverage really is 100% —
coverage cannot see which part of the question is unanswerable. And the IDF sample
costs latency on a store's first query after a write: +25 ms cold on a 3,000-atom
tenant, +0.20 ms warm (+1.7%), which is why the lexical calibration run shows
recall p50 25.1 ms against the published 13.3 ms — that harness queries each fresh
store once.


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
quiet-host number. Re-measured one run at a time, the reranker costs **616 ms p50**
and the affordable configuration in the next section costs **135 ms**.

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


### Making that reranker affordable: eight candidates and INT8

76.80% recall is worth little if a query costs 1.3 s. Two changes cut that by
4.5×, and they were gated numerically before the runs: **search p50 ≤ 300 ms**
and **evidence recall ≥ 75.0%** (at least 90% of the 6.53-point rerank gain over
the 69.07% lexical default).

- **`recall(rerank_top_k=8)`** — the engine assembles up to `3*k = 30` chunks, and
the first version cross-encoded every one of them. A cross-encoder pays one
forward pass per passage, so reranking only the eight chunks lexical scoring
already ranks highest is ~4× less work; the lexical tail keeps its order and only
fills slots the reranked head does not.
- **`local_reranker(quantize=True)`** — dynamic INT8 quantization of the same
MiniLM model with `onnxruntime.quantization`, built once and cached next to the
model. It changes the arithmetic per pair, not the number of pairs.

Same 584 questions, same budget, four configurations, one run at a time on the
same host ([locomo-rerank-latency-results.json](locomo-rerank-latency-results.json),
per-question logs for all four):

| configuration | budget | tokens/question | evidence recall@10 | search p50 | search p95 | ingest p50 |
| --- | --- | --- | --- | --- | --- | --- |
| rerank all chunks, fp32 | 420 | 406.0 | 76.80% | 616.1 ms | 969.5 ms | 30.4 ms |
| rerank all chunks, INT8 | 420 | 405.1 | 76.29% | 453.1 ms | 765.8 ms | 29.9 ms |
| rerank top 8, fp32 | 420 | 411.6 | 76.46% | 176.3 ms | 283.6 ms | 31.3 ms |
| **rerank top 8, INT8** | **420** | **411.8** | **75.60%** | **135.5 ms** | **225.0 ms** | **31.4 ms** |
| default, no model | 420 | 409.6 | 69.07% | 30.3 ms | — | 53.2 ms |

Both gates pass: **135.5 ms p50** and **75.60%** recall. Against Mem0
`infer=False` (74.40% at 418.7 tokens, 257 ms search) the cheap configuration is
still **1.20 points ahead at 411.8 tokens**, 2% fewer than Mem0's, and it is the
row to reach for when a query must stay interactive. The full reranker remains the
accuracy row at +2.40 points.

What the two levers cost in recall, paired per question on the same 584 questions:
top-8 + INT8 fixes **66** the lexical path misses and breaks **28**
(McNemar exact **p = 1.1e-4**), and against reranking everything it is 10 fixed
vs 17 broken (**p = 0.25**) — a 4.5× latency cut that this harness cannot
distinguish from the expensive row. Quantizing without capping changes recall by
6 fixed vs 9 broken; capping without quantizing by 11 vs 13. Neither lever alone
is the gain, and neither is a measurable loss.

The costs, stated rather than buried: 135 ms is still **4.5× the lexical path's
30 ms**, so the cheap reranker is a latency-priced accuracy feature, not a free
one; the INT8 model is an extra build step and a second cached file; and this is
still one dataset on one host with no independent reproduction. `rerank_top_k`
below `k` means the bottom of the returned list is lexical, which is why the
benchmark's `rerank_top_k` is 8 and `k` is 10 rather than the other way round.

```bash
pip install "infinitydb-ai[onnx]"
python benchmarks/locomo_local_compare.py --engines infinitydb \
  --max-convos 4 --max-qa 500 --top-k 10 --budget-tokens 420 --budget-only \
  --rerank-model Xenova/ms-marco-MiniLM-L-6-v2 --rerank-top-k 8 --rerank-quantize \
  --output benchmarks/locomo-rerank-latency-results.json
```

## Multi-tenant scale on PostgreSQL

`benchmarks/postgres_scale_bench.py` writes the same synthetic fact shape into
every tenant's schema with a marker token unique to that tenant, then reads it
back through `InfinityDB` and checks storage, unqualified reads and `recall()`
for foreign markers ([results](postgres-scale-results.json)):

```bash
pip install "infinitydb-ai[postgres]" pgserver
python benchmarks/postgres_scale_bench.py --total-atoms 100000 --tenants 100 --skip-locomo
```

| metric | value |
| --- | --- |
| atoms stored / tenant schemas | 100,000 / 100 (1,000 each) |
| `add()` p50 / p95 / p99 | 2.20 / 4.43 / 7.01 ms |
| `recall(k=10)` p50 / p95 / p99 | 25.37 / 33.71 / 36.42 ms |
| foreign-marker rows / recall results | 0 / 0 |
| rows seen by unqualified reads | 1,000 per tenant |

Half the queries omit the tenant marker on purpose, and every row is scanned for
every other tenant's marker, so this is a leak check rather than a marker-match
check. Context for the numbers: ingest averaged 220 atoms/s because each `add()`
is its own committed transaction, one add took 29 s (a checkpoint; p99 is 7 ms),
and the run is one host, one process and synthetic facts. It is evidence that
isolation holds and a floor for production latency, not a capacity guarantee.
