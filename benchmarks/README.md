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
[its README](arena/README.md) for the metric definitions, per-category results, the atom-only ablation and the equal-context table, and the limitations (retrieval proxies without a reader LLM, one dataset, tuned heuristics, one host, no independent reproduction). Systems that need credentials (Zep/Graphiti, TencentDB) are deliberately absent rather than emulated.

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
