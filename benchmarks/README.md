# InfinityDB Benchmark Suite

This repository contains a local benchmark harness for development and regression testing. **It is not currently an independent comparison against the TencentDB, Mem0, Zep, or Letta products.** The comparison classes in `run_benchmark.py` are local architecture emulators and must not be described as measurements of those products.

## Current checked-in artifacts

The committed artifacts are historical snapshots, not a release certification:

- `benchmark_results.json` reports InfinityDB PersonaMem accuracy as 0.0, LongMemEval and LoCoMo accuracy as 100.0, and measured token saving as 63.2%.
- `stress_test_2000_results.json` reports 2,000 recall iterations, 100% success in that run, and p95 recall latency of 8.238 ms.
- The stress-test result is specific to its recorded environment and workload. It does not establish a universal latency guarantee.

Do not combine values from different workloads or replace measured values with target values. Preserve raw output and report failed runs.

## Run locally

```bash
python -m pip install -e ".[dev]"
python benchmarks/run_benchmark.py
python benchmarks/stress_test_2000.py
```

The scripts write local result artifacts. Record the commit SHA, Python and dependency versions, operating system, CPU/RAM, dataset revision, random seed, workload size, and command used alongside any result shared publicly.

## Requirements for a fair external comparison

Before publishing competitor rankings, each system must run using its actual pinned implementation and documented configuration. Use the same hardware, dataset revision, question set, scoring rules, warm-up policy, concurrency, and repetition count. Report accuracy/recall quality and latency together; include p50, p95, p99, throughput, ingestion cost, storage footprint, and failure counts. Publish raw per-example outputs and scripts so others can reproduce the result.

Until those requirements are met, this suite should be used for InfinityDB regression testing only—not as evidence that InfinityDB is number one.
