# Benchmark integrity audit

Date: 2026-11-01

## Scope

This audit inspected the default-branch README, benchmark implementation, committed benchmark result JSON, stress-test result JSON, core retrieval implementation, proof implementation, tests, and repository metadata.

## Findings

1. `benchmarks/run_benchmark.py` labels its comparison implementations as architecture emulators. They are not the actual TencentDB, Mem0, Zep, or Letta products. Results from these emulators must not be described as direct competitor benchmarks.
2. `benchmarks/benchmark_results.json` reports InfinityDB PersonaMem accuracy as `0.0`, while the README advertises 89.7%. The committed artifacts do not support that README claim.
3. The committed benchmark result JSON reports measured token saving of 63.2%, while the README headline advertises 81.4%. The two numbers need a clear distinction and a reproducible calculation.
4. The stress-test artifact reports p95 recall of 8.238ms, while other published material advertises 4.2ms. These may be different workloads, but the workload definitions must be explicit.
5. The previous `infinitydb/core/proof.py` implementation returned `True` unconditionally. This branch replaces that stub with verification of the SHA-256 digest and the exact source character range, and adds a tamper-detection test.
6. The current `InfinityDB.recall` implementation reads all L1 atoms for a user (or all users when no user ID is supplied) and scores them in Python. It is not currently an indexed BM25/FTS5 retrieval implementation.

## Required before public superiority claims

- Run the benchmark harness against the actual pinned competitor releases, not local emulators.
- Publish exact commit hashes, package versions, dataset versions, hardware, model/provider settings, seeds, prompts, and raw outputs.
- Report accuracy and retrieval quality alongside latency; do not optimize latency by silently excluding older relevant memories.
- Reconcile all README claims with raw result artifacts.
- Add independent reproduction instructions and preserve unsuccessful runs.

## Branch changes

- Implement cryptographic proof verification against the source log and extracted fact.
- Add a test that tampers with the original source and expects verification to fail.

This document is an engineering audit, not a certification or independent third-party evaluation.


## Public-claims remediation (v1.1)

The README and benchmark guide are being aligned with the committed artifacts. Historical cross-vendor score tables and unsupported latency/token-saving guarantees are removed from the headline presentation. The local harness is explicitly described as a regression tool with architecture emulators, not an independent product shootout.

A future benchmark release must include pinned real implementations, dataset revisions, environment metadata, per-example outputs, and repeatable scoring before publishing comparative rankings.

## Real-implementation arena (2026-11-10)

A second, separate harness now exists: `benchmarks/arena/`. It runs actual
memory systems on the public LoCoMo dataset, each in its own pinned
environment, and keeps every retrieved-memory row:

- `mem0ai==2.2.1` (OSS, `infer=False`, FastEmbed `thenlper/gte-large`, on-disk
  Qdrant) versus InfinityDB on the same turns, questions and `top_k=10`.
- A free control that is not a product: `hybrid` (porter-stemmed BM25 plus the
  same FastEmbed embeddings, reciprocal rank fusion k=60). It exists so that a
  reader can see what a competent bag-of-words + dense stack gets from the same
  verbatim turns, with no LLM and no tuning for this dataset.
- Measured on one 8-vCPU CPU-only host, all 10 conversations, 1,986 questions,
  no reader LLM: InfinityDB 0.780 evidence recall@10 / 0.545 answer-in-context,
  mem0 0.678 / 0.386, hybrid 0.676 / 0.389. InfinityDB ingested ~72× faster and
  searched ~14× faster, at ~2.2× the median context tokens. With the returned
  context held to mem0's size (`budget_tokens=320`) InfinityDB still leads on
  evidence recall (0.696 vs 0.678) and on answer-in-context (0.479 vs 0.386).
  Two further probes are reported separately and labelled as such: a
  contamination probe (pooled store, 0.733 unscoped → 0.780 scoped) and a
  180-question stratified LongMemEval sample (R@5 0.933, against MemPalace's
  published 96.6% on all 500).
  mem0 leads on open-domain questions.
- One optional configuration was measured and **rejected as a default**: a
  cross-encoder reranker (`ms-marco-MiniLM-L-6`) adds +0.7% relative recall for
  13× the latency. It is shipped as `recall(..., rerank=fn)` and documented as
  off by default, with its row in the results table.
- Zep/Graphiti and TencentDB memory were **not** run: they require an LLM
  provider key plus a graph database, or Tencent Cloud credentials. No adapter,
  emulator or comparison is included for them.

What this does and does not fix, relative to the findings above:

1. Finding 1 (emulator comparisons) is addressed for mem0 only, and by a real
   package rather than an emulator. Zep, TencentDB and Letta remain unmeasured.
2. Findings 2–4 (README claims vs committed artifacts) remain resolved on the
   current branch by removing the unsupported figures; the arena adds measured
   retrieval numbers, not answer accuracy.
3. Finding 5 (proof stub) and finding 6 (non-indexed retrieval) were addressed
   in v1.1.0; retrieval is now turn-window BM25 with an FTS5 index.

Still required before any leadership claim: a reader-model column
(`qa_f1`/`gold_in_answer`), a held-out or additional dataset, more than one
host, runs of Zep and TencentDB with documented configurations, and independent
reproduction. Until then the arena's own README states that its numbers are
retrieval proxies and not a leaderboard.
