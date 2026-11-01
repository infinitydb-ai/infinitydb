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
