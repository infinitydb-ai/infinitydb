# Reproducible benchmark harness

Run from the repository root:

```bash
python benchmarks/reproducible.py --iterations 1000 --output benchmark-results.json
```

The harness exercises the checked-out Python InfinityDB SQLite implementation using a
fixed synthetic fact workload across ten tenant IDs. It reports ingestion throughput,
recall p50/p95/p99, SQLite database footprint after clean close, environment metadata,
the dataset SHA-256, and correctness checks for tenant isolation and source-proof
verification.

The harness deliberately does **not**:
- compare InfinityDB with third-party products or architecture emulators;
- estimate token savings or extraction costs;
- assert a universal latency target;
- claim that synthetic data represents production customer workloads.

Results are hardware-, OS-, filesystem-, SQLite-build-, and workload-dependent. Preserve
the JSON output together with the exact Git commit and environment when publishing a
measurement. Run repeated trials on an otherwise idle machine before drawing conclusions.

## Node.js SDK benchmark

Run the matching deterministic workload against the Node.js SQLite implementation:

```bash
node infinitydb-js/benchmark.js --iterations 1000 --output benchmark-results-node.json
```

The Node.js harness uses the same five fact templates, ten tenant IDs, dataset SHA-256,
correctness checks, and nearest-rank p50/p95/p99 recall percentiles as the Python harness.
It reports Node.js and SQLite environment metadata plus the file-backed SQLite footprint.
The Python and Node.js measurements are separate runs and should not be treated as a
head-to-head comparison unless performed on the same machine under controlled conditions.


## Shared JSON result contract

Both SDK harnesses emit the same top-level result contract (schema version 1):

- `benchmark`: always `infinitydb-reproducible`;
- `schema_version`: result contract version, independent of SDK version;
- `engine`: `python` or `node`;
- common `iterations`, `recall_samples`, `correctness`, `ingestion`,
  `recall_latency_ms`, `storage`, `environment`, `dataset`,
  `engine_stats`, and `limitations` fields;
- `environment.runtime.name` and `environment.runtime.version` identify the
  runtime without changing the rest of the environment field names.

The machine-specific runtime version and engine statistics are intentionally values,
not schema differences. The contract makes outputs structurally interoperable; it does
not make independent runs statistically comparable. Use the same hardware, OS, SQLite
build, dataset, iteration count, and controlled run conditions before comparing results.
The schema is published at `benchmarks/result-schema-v1.json`.
