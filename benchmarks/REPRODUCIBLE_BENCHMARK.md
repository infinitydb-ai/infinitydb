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
