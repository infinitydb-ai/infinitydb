# Vendor-neutral memory benchmark

This harness provides a common adapter contract and deterministic workload for
comparing memory systems. It does **not** ship unverified Mem0, Zep, or Letta
integrations or claim benchmark results for them. Each integration must map the
same add/search contract to the vendor's supported API and document its
configuration.

## Adapter contract

Implement a Python factory returning an object with:

- `add(memory_id, tenant_id, content, vector)`
- `search(tenant_id, query, vector, limit) -> list[memory_id]`
- `metadata() -> dict`
- `close()`

The runner creates stable 128-dimensional feature-hash vectors, ingests
multi-tenant records, then queries for the corresponding exact memory ID.
This measures plumbing, tenant isolation, ingest/query latency, throughput,
top-k recall on known exact-match queries, and Python-tracked peak allocations.
It is intentionally not a semantic-recall or long-term-memory reasoning test.

## Run

From `alpha-v2/`:

```bash
python -m benchmarks.vendor_neutral.runner \
  --adapter my_adapter:build \
  --records 10000 --queries 1000 --tenants 10 --top-k 10 \
  --output results/my-system.json
```

Add an empty `__init__.py` in the package directories if your environment
does not support namespace packages. The adapter module must be importable on
`PYTHONPATH`.

## Fair-comparison protocol

1. Pin package versions, model/provider revisions, index parameters and source
   commit for every system.
2. Run on the same host, OS, CPU, RAM, storage and network topology.
3. Use identical records, tenants, query order, top-k and concurrency.
4. Report cold-start and warmed runs separately; repeat at least five times.
5. Report p50/p95/p99, throughput, peak RSS (including native/server process),
   storage size, cost, recall, and failure rate. This runner's tracemalloc metric
   is only Python allocation data, not total process memory.
6. Validate tenant isolation with adversarial cross-tenant queries before
   publishing results.
7. Publish raw JSON, adapter source, exact commands, environment manifest and
   logs. Never compare a local embedded database against a remote service
   without disclosing network placement.

## Extending the suite

The deterministic smoke workload is a systems sanity check. A credible
leaderboard must additionally include semantic retrieval, multi-hop questions,
temporal updates, conflicting facts, long conversations, deletion/forgetting,
crash recovery, concurrent writers, and tenant-isolation attack cases. Use
public datasets under their respective licenses and publish dataset hashes.
Do not combine scores from incomparable datasets into a single ranking.
