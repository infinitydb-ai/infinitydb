# v2 alpha benchmark protocol

The checked-in microbenchmarks are synthetic, single-process, local SQLite measurements. They are not independent product comparisons and must not be used to claim a leaderboard position or production latency guarantee.

For a defensible evaluation, pin code and dependency versions; disclose hardware, OS, cache state, corpus generation, index-build and ingestion time; report cold and warm p50/p95/p99; and measure recall@k, MRR/nDCG, database size, RAM, CPU, context tokens and end-to-end task cost. Compare identical datasets, query sets, resource limits and embedding models. Publish raw outputs and failed queries.

The current lexical candidate selector is latency-oriented and can prioritize recently inserted matching rows. Evaluate historical-query recall before relying on it for exhaustive long-term retrieval.


## Optional HNSW ANN evaluation

Install the optional backend with `pip install -e '.[ann]'`. The Python store exposes
`MemoryStore.search_vector_ann()` separately from exact `search_vector()`. Run
`infinitydb-ann-bench --records 3000 --queries 100` for a reproducible local comparison.
The ANN index is built lazily for one tenant, dimension, provider, model and
model revision. By default it is in-memory. For restart persistence, construct
`MemoryStore(database_path, ann_index_dir=directory)`. Each partition uses a
hashed filename (not a tenant-name filename), a JSON label manifest, a source
fingerprint derived from ordered SQLite embedding rows, and an SHA-256 checksum
of the native index. Index and manifest files are written through temporary
files and atomic replacement; the manifest is replaced last. On open/search,
a missing, stale, version-mismatched or checksum-invalid index is rebuilt from
SQLite, which remains authoritative. This persistence mode is single-process;
cross-process writer coordination is not yet implemented. `delete_embedding()`
removes a tenant-owned vector while preserving its memory and evidence, then
invalidates the local ANN cache. `compact_ann_index()` explicitly rebuilds one
tenant/model/dimension partition from SQLite and removes empty persisted
partitions. Candidate IDs are filtered against current SQLite
tenant, temporal and kind constraints, then rescored with exact cosine
similarity and memory confidence. Updating an embedding invalidates that
tenant's cached ANN partitions. New provider-backed single and batch ingestion
is appended to a warm partition when the backend supports resizing; otherwise
the cache is discarded and rebuilt on the next query.

Report recall@k against exact `search_vector()` on the same corpus and query
set, along with index build time, warm/cold p50/p95/p99, RAM, CPU and index
parameters. The optional backend is not a distributed index, and no performance or recall
claim should be made without measured results. The persistence path is covered
by tests for restart reload, source-change invalidation and corrupt-file recovery.
