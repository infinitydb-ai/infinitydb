# v2 alpha benchmark protocol

The checked-in microbenchmarks are synthetic, single-process, local SQLite measurements. They are not independent product comparisons and must not be used to claim a leaderboard position or production latency guarantee.

For a defensible evaluation, pin code and dependency versions; disclose hardware, OS, cache state, corpus generation, index-build and ingestion time; report cold and warm p50/p95/p99; and measure recall@k, MRR/nDCG, database size, RAM, CPU, context tokens and end-to-end task cost. Compare identical datasets, query sets, resource limits and embedding models. Publish raw outputs and failed queries.

The current lexical candidate selector is latency-oriented and can prioritize recently inserted matching rows. Evaluate historical-query recall before relying on it for exhaustive long-term retrieval.


## Optional HNSW ANN evaluation

Install the optional backend with `pip install -e '.[ann]'`. The v2 API exposes
`MemoryStore.search_vector_ann()` separately from exact `search_vector()`.
The ANN index is built lazily in memory for one tenant, dimension, provider,
model and model revision. Candidate IDs are filtered against current SQLite
tenant, temporal and kind constraints, then rescored with exact cosine
similarity and memory confidence. Updating an embedding invalidates that
tenant's cached ANN partitions.

Report recall@k against exact `search_vector()` on the same corpus and query
set, along with index build time, warm/cold p50/p95/p99, RAM, CPU and index
parameters. The optional backend is not a persistent or distributed index,
and no performance or recall claim should be made without measured results.
