# v2 alpha benchmark protocol

The checked-in microbenchmarks are synthetic, single-process, local SQLite measurements. They are not independent product comparisons and must not be used to claim a leaderboard position or production latency guarantee.

For a defensible evaluation, pin code and dependency versions; disclose hardware, OS, cache state, corpus generation, index-build and ingestion time; report cold and warm p50/p95/p99; and measure recall@k, MRR/nDCG, database size, RAM, CPU, context tokens and end-to-end task cost. Compare identical datasets, query sets, resource limits and embedding models. Publish raw outputs and failed queries.

The current lexical candidate selector is latency-oriented and can prioritize recently inserted matching rows. Evaluate historical-query recall before relying on it for exhaustive long-term retrieval.
