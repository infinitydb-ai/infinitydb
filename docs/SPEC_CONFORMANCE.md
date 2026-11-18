# Spec conformance

The project brief describes a Rust core with PyO3/NAPI bindings, ONNX models, HNSW,
background consolidation workers and a bi-temporal graph. This page maps each of those
requirements onto the code that exists in this repository **today**, with the file that
implements it and the measurement that backs the claim. Where something is not built,
it says so — a spec is a target, not a description, and the difference is the only
useful thing this document can contain.

Status vocabulary: **present** (works and is measured), **partial** (a working subset,
with the gap named), **absent** (not implemented).

| Requirement | Status | Where it lives / what is missing |
|---|---|---|
| Rust core, single binary + embedded library | **absent** | The only Rust in the repo is `alpha-v2/rust/infinitydb-core/src/lib.rs`: dependency-free score-fusion primitives (about 40 lines). There is no Rust storage engine, no PyO3, no NAPI-RS. The shipped engine is Python on SQLite. |
| Python binding | **present** | The engine *is* Python (`infinitydb/core/engine.py`), not a binding over something else. |
| TypeScript binding | **partial** | `infinitydb-js` is a client SDK over the Python/REST surface, not a native NAPI-RS module. |
| Embedded disk + memory store | **present** | SQLite in WAL mode, `:memory:` supported (`InfinityDB(":memory:")`), no server process required. |
| Multi-tenancy `org_id`/`user_id`/`agent_id`/`session_id` | **partial** | `user_id` (required scope on every table and query) and `session_id` exist and are enforced. `org_id` and `agent_id` do not exist as columns; tenants nesting and per-agent scoping would need a migration. |
| Turn-level provenance log | **present** | `l0_logs` stores every raw turn with `session_id` and `ts`; `l6_proof` is a SHA-256 hash chain over evidence, and `get_proof`/`audit_verify` verify it (`tests/test_audit.py`). |
| Zero-LLM ingestion | **present** | The write path never calls a model or the network: rule-based extraction plus deterministic state diffing in `_add` / `_extract`. |
| Sub-5 ms P99 write | **present, measured** | 3.53 / 3.76 / 3.77 ms P99 across three runs, 2,000 facts, 10 tenants (`benchmarks/latency_targets.py`). |
| Sub-100M ONNX models (embeddings + NER) | **absent** | `pyproject.toml` has `dependencies = []` and no ONNX runtime. Extraction is regex/rule based, and dense vectors are opt-in through a caller-supplied embedder (`index_dense`). This is *why* writes are fast; it is also why the engine has no semantic similarity by default. |
| Async "dreaming" workers (dedupe, summarize, consolidate) | **absent** | There is no background worker, no summary table, no consolidation pass. Correction is explicit and synchronous (`supersede_fact`, `correct_fact`). |
| Bi-temporal facts: `valid_at` + `recorded_at` | **partial** | The behaviour exists under different names. `l1_atoms.ts` is *event* time and `recall(as_of=...)` reads history at a point in time; `l1_atom_relations.created_at` is *recorded* time; late-arriving facts are recorded without rewriting the past (`add(..., when=)`). The literal columns `valid_at`/`recorded_at` do not exist, and `l1_entity_edges` carries only `ts`. |
| Supersede without deleting audit history | **present** | `l1_atom_relations` with `relation='supersedes'`; superseded atoms stay queryable via `as_of` and are excluded from current recall by a `NOT EXISTS` clause (`tests/test_v110_temporal.py`, `tests/test_graph.py`). |
| HNSW dense index | **partial** | In `alpha-v2` only (optional `hnswlib`, persistent indexes with fingerprint recovery, `infinitydb-ann-bench`). The stable core stores normalised vectors in `l2_vectors` and scores them **exactly** — no ANN index. |
| BM25 lexical search | **present** | SQLite FTS5 with real IDF-weighted BM25 scoring in `_recall_atoms` (`k1=1.2`, `b=0.3`), plus a stemmed-prefix fallback for inflection mismatches. |
| Graph traversal in the same pipeline | **present** | `l1_entity_edges` (subject/predicate/object, indexed both ways) is traversed inside `recall` for bridge/multi-hop assembly; this is what Phase 2i–2k bought (multi-hop 0.4939 → 0.6260). |
| Single hybrid query pipeline | **present** | Lexical candidates, optional dense ranking, category routing, temporal filters and chunk assembly are one `recall` call returning ranked, deduplicated memories. |
| Token-budgeted context compiler | **present** | `recall(..., budget_tokens=N)` packs highest-ranked memories into the budget, and `max_tokens`-style callers get an explicit word count back. Measured: 0.8496 @1,170 words, 0.8642 @1,392, 0.8729 @1,426 (`frontier_2l.json`). |
| MCP server | **present** | `infinitydb/mcp_server.py`, 13 tools (`recall_memory`, `add_memory`, `correct_fact`, `forget`, `audit_export`, `verify_proof`, …), protocol tests in `tests/test_mcp_protocol.py`. |
| REST/gRPC endpoints | **partial** | `server.py` is a FastAPI REST service. gRPC is absent. |
| Reproducible benchmarks incl. P99 latency | **present** | `benchmarks/latency_targets.py` (this work), `benchmarks/reproducible.py`, `benchmarks/arena/` with `verify` re-deriving every metric from raw rows. |
| LoCoMo accuracy | **present** | 1,986 questions: evidence recall 0.8480, in-context 0.5935, 1,426 words median, p50 13.11 ms. |
| LongMemEval accuracy | **partial** | Recall@5 **0.9333** on 180 of 500 questions, 534 words median context, 15.16 ms median latency (`results/longmemeval/infinitydb-k5.jsonl`). The other 320 questions have not been run. |

## The two latency targets, stated as measured

Measured on a 2,000-fact synthetic corpus, 10 tenants, one process, this VM
(`benchmarks/latency_targets.py`, results in
`benchmarks/benchmark_latency_targets.json`):

| Target from the brief | Result | Verdict |
|---|---|---|
| P99 write < 5 ms | 3.53 / 3.76 / 3.77 ms (three runs) | **met** |
| P50 multi-hop read < 10 ms | 5.46 / 5.50 / 5.54 ms | **met** |
| P95 multi-hop read < 10 ms | 6.14 / 6.29 / 7.40 ms | **met** |
| P99 multi-hop read < 10 ms | 8.20 / 10.76 / 12.77 ms (three runs, same code) | **not met, and not stable** |

The p99 read tail is dominated by host scheduling noise on a shared VM: repeat passes
of the identical query set share **zero** of their slowest 1%, and the slow calls
scatter across queries rather than clustering in a code path. It would be dishonest to
quote the 8.20 ms run and call the target met. Two real costs were removed on the way
(see CHANGELOG: memoised `_stem`, one round trip for BM25 corpus stats), which moved
median read latency from 6.05 ms to about 5.5 ms; the remaining tail is not something
caching fixes.

On the LoCoMo conversation corpus (hundreds of turns per question) the same read path
runs at p50 13.11 ms / p95 18.25 ms, because assembly traverses far more text. **The
10 ms target is a small-corpus property, not a general one.**

## What "100x faster than Mem0/Zep/Cognee" can and cannot be said

The repository measures this engine. It contains adapters that run competitors in
`benchmarks/arena/systems/`, and their numbers are only ever reported as what those
adapters produced on this machine, with the version pinned. No multiplicative speedup
claim is made here, because a fair wall-clock comparison also has to weigh the LLM
calls those systems make in their write path for extraction — a cost this engine does
not pay and cannot simulate.
