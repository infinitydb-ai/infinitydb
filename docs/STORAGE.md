# Storage backends (SQLite · PostgreSQL + pgvector)

`infinitydb.storage` is the persistence layer for the OAMS-1.0 memory hierarchy
(L0 raw turns, L1 atomic facts, L2 causal edges, L3 persona, L4 policies,
L5 retention state, L6 SHA-256 provenance chain). It is a separate module with a
single contract and two implementations:

| Backend | Install | Notes |
| --- | --- | --- |
| `SqliteStorage` | default, `dependencies = []` | standard library only, FTS5 lexical search |
| `PostgresStorage` | `pip install "infinitydb-ai[postgres]"` | PostgreSQL 14+ with the `vector` extension |

```python
from infinitydb.storage import open_backend

backend = open_backend(db_path="memory.db")            # SQLite, no dependencies
backend = open_backend(dsn="postgresql://…/infinitydb")  # PostgreSQL + pgvector

backend.append_turns([...])
backend.insert_atoms([...])
```

`open_backend()` runs the idempotent `initialize()` migrations for you. Pass
`initialize=False` to control that yourself.

## The contract

`StorageBackend` (`infinitydb/storage/base.py`) defines typed, tenant-scoped
operations:

- **L0** — `append_turns`, `get_turn_window`
- **L1 + L6** — `insert_atoms`, `invalidate_atom`, `get_atom_proof_chain`,
  `get_latest_chain_hash`
- **L2** — `upsert_edges`, `traverse_neighbors`
- **Retrieval** — `search_lexical`, `upsert_embeddings`, `search_dense`
- **L3** — `get_persona_state`, `upsert_persona_state`
- **L4** — `list_active_policies`, `upsert_policy`
- **L5** — `reinforce_atoms`
- **Erasure** — `delete_user_data`

Every method takes `(tenant_id, user_id)` explicitly and filters on both columns.
An empty identifier raises `ValueError` rather than widening the scope.

### The L6 chain

`insert_atoms` assigns `prev_hash`/`payload_hash` itself, inside the write
transaction, so callers never compose hashes by hand:

```python
previous = GENESIS_HASH or the latest payload_hash for that (tenant, user)
payload_hash = sha256(f"{previous}\n{canonical_payload(atom)}")
```

`canonical_payload` covers the source-bearing fields only — not `access_count`,
`stability` or `invalid_at`, which mutate after the fact. `get_atom_proof_chain`
returns the ordered prefix ending at an atom and `verify_chain(chain)` checks it,
so tampering with any earlier atom invalidates every later hash.

### Hybrid retrieval

- SQLite: FTS5 (`unicode61`) ranked by `bm25`, joined back to the atoms table.
- PostgreSQL: a stored generated `tsvector` column with a GIN index, ranked by
  `ts_rank_cd` over `websearch_to_tsquery('english', …)` — the forgiving parser,
  so punctuation in a user question never raises a syntax error.
- Dense: `vector({dim})` (384 by default) with a partial HNSW cosine index over
  active, embedded atoms on PostgreSQL; a normalised dot-product scan on SQLite.
  Both store unit vectors, so cosine similarity is a dot product.

## Multi-tenant isolation on PostgreSQL

1. Every transaction binds the tenant with `SET LOCAL`, using
   `set_config('app.current_tenant_id', %s, true)`.
2. `initialize()` enables **and forces** Row-Level Security on all six tables,
   with one policy comparing `tenant_id` to that setting. A connection that never
   binds a tenant sees zero rows instead of everything. Superusers still bypass
   RLS — the explicit `WHERE tenant_id = …` predicates are the primary defence
   and RLS is the second.
3. `insert_atoms` takes `pg_advisory_xact_lock(hashtext(tenant || ':' || user))`
   before reading the chain tip, so concurrent appends to one user's chain are
   serialised while different users and tenants never block each other.

## Tested

`tests/test_storage.py` runs the same contract suite against both backends;
`tests/test_storage_postgres.py` covers the PostgreSQL-specific guarantees
(fixed-width vector rejection, GIN + HNSW index presence, and RLS enforcement
through a non-owner role). The PostgreSQL arms use the `pgserver` test wheel,
which bundles PostgreSQL 16 and pgvector 0.6.2 with no server setup; where it is
not installed they are skipped and the SQLite arm still runs on a bare install.

## Engine wiring (`InfinityDB(dsn=...)`)

The contract above is the portable L0-L6 API. The core engine has a second,
narrower seam in `infinitydb/storage/engine.py`: the persistence operations the
engine actually performs, written once and translated per dialect.

```python
from infinitydb import InfinityDB

db = InfinityDB(db_path="memory.db")                          # stdlib only
db = InfinityDB(dsn="postgresql://user@host/infinitydb",
                tenant_id="acme")                             # [postgres] extra
```

- **One ranker.** Extraction, turn-window scoring, expansion, chunk assembly,
  `budget_tokens` packing, persona resolution and provenance run in Python on
  both backends. Only candidate narrowing is SQL: FTS5 `bm25` on SQLite,
  `tsvector` + GIN on PostgreSQL.
- **SQLite is unchanged.** `SqliteEngineStore` executes the engine's existing
  statements verbatim, so `InfinityDB(db_path=...)` still reproduces the
  published LoCoMo row (87.97% evidence recall@10 at 1,748 tokens/question).
- **PostgreSQL fetches wide and caps in Python.** `ts_rank_cd` and `bm25` order
  borderline candidates differently, so a SQL `LIMIT 400` in each backend would
  score different candidate sets. The PostgreSQL store returns all lexical
  matches and `_RANKED_CANDIDATE_CAP` applies the historical 400-candidate cap
  to the shared ranking instead.
- **Dialect translation is explicit.** `?` → `%s`, `INSERT OR REPLACE` /
  `INSERT OR IGNORE` → `ON CONFLICT`, SQLite `rowid` → an identity `rid`, FTS5
  MATCH → `tsquery`. `schema_engine_pg.sql` creates the engine's own tables with
  generated `tsvector` columns, GIN indexes and identity rows; FTS5 virtual
  tables and triggers do not exist on PostgreSQL.
- **Text search is configured to match FTS5.** FTS5's porter tokenizer stems
  without dropping stopwords. `initialize()` therefore creates a dictionary
  (`snowball`, `StopWords = ''`) and a configuration named
  `infinitydb_porter`, and the window index uses it. With PostgreSQL's stock
  `english` configuration, words like "her" and "has" disappear from the index,
  their document frequency becomes zero and their IDF inflates enough to
  reorder results. Creating the configuration is a one-time DDL step and needs
  `CREATE` on the database/schema; the error says so if the role lacks it.
- **Tenant isolation.** Each PostgreSQL tenant gets its own schema
  (`t_<slug>_<digest>`) and the connection's `search_path` points at it, so the
  engine's unqualified table names are tenant-local. This is stronger than the
  `oams_*` contract's RLS for this path because the engine derives ids from
  content: two tenants can hold the identical fact at the identical timestamp,
  and a superuser — which bypasses RLS — still cannot see another tenant's rows.
  `test_engine_storage.py` checks that `public.l1_atoms` does not exist, that
  two tenants with identical memory both keep their rows, and that a role granted
  only one schema cannot read the other's. The shared `infinitydb_porter`
  text-search configuration lives in `public`.
- **Tenant is explicit on SQLite.** One file is one tenant; passing another
  `tenant_id` raises instead of silently ignoring the scope.

The `oams_*` contract keeps its own model: pooled connections, `SET LOCAL` per
transaction, and forced RLS on a `tenant_id` column. The engine store uses a
dedicated connection per tenant and schema isolation instead.

**Permission requirements.** `initialize()` needs `CREATE` on the database (to
add the tenant schema) and on `public` (to create the shared text-search
configuration once); the error message names both when either is missing.

### Measured parity

`benchmarks/arena/backend_parity.py` ingests the public LoCoMo conversations
into fresh SQLite and PostgreSQL stores, asks the same 584 questions at `k=10`
and compares the evidence decision the benchmark scores. One run, PostgreSQL
16.2 + pgvector 0.6.2 from the `pgserver` wheel, local Unix socket
([results](../benchmarks/arena/backend-parity-results.json)):

| metric | SQLite | PostgreSQL |
| --- | --- | --- |
| evidence recall@10 (unbudgeted) | 87.97% | **88.14%** |
| evidence recall@10, `budget_tokens=420` | 69.07% | 69.07% |
| recall p50 / p95 | 15.7 / 23.3 ms | 63.8 / 102.8 ms |
| ingest p50 per session | 14.9 ms | 50.7 ms |

Per question, the unbudgeted run agrees on 573/582 evidence decisions
(PostgreSQL gains 5, loses 4) and returns byte-identical memories on 84/584
questions. At 420 tokens it agrees on 564/582 with 9 gains and 9 losses and
returns identical memories on 222/584. Porter and Snowball stemming, plus the
different candidate ordering, explain the remaining flips; they are reported
rather than hidden. These are retrieval proxies, not reader accuracy; one run,
one host.

The gate fails if PostgreSQL's aggregate evidence recall is below SQLite's on
the same run. `scripts/verify.sh` runs a 40-question subset when the extra and
`pgserver` are installed, and says it skipped otherwise.

### Measured scale and isolation

`benchmarks/postgres_scale_bench.py` writes 1,000 facts into each of 100
schemas, every fact carrying a token unique to its tenant, then reads back
through the engine ([results](../benchmarks/postgres-scale-results.json)):

| metric | value |
| --- | --- |
| atoms stored / tenant schemas | 100,000 / 100 |
| `add()` p50 / p95 / p99 | 2.20 / 4.43 / 7.01 ms |
| `recall(k=10)` p50 / p95 / p99 | 25.37 / 33.71 / 36.42 ms |
| foreign-marker rows in storage | 0 |
| foreign-marker results in 500 recalls | 0 |
| rows seen by unqualified reads | 1,000 per tenant |

```bash
pip install "infinitydb-ai[postgres]" pgserver
python benchmarks/postgres_scale_bench.py --total-atoms 100000 --tenants 100 --skip-locomo
```

The isolation check is deliberately not marker-only: half the queries omit the
tenant marker, and every tenant's rows are scanned for every other tenant's
marker, so a leak would surface even where ranking happens to return nothing.
This is schema isolation verified through `search_path`, not an RLS policy, and
it is single-host, single-process evidence with synthetic facts. Ingest averaged
220 atoms/s because each `add()` commits its own transaction, and one add took
29 s (a checkpoint); the p99 is 7 ms. Read it as a floor for a production
deployment, not a capacity guarantee.
