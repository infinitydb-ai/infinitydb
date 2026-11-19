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
2. `initialize()` enables **and forces** Row-Level Security on all five tables,
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

## Scope and current boundary

The default `InfinityDB` engine — the write path and the recall path measured in
the benchmarks — still uses its own SQLite schema and is **not** routed through
this module yet. These tables are prefixed (`oams_` by default) so both layers can
share one database file; `test_sqlite_storage_shares_a_file_with_the_core_engine`
verifies that they coexist. Moving engine recall onto this contract is a separate
change: the engine's retrieval depends on turn windows, feedback learning and
chunk assembly that the contract does not yet model.
