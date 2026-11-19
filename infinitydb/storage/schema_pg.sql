-- ============================================================================
-- InfinityDB OAMS-1.0 storage schema for PostgreSQL 14+ with pgvector.
--
-- This file is a template: PostgresStorage substitutes two placeholders before
-- executing it.
--   {p}   table prefix, default "oams_" (so the tables can share a database
--         with the SQLite-shaped core engine tables without name collisions)
--   {dim} embedding dimension, default 384 (BAAI/bge-small-en-v1.5)
--
-- Every table, primary key and secondary index leads with (tenant_id, user_id)
-- so a scoped query never scans another tenant's rows. Row-Level Security is
-- enabled separately by PostgresStorage.initialize() so the policy expressions
-- are visible as real statements rather than a dollar-quoted block.
-- ============================================================================

CREATE TABLE IF NOT EXISTS {p}schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  DOUBLE PRECISION NOT NULL,
    description TEXT NOT NULL
);

-- ----------------------------------------------------------------------------
-- L0: Append-Only Raw Conversation Log
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS {p}turns (
    tenant_id   TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    turn_id     TEXT NOT NULL,
    speaker     TEXT NOT NULL,
    content     TEXT NOT NULL,
    occurred_at DOUBLE PRECISION NOT NULL,
    recorded_at DOUBLE PRECISION NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, user_id, session_id, turn_index),
    UNIQUE (tenant_id, user_id, turn_id)
);

CREATE INDEX IF NOT EXISTS idx_{p}turns_occurred
    ON {p}turns (tenant_id, user_id, occurred_at DESC);

-- ----------------------------------------------------------------------------
-- L1 + L5 + L6: Atomic facts, bi-temporal validity, Ebbinghaus state,
-- SHA-256 provenance chain, optional dense embedding, generated FTS vector.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS {p}atoms (
    tenant_id        TEXT NOT NULL,
    user_id          TEXT NOT NULL,
    atom_id          TEXT NOT NULL,
    chain_seq        BIGINT NOT NULL,
    session_id       TEXT NOT NULL,
    source_turn_id   TEXT NOT NULL,
    fact_text        TEXT NOT NULL,
    subject          TEXT,
    predicate        TEXT,
    object_val       TEXT,
    valid_at         DOUBLE PRECISION NOT NULL,
    invalid_at       DOUBLE PRECISION,
    recorded_at      DOUBLE PRECISION NOT NULL,
    confidence       REAL NOT NULL DEFAULT 1.0,
    stability        REAL NOT NULL DEFAULT 1.0,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at DOUBLE PRECISION NOT NULL,
    prev_hash        TEXT NOT NULL,
    payload_hash     TEXT NOT NULL,
    embedding        vector({dim}),
    fact_tsv         tsvector GENERATED ALWAYS AS (
                         to_tsvector('english', coalesce(fact_text, ''))
                     ) STORED,
    PRIMARY KEY (tenant_id, user_id, atom_id),
    UNIQUE (tenant_id, user_id, chain_seq),
    UNIQUE (tenant_id, user_id, payload_hash)
);

-- Active-memory filtering and L6 chain-slice traversal.
CREATE INDEX IF NOT EXISTS idx_{p}atoms_active_temporal
    ON {p}atoms (tenant_id, user_id, valid_at DESC)
    WHERE invalid_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_{p}atoms_chain_seq
    ON {p}atoms (tenant_id, user_id, chain_seq DESC);

CREATE INDEX IF NOT EXISTS idx_{p}atoms_source_turn
    ON {p}atoms (tenant_id, user_id, source_turn_id);

-- Lexical candidate narrowing (the SQLite FTS5 equivalent).
CREATE INDEX IF NOT EXISTS idx_{p}atoms_fact_tsv_gin
    ON {p}atoms USING GIN (fact_tsv);

-- Partial HNSW cosine index kept compact by indexing active vectors only.
CREATE INDEX IF NOT EXISTS idx_{p}atoms_embedding_hnsw
    ON {p}atoms USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL AND invalid_at IS NULL;

-- ----------------------------------------------------------------------------
-- L2: Causal / temporal / supersession DAG
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS {p}edges (
    tenant_id     TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    edge_id       TEXT NOT NULL,
    src_atom_id   TEXT NOT NULL,
    dst_atom_id   TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight        REAL NOT NULL DEFAULT 1.0,
    valid_at      DOUBLE PRECISION NOT NULL,
    recorded_at   DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (tenant_id, user_id, edge_id),
    FOREIGN KEY (tenant_id, user_id, src_atom_id)
        REFERENCES {p}atoms (tenant_id, user_id, atom_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, user_id, dst_atom_id)
        REFERENCES {p}atoms (tenant_id, user_id, atom_id) ON DELETE CASCADE,
    CHECK (src_atom_id <> dst_atom_id)
);

CREATE INDEX IF NOT EXISTS idx_{p}edges_src
    ON {p}edges (tenant_id, user_id, src_atom_id);

CREATE INDEX IF NOT EXISTS idx_{p}edges_dst
    ON {p}edges (tenant_id, user_id, dst_atom_id);

-- ----------------------------------------------------------------------------
-- L3: Materialised dynamic persona state
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS {p}personas (
    tenant_id    TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    persona_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at   DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (tenant_id, user_id)
);

-- ----------------------------------------------------------------------------
-- L4: Enterprise policy guardrails
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS {p}policies (
    tenant_id  TEXT NOT NULL,
    policy_id  TEXT NOT NULL,
    user_id    TEXT,
    rule_type  TEXT NOT NULL,
    pattern    TEXT NOT NULL,
    action     TEXT NOT NULL,
    priority   INTEGER NOT NULL DEFAULT 100,
    enabled    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (tenant_id, policy_id)
);

CREATE INDEX IF NOT EXISTS idx_{p}policies_lookup
    ON {p}policies (tenant_id, enabled, priority ASC);

-- ----------------------------------------------------------------------------
-- L5: Caller feedback appended by record_feedback(). One row per judgement,
-- kept apart from the atoms so the signal can be re-derived or dropped without
-- touching memory content.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS {p}feedback (
    id         BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    tenant_id  TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    query      TEXT NOT NULL,
    atom_id    TEXT NOT NULL,
    label      REAL NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_{p}feedback_lookup
    ON {p}feedback (tenant_id, user_id, atom_id);
