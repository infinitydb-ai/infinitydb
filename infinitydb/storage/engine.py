"""The core engine's persistence port: one schema, two SQL dialects.

``infinitydb.storage`` (see :mod:`infinitydb.storage.base`) is the portable
L0-L6 contract. This module is a different, narrower seam: the storage
operations the 2,700-line engine actually performs, expressed once so the
Python scoring, chunk assembly, budget packing, persona resolution and
provenance code is identical on SQLite and PostgreSQL.

``SqliteEngineStore`` executes the engine's existing SQL verbatim, so the
default ``InfinityDB(db_path=...)`` path is byte-for-byte the code that produced
the published LoCoMo numbers. ``PostgresEngineStore`` translates the same
statements for PostgreSQL 14+:

* ``?`` placeholders become ``%s``;
* ``INSERT OR REPLACE`` / ``INSERT OR IGNORE`` become ``ON CONFLICT`` upserts;
* the implicit SQLite ``rowid`` becomes an identity column ``rid``;
* FTS5 candidate narrowing becomes ``tsvector`` + GIN (``simple`` for atoms,
  the no-stopword ``public.infinitydb_porter`` configuration for turn windows);
* each tenant gets its own PostgreSQL schema, selected by the connection's
  ``search_path``, so the engine's unqualified table names are tenant-local even
  for a superuser connection that bypasses Row-Level Security.

The engine supplies its own ranking; this module only narrows candidates and
persists rows. Nothing here ships to the default install: ``psycopg`` is
imported lazily by :class:`PostgresEngineStore`.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_ENGINE_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Primary keys used to turn ``INSERT OR REPLACE`` into an explicit upsert. They
# mirror the SQLite DDL in ``infinitydb/core/engine.py``.
_ENGINE_PKS: dict[str, tuple[str, ...]] = {
    "l0_logs": ("id",),
    "l1_atoms": ("id",),
    "l1_atom_relations": ("old_atom_id", "new_atom_id"),
    "l1_entity_edges": ("id",),
    "l1_turn_windows": ("atom_id",),
    "l2_episodes": ("id",),
    "l2_vectors": ("model", "window_rowid"),
    "l3_persona": ("user_id", "key"),
    "l4_policies": ("id",),
    "l6_proof": ("atom_id",),
    "legacy_memories": ("memory_key",),
}

def tenant_schema(tenant_id: str) -> str:
    """A deterministic, quoted-safe PostgreSQL schema name for one tenant.

    A readable slug keeps the schema operable, and the digest keeps two tenants
    whose slugs collide (``acme`` / ``ACME!``) in separate schemas.
    """
    digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9_]+", "_", tenant_id.lower()).strip("_")[:32] or "tenant"
    return f"t_{slug}_{digest}"

_INSERT_REPLACE_RE = re.compile(
    r"^\s*INSERT\s+OR\s+REPLACE\s+INTO\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)",
    re.IGNORECASE | re.DOTALL,
)
_INSERT_IGNORE_RE = re.compile(
    r"^\s*INSERT\s+OR\s+IGNORE\s+INTO\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)",
    re.IGNORECASE | re.DOTALL,
)


def build_tsquery(terms: Sequence[str]) -> str:
    """Build an OR-of-prefixes tsquery from the ranker's query terms.

    The SQLite path lets FTS5 tokenise the terms itself (``_fts_match_terms``);
    PostgreSQL needs an explicit query string. Prefix matching mirrors the
    FTS5 ``"term"*`` atoms, and non-alphanumeric input is stripped so user text
    cannot inject tsquery operators.
    """
    lexemes: list[str] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = re.sub(r"[^\w]+", "", str(term).lower())
        if len(cleaned) < 2 or cleaned in seen:
            continue
        seen.add(cleaned)
        lexemes.append(f"{cleaned}:*")
    return " | ".join(lexemes)


class EngineRow:
    """A result row addressable by column name or by position.

    The engine uses both idioms (``row["fact"]`` and ``row[0]``), and
    ``dict(row)`` appears in the audit export, so a Postgres row is wrapped
    once instead of rewriting every call site.
    """

    __slots__ = ("_mapping", "_values")

    def __init__(self, columns: Sequence[str], values: Sequence[Any]) -> None:
        self._mapping = dict(zip(columns, values))
        self._values = tuple(values)

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, str):
            return self._mapping[key]
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def keys(self):
        return self._mapping.keys()

    def get(self, key: str, default: Any = None) -> Any:
        return self._mapping.get(key, default)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"EngineRow({self._mapping!r})"


class _Cursor:
    """Minimal DB-API cursor surface the engine relies on."""

    def __init__(
        self,
        rows: Sequence[Any] = (),
        rowcount: int = 0,
        lastrowid: int = 0,
    ) -> None:
        self._rows = list(rows)
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    def fetchall(self) -> list[Any]:
        return list(self._rows)

    def fetchone(self) -> Any | None:
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


class SqliteEngineStore:
    """The engine's original persistence path, unchanged.

    The engine still owns its DDL (``InfinityDB._init_schema``); this store only
    adds the batched read primitives and forwards every other statement to the
    wrapped ``sqlite3`` connection.
    """

    backend_name = "sqlite"
    row_alias = "rowid"
    # The published SQLite rows were measured with FTS5 narrowing to 400
    # candidates; keep that number exact rather than re-selecting the same count
    # with a slightly different metric.
    candidate_limit = 400

    def __init__(self, db_path: str = ":memory:", *, tenant_id: str = "default") -> None:
        if tenant_id != "default":
            raise ValueError(
                "the SQLite engine path is single-tenant: pass tenant_id='default' "
                "and use InfinityDB(dsn=...) for multiple tenants"
            )
        self.db_path = db_path
        self.location = db_path
        self.tenant_id = tenant_id
        self._is_memory = db_path == ":memory:"
        raw = sqlite3.connect(db_path, check_same_thread=False)
        raw.row_factory = sqlite3.Row
        if not self._is_memory:
            raw.execute("PRAGMA journal_mode = WAL;")
            raw.execute("PRAGMA synchronous = FULL;")
        raw.execute("PRAGMA foreign_keys = ON;")
        self.connection = _SqliteConnection(raw)

    def initialize(self) -> None:
        """No-op: the engine creates its own tables through ``_init_schema``."""

    def close(self) -> None:
        self.connection.close()

    # -- retrieval primitives ---------------------------------------------

    def lexical_windows(
        self,
        user_id: str | None,
        session_id: str | None,
        as_of: float | None,
        terms: Sequence[str],
        match_query: str,
        limit: int,
    ) -> list[Any]:
        """Dialogue turns whose neighbour window matches, best BM25 first."""
        if not match_query:
            return []
        sql = """
            SELECT w.rowid AS window_rowid, w.atom_id, w.session_id, w.seq,
                   w.turn_text, w.window_text, w.ts,
                   a.category, a.source_id, a.importance, a.access_count
            FROM l1_turn_windows_fts f
            JOIN l1_turn_windows w ON w.rowid = f.rowid
            JOIN l1_atoms a ON a.id = w.atom_id
            WHERE l1_turn_windows_fts MATCH ?
              AND (? IS NULL OR w.ts <= ?)
              AND NOT EXISTS (
                  SELECT 1 FROM l1_atom_relations r
                  WHERE r.old_atom_id = w.atom_id
                    AND (? IS NULL OR r.created_at <= ?)
              )
        """
        params: list[Any] = [match_query, as_of, as_of, as_of, as_of]
        if user_id:
            sql += " AND w.user_id = ?"
            params.append(user_id)
        if session_id is not None:
            sql += " AND w.session_id = ?"
            params.append(session_id)
        sql += f" ORDER BY bm25(l1_turn_windows_fts) LIMIT {int(limit)}"
        return list(self.connection.execute(sql, params).fetchall())

    def window_corpus_stats(
        self, user_id: str | None, session_id: str | None
    ) -> tuple[int, float]:
        join = "" if session_id is None else " JOIN l1_turn_windows w ON w.rowid = f.rowid"
        where = "" if session_id is None else " AND w.session_id = ?"
        params: list[Any] = [] if session_id is None else [session_id]
        total = int(
            self.connection.execute(
                "SELECT count(*) FROM l1_turn_windows_fts f" + join + " WHERE 1 = 1" + where,
                params,
            ).fetchone()[0]
        )
        row = self.connection.execute(
            "SELECT AVG(LENGTH(w.window_text) - LENGTH(REPLACE(w.window_text, ' ', '')) + 1) "
            "FROM l1_turn_windows w"
            + (" WHERE w.session_id = ?" if session_id is not None else ""),
            params,
        ).fetchone()
        avg = float(row[0]) if row and row[0] else 1.0
        return max(1, total), avg

    def window_document_frequency(
        self, term: str, user_id: str | None, session_id: str | None
    ) -> int:
        join = "" if session_id is None else " JOIN l1_turn_windows w ON w.rowid = f.rowid"
        where = "" if session_id is None else " AND w.session_id = ?"
        params: list[Any] = ['"' + term.replace('"', '""') + '"']
        if session_id is not None:
            params.append(session_id)
        return int(
            self.connection.execute(
                "SELECT count(*) FROM l1_turn_windows_fts f"
                + join
                + " WHERE l1_turn_windows_fts MATCH ?"
                + where,
                params,
            ).fetchone()[0]
        )

    def turn_windows_by_ids(
        self, window_ids: Sequence[int], user_id: str | None, session_id: str | None
    ) -> list[Any]:
        """Batched window fetch used by dense fusion (no per-id round trip)."""
        ids = [int(value) for value in window_ids]
        if not ids:
            return []
        clause, params = "", []
        if session_id is not None:
            clause += " AND w.session_id = ?"
            params.append(session_id)
        if user_id:
            clause += " AND w.user_id = ?"
            params.append(user_id)
        placeholders = ",".join("?" for _ in ids)
        return list(
            self.connection.execute(
                "SELECT w.rowid AS window_rowid, w.atom_id, w.session_id, w.seq, "
                "w.turn_text, w.window_text, w.ts, a.category, a.source_id, "
                "a.importance, a.access_count FROM l1_turn_windows w "
                "JOIN l1_atoms a ON a.id = w.atom_id "
                "WHERE w.rowid IN (" + placeholders + ")" + clause,
                [*ids, *params],
            ).fetchall()
        )

    def lexical_atoms(
        self,
        user_id: str | None,
        session_id: str | None,
        as_of: float | None,
        terms: Sequence[str],
        match_query: str,
        limit: int,
    ) -> list[Any]:
        """Atom-level FTS5 candidate narrowing (prose, tools, extracted facts)."""
        if not match_query:
            return []
        sql = """
            SELECT a.id, a.user_id, a.fact, a.category, a.source_id,
                   a.importance, a.access_count, a.ts
            FROM l1_atoms_fts f
            JOIN l1_atoms a ON a.rowid = f.rowid
            WHERE l1_atoms_fts MATCH ?
              AND (? IS NULL OR a.ts <= ?)
              AND NOT EXISTS (
                  SELECT 1 FROM l1_atom_relations r
                  WHERE r.old_atom_id = a.id
                    AND (? IS NULL OR r.created_at <= ?)
              )
        """
        params: list[Any] = [match_query, as_of, as_of, as_of, as_of]
        if user_id:
            sql += " AND a.user_id = ?"
            params.append(user_id)
        if session_id is not None:
            sql += " AND a.source_id IN (SELECT id FROM l0_logs WHERE session_id = ?)"
            params.append(session_id)
        sql += f" ORDER BY bm25(l1_atoms_fts) LIMIT {int(limit)}"
        return list(self.connection.execute(sql, params).fetchall())

    def category_candidates(
        self,
        categories: Sequence[str],
        user_id: str | None,
        session_id: str | None,
        as_of: float | None,
        limit: int,
    ) -> list[Any]:
        """Current facts in categories the question points at, newest first."""
        if not categories:
            return []
        marks = ",".join("?" for _ in categories)
        sql = f"""
            SELECT a.id, a.user_id, a.fact, a.category, a.source_id,
                   a.importance, a.access_count, a.ts
            FROM l1_atoms a
            WHERE a.category IN ({marks})
              AND (? IS NULL OR a.ts <= ?)
              AND NOT EXISTS (
                  SELECT 1 FROM l1_atom_relations r
                  WHERE r.old_atom_id = a.id
                    AND (? IS NULL OR r.created_at <= ?)
              )
        """
        params: list[Any] = [*categories, as_of, as_of, as_of, as_of]
        if user_id:
            sql += " AND a.user_id = ?"
            params.append(user_id)
        if session_id is not None:
            sql += " AND a.source_id IN (SELECT id FROM l0_logs WHERE session_id = ?)"
            params.append(session_id)
        sql += f" ORDER BY a.ts DESC LIMIT {int(limit)}"
        return list(self.connection.execute(sql, params).fetchall())

    def atom_corpus_stats(self) -> tuple[int, float]:
        row = self.connection.execute(
            "SELECT count(*), AVG(LENGTH(fact) - LENGTH(REPLACE(fact, ' ', '')) + 1) "
            "FROM l1_atoms"
        ).fetchone()
        total = max(1, int(row[0] or 0))
        avg = float(row[1]) if row[1] and row[1] > 0 else 1.0
        return total, avg

    def atom_document_frequency(self, term: str) -> int:
        try:
            cur = self.connection.execute(
                "SELECT count(*) FROM l1_atoms_fts WHERE l1_atoms_fts MATCH ?",
                ('"' + term.replace('"', '""') + '"',),
            )
            return int(cur.fetchone()[0])
        except sqlite3.Error:
            return 0

    def record_feedback_row(
        self,
        user_id: str,
        query: str,
        query_stem: str,
        atom_id: str,
        label: float,
        created_at: float,
    ) -> int:
        with self.connection:
            cur = self.connection.execute(
                """INSERT INTO recall_feedback (user_id, query, query_stem, atom_id, label, ts)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, query, query_stem, atom_id, label, created_at),
            )
        return int(cur.lastrowid or 0)


class _SqliteConnection:
    """Thin passthrough so both stores expose one connection interface."""

    def __init__(self, raw: sqlite3.Connection) -> None:
        self._raw = raw
        self._lock = threading.RLock()

    def execute(self, sql: str, params: Sequence[Any] = ()):
        with self._lock:
            return self._raw.execute(sql, params)

    def executescript(self, script: str):
        with self._lock:
            return self._raw.executescript(script)

    def executemany(self, sql: str, seq: Sequence[Sequence[Any]]):
        with self._lock:
            return self._raw.executemany(sql, seq)

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    def close(self) -> None:
        self._raw.close()

    def __enter__(self) -> "_SqliteConnection":
        self._raw.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._raw.__exit__(exc_type, exc, tb)


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------


class PostgresEngineStore:
    """Engine persistence on PostgreSQL 14+ (``infinitydb-ai[postgres]``)."""

    backend_name = "postgres"
    row_alias = "rid"
    # ts_rank_cd and bm25 order candidates differently, so a SQL LIMIT here would
    # truncate a different set than FTS5 does. Fetch wide and let the shared
    # Python ranking apply the cap instead.
    candidate_limit = 20000

    def __init__(self, dsn: str, *, tenant_id: str = "default") -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("dsn must be a non-empty PostgreSQL connection string")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - the install-error test
            raise ImportError(
                'The PostgreSQL engine backend needs the optional extra: '
                'pip install "infinitydb-ai[postgres]"'
            ) from exc
        self.dsn = dsn
        self.location = dsn
        self.tenant_id = tenant_id
        self.schema_name = tenant_schema(tenant_id)
        self._psycopg = psycopg
        self._raw = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        # One schema per tenant. The connection is dedicated to it, so every
        # unqualified table name in the engine's SQL resolves inside this
        # tenant only, even when the role is a superuser that would bypass RLS.
        quoted = '"' + self.schema_name.replace('"', '""') + '"'
        self._raw.execute(
            "SELECT set_config('search_path', %s, false)", (f"{quoted}, public",)
        )
        self.connection = _PostgresConnection(self._raw)
        self._ready = False

    def initialize(self) -> None:
        if self._ready:
            return
        quoted = '"' + self.schema_name.replace('"', '""') + '"'
        try:
            self._raw.execute(f"CREATE SCHEMA IF NOT EXISTS {quoted}")
        except self._psycopg.Error as exc:  # pragma: no cover - permissions
            raise RuntimeError(
                "InfinityDB(dsn=...) creates one PostgreSQL schema per tenant; "
                "grant the role CREATE on this database (and on schema public for "
                "the shared text-search configuration) so it can be created"
            ) from exc

        # FTS5's porter tokenizer stems without dropping stopwords, and its
        # document frequencies count words like "her" and "has". PostgreSQL's
        # stock `english` configuration removes those words, which would inflate
        # their IDF (df=0) and reorder the shared Python ranking. This
        # configuration copies `simple` (no stopwords) and applies the English
        # stemmer to every word token, so the two backends see the same corpus.
        # It lives in `public` because it is shared by every tenant schema.
        if (
            self._raw.execute(
                "SELECT 1 FROM pg_ts_config c JOIN pg_namespace n ON n.oid = c.cfgnamespace "
                "WHERE c.cfgname = 'infinitydb_porter' AND n.nspname = 'public'"
            ).fetchone()
            is None
        ):
            try:
                # The stock `english_stem` dictionary carries a stopword list of
                # its own; an empty StopWords option keeps "her" and "has" as
                # lexemes, exactly like FTS5's porter tokenizer.
                self._raw.execute(
                    "CREATE TEXT SEARCH DICTIONARY public.infinitydb_stem "
                    "(TEMPLATE = snowball, Language = english, StopWords = '')"
                )
                self._raw.execute(
                    "CREATE TEXT SEARCH CONFIGURATION public.infinitydb_porter "
                    "(COPY = simple)"
                )
                self._raw.execute(
                    "ALTER TEXT SEARCH CONFIGURATION public.infinitydb_porter "
                    "ALTER MAPPING FOR asciiword, asciihword, hword_asciipart, "
                    "word, hword, hword_part, numword WITH public.infinitydb_stem"
                )
            except self._psycopg.Error as exc:  # pragma: no cover - permissions
                raise RuntimeError(
                    "the engine's PostgreSQL schema needs a text search "
                    "configuration named infinitydb_porter; grant the role CREATE "
                    "on schema public so it can be created once"
                ) from exc
        schema = Path(__file__).with_name("schema_engine_pg.sql").read_text(encoding="utf-8")
        for statement in _statements(schema):
            self._raw.execute(statement)
        self._ready = True

    def close(self) -> None:
        self._raw.close()

    # -- retrieval primitives ---------------------------------------------

    def lexical_windows(
        self,
        user_id: str | None,
        session_id: str | None,
        as_of: float | None,
        terms: Sequence[str],
        match_query: str,
        limit: int,
    ) -> list[Any]:
        query = build_tsquery(terms)
        if not query:
            return []
        rows = self._raw.execute(
            """
            SELECT w.rid AS window_rowid, w.atom_id, w.session_id, w.seq,
                   w.turn_text, w.window_text, w.ts,
                   a.category, a.source_id, a.importance, a.access_count
            FROM l1_turn_windows w
            JOIN l1_atoms a ON a.id = w.atom_id
            WHERE w.window_tsv @@ to_tsquery('public.infinitydb_porter'::regconfig, %s)
              AND (%s::double precision IS NULL OR w.ts <= %s::double precision)
              AND NOT EXISTS (
                  SELECT 1 FROM l1_atom_relations r
                  WHERE r.old_atom_id = w.atom_id
                    AND (%s::double precision IS NULL OR r.created_at <= %s::double precision)
              )
              AND (%s::text IS NULL OR w.user_id = %s)
              AND (%s::text IS NULL OR w.session_id = %s)
            ORDER BY ts_rank_cd(
                         w.window_tsv, to_tsquery('public.infinitydb_porter'::regconfig, %s)
                     ) DESC,
                     w.ts DESC
            LIMIT %s
            """,
            (
                query,
                as_of,
                as_of,
                as_of,
                as_of,
                user_id,
                user_id,
                session_id,
                session_id,
                query,
                int(limit),
            ),
        ).fetchall()
        return [_row(row) for row in rows]

    def window_corpus_stats(
        self, user_id: str | None, session_id: str | None
    ) -> tuple[int, float]:
        row = self._raw.execute(
            "SELECT count(*) AS total, "
            "AVG(length(window_text) - length(replace(window_text, ' ', '')) + 1) AS avg "
            "FROM l1_turn_windows WHERE (%s::text IS NULL OR session_id = %s)",
            (session_id, session_id),
        ).fetchone()
        row = _row(row)
        total = max(1, int(row[0] or 0))
        avg = float(row[1]) if row[1] and row[1] > 0 else 1.0
        return total, avg

    def window_document_frequency(
        self, term: str, user_id: str | None, session_id: str | None
    ) -> int:
        cleaned = re.sub(r"[^\w]+", "", term.lower())
        if not cleaned:
            return 0
        # Exact lexeme, not a prefix: FTS5's document frequency counts windows
        # that contain the term, and a prefix count (go:* -> good/going) inflates
        # IDF enough to reorder the ranking.
        row = self._raw.execute(
            "SELECT count(*) FROM l1_turn_windows "
            "WHERE window_tsv @@ to_tsquery('public.infinitydb_porter'::regconfig, %s) "
            "AND (%s::text IS NULL OR session_id = %s)",
            (cleaned, session_id, session_id),
        ).fetchone()
        return int(_row(row)[0])

    def turn_windows_by_ids(
        self, window_ids: Sequence[int], user_id: str | None, session_id: str | None
    ) -> list[Any]:
        ids = [int(value) for value in window_ids]
        if not ids:
            return []
        rows = self._raw.execute(
            "SELECT w.rid AS window_rowid, w.atom_id, w.session_id, w.seq, "
            "w.turn_text, w.window_text, w.ts, a.category, a.source_id, "
            "a.importance, a.access_count FROM l1_turn_windows w "
            "JOIN l1_atoms a ON a.id = w.atom_id "
            "WHERE w.rid = ANY(%s::bigint[]) "
            "AND (%s::text IS NULL OR w.user_id = %s) "
            "AND (%s::text IS NULL OR w.session_id = %s)",
            (ids, user_id, user_id, session_id, session_id),
        ).fetchall()
        return [_row(row) for row in rows]

    def lexical_atoms(
        self,
        user_id: str | None,
        session_id: str | None,
        as_of: float | None,
        terms: Sequence[str],
        match_query: str,
        limit: int,
    ) -> list[Any]:
        query = build_tsquery(terms)
        if not query:
            return []
        rows = self._raw.execute(
            """
            SELECT a.id, a.user_id, a.fact, a.category, a.source_id,
                   a.importance, a.access_count, a.ts
            FROM l1_atoms a
            WHERE a.fact_tsv @@ to_tsquery('simple', %s)
              AND (%s::double precision IS NULL OR a.ts <= %s::double precision)
              AND NOT EXISTS (
                  SELECT 1 FROM l1_atom_relations r
                  WHERE r.old_atom_id = a.id
                    AND (%s::double precision IS NULL OR r.created_at <= %s::double precision)
              )
              AND (%s::text IS NULL OR a.user_id = %s)
              AND (%s::text IS NULL OR a.source_id IN (
                    SELECT id FROM l0_logs WHERE session_id = %s))
            ORDER BY ts_rank_cd(a.fact_tsv, to_tsquery('simple', %s)) DESC,
                     a.ts DESC
            LIMIT %s
            """,
            (
                query,
                as_of,
                as_of,
                as_of,
                as_of,
                user_id,
                user_id,
                session_id,
                session_id,
                query,
                int(limit),
            ),
        ).fetchall()
        return [_row(row) for row in rows]

    def category_candidates(
        self,
        categories: Sequence[str],
        user_id: str | None,
        session_id: str | None,
        as_of: float | None,
        limit: int,
    ) -> list[Any]:
        if not categories:
            return []
        rows = self._raw.execute(
            """
            SELECT a.id, a.user_id, a.fact, a.category, a.source_id,
                   a.importance, a.access_count, a.ts
            FROM l1_atoms a
            WHERE a.category = ANY(%s::text[])
              AND (%s::double precision IS NULL OR a.ts <= %s::double precision)
              AND NOT EXISTS (
                  SELECT 1 FROM l1_atom_relations r
                  WHERE r.old_atom_id = a.id
                    AND (%s::double precision IS NULL OR r.created_at <= %s::double precision)
              )
              AND (%s::text IS NULL OR a.user_id = %s)
              AND (%s::text IS NULL OR a.source_id IN (
                    SELECT id FROM l0_logs WHERE session_id = %s))
            ORDER BY a.ts DESC
            LIMIT %s
            """,
            (
                list(categories),
                as_of,
                as_of,
                as_of,
                as_of,
                user_id,
                user_id,
                session_id,
                session_id,
                int(limit),
            ),
        ).fetchall()
        return [_row(row) for row in rows]

    def atom_corpus_stats(self) -> tuple[int, float]:
        row = self._raw.execute(
            "SELECT count(*) AS total, "
            "AVG(length(fact) - length(replace(fact, ' ', '')) + 1) AS avg "
            "FROM l1_atoms"
        ).fetchone()
        row = _row(row)
        total = max(1, int(row[0] or 0))
        avg = float(row[1]) if row[1] and row[1] > 0 else 1.0
        return total, avg

    def atom_document_frequency(self, term: str) -> int:
        cleaned = re.sub(r"[^\w]+", "", term.lower())
        if not cleaned:
            return 0
        try:
            row = self._raw.execute(
                "SELECT count(*) FROM l1_atoms WHERE fact_tsv @@ to_tsquery('simple', %s)",
                (cleaned,),
            ).fetchone()
            return int(_row(row)[0])
        except self._psycopg.Error:
            return 0

    def record_feedback_row(
        self,
        user_id: str,
        query: str,
        query_stem: str,
        atom_id: str,
        label: float,
        created_at: float,
    ) -> int:
        row = self._raw.execute(
            """INSERT INTO recall_feedback (user_id, query, query_stem, atom_id, label, ts)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (user_id, query, query_stem, atom_id, label, created_at),
        ).fetchone()
        return int(_row(row)[0])


class _PostgresConnection:
    """Translates the engine's SQLite-shaped SQL for PostgreSQL."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self._lock = threading.RLock()

    # -- translation --------------------------------------------------------

    @staticmethod
    def _columns(columns: str) -> list[str]:
        return [part.strip().strip('"') for part in columns.split(",") if part.strip()]

    @classmethod
    def _upsert(cls, table: str, columns: str, tail: str, *, ignore: bool) -> str:
        pks = _ENGINE_PKS.get(table)
        if ignore or not pks:
            return f"INSERT INTO {table} ({columns}){tail} ON CONFLICT DO NOTHING"
        names = cls._columns(columns)
        updates = [f"{name} = EXCLUDED.{name}" for name in names if name not in pks]
        conflict = f"({', '.join(pks)})"
        if not updates:
            return f"INSERT INTO {table} ({columns}){tail} ON CONFLICT {conflict} DO NOTHING"
        return (
            f"INSERT INTO {table} ({columns}){tail} ON CONFLICT {conflict} "
            "DO UPDATE SET " + ", ".join(updates)
        )

    @classmethod
    def translate(cls, sql: str) -> str:
        match = _INSERT_REPLACE_RE.match(sql)
        if match:
            tail = sql[match.end() :].rstrip().rstrip(";")
            sql = cls._upsert(match.group(1), match.group(2), tail, ignore=False)
        else:
            match = _INSERT_IGNORE_RE.match(sql)
            if match:
                tail = sql[match.end() :].rstrip().rstrip(";")
                sql = cls._upsert(match.group(1), match.group(2), tail, ignore=True)
        if sql.strip().upper().startswith("BEGIN IMMEDIATE"):
            return ""
        # Only SQLite's implicit rowid is renamed; `window_rowid` is a real
        # column name and must survive translation.
        sql = re.sub(r"\browid\b", "rid", sql)
        return sql.replace("?", "%s")

    # -- DB-API surface -----------------------------------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> _Cursor:
        with self._lock:
            translated = self.translate(sql)
            if not translated.strip():
                return _Cursor()
            cur = self._raw.execute(translated, tuple(params))
            if cur.description is None:
                return _Cursor(rowcount=int(cur.rowcount or 0))
            columns = [column.name for column in cur.description]
            rows = [EngineRow(columns, [row[name] for name in columns]) for row in cur.fetchall()]
            return _Cursor(rows, rowcount=int(cur.rowcount or 0))

    def executemany(self, sql: str, seq: Sequence[Sequence[Any]]):
        with self._lock:
            cur = self._raw.executemany(self.translate(sql), [tuple(row) for row in seq])
            return _Cursor(rowcount=int(cur.rowcount or 0))

    def executescript(self, script: str) -> None:  # pragma: no cover - not used
        raise NotImplementedError("the PostgreSQL engine store creates its own schema")

    def commit(self) -> None:
        # autocommit=True: each statement is already durable.
        return None

    def rollback(self) -> None:
        self._raw.execute("ROLLBACK")

    def close(self) -> None:
        self._raw.close()

    def __enter__(self) -> "_PostgresConnection":
        self._raw.execute("BEGIN")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self._raw.execute("COMMIT")
        else:
            self._raw.execute("ROLLBACK")


def _row(row: Any) -> EngineRow:
    """psycopg dict_row rows are dicts; wrap them with positional access."""
    if isinstance(row, dict):
        return EngineRow(list(row.keys()), list(row.values()))
    return row


def _statements(schema: str) -> list[str]:
    lines = [line for line in schema.splitlines() if not line.lstrip().startswith("--")]
    return [statement.strip() for statement in "\n".join(lines).split(";") if statement.strip()]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def open_engine_store(
    *,
    db_path: str | None = None,
    dsn: str | None = None,
    tenant_id: str = "default",
    initialize: bool = True,
) -> SqliteEngineStore | PostgresEngineStore:
    """Open the engine's persistence port for SQLite (``db_path``) or Postgres (``dsn``).

    Passing both, or neither, is a programming error rather than a silent choice:
    a caller who thought it was writing to Postgres must never end up in a
    local SQLite file.
    """
    if db_path is not None and dsn is not None:
        raise ValueError("pass either db_path (SQLite) or dsn (PostgreSQL), not both")
    if db_path is None and dsn is None:
        raise ValueError("pass db_path (SQLite) or dsn (PostgreSQL)")

    store: SqliteEngineStore | PostgresEngineStore
    if dsn is not None:
        store = PostgresEngineStore(dsn, tenant_id=tenant_id)
    else:
        store = SqliteEngineStore(db_path or ":memory:", tenant_id=tenant_id)
    if initialize:
        store.initialize()
    return store
