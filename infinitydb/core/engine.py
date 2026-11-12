"""InfinityDB - local agent memory engine.

Architecture:
- L0: Raw conversation and tool logs
- L1: Atomic facts extracted by deterministic rules
- L2: Episode graph for extracted causal and temporal relationships
- L3: Structured persona records
- L4: Tenant-scoped policies
- L5: Time-decay and access-reinforcement scoring
- L6: SHA-256 provenance records linking facts to source logs

Extraction coverage and retrieval performance depend on input and workload.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from array import array
import sqlite3
from collections import Counter


def normalise_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Memory:
    """Legacy key-value memory item for backwards compatibility."""

    key: str
    content: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# A chat transcript line: "Caroline: I moved to Madrid". Speaker names are short
# labels, never timestamps (which start with a digit) or URLs.
_DIALOGUE_LINE_RE = re.compile(r"^(?P<speaker>[A-Z][A-Za-z .'\-]{0,40}?)\s*:\s+(?P<text>\S.*)$")


# Relative time expressions that carry no absolute meaning on their own. The
# offset is in days; year phrases are handled separately because a year is what
# people actually answer ("2022"), not a date.
_RELATIVE_DAY_PHRASES: dict[str, int] = {
    "yesterday": -1,
    "today": 0,
    "tomorrow": 1,
    "last night": -1,
    "last week": -7,
    "next week": 7,
    "last month": -30,
    "next month": 30,
}

_NUMBER_WORDS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "few": 3,
    "a few": 3,
    "couple": 2,
    "couple of": 2,
}


def normalise_relative_dates(text: str, ts: float) -> str:
    """Rewrite relative time expressions as absolute dates using event time.

    "last year" cannot answer a question about the year until it is anchored to
    the moment it was said. This resolves the fixed phrases below against ``ts``
    and leaves everything else alone:

        "I painted a sunrise last year"  (2023-05-08)  -> "... in 2022"
        "my sister visited yesterday"                   -> "... 2023-05-07"
        "three weeks ago I moved"                       -> "2023-04-17 I moved"

    Deliberately conservative: "the last year" and "past year" are durations,
    not points in time, and are not rewritten. Ambiguous cases ("last summer")
    are left untouched rather than guessed at. The caller's raw L0 log keeps the
    original wording, so this transform is derived data and can be rebuilt.
    """
    import calendar
    from datetime import date, datetime, timedelta, timezone

    if not isinstance(text, str) or not text:
        return text
    if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(float(ts)):
        return text

    day = datetime.fromtimestamp(float(ts), timezone.utc).date()
    guard = r"(?<!\bthe )(?<!\bpast )"
    out = text

    for phrase, offset in _RELATIVE_DAY_PHRASES.items():
        pattern = re.compile(guard + r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        out = pattern.sub(
            lambda match, days=offset: (day + timedelta(days=days)).isoformat(), out
        )

    out = re.sub(
        guard + r"\blast year\b", str(day.year - 1), out, flags=re.IGNORECASE
    )
    out = re.sub(
        guard + r"\bnext year\b", str(day.year + 1), out, flags=re.IGNORECASE
    )

    def shift_months(start: date, months: int) -> date:
        index = start.year * 12 + (start.month - 1) + months
        year, zero_based_month = divmod(index, 12)
        month = zero_based_month + 1
        return date(year, month, min(start.day, calendar.monthrange(year, month)[1]))

    def rewrite_ago(match: "re.Match[str]") -> str:
        raw_count = (match.group(1) or "a").lower()
        unit = match.group(2).lower().rstrip("s")
        count = int(raw_count) if raw_count.isdigit() else _NUMBER_WORDS.get(raw_count, 0)
        if not count:
            return match.group(0)
        if unit == "day":
            return (day - timedelta(days=count)).isoformat()
        if unit == "week":
            return (day - timedelta(weeks=count)).isoformat()
        if unit == "month":
            return shift_months(day, -count).isoformat()
        if unit == "year":
            return str(day.year - count)
        return match.group(0)

    out = re.sub(
        r"\b(a few|couple of|\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(days?|weeks?|months?|years?)\s+ago\b",
        rewrite_ago,
        out,
        flags=re.IGNORECASE,
    )
    return out


class InfinityDB:
    """InfinityDB L0-L6 Brain Engine.

    Supports both in-memory and persistent SQLite with WAL mode,
    FTS5 candidate retrieval with importance and time-decay scoring, causal relation tracking,
    persona conflict resolution, enterprise policies, and cryptographic proof chains.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._is_memory = db_path == ":memory:"
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._seq = 0
        self._last_auto_ts = 0.0
        # Stemmed token counts per turn window, keyed by rowid. Windows are
        # immutable between writes, so this is cleared whenever turns change.
        self._window_terms: dict[int, Counter] = {}

        if not self._is_memory:
            self.conn.execute("PRAGMA journal_mode = WAL;")
            # Match the Node.js SDK's FULL synchronous setting. In WAL mode this
            # requests an fsync of the WAL on each committed transaction.
            self.conn.execute("PRAGMA synchronous = FULL;")
        self.conn.execute("PRAGMA foreign_keys = ON;")

        self._init_schema()

        # Legacy key-value cache; SQLite remains the source of truth.
        self._legacy_memories: dict[str, Memory] = {
            row["memory_key"]: Memory(
                key=row["memory_key"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in self.conn.execute(
                "SELECT memory_key, content, created_at FROM legacy_memories"
            )
        }

    def _init_schema(self) -> None:
        """Initialize relational tables and indices for L0-L6 layers."""
        self.conn.executescript(
            """
            -- L0: Raw conversation logs and tool offloads
            CREATE TABLE IF NOT EXISTS l0_logs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT,
                raw TEXT NOT NULL,
                metadata_json TEXT,
                ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_l0_user_ts ON l0_logs(user_id, ts);

            -- L1: Atomic facts extracted deterministically
            CREATE TABLE IF NOT EXISTS l1_atoms (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                category TEXT NOT NULL,
                source_id TEXT NOT NULL,
                importance REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_l1_user_cat ON l1_atoms(user_id, category);
            CREATE INDEX IF NOT EXISTS idx_l1_fact ON l1_atoms(fact);

            -- Indexed full-text retrieval for L1 atoms. Triggers keep the index in sync.
            CREATE VIRTUAL TABLE IF NOT EXISTS l1_atoms_fts
            USING fts5(fact, content='l1_atoms', content_rowid='rowid',
                       tokenize='unicode61 remove_diacritics 2');
            CREATE TRIGGER IF NOT EXISTS l1_atoms_fts_insert AFTER INSERT ON l1_atoms BEGIN
                INSERT INTO l1_atoms_fts(rowid, fact) VALUES (new.rowid, new.fact);
            END;
            CREATE TRIGGER IF NOT EXISTS l1_atoms_fts_delete AFTER DELETE ON l1_atoms BEGIN
                INSERT INTO l1_atoms_fts(l1_atoms_fts, rowid, fact)
                VALUES ('delete', old.rowid, old.fact);
            END;
            CREATE TRIGGER IF NOT EXISTS l1_atoms_fts_update AFTER UPDATE OF fact ON l1_atoms BEGIN
                INSERT INTO l1_atoms_fts(l1_atoms_fts, rowid, fact)
                VALUES ('delete', old.rowid, old.fact);
                INSERT INTO l1_atoms_fts(rowid, fact) VALUES (new.rowid, new.fact);
            END;

            CREATE TABLE IF NOT EXISTS infinitydb_schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Legacy key-value API records are durable across process restarts.
            CREATE TABLE IF NOT EXISTS legacy_memories (
                memory_key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            -- L2: Episode Graph (Causal & Temporal edges)
            CREATE TABLE IF NOT EXISTS l2_episodes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                cause TEXT NOT NULL,
                effect TEXT NOT NULL,
                relation TEXT NOT NULL,
                source_id TEXT NOT NULL,
                ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_l2_user ON l2_episodes(user_id);

            -- L3: Dynamic Persona with Conflict Resolution
            CREATE TABLE IF NOT EXISTS l3_persona (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_atom_id TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (user_id, key)
            );

            -- L4: Enterprise Policies and SOPs
            CREATE TABLE IF NOT EXISTS l4_policies (
                id TEXT PRIMARY KEY,
                rule TEXT NOT NULL,
                domain TEXT NOT NULL,
                priority REAL NOT NULL,
                created_at REAL NOT NULL,
                tenant_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_l4_domain ON l4_policies(domain);

            -- Explicit, auditable fact supersession. Facts remain immutable in L1.
            CREATE TABLE IF NOT EXISTS l1_atom_relations (
                old_atom_id TEXT NOT NULL,
                new_atom_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                relation TEXT NOT NULL CHECK (relation = 'supersedes'),
                created_at REAL NOT NULL,
                PRIMARY KEY (old_atom_id, new_atom_id),
                CHECK (old_atom_id <> new_atom_id)
            );
            CREATE INDEX IF NOT EXISTS idx_l1_relations_user_time
                ON l1_atom_relations(user_id, created_at);

            -- Typed relation edges, written at ingest. Lexical co-occurrence
            -- says two words appear together; an edge says *how* two things are
            -- related (works at, lives in, studied at, likes), which is what
            -- read-time traversal needs to follow without guessing. Atoms stay
            -- immutable: an edge only points at the atom that asserted it.
            CREATE TABLE IF NOT EXISTS l1_entity_edges (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT,
                atom_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_l1_edges_subject
                ON l1_entity_edges(user_id, subject);
            CREATE INDEX IF NOT EXISTS idx_l1_edges_object
                ON l1_entity_edges(user_id, object);

            -- Optional dense index over turn windows. Vectors are stored
            -- normalised so cosine similarity is a dot product, and are derived
            -- data: they can be rebuilt from the turn text at any time.
            CREATE TABLE IF NOT EXISTS l2_vectors (
                model TEXT NOT NULL,
                window_rowid INTEGER NOT NULL,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL,
                PRIMARY KEY (model, window_rowid)
            );

            -- Turn context: one row per dialogue turn, holding the turn and its
            -- immediate neighbours in the same session. Recall scores this window
            -- (a follow-up turn often only makes sense next to the turn before
            -- it) and resolves hits back to exact turns for rendering.
            CREATE TABLE IF NOT EXISTS l1_turn_windows (
                atom_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT,
                seq INTEGER NOT NULL,
                turn_text TEXT NOT NULL,
                window_text TEXT NOT NULL,
                ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_l1_windows_user_seq
                ON l1_turn_windows(user_id, session_id, seq);
            CREATE VIRTUAL TABLE IF NOT EXISTS l1_turn_windows_fts
            USING fts5(window_text, content='l1_turn_windows', content_rowid='rowid',
                       tokenize='porter unicode61 remove_diacritics 2');
            CREATE TRIGGER IF NOT EXISTS l1_turn_windows_fts_insert AFTER INSERT ON l1_turn_windows BEGIN
                INSERT INTO l1_turn_windows_fts(rowid, window_text)
                VALUES (new.rowid, new.window_text);
            END;
            CREATE TRIGGER IF NOT EXISTS l1_turn_windows_fts_delete AFTER DELETE ON l1_turn_windows BEGIN
                INSERT INTO l1_turn_windows_fts(l1_turn_windows_fts, rowid, window_text)
                VALUES ('delete', old.rowid, old.window_text);
            END;
            CREATE TRIGGER IF NOT EXISTS l1_turn_windows_fts_update AFTER UPDATE OF window_text ON l1_turn_windows BEGIN
                INSERT INTO l1_turn_windows_fts(l1_turn_windows_fts, rowid, window_text)
                VALUES ('delete', old.rowid, old.window_text);
                INSERT INTO l1_turn_windows_fts(rowid, window_text)
                VALUES (new.rowid, new.window_text);
            END;

            -- L6: Cryptographic Proof Chain linking atoms to raw logs
            CREATE TABLE IF NOT EXISTS l6_proof (
                atom_id TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                source_log_id TEXT NOT NULL,
                raw_snippet TEXT NOT NULL,
                char_start INTEGER,
                char_end INTEGER,
                ts REAL NOT NULL
            );
            """
        )
        # Migrate existing L4 policy tables without changing legacy global policies.
        policy_columns = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(l4_policies)")
        }
        if "tenant_id" not in policy_columns:
            self.conn.execute("ALTER TABLE l4_policies ADD COLUMN tenant_id TEXT")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_l4_tenant_priority "
            "ON l4_policies(tenant_id, priority DESC)"
        )

        # The window index is porter-stemmed; rebuild it once if an earlier
        # build of this table used unstemmed tokens.
        windows_version = self.conn.execute(
            "SELECT value FROM infinitydb_schema_meta WHERE key = 'l1_turn_windows_fts_version'"
        ).fetchone()
        if windows_version is None:
            self.conn.execute("DROP TABLE IF EXISTS l1_turn_windows_fts")
            self.conn.execute(
                """CREATE VIRTUAL TABLE l1_turn_windows_fts
                   USING fts5(window_text, content='l1_turn_windows', content_rowid='rowid',
                              tokenize='porter unicode61 remove_diacritics 2')"""
            )
            self.conn.execute("INSERT INTO l1_turn_windows_fts(l1_turn_windows_fts) VALUES ('rebuild')")
            self.conn.execute(
                "INSERT INTO infinitydb_schema_meta(key, value) VALUES (?, ?)",
                ("l1_turn_windows_fts_version", "1"),
            )

        # Build the FTS index once for databases created before FTS was introduced.
        # Triggers maintain it transactionally for all subsequent writes/deletes.
        fts_version = self.conn.execute(
            "SELECT value FROM infinitydb_schema_meta WHERE key = 'l1_atoms_fts_version'"
        ).fetchone()
        if fts_version is None:
            self.conn.execute("INSERT INTO l1_atoms_fts(l1_atoms_fts) VALUES ('rebuild')")
            self.conn.execute(
                "INSERT INTO infinitydb_schema_meta(key, value) VALUES (?, ?)",
                ("l1_atoms_fts_version", "1"),
            )
        self.conn.commit()

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _stem(token: str) -> str:
        """Conservative suffix stripping, mirroring the index's porter tokens.

        Full Porter is not reimplemented here; these rules cover the
        plural/gerund/past forms that otherwise split "walks" from "walking"
        in a question/answer pair.
        """
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 5 and token.endswith(("sses", "shes", "ches", "xes")):
            return token[:-2]
        if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            return token[:-1]
        if len(token) > 5 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 4 and token.endswith("ed"):
            return token[:-2]
        return token

    # -------------------------------------------------------------------------
    # Legacy Key-Value Compatibility (Passes existing tests)
    # -------------------------------------------------------------------------
    @staticmethod
    def _dialogue_units(conversation: str) -> list[tuple[str | None, str, int]]:
        """Split a transcript into (speaker, text, absolute offset) units.

        Only used when every non-empty line is ``Speaker: utterance``; anything
        else (prose, tool logs, mixed text) is returned as one anonymous unit so
        the previous extraction behaviour is preserved.
        """
        units: list[tuple[str | None, str, int]] = []
        offset = 0
        non_empty = 0
        for raw_line in conversation.splitlines(keepends=True):
            line = raw_line.rstrip("\r\n")
            if line.strip():
                non_empty += 1
                match = _DIALOGUE_LINE_RE.match(line)
                if match:
                    units.append(
                        (
                            match.group("speaker").strip(),
                            match.group("text"),
                            offset + match.start("text"),
                        )
                    )
            offset += len(raw_line)
        if non_empty and len(units) == non_empty:
            return units
        return [(None, conversation, 0)]

    def _document_frequency(self, term: str) -> int:
        """Number of indexed atoms containing ``term`` (for IDF weighting)."""
        try:
            cur = self.conn.execute(
                "SELECT count(*) FROM l1_atoms_fts WHERE l1_atoms_fts MATCH ?",
                ('"' + term.replace('"', '""') + '"',),
            )
            return int(cur.fetchone()[0])
        except sqlite3.Error:
            return 0

    def _average_fact_tokens(self) -> float:
        row = self.conn.execute(
            "SELECT AVG(LENGTH(fact) - LENGTH(REPLACE(fact, ' ', '')) + 1) FROM l1_atoms"
        ).fetchone()
        value = row[0] if row and row[0] else None
        return float(value) if value and value > 0 else 1.0

    def put(self, key: str, content: str) -> Memory:
        """Atomically persist a legacy key-value item and its L0-L6 record."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key cannot be empty")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content cannot be empty")
        memory = Memory(key=key, content=content)
        with self.conn:
            # Call the internal writer directly: nested sqlite3 connection
            # context managers can commit independently and break atomicity.
            self._add(f"Memory key {key}: {content}", user_id="legacy")
            self.conn.execute(
                """
                INSERT INTO legacy_memories (memory_key, content, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(memory_key) DO UPDATE SET
                    content = excluded.content,
                    created_at = excluded.created_at
                """,
                (memory.key, memory.content, memory.created_at),
            )
        self._legacy_memories[key] = memory
        return memory

    def get(self, key: str) -> Memory | None:
        """Retrieve a legacy key-value item from the authoritative SQLite store."""
        row = self.conn.execute(
            "SELECT memory_key, content, created_at FROM legacy_memories WHERE memory_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            self._legacy_memories.pop(key, None)
            return None
        memory = Memory(key=row["memory_key"], content=row["content"], created_at=row["created_at"])
        self._legacy_memories[key] = memory
        return memory

    def delete(self, key: str) -> bool:
        """Atomically delete a legacy key-value item."""
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM legacy_memories WHERE memory_key = ?", (key,)
            )
        self._legacy_memories.pop(key, None)
        return cursor.rowcount > 0

    def __len__(self) -> int:
        return int(self.conn.execute(
            "SELECT count(*) FROM legacy_memories"
        ).fetchone()[0])

    # -------------------------------------------------------------------------
    # Core L0-L6 Engine API
    # -------------------------------------------------------------------------
    def add(
        self,
        conversation: str,
        user_id: str = "default",
        session_id: str | None = None,
        importance: float = 0.9,
        ts: float | None = None,
        auto_supersede: bool = True,
        extractor: Any = None,
        normalise_dates: bool = False,
    ) -> str:
        """Atomically ingest one conversation across all memory layers.

        ``ts`` is the Unix time the conversation happened (default: now). Pass it
        when importing history so recall(as_of=...) and decay use event time.
        ``auto_supersede`` marks an older single-valued fact (location, role,
        employer, identity, budget) as superseded when a newer one arrives.

        ``extractor`` is an optional callable ``extractor(text) -> [facts]`` for
        callers that want model-written facts (an LLM, a fine-tuned tagger, a
        batch pipeline). Each fact is a dict with ``fact`` (required), and
        optional ``category``, ``importance`` and ``speaker``. The engine keeps
        zero dependencies: the caller supplies the callable. Without one,
        ingestion stays fully deterministic and offline.

        ``normalise_dates`` resolves relative expressions ("last year", "three
        days ago") into absolute dates using ``ts`` before extraction. Off by
        default: measured on LoCoMo it cost 0.0024 evidence recall (0.7771 vs
        0.7795) because the rewritten tokens change BM25 statistics, while fixing
        reader answers about dates. That trade is the caller's to make.
        """
        with self.conn:
            return self._add(
                conversation=conversation,
                user_id=user_id,
                session_id=session_id,
                importance=importance,
                ts=ts,
                auto_supersede=auto_supersede,
                extractor=extractor,
                normalise_dates=normalise_dates,
            )

    def _add(
        self,
        conversation: str,
        user_id: str = "default",
        session_id: str | None = None,
        importance: float = 0.9,
        ts: float | None = None,
        auto_supersede: bool = True,
        extractor: Any = None,
        normalise_dates: bool = False,
    ) -> str:
        """Ingest raw conversation or tool log into L0-L6 Brain pipeline.

        Executes deterministic rule-based extraction without an LLM call, constructs causal
        edges, updates persona profile with conflict auto-resolution, and links
        cryptographic proof records.
        """
        if not isinstance(conversation, str) or not conversation.strip():
            raise ValueError("conversation cannot be empty")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not isinstance(importance, (int, float)) or isinstance(importance, bool):
            raise TypeError("importance must be a number between 0 and 1")
        if not 0.0 <= float(importance) <= 1.0:
            raise ValueError("importance must be between 0 and 1")
        if extractor is not None and not callable(extractor):
            raise TypeError("extractor must be a callable or None")

        if ts is None:
            # Keep successive event times strictly increasing even when the OS
            # clock has coarse resolution, so two memories written in the same
            # tick can still be ordered and as_of recall is deterministic.
            now = max(time.time(), self._last_auto_ts + 1e-5)
            self._last_auto_ts = now
        else:
            if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
                raise ValueError("ts must be a finite Unix timestamp")
            now = float(ts)
        self._seq += 1
        log_id = f"L0_{self._sha256(conversation + str(now) + str(self._seq))[:16]}"

        # L0: Persist raw log
        self.conn.execute(
            "INSERT INTO l0_logs VALUES (?, ?, ?, ?, ?, ?)",
            (log_id, user_id, session_id or "", conversation, "{}", now),
        )

        # L1 onward sees absolute dates. "last year" is a duration-free reference
        # that means nothing without the event time, and a later question about
        # the year can never match the phrase: the reader answered "Last year."
        # where the gold answer was "2022". The raw L0 log above keeps the
        # original wording, so this is derived data, not a rewrite of the record.
        # Conservative by design: "the last year" and "past year" are durations
        # and are left alone.
        if normalise_dates:
            conversation = normalise_relative_dates(conversation, now)

        # L1: Deterministic Zero-LLM Fact Extraction (Regex & Semantic Patterns)
        # Lead-in phrases are case-insensitive via scoped (?i:...); captured values are not,
        # so "I'm so tired" is NOT an identity fact, while "I'm Maria" is.
        patterns: list[tuple[str, str, float]] = [
            (r"(?i:I am|I['\u2019]m|my name is)\s+([A-Z][a-zA-Z0-9_\-]*(?:\s+[A-Z][a-zA-Z0-9_\-]*)*)", "identity", 0.95),
            (r"(?i:I work as|my role is|I am an?|I['\u2019]m an?)\s+(?!(?i:bit|little|lot|few|big|huge|fan|bad|good|great|real|total|very|so|kind|sort)\b)([^.,\n]+)", "role", 0.90),
            (r"(?i:I work (?:at|for)|I['\u2019]m working (?:at|for)|I joined|I am employed (?:at|by))\s+([^.,\n]+)", "employer", 0.90),
            (r"(?i:I prefer|I like|I love|I use|my preference is)\s+([^.,\n]+)", "preference", 0.92),
            (r"(?i:I hate|I dislike|I avoid|never use)\s+([^.,\n]+)", "dislike", 0.88),
            (r"(?i:my budget is|budget of|budget:\s*)\s*([$\u20ac\u00a3\u00a5]?[0-9,]+(?:\s*[kKmMbB])?)", "budget", 0.94),
            (r"(?i:I live in|I am living in|I['\u2019]m living in|I['\u2019]m based in|based in|located in|I moved to|I['\u2019]ve moved to|I relocated to)\s+([^.,\n]+)", "location", 0.85),
            (r"(?i:our goal is|aiming to|target is|deadline is)\s+([^.,\n]+)", "goal", 0.91),
            (r"(?i:must comply with|required to|constraint:\s*)\s+([^.,\n]+)", "constraint", 0.95),
            (r"(?i:my (?:sister|brother|mother|father|mom|dad|wife|husband|partner|daughter|son|friend|colleague|boss|manager))\s+(?i:lives in|works (?:at|for)|moved to|is based in)\s+([^.,\n]+)", "relation", 0.80),
        ]

        extracted_atoms = []
        # (atom id, turn text) for every dialogue turn, used to maintain the
        # neighbour-window index below.
        dialogue_turns: list[tuple[str, str]] = []
        for speaker, text, base_start in self._dialogue_units(conversation):
            unit_atoms: list[tuple[str, str, float, int, int]] = []
            for pat, cat, weight in patterns:
                for match in re.finditer(pat, text):
                    raw_fact = match.group(0)
                    leading = len(raw_fact) - len(raw_fact.lstrip())
                    fact = raw_fact.strip()
                    start_idx = base_start + match.start() + leading
                    unit_atoms.append(
                        (fact, cat, weight, start_idx, start_idx + len(fact))
                    )

            if speaker:
                # A chat turn is already an atomic, speaker-attributed memory. Keep
                # the turn verbatim so its L6 proof range stays exact instead of
                # storing a pronoun-only fragment that loses who said it.
                stripped = text.strip()
                if stripped:
                    text_start = base_start + (len(text) - len(text.lstrip()))
                    unit_atoms.insert(
                        0, (stripped, "utterance", 0.82, text_start, text_start + len(stripped))
                    )
            elif not unit_atoms:
                # Fallback for prose input: sentences with no recognised pattern.
                for sentence_match in re.finditer(r"[^.!?\n]+", text):
                    raw_sentence = sentence_match.group(0)
                    leading_whitespace = len(raw_sentence) - len(raw_sentence.lstrip())
                    fact = raw_sentence.strip()
                    if len(fact) <= 8:
                        continue
                    start_idx = base_start + sentence_match.start() + leading_whitespace
                    unit_atoms.append((fact, "general", 0.75, start_idx, start_idx + len(fact)))
                    if len(unit_atoms) == 3:
                        break

            for fact, cat, weight, start_idx, end_idx in unit_atoms:
                atom_id = f"L1_{self._sha256(log_id + fact)[:16]}"
                extracted_atoms.append((atom_id, fact, cat, weight, start_idx, end_idx))
                if cat == "utterance" and speaker:
                    dialogue_turns.append((atom_id, f"{speaker}: {fact}"))

        # Optional model-written facts. They are stored as atoms like any other
        # fact, and additionally indexed under a ``::facts`` session so recall
        # can return them without them being mistaken for dialogue turns.
        extractor_turns: list[tuple[str, str]] = []
        if extractor is not None:
            for item in self._run_extractor(extractor, conversation):
                fact = item["fact"]
                cat = item["category"]
                weight = item["importance"]
                atom_id = f"L1_{self._sha256(log_id + fact)[:16]}"
                # A model-written fact is usually a paraphrase, so its offsets are
                # only meaningful when it happens to occur verbatim. -1 marks
                # "this fact was written by the extractor, not copied from text".
                pos = conversation.find(fact)
                if pos >= 0:
                    extracted_atoms.append(
                        (atom_id, fact, cat, weight, pos, pos + len(fact))
                    )
                else:
                    extracted_atoms.append((atom_id, fact, cat, weight, -1, -1))
                if cat in ("fact", "preference", "event", "profile"):
                    extractor_turns.append((atom_id, fact))

        for atom_id, fact, cat, weight, start_idx, end_idx in extracted_atoms:
            # Store L1 Atom
            self.conn.execute(
                """
                INSERT OR REPLACE INTO l1_atoms
                (id, user_id, fact, category, source_id, importance, access_count, ts)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (atom_id, user_id, fact, cat, log_id, weight * float(importance), now),
            )

            # L6: Cryptographic Proof Record
            proof_hash = self._sha256(fact + log_id)
            snippet = (
                conversation[max(0, start_idx - 20) : min(len(conversation), end_idx + 20)]
                if start_idx >= 0
                else fact
            )
            self.conn.execute(
                """
                INSERT OR REPLACE INTO l6_proof
                (atom_id, hash, source_log_id, raw_snippet, char_start, char_end, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (atom_id, proof_hash, log_id, snippet, start_idx, end_idx, now),
            )

            # Automatic supersession of single-valued facts (event-time aware).
            if auto_supersede and cat in self._SINGLE_VALUED:
                self._auto_supersede(user_id, cat, atom_id, fact, now)

            # L3: Persona extraction with Conflict Auto-Resolution
            if cat in ("identity", "role", "employer", "preference", "budget", "location", "dislike"):
                self._update_persona_with_conflict_resolution(
                    user_id=user_id,
                    category=cat,
                    fact=fact,
                    confidence=weight,
                    source_atom_id=atom_id,
                    timestamp=now,
                )

        # Typed edges, written before the window index because supersession
        # from a contradictory edge must be visible to the same transaction.
        session_key = session_id or ""
        if dialogue_turns:
            for atom_id, turn_text in dialogue_turns:
                speaker = turn_text.split(":", 1)[0].strip().lower()
                if not speaker:
                    continue
                for predicate, obj in self._relation_edges(turn_text):
                    edge_id = f"E_{self._sha256(atom_id + predicate + obj)[:16]}"
                    self.conn.execute(
                        """INSERT OR REPLACE INTO l1_entity_edges
                           (id, user_id, session_id, atom_id, subject, predicate, object, ts)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (edge_id, user_id, session_key, atom_id, speaker, predicate, obj, now),
                    )
                    if auto_supersede:
                        self._supersede_conflicting_edge(
                            user_id, atom_id, speaker, predicate, obj, now
                        )

        # Turn-context index: store each dialogue turn next to its neighbours.
        # A follow-up turn often only carries meaning beside the turn before it
        # ("Caroline: yes, that too"), so recall scores the window and resolves
        # back to the exact turns. Windows never cross a session boundary.
        if dialogue_turns:
            self._window_terms.clear()
            session_key = session_id or ""
            last = self.conn.execute(
                """SELECT rowid, seq, turn_text, window_text FROM l1_turn_windows
                   WHERE user_id = ? AND session_id = ?
                   ORDER BY seq DESC LIMIT 1""",
                (user_id, session_key),
            ).fetchone()
            seq = last["seq"] if last else 0
            for atom_id, turn_text in dialogue_turns:
                seq += 1
                if last is not None:
                    window_text = f"{last['turn_text']}\n{turn_text}"
                    # The previous turn now has a successor, so widen its window
                    # as well. This keeps the window symmetric for every turn.
                    self.conn.execute(
                        "UPDATE l1_turn_windows SET window_text = ? WHERE rowid = ?",
                        (f"{last['window_text']}\n{turn_text}", last["rowid"]),
                    )
                else:
                    window_text = turn_text
                self.conn.execute(
                    """INSERT OR REPLACE INTO l1_turn_windows
                       (atom_id, user_id, session_id, seq, turn_text, window_text, ts)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (atom_id, user_id, session_key, seq, turn_text, window_text, now),
                )
                last = {
                    "rowid": self.conn.execute(
                        "SELECT rowid FROM l1_turn_windows WHERE atom_id = ?", (atom_id,)
                    ).fetchone()["rowid"],
                    "seq": seq,
                    "turn_text": turn_text,
                    "window_text": window_text,
                }

        if extractor_turns:
            # Extracted facts share a session of their own: they are already
            # self-contained statements, so they should never be merged into a
            # neighbouring dialogue chunk.
            self._window_terms.clear()
            fact_session = f"{session_id or ''}::facts"
            row = self.conn.execute(
                """SELECT seq FROM l1_turn_windows
                   WHERE user_id = ? AND session_id = ?
                   ORDER BY seq DESC LIMIT 1""",
                (user_id, fact_session),
            ).fetchone()
            seq = row["seq"] if row else 0
            for atom_id, fact in extractor_turns:
                seq += 1
                self.conn.execute(
                    """INSERT OR REPLACE INTO l1_turn_windows
                       (atom_id, user_id, session_id, seq, turn_text, window_text, ts)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (atom_id, user_id, fact_session, seq, fact, fact, now),
                )

        # L2: Causal Episode Extraction (Causal & Temporal Links)
        # e.g., "X because Y", "X so Y", "due to X, Y"
        causal_matches = re.finditer(
            r"(.+?)\s+(?:because|since|as a result of|so that)\s+([^.,\n]+)",
            conversation,
            re.IGNORECASE,
        )
        for c_match in causal_matches:
            effect_part = c_match.group(1).strip()[-80:]
            cause_part = c_match.group(2).strip()[:80]
            if effect_part and cause_part:
                ep_id = f"L2_{self._sha256(log_id + cause_part + effect_part)[:16]}"
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO l2_episodes
                    (id, user_id, cause, effect, relation, source_id, ts)
                    VALUES (?, ?, ?, ?, 'causes', ?, ?)
                    """,
                    (ep_id, user_id, cause_part, effect_part, log_id, now),
                )

        return log_id

    def _run_extractor(self, extractor: Any, conversation: str) -> list[dict[str, Any]]:
        """Normalise whatever the caller's extractor returned into facts.

        Accepts a list of strings, a list of dicts, a single dict, or None.
        Errors raised by the extractor propagate: a silent fallback would make
        an extraction outage look like an empty conversation.
        """
        try:
            raw = extractor(conversation)
        except Exception as exc:
            raise RuntimeError(f"extractor failed: {exc}") from exc
        if raw is None:
            return []
        if isinstance(raw, (str, dict)):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            raise TypeError("extractor must return a string, dict, or list of them")
        facts: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, str):
                item = {"fact": item}
            if not isinstance(item, dict):
                raise TypeError(
                    "extractor returned " + type(item).__name__ + ", expected str or dict"
                )
            text = str(item.get("fact", "")).strip()
            if not text:
                continue
            category = str(item.get("category") or "fact").strip().lower()
            if not re.fullmatch(r"[a-z_]{1,32}", category):
                raise ValueError("extractor category must be lowercase snake_case")
            importance = item.get("importance", 0.9)
            if isinstance(importance, bool) or not isinstance(importance, (int, float)):
                raise TypeError("extractor importance must be a number")
            if not 0.0 <= float(importance) <= 1.0:
                raise ValueError("extractor importance must be between 0 and 1")
            facts.append(
                {"fact": text[:2000], "category": category, "importance": float(importance)}
            )
        return facts

    def _update_persona_with_conflict_resolution(
        self,
        user_id: str,
        category: str,
        fact: str,
        confidence: float,
        source_atom_id: str,
        timestamp: float,
    ) -> None:
        """Resolve conflicts when user preferences change over time.

        If a conflicting preference exists (e.g., previously 'I like Python',
        now 'I prefer Rust'), newer fact with sufficient confidence overrides
        and records the resolved state.
        """
        cur = self.conn.execute(
            "SELECT value, confidence, updated_at FROM l3_persona WHERE user_id = ? AND key = ?",
            (user_id, category),
        )
        row = cur.fetchone()

        if row is None:
            # First time recording this persona facet
            self.conn.execute(
                """
                INSERT INTO l3_persona (user_id, key, value, confidence, source_atom_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, category, fact, confidence, source_atom_id, timestamp),
            )
        else:
            # Conflict auto-resolution: if newer or higher confidence, replace
            prev_updated = row["updated_at"]
            is_newer = timestamp >= prev_updated
            if is_newer:
                self.conn.execute(
                    """
                    UPDATE l3_persona
                    SET value = ?, confidence = ?, source_atom_id = ?, updated_at = ?
                    WHERE user_id = ? AND key = ?
                    """,
                    (fact, confidence, source_atom_id, timestamp, user_id, category),
                )

    # Deterministic typed-relation patterns. Only first-person predicates are
    # matched, so the subject is always known: the speaker of the turn. Fancier
    # parsing would need a model, and a wrong edge is worse than a missing one.
    _EDGE_PATTERNS: tuple[tuple[str, str], ...] = (
        ("works_at", r"(?i:\b(?:i work (?:at|for)|i am working (?:at|for)|i'm working (?:at|for)|i joined|i started working (?:at|for))\s+)([A-Za-z0-9][\w'&.\-]*(?:\s+[A-Za-z0-9][\w'&.\-]*){0,3})"),
        ("lives_in", r"(?i:\b(?:i live in|i am living in|i'm living in|i moved to|i have moved to|i've moved to|i relocated to|i am based in|i'm based in)\s+)([A-Za-z0-9][\w'&.\-]*(?:\s+[A-Za-z0-9][\w'&.\-]*){0,2})"),
        ("studied_at", r"(?i:\b(?:i studied at|i study at|i am studying at|i'm studying at|i went to|i graduated from)\s+)([A-Za-z0-9][\w'&.\-]*(?:\s+[A-Za-z0-9][\w'&.\-]*){0,3})"),
        ("likes", r"(?i:\b(?:i love|i like|i really like|i enjoy)\s+)([A-Za-z0-9][\w'&.\-]*(?:\s+[A-Za-z0-9][\w'&.\-]*){0,3})"),
        ("dislikes", r"(?i:\b(?:i hate|i dislike|i don't like|i do not like|i can't stand)\s+)([A-Za-z0-9][\w'&.\-]*(?:\s+[A-Za-z0-9][\w'&.\-]*){0,3})"),
    )
    _EDGE_OBJECT_TAIL = frozenset({
        "a", "an", "the", "and", "but", "so", "because", "with", "too", "also",
        "now", "again", "there", "here", "this", "that", "it", "them", "him", "her",
    })
    _EDGE_OPPOSITE = {"likes": "dislikes", "dislikes": "likes"}
    _EDGE_PREDICATE_HINTS = {
        "works_at": {"work", "works", "working", "job", "employer", "company", "employed", "employment", "role"},
        "lives_in": {"live", "lives", "living", "home", "city", "town", "address", "moved", "relocated", "based", "reside"},
        "studied_at": {"study", "studied", "school", "college", "university", "degree", "graduated", "class"},
        "likes": {"like", "likes", "love", "loves", "enjoy", "enjoys", "prefer", "favourite", "favorite", "hobby"},
        "dislikes": {"hate", "hates", "dislike", "dislikes", "avoid", "avoids", "allergic"},
    }

    def _relation_edges(self, text: str) -> list[tuple[str, str]]:
        """Return ``(predicate, object)`` pairs asserted in one utterance.

        Objects are trimmed of trailing connectives ("the gym because ..." ->
        "the gym") so an edge stores a thing, not half a sentence.
        """
        found: list[tuple[str, str]] = []
        for predicate, pattern in self._EDGE_PATTERNS:
            for match in re.finditer(pattern, text):
                words = match.group(1).split()
                while len(words) > 1 and words[-1].lower().strip(".,!?;:") in self._EDGE_OBJECT_TAIL:
                    words.pop()
                obj = " ".join(words).strip(" .,!?;:")
                if len(obj) < 2:
                    continue
                if (predicate, obj.lower()) not in [(p, o.lower()) for p, o in found]:
                    found.append((predicate, obj))
        return found

    def _record_supersession(
        self, user_id: str, old_atom_id: str, new_atom_id: str, ts: float
    ) -> None:
        if old_atom_id == new_atom_id:
            return
        self.conn.execute(
            "INSERT OR IGNORE INTO l1_atom_relations "
            "(old_atom_id, new_atom_id, user_id, relation, created_at) "
            "VALUES (?, ?, ?, 'supersedes', ?)",
            (old_atom_id, new_atom_id, user_id, ts),
        )

    def _supersede_conflicting_edge(
        self,
        user_id: str,
        atom_id: str,
        subject: str,
        predicate: str,
        obj: str,
        ts: float,
    ) -> None:
        """Resolve a contradiction the atom layer cannot see.

        "I work at the clinic" and "I work at the hospital" are different atoms
        in different logs with no shared category, so nothing supersedes them.
        A typed edge makes the conflict explicit: same subject, same predicate,
        different object. Opposite polarity (like/dislike of the same object) is
        the same rule with the predicates swapped.
        """
        rows = self.conn.execute(
            "SELECT atom_id, predicate, object, ts FROM l1_entity_edges "
            "WHERE user_id = ? AND subject = ? AND atom_id <> ?",
            (user_id, subject, atom_id),
        ).fetchall()
        for row in rows:
            conflicts = (
                row["predicate"] == predicate
                and row["object"].lower() != obj.lower()
            ) or (
                self._EDGE_OPPOSITE.get(predicate) == row["predicate"]
                and row["object"].lower() == obj.lower()
            )
            if not conflicts:
                continue
            if row["ts"] <= ts:
                self._record_supersession(user_id, row["atom_id"], atom_id, ts)
            else:
                self._record_supersession(user_id, atom_id, row["atom_id"], row["ts"])

    _SINGLE_VALUED = frozenset({"location", "role", "employer", "identity", "budget"})

    # Question words that point at a fact category even when the stored fact shares no words
    # with the question ("where do I live" -> "I moved to Madrid").
    _CATEGORY_HINTS = {
        "location": {"live", "lives", "living", "lived", "reside", "resides", "located", "based", "city", "address", "moved", "relocated"},
        "employer": {"employer", "company", "employed", "work", "works", "working", "joined", "workplace"},
        "role": {"role", "job", "title", "position", "profession", "occupation", "work", "works"},
        "identity": {"name", "called"},
        "budget": {"budget", "spend"},
    }

    def _auto_supersede(self, user_id: str, category: str, atom_id: str, fact: str, ts: float) -> None:
        """Keep one current value per single-valued category, using event time.

        A newer fact supersedes older ones (relation time = the newer fact's event time, so
        recall(as_of=...) sees history correctly). An older fact arriving late is recorded as
        already superseded by the newer one that exists. Identical text is treated as a repeat.
        """
        rows = self.conn.execute(
            "SELECT id, fact, ts FROM l1_atoms WHERE user_id = ? AND category = ? AND id <> ?",
            (user_id, category, atom_id),
        ).fetchall()
        for r in rows:
            if r["fact"].strip().lower() == fact.strip().lower():
                continue
            if r["ts"] <= ts:
                old, new, when = r["id"], atom_id, ts
            else:
                old, new, when = atom_id, r["id"], r["ts"]
            self.conn.execute(
                "INSERT OR IGNORE INTO l1_atom_relations (old_atom_id, new_atom_id, user_id, relation, created_at) "
                "VALUES (?, ?, ?, 'supersedes', ?)",
                (old, new, user_id, when),
            )

    def supersede_fact(
        self, old_atom_id: str, new_atom_id: str, user_id: str
    ) -> str:
        """Record supersession with validation and insertion in one write transaction."""
        if not all(isinstance(v, str) and v.strip() for v in
                   (old_atom_id, new_atom_id, user_id)):
            raise ValueError("atom ids and user_id must be non-empty strings")
        if old_atom_id == new_atom_id:
            raise ValueError("a fact cannot supersede itself")

        # Acquire the SQLite writer reservation before validation so another
        # connection cannot delete or reassign either fact between check and insert.
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            return self._supersede_fact_in_transaction(
                old_atom_id, new_atom_id, user_id
            )

    def _supersede_fact_in_transaction(
        self, old_atom_id: str, new_atom_id: str, user_id: str
    ) -> str:
        old = self.conn.execute(
            "SELECT user_id FROM l1_atoms WHERE id = ?", (old_atom_id,)
        ).fetchone()
        new = self.conn.execute(
            "SELECT user_id FROM l1_atoms WHERE id = ?", (new_atom_id,)
        ).fetchone()
        if old is None or new is None:
            raise ValueError("both facts must exist")
        if old["user_id"] != user_id or new["user_id"] != user_id:
            raise ValueError("both facts must belong to the supplied user_id")
        cycle = self.conn.execute(
            """
            WITH RECURSIVE descendants(atom_id) AS (
                SELECT new_atom_id FROM l1_atom_relations WHERE old_atom_id = ?
                UNION
                SELECT r.new_atom_id FROM l1_atom_relations r
                JOIN descendants d ON r.old_atom_id = d.atom_id
            )
            SELECT 1 FROM descendants WHERE atom_id = ? LIMIT 1
            """, (new_atom_id, old_atom_id)
        ).fetchone()
        if cycle:
            raise ValueError("supersession would create a cycle")
        created_at = time.time()
        self.conn.execute(
            """INSERT INTO l1_atom_relations
               (old_atom_id, new_atom_id, user_id, relation, created_at)
               VALUES (?, ?, ?, 'supersedes', ?)""",
            (old_atom_id, new_atom_id, user_id, created_at),
        )
        return self._sha256(
            f"{user_id}:{old_atom_id}:{new_atom_id}:{created_at}"
        )[:24]

    def _recall_turn_windows(
        self,
        q_terms: list[str],
        n: int,
        user_id: str | None,
        as_of: float | None,
        now: float,
        expansion: float = 0.0,
        rerank: Any = None,
        query: str = "",
        session_id: str | None = None,
        entity_stems: list[set[str]] | None = None,
        dense: Any = None,
    ) -> list[dict[str, Any]]:
        """Recall dialogue turns by scoring each turn inside its neighbour window.

        Consecutive turns are merged into one chunk so the caller receives an
        uninterrupted piece of conversation (with speaker names and one event
        date) instead of overlapping fragments. The window index only contains
        dialogue turns; prose and tool-log memories are handled by
        ``_recall_atoms``.
        """
        match_query = " OR ".join(
            '"' + term.replace('"', '""') + '"' for term in q_terms
        )
        # Corpus statistics are computed *inside the scope*. A scoped query that
        # still divides by the global document count ranks as if the whole store
        # competed for the slot, which is exactly the contamination scoping is
        # meant to remove.
        scope_sql = "" if session_id is None else " JOIN l1_turn_windows w ON w.rowid = f.rowid"
        scope_where = "" if session_id is None else " AND w.session_id = ?"
        scope_params: list[Any] = [] if session_id is None else [session_id]

        def fetch_rows(match_query: str, limit: int) -> list[Any]:
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
                # Scoping is the cheapest recall win there is: with a session or
                # entity filter the ranker never sees the distractors, instead of
                # ranking them and hoping they lose.
                sql += " AND w.session_id = ?"
                params.append(session_id)
            sql += f" ORDER BY bm25(l1_turn_windows_fts) LIMIT {int(limit)}"
            return self.conn.execute(sql, params).fetchall()

        rows = fetch_rows(match_query, 400)
        if not rows and dense is None:
            # With dense fusion a query that shares no term with the store is
            # still answerable, so the early exit only applies to lexical-only
            # recall.
            return []
        total_docs = max(
            1,
            int(
                self.conn.execute(
                    "SELECT count(*) FROM l1_turn_windows_fts f"
                    + scope_sql
                    + " WHERE 1 = 1"
                    + scope_where,
                    scope_params,
                ).fetchone()[0]
            ),
        )

        def document_frequency(term: str) -> int:
            return int(
                self.conn.execute(
                    "SELECT count(*) FROM l1_turn_windows_fts f"
                    + scope_sql
                    + " WHERE l1_turn_windows_fts MATCH ?"
                    + scope_where,
                    ['"' + term.replace('"', '""') + '"', *scope_params],
                ).fetchone()[0]
            )

        idf: dict[str, float] = {}
        for term in {self._stem(t) for t in q_terms}:
            df = document_frequency(term)
            idf[term] = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
        avg_row = self.conn.execute(
            "SELECT AVG(LENGTH(w.window_text) - LENGTH(REPLACE(w.window_text, ' ', '')) + 1) "
            "FROM l1_turn_windows w"
            + (" WHERE w.session_id = ?" if session_id is not None else ""),
            scope_params,
        ).fetchone()
        avg_tokens = float(avg_row[0]) if avg_row and avg_row[0] else 1.0
        k1, b = 1.2, 0.75

        def doc_score(tokens: Counter, doc_len: int, weights: dict[str, float]) -> float:
            total = 0.0
            for term, weight in weights.items():
                # ``_stem`` strips "ed"/"ing" from the stored side ("moved" ->
                # "mov") but leaves a bare "move" alone, so a question using the
                # base verb would score zero against the past tense and the
                # candidate would be dropped. Try the e-final variant too.
                tf = tokens.get(term, 0)
                if tf == 0 and len(term) > 3 and term.endswith("e"):
                    tf = tokens.get(term[:-1], 0)
                if tf == 0:
                    continue
                total += weight * (tf * (k1 + 1.0)) / (
                    tf + k1 * (1.0 - b + b * doc_len / avg_tokens)
                )
            return total

        candidates: list[tuple[Any, Counter, int, float]] = []
        for row in rows:
            tokens = self._window_terms.get(row["window_rowid"])
            if tokens is None:
                tokens = Counter(
                    self._stem(t) for t in re.findall(r"\w+", row["window_text"].lower())
                )
                self._window_terms[row["window_rowid"]] = tokens
            if entity_stems and not all(
                any(stem in tokens for stem in group) for group in entity_stems
            ):
                continue
            doc_len = max(1, sum(tokens.values()))
            score = doc_score(tokens, doc_len, idf)
            if score == 0.0:
                continue
            candidates.append((row, tokens, doc_len, score))

        if expansion > 0.0 and candidates:
            # Light pseudo-relevance feedback: the best few passages often use
            # the vocabulary the question paraphrases ("job" vs "work"). Their
            # rare terms are added at a fraction of the original term weight,
            # so the question still dominates the ranking.
            candidates.sort(key=lambda item: -item[3])
            weights = dict(idf)
            extra: list[str] = []
            for row, tokens, _doc_len, _score in candidates[:3]:
                for term, tf in tokens.items():
                    if term in weights or tf < 1 or len(term) < 3:
                        continue
                    df = document_frequency(term)
                    if 0 < df <= max(2, total_docs * 0.2) and term not in extra:
                        extra.append(term)
                        weights[term] = expansion * math.log(
                            1.0 + (total_docs - df + 0.5) / (df + 0.5)
                        )
                if len(extra) >= 8:
                    break
            if extra:
                candidates = [
                    (row, tokens, doc_len, doc_score(tokens, doc_len, weights))
                    for row, tokens, doc_len, _score in candidates
                ]


        if dense is not None:
            # Dense + lexical fusion. Lexical scoring cannot reach a paraphrase
            # ("car" for "vehicle") at any weight, so the dense ranking supplies
            # candidates the query terms never produced. Fusion is reciprocal
            # rank fusion, which needs no score calibration between the two.
            embedder, model_name, dense_weight = dense
            dense_ranked = self._dense_ranking(
                query or " ".join(q_terms), embedder, model_name, user_id, session_id
            )
            known = {row["window_rowid"] for row, *_ in candidates}
            dense_rank = {}
            for rank, (rid, _score) in enumerate(dense_ranked):
                dense_rank.setdefault(rid, rank)
            missing = [rid for rid in dense_rank if rid not in known][:25]
            if missing:
                clause, params = "", []
                if session_id is not None:
                    clause += " AND w.session_id = ?"
                    params.append(session_id)
                if user_id:
                    clause += " AND w.user_id = ?"
                    params.append(user_id)
                placeholders = ",".join("?" for _ in missing)
                extra_rows = self.conn.execute(
                    "SELECT w.rowid AS window_rowid, w.atom_id, w.session_id, w.seq, "
                    "w.turn_text, w.window_text, w.ts, a.category, a.source_id, "
                    "a.importance, a.access_count FROM l1_turn_windows w "
                    "JOIN l1_atoms a ON a.id = w.atom_id "
                    "WHERE w.rowid IN (" + placeholders + ")" + clause,
                    [*missing, *params],
                ).fetchall()
                for row in extra_rows:
                    tokens = self._window_terms.get(row["window_rowid"])
                    if tokens is None:
                        tokens = Counter(
                            self._stem(t) for t in re.findall(r"\w+", row["window_text"].lower())
                        )
                        self._window_terms[row["window_rowid"]] = tokens
                    if entity_stems and not all(
                        any(stem in tokens for stem in group) for group in entity_stems
                    ):
                        continue
                    candidates.append((row, tokens, max(1, sum(tokens.values())), 0.0))
            lexical_rank = {
                row["window_rowid"]: rank
                for rank, row in enumerate(
                    r for r, _t, _d, score in sorted(candidates, key=lambda item: -item[3])
                    if score > 0.0
                )
            }
            fused = []
            for row, tokens, doc_len, score in candidates:
                rid = row["window_rowid"]
                value = 0.0
                if rid in lexical_rank:
                    value += 1.0 / (60 + lexical_rank[rid])
                if rid in dense_rank:
                    value += dense_weight / (60 + dense_rank[rid])
                fused.append((row, tokens, doc_len, value))
            candidates = fused

        ranked: list[dict[str, Any]] = []
        for row, _tokens, _doc_len, raw in candidates:
            if raw == 0.0:
                continue
            # Turn windows are ranked by lexical relevance alone. Recency is
            # already expressed in the returned event dates, and a decay
            # multiplier here measurably demotes the older sessions that
            # long-horizon questions ask about.
            retention = 1.0
            ranked.append({
                "id": row["atom_id"],
                "category": row["category"],
                "source_id": row["source_id"],
                "score": round(raw, 4),
                "retention": round(retention, 3),
                "ts": row["ts"],
                "_raw": raw,
                "_session": row["session_id"],
                "_seq": row["seq"],
                "_turn": row["turn_text"],
            })
        ranked.sort(key=lambda item: (item["_raw"], item["ts"]), reverse=True)

        # Merge adjacent turns into contiguous chunks. Ranking order decides
        # which chunk "owns" a turn, so a chunk may still grow after it is
        # created; chunks are emitted in the order of their best member.
        # Only the head of the ranking may extend a chunk, and no chunk exceeds
        # three turns (the union of two overlapping one-turn windows). Without
        # these caps a weak tail hit keeps appending unrelated turns and the
        # caller pays for context it will never use.
        chunks: list[dict[str, Any]] = []
        max_chunks = 3 * n if rerank is not None else n
        for item in ranked[: 3 * n]:
            for chunk in chunks:
                if chunk["_session"] != item["_session"]:
                    continue
                if chunk["_hi"] - chunk["_lo"] + 1 >= 4:
                    continue
                if item["_seq"] == chunk["_hi"] + 1:
                    chunk["_hi"] = item["_seq"]
                    chunk["_turns"].append(item["_turn"])
                    break
                if item["_seq"] == chunk["_lo"] - 1:
                    chunk["_lo"] = item["_seq"]
                    chunk["_turns"].insert(0, item["_turn"])
                    chunk["_ts"] = item["ts"]
                    break
            else:
                if len(chunks) >= max_chunks:
                    continue
                chunks.append({
                    "id": item["id"],
                    "category": item["category"],
                    "source_id": item["source_id"],
                    "score": item["score"],
                    "retention": item["retention"],
                    "ts": item["ts"],
                    "_raw": item["_raw"],
                    "_session": item["_session"],
                    "_lo": item["_seq"],
                    "_hi": item["_seq"],
                    "_turns": [item["_turn"]],
                })

        chunks.sort(key=lambda chunk: chunk["_raw"], reverse=True)
        if rerank is not None and len(chunks) > 1:
            # Optional cross-encoder pass over the assembled chunks. Lexical
            # scoring decides which chunks exist; the reranker decides which of
            # them are worth the caller's context window.
            scores = list(rerank(query, ["\n".join(c["_turns"]) for c in chunks]))
            if len(scores) == len(chunks):
                order = sorted(
                    range(len(chunks)), key=lambda i: -float(scores[i])
                )[:n]
                chunks = [chunks[i] for i in order]
        out = [
            {
                "id": chunk["id"],
                "fact": "\n".join(chunk["_turns"]),
                "category": chunk["category"],
                "source_id": chunk["source_id"],
                "score": chunk["score"],
                "retention": chunk["retention"],
                "ts": chunk["ts"],
            }
            for chunk in chunks
        ]
        return out

    def _recall_atoms(
        self,
        query: str,
        q_terms: list[str],
        n: int,
        user_id: str | None,
        as_of: float | None,
        now: float,
        session_id: str | None = None,
        entity_stems: list[set[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Atom-level BM25 recall, used for prose input and legacy stores."""

        def in_scope(fact: str) -> bool:
            if not entity_stems:
                return True
            tokens = {self._stem(t) for t in re.findall(r"\w+", fact.lower())}
            return all(bool(group & tokens) for group in entity_stems)

        if not q_terms:
            return []
        # FTS5 narrows candidates before Python applies importance and time-decay.
        # Quote each token so user input cannot inject FTS operators.
        scored_results = []
        atom_ids_accessed = []
        if q_terms:
            match_query = " OR ".join(
                '"' + term.replace('"', '""') + '"' for term in q_terms
            )
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
            sql += " ORDER BY bm25(l1_atoms_fts) LIMIT 200"
            cur = self.conn.execute(sql, params)
            rows = [row for row in cur.fetchall() if in_scope(row["fact"])]

            # Category routing: add the user's current facts of any category the question points at.
            hinted = sorted(c for c, words in self._CATEGORY_HINTS.items() if words & set(q_terms))
            cat_hits: set[str] = set()
            if hinted:
                marks = ",".join("?" for _ in hinted)
                sql2 = f"""
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
                params2: list[Any] = [*hinted, as_of, as_of, as_of, as_of]
                if user_id:
                    sql2 += " AND a.user_id = ?"
                    params2.append(user_id)
                if session_id is not None:
                    sql2 += " AND a.source_id IN (SELECT id FROM l0_logs WHERE session_id = ?)"
                    params2.append(session_id)
                sql2 += " ORDER BY a.ts DESC LIMIT 50"
                seen_ids = {r["id"] for r in rows}
                for r in self.conn.execute(sql2, params2).fetchall():
                    if not in_scope(r["fact"]):
                        continue
                    cat_hits.add(r["id"])
                    if r["id"] not in seen_ids:
                        rows.append(r)

            total_atoms = max(
                1, int(self.conn.execute("SELECT count(*) FROM l1_atoms").fetchone()[0])
            )
            idf: dict[str, float] = {}
            for term in set(q_terms):
                df = self._document_frequency(term)
                idf[term] = math.log(1.0 + (total_atoms - df + 0.5) / (df + 0.5))
            avg_tokens = self._average_fact_tokens()
            k1, b = 1.2, 0.3

            for row in rows:
                fact = row["fact"]
                fact_lower = fact.lower()
                tokens = Counter(re.findall(r"\w+", fact_lower))
                doc_len = max(1, sum(tokens.values()))

                # Real IDF-weighted BM25: terms that occur in almost every memory
                # (the speakers' own names in a two-person chat, say) carry almost
                # no weight, so content words decide the ranking.
                term_score = 0.0
                for term, term_idf in idf.items():
                    tf = tokens.get(term, 0)
                    if tf == 0:
                        tf = sum(
                            count for token, count in tokens.items()
                            if len(token) > 3 and term in token
                        )
                    if tf == 0:
                        continue
                    term_score += term_idf * (tf * (k1 + 1.0)) / (
                        tf + k1 * (1.0 - b + b * doc_len / avg_tokens)
                    )
                if term_score == 0.0 and row["id"] in cat_hits:
                    term_score = 1.0          # matched by category, not by words
                if query.lower() in fact_lower:
                    term_score += 1.5

                # L5 Ebbinghaus retention: recency nudges the ranking, but the 50%
                # floor keeps long-horizon facts retrievable instead of decaying
                # to zero after a few months.
                delta_days = (now - row["ts"]) / 86400.0
                stability_days = 30.0 * (
                    1.0 + row["importance"] * 2.0 + math.log1p(row["access_count"])
                )
                retention_decay = 0.5 + 0.5 * math.exp(-delta_days / max(stability_days, 1.0))
                total_score = (term_score + row["importance"] * 0.6) * retention_decay

                if term_score > 0 or row["id"] in cat_hits:
                    scored_results.append({
                        "id": row["id"],
                        "fact": fact,
                        "category": row["category"],
                        "source_id": row["source_id"],
                        "score": round(total_score, 4),
                        "retention": round(retention_decay, 3),
                        "ts": row["ts"],
                        "_raw": total_score,
                    })


        # Sort descending by composite score
        # Unrounded score first, then newest event time, so ties never favour stale facts.
        scored_results.sort(key=lambda x: (x["_raw"], x["ts"]), reverse=True)
        # Collapse duplicate memory text (repeated greetings, the same fact stored
        # by several rules) so every returned slot carries new information.
        top_k = []
        seen_facts: set[str] = set()
        for item in scored_results:
            key = re.sub(r"\W+", " ", item["fact"].lower()).strip()
            if key in seen_facts:
                continue
            seen_facts.add(key)
            top_k.append(item)
            if len(top_k) == n:
                break
        for item in top_k:
            item.pop("_raw", None)
        return top_k

    def recall(
        self,
        query: str,
        n: int = 5,
        user_id: str | None = None,
        include_policies: bool = True,
        as_of: float | None = None,
        now: float | None = None,
        reinforce: bool = True,
        turn_windows: bool = True,
        budget_tokens: int | None = None,
        expansion: float = 0.15,
        rerank: Any = None,
        session_id: str | None = None,
        entities: list[str] | None = None,
        dense: Any = None,
        dense_model: str = "default",
        dense_weight: float = 1.0,
    ) -> dict[str, Any]:
        """Hybrid memory recall combining BM25 keyword matching, causal links,

        importance weighting, and L5 Ebbinghaus time-decay scoring.

        Dialogue turns are scored together with the turn before and after them,
        then merged into contiguous chunks; ``turn_windows=False`` scores each
        stored atom on its own text instead (useful for ablation).

        ``expansion`` adds rare terms from the best few passages back into the
        query (light pseudo-relevance feedback), which recovers paraphrases such
        as "job" for "work". Default 0.15; set 0 to score the query alone.

        ``budget_tokens`` caps the words returned: the highest-ranked memories
        are packed until the budget is reached, and at least one memory is
        always returned. This lets a caller trade recall for prompt cost
        explicitly instead of guessing a value of ``n``.

        ``rerank`` is an optional callable ``rerank(query, passages) -> scores``
        (for example a cross-encoder) applied to the merged turn chunks. It is a
        plug-in so the engine keeps zero third-party dependencies; the same
        query with no reranker is the default and is measurably faster.

        ``session_id`` and ``entities`` restrict the search space before ranking:
        only memories from that session, and only memories mentioning every
        given entity, are considered. Scoping by session or entity is how a
        store with many conversations stays as accurate as one conversation.
        Typed relation edges are written at ingest and exposed through
        ``relations()``; they are deliberately not used to re-rank or expand the
        query, because every measured attempt to do so harmed recall (see the
        teardown's experiment log).

        ``dense`` enables semantic fusion: pass ``(embedder, model_name, weight)``
        or call ``recall(..., dense=(embedder, "minilm", 1.0))`` after
        ``index_dense(embedder, "minilm")``. Lexical and dense rankings are fused
        with reciprocal rank fusion, so a paraphrase the query never mentions can
        still surface. Off by default; the core ships no model.
        Recall latency depends on the database size, hardware, and workload.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("n must be a positive integer")
        if n < 1:
            raise ValueError("n must be a positive integer")
        if user_id is not None and (not isinstance(user_id, str) or not user_id.strip()):
            raise ValueError("user_id cannot be empty")
        if not isinstance(turn_windows, bool):
            raise TypeError("turn_windows must be a boolean")
        if budget_tokens is not None and (
            isinstance(budget_tokens, bool)
            or not isinstance(budget_tokens, int)
            or budget_tokens < 1
        ):
            raise ValueError("budget_tokens must be a positive integer or None")
        if not isinstance(expansion, (int, float)) or isinstance(expansion, bool):
            raise TypeError("expansion must be a number between 0 and 1")
        if not 0.0 <= float(expansion) <= 1.0:
            raise ValueError("expansion must be between 0 and 1")
        if rerank is not None and not callable(rerank):
            raise TypeError("rerank must be a callable or None")
        if dense is not None:
            if not (isinstance(dense, (tuple, list)) and len(dense) == 3):
                raise TypeError("dense must be a (embedder, model, weight) tuple or None")
            if not callable(dense[0]):
                raise TypeError("dense embedder must be callable")
            if not isinstance(dense[1], str) or not dense[1].strip():
                raise ValueError("dense model must be a non-empty string")
            if not isinstance(dense[2], (int, float)) or isinstance(dense[2], bool):
                raise TypeError("dense weight must be a number")
            if not 0.0 <= float(dense[2]) <= 5.0:
                raise ValueError("dense weight must be between 0 and 5")
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id.strip()
        ):
            raise ValueError("session_id must be a non-empty string or None")
        entity_stems: list[set[str]] | None = None
        if entities is not None:
            if isinstance(entities, str) or not isinstance(entities, (list, tuple)):
                raise TypeError("entities must be a list of strings or None")
            entity_stems = []
            for entity in entities:
                if not isinstance(entity, str) or not entity.strip():
                    raise ValueError("entities must be non-empty strings")
                stems = {self._stem(t) for t in re.findall(r"\w+", entity.lower())}
                if stems:
                    entity_stems.append(stems)

        if as_of is not None and (
            isinstance(as_of, bool) or not isinstance(as_of, (int, float))
            or not math.isfinite(as_of)
        ):
            raise ValueError("as_of must be a finite Unix timestamp")

        start_time = time.perf_counter()
        _STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "at", "for",
                 "and", "or", "does", "do", "did", "what", "which", "who", "whom", "where", "when", "how", "why",
                 "my", "me", "i", "it", "its", "this", "that", "with", "as", "by", "from", "one"}
        q_all = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
        q_terms = [t for t in q_all if t not in _STOP] or q_all
        # Decay is measured against as_of when given, else an explicit ``now``, else wall-clock.
        now = as_of if as_of is not None else (now if now is not None else time.time())

        # Dialogue turns are retrieved through their neighbour window (a
        # follow-up turn often only makes sense beside the turn before it);
        # prose and legacy facts fall back to atom-level scoring.
        top_k: list[dict[str, Any]] = []
        if q_terms and turn_windows:
            top_k = self._recall_turn_windows(
                q_terms,
                n,
                user_id,
                as_of,
                now,
                expansion=float(expansion),
                rerank=rerank,
                query=query,
                session_id=session_id,
                entity_stems=entity_stems,
                dense=tuple(dense) if dense is not None else None,
            )
        if not top_k:
            top_k = self._recall_atoms(
                query, q_terms, n, user_id, as_of, now, session_id, entity_stems
            )
        elif len(top_k) < n:
            # Merged turn chunks can leave slots unused. Fill them with prose,
            # tool-log and extracted-fact memories that are not already inside
            # a returned chunk, so a mixed store never loses whole modalities.
            seen = "\n".join(item["fact"] for item in top_k).lower()
            for item in self._recall_atoms(
                query, q_terms, n, user_id, as_of, now, session_id, entity_stems
            ):
                if item["fact"].lower() in seen:
                    continue
                top_k.append(item)
                if len(top_k) == n:
                    break

        # Pack the highest-ranked memories into an explicit word budget.
        if budget_tokens is not None and top_k:
            packed: list[dict[str, Any]] = []
            used = 0
            for item in top_k:
                item_tokens = len(re.findall(r"\w+", item["fact"]))
                if packed and used + item_tokens > budget_tokens:
                    continue
                packed.append(item)
                used += item_tokens
                if used >= budget_tokens:
                    break
            top_k = packed

        # Reinforce only the memories actually returned to the caller.
        top_ids = [item["id"] for item in top_k]
        if top_ids and reinforce:
            id_placeholders = ",".join("?" for _ in top_ids)
            self.conn.execute(
                f"UPDATE l1_atoms SET access_count = access_count + 1 WHERE id IN ({id_placeholders})",
                top_ids,
            )
            self.conn.commit()

        # Check L2 Causal edges
        causal_insights = []
        for item in top_k:
            c_cur = self.conn.execute(
                "SELECT cause, effect, relation FROM l2_episodes WHERE source_id = ?",
                (item["source_id"],),
            )
            for c_row in c_cur.fetchall():
                causal_insights.append({
                    "cause": c_row["cause"],
                    "effect": c_row["effect"],
                    "relation": c_row["relation"],
                })

        # Check L4 Enterprise Policies if requested
        relevant_policies = []
        if include_policies:
            if user_id is None:
                p_cur = self.conn.execute(
                    "SELECT id, rule, domain, priority FROM l4_policies "
                    "WHERE tenant_id IS NULL ORDER BY priority DESC LIMIT 3"
                )
            else:
                p_cur = self.conn.execute(
                    "SELECT id, rule, domain, priority FROM l4_policies "
                    "WHERE tenant_id IS NULL OR tenant_id = ? "
                    "ORDER BY priority DESC LIMIT 3",
                    (user_id,),
                )
            for p_row in p_cur.fetchall():
                relevant_policies.append({
                    "id": p_row["id"],
                    "rule": p_row["rule"],
                    "domain": p_row["domain"],
                    "priority": p_row["priority"],
                })

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "results": top_k,
            "causal_graph": causal_insights,
            "policies": relevant_policies,
            "latency_ms": round(elapsed_ms, 2),
            # Token savings depend on the caller's prompt construction and tokenizer;
            # the engine cannot truthfully report a universal percentage.
            "tokens_saved": None,
            "proof_available": True,
        }

    def get_proof(self, atom_id: str) -> dict[str, Any] | None:
        """Retrieve provenance and cryptographically verify it against L0."""
        from infinitydb.core.proof import verify_proof

        cur = self.conn.execute(
            """
            SELECT p.atom_id, p.hash, p.source_log_id, p.raw_snippet,
                   p.char_start, p.char_end, p.ts, l.raw, a.fact
            FROM l6_proof p
            JOIN l0_logs l ON p.source_log_id = l.id
            JOIN l1_atoms a ON a.id = p.atom_id
            WHERE p.atom_id = ?
            """,
            (atom_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        verified = verify_proof(
            row["hash"],
            row["fact"],
            row["source_log_id"],
            row["raw"],
            row["char_start"],
            row["char_end"],
        )
        return {
            "atom_id": row["atom_id"],
            "hash": row["hash"],
            "source_log_id": row["source_log_id"],
            "raw_snippet": row["raw_snippet"],
            "char_range": [row["char_start"], row["char_end"]],
            "verified": verified,
            "timestamp": row["ts"],
        }

    @staticmethod
    def _normalise(vector: list[float]) -> "array":
        return array("f", normalise_vector(vector))

    @staticmethod
    def _dot(query: "array", blob: bytes) -> float:
        vec = array("f")
        vec.frombytes(blob)
        total = 0.0
        for a, b in zip(query, vec):
            total += a * b
        return total

    def index_dense(
        self,
        embedder: Any,
        model: str = "default",
        batch_size: int = 64,
        limit: int | None = None,
    ) -> int:
        """Embed every turn window that has no vector yet for ``model``.

        ``embedder`` is any callable mapping a list of strings to a list of
        float vectors (fastembed, sentence-transformers, or an API client). The
        engine ships none: a dense index is opt-in, so the zero-dependency core
        stays zero-dependency.
        """
        if not callable(embedder):
            raise TypeError("embedder must be a callable")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        rows = self.conn.execute(
            """SELECT w.rowid AS rid, w.window_text AS text
               FROM l1_turn_windows w
               WHERE NOT EXISTS (
                   SELECT 1 FROM l2_vectors v
                   WHERE v.model = ? AND v.window_rowid = w.rowid
               )
               ORDER BY w.rowid""",
            (model,),
        ).fetchall()
        if limit is not None:
            rows = rows[:limit]
        written = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            try:
                vectors = list(embedder([row["text"] for row in batch]))
            except Exception as exc:
                raise RuntimeError(f"dense embedder failed: {exc}") from exc
            if len(vectors) != len(batch):
                raise RuntimeError("dense embedder returned the wrong number of vectors")
            for row, vector in zip(batch, vectors):
                values = [float(x) for x in vector]
                self.conn.execute(
                    "INSERT OR REPLACE INTO l2_vectors (model, window_rowid, dim, vec) "
                    "VALUES (?, ?, ?, ?)",
                    (model, row["rid"], len(values), self._normalise(values).tobytes()),
                )
                written += 1
            self.conn.commit()
        return written

    def _dense_ranking(
        self,
        query: str,
        embedder: Any,
        model: str,
        user_id: str | None,
        session_id: str | None,
        limit: int = 50,
    ) -> list[tuple[int, float]]:
        try:
            vector = list(embedder([query]))[0]
        except Exception as exc:
            raise RuntimeError(f"dense embedder failed: {exc}") from exc
        query_vec = self._normalise([float(x) for x in vector])
        clause, params = "", []
        if session_id is not None:
            clause += " AND w.session_id = ?"
            params.append(session_id)
        if user_id:
            clause += " AND w.user_id = ?"
            params.append(user_id)
        rows = self.conn.execute(
            "SELECT v.window_rowid AS rid, v.vec AS vec FROM l2_vectors v "
            "JOIN l1_turn_windows w ON w.rowid = v.window_rowid "
            "WHERE v.model = ?" + clause,
            [model, *params],
        ).fetchall()
        if not rows:
            raise RuntimeError(
                "dense index is empty for this scope: call index_dense() before recall(dense=...)"
            )
        scored = [(row["rid"], self._dot(query_vec, row["vec"])) for row in rows]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:limit]

    def relations(
        self,
        entity: str,
        user_id: str = "default",
        hops: int = 1,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Typed relation paths out of ``entity``, as stored at ingest.

        ``hops=1`` returns the edges the entity is the subject of;
        ``hops=2`` also follows each object as a subject, which is the
        multi-hop shape ("Caroline -> works_at -> the gym -> located_in ->
        Bristol"). This is an explicit graph query, not a ranking hint: the
        measured attempt to feed these terms into recall is in the teardown's
        experiment log and was reverted.
        """
        if not isinstance(entity, str) or not entity.strip():
            raise ValueError("entity must be a non-empty string")
        if isinstance(hops, bool) or not isinstance(hops, int) or not 1 <= hops <= 3:
            raise ValueError("hops must be an integer between 1 and 3")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        key = entity.strip().lower()
        # Every edge of the tenant is loaded because a hop can arrive from an
        # entity the query never named: a shared object links the two.
        rows = self.conn.execute(
            "SELECT subject, predicate, object, atom_id, ts FROM l1_entity_edges "
            "WHERE user_id = ? ORDER BY ts",
            (user_id,),
        ).fetchall()
        paths: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        frontier = {key}
        for hop in range(1, hops + 1):
            nxt: set[str] = set()
            for row in rows:
                subject = row["subject"].lower()
                obj = row["object"].lower()
                if subject in frontier:
                    direction, other = "out", obj
                elif obj in frontier:
                    # A shared object *is* the link: Caroline works at the gym
                    # and Tom works at the gym relates Caroline to Tom.
                    direction, other = "in", subject
                else:
                    continue
                fingerprint = (row["subject"], row["predicate"], row["object"])
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    paths.append({
                        "hop": hop,
                        "direction": direction,
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                        "atom_id": row["atom_id"],
                        "ts": row["ts"],
                    })
                nxt.add(other)
            frontier = nxt - {key}
            if not frontier or len(paths) >= limit:
                break
        return paths[:limit]

    def rebuild_edges(self, user_id: str | None = None) -> int:
        """Re-extract typed edges from the turn index (migration / repair).

        Databases written before the edge index existed have no edges. Edges are
        derived data, so they can be rebuilt from the immutable turns at any
        time; the atoms, proofs and event times they point at are untouched.
        Returns the number of edges written.
        """
        sql = "SELECT atom_id, user_id, session_id, turn_text, ts FROM l1_turn_windows"
        params: list[Any] = []
        if user_id is not None:
            sql += " WHERE user_id = ?"
            params.append(user_id)
        rows = self.conn.execute(sql, params).fetchall()
        written = 0
        for row in rows:
            speaker = row["turn_text"].split(":", 1)[0].strip().lower() if ":" in row["turn_text"] else ""
            if not speaker or speaker.startswith("user"):
                continue
            for predicate, obj in self._relation_edges(row["turn_text"]):
                edge_id = f"E_{self._sha256(row['atom_id'] + predicate + obj)[:16]}"
                self.conn.execute(
                    """INSERT OR REPLACE INTO l1_entity_edges
                       (id, user_id, session_id, atom_id, subject, predicate, object, ts)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (edge_id, row["user_id"], row["session_id"], row["atom_id"],
                     speaker, predicate, obj, row["ts"]),
                )
                written += 1
        self.conn.commit()
        return written

    def contradictions(self, user_id: str = "default", limit: int = 50) -> list[dict[str, Any]]:
        """List detected contradictions as superseded fact pairs, newest first.

        Every row is a real audit trail: which fact was replaced, by which fact,
        and when the replacement took effect. Nothing is deleted, so an
        ``as_of`` query still sees the history.
        """
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        rows = self.conn.execute(
            """
            SELECT r.old_atom_id, r.new_atom_id, r.created_at,
                   old.fact AS old_fact, new.fact AS new_fact,
                   old.category AS category
            FROM l1_atom_relations r
            JOIN l1_atoms old ON old.id = r.old_atom_id
            JOIN l1_atoms new ON new.id = r.new_atom_id
            WHERE r.user_id = ?
            ORDER BY r.created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [
            {
                "old_atom_id": row["old_atom_id"],
                "new_atom_id": row["new_atom_id"],
                "old_fact": row["old_fact"],
                "new_fact": row["new_fact"],
                "category": row["category"],
                "resolved_at": row["created_at"],
            }
            for row in rows
        ]

    def get_persona(self, user_id: str = "default") -> dict[str, str]:
        """L3 Dynamic Persona: return current resolved user profile."""
        cur = self.conn.execute(
            "SELECT key, value FROM l3_persona WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        return {row["key"]: row["value"] for row in cur.fetchall()}

    def persona_summary(self, user_id: str = "default", max_facts: int = 8) -> str:
        """Compact one-paragraph profile built from the resolved persona.

        Long agent sessions spend thousands of tokens re-reading raw turns to
        answer "who is this user". This returns the distilled state instead.
        It is a *summary* of already-resolved facts, so it is cheap and
        auditable; whether it improves retrieval is a separate question and is
        not claimed here.
        """
        if not isinstance(max_facts, int) or isinstance(max_facts, bool) or max_facts < 1:
            raise ValueError("max_facts must be a positive integer")
        persona = self.get_persona(user_id)
        if not persona:
            return ""
        order = ["identity", "role", "employer", "location", "budget", "goal", "preference", "dislike"]
        keys = [k for k in order if k in persona] + [k for k in persona if k not in order]
        parts = [f"{key}: {persona[key]}" for key in keys[:max_facts]]
        return "Profile — " + "; ".join(parts)

    def add_policy(
        self,
        rule: str,
        domain: str = "general",
        priority: float = 1.0,
        tenant_id: str | None = None,
    ) -> str:
        """Register a global policy or one scoped to a single tenant.

        Global policies (tenant_id=None) remain visible to every tenant.
        Tenant-scoped policies are returned only to the matching tenant.
        """
        if not isinstance(rule, str) or not rule.strip():
            raise ValueError("rule cannot be empty")
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("domain cannot be empty")
        if not isinstance(priority, (int, float)) or isinstance(priority, bool):
            raise TypeError("priority must be a number")
        if not math.isfinite(float(priority)):
            raise ValueError("priority must be finite")
        if tenant_id is not None and (
            not isinstance(tenant_id, str) or not tenant_id.strip()
        ):
            raise ValueError("tenant_id cannot be empty")

        scope = tenant_id or "global"
        pid = f"L4_{self._sha256(rule + domain + scope)[:16]}"
        # Update mutable policy fields without deleting/recreating the row.
        # In particular, created_at records the policy's original registration time.
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO l4_policies
                (id, rule, domain, priority, created_at, tenant_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    rule = excluded.rule,
                    domain = excluded.domain,
                    priority = excluded.priority,
                    tenant_id = excluded.tenant_id
                """,
                (pid, rule.strip(), domain.strip(), float(priority), time.time(), tenant_id),
            )
        return pid

    def export_graph(self, user_id: str = "default", format: str = "mermaid") -> str:
        """Export episodic memory graph to Mermaid or text representation."""
        cur = self.conn.execute(
            "SELECT cause, effect, relation FROM l2_episodes WHERE user_id = ?",
            (user_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return "graph TD\n  Empty[No causal memory recorded]"

        lines = ["graph TD"]
        for idx, row in enumerate(rows):
            c_safe = re.sub(r'["\n]', "", row["cause"])[:30]
            e_safe = re.sub(r'["\n]', "", row["effect"])[:30]
            lines.append(f'  C{idx}["{c_safe}"] -->|{row["relation"]}| E{idx}["{e_safe}"]')
        return "\n".join(lines)

    def delete_user(self, user_id: str) -> dict[str, int]:
        """Atomically erase one tenant without building unbounded SQL parameter lists."""
        with self.conn:
            # A subquery keeps proof deletion independent of SQLite's host-parameter
            # limit, even for tenants with very large fact histories.
            self.conn.execute(
                """DELETE FROM l6_proof
                   WHERE atom_id IN (
                       SELECT id FROM l1_atoms WHERE user_id = ?
                   )""",
                (user_id,),
            )
            self.conn.execute(
                "DELETE FROM l1_atom_relations WHERE user_id = ?", (user_id,)
            )
            c5 = self.conn.execute(
                "DELETE FROM l1_turn_windows WHERE user_id = ?", (user_id,)
            ).rowcount
            self._window_terms.clear()
            c1 = self.conn.execute(
                "DELETE FROM l1_atoms WHERE user_id = ?", (user_id,)
            ).rowcount
            c0 = self.conn.execute(
                "DELETE FROM l0_logs WHERE user_id = ?", (user_id,)
            ).rowcount
            c2 = self.conn.execute(
                "DELETE FROM l2_episodes WHERE user_id = ?", (user_id,)
            ).rowcount
            c3 = self.conn.execute(
                "DELETE FROM l3_persona WHERE user_id = ?", (user_id,)
            ).rowcount
            c4 = self.conn.execute(
                "DELETE FROM l4_policies WHERE tenant_id = ?", (user_id,)
            ).rowcount

        return {
            "l0_logs_deleted": c0,
            "l1_turn_windows_deleted": c5,
            "l1_atoms_deleted": c1,
            "l2_episodes_deleted": c2,
            "l3_persona_deleted": c3,
            "l4_tenant_policies_deleted": c4,
        }

    def close(self) -> None:
        """Close the SQLite connection and release its file and WAL handles."""
        self.conn.close()

    def __enter__(self) -> "InfinityDB":
        """Allow deterministic connection cleanup with a context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def stats(self) -> dict[str, Any]:
        """Return system-wide statistics for L0-L6 memory layers."""
        l0_count = self.conn.execute("SELECT count(*) FROM l0_logs").fetchone()[0]
        l1_count = self.conn.execute("SELECT count(*) FROM l1_atoms").fetchone()[0]
        l2_count = self.conn.execute("SELECT count(*) FROM l2_episodes").fetchone()[0]
        l3_count = self.conn.execute("SELECT count(*) FROM l3_persona").fetchone()[0]
        l4_count = self.conn.execute("SELECT count(*) FROM l4_policies").fetchone()[0]
        l6_count = self.conn.execute("SELECT count(*) FROM l6_proof").fetchone()[0]
        windows_count = self.conn.execute(
            "SELECT count(*) FROM l1_turn_windows"
        ).fetchone()[0]
        return {
            "l0_raw_logs": l0_count,
            "l1_atom_facts": l1_count,
            "l1_turn_windows": windows_count,
            "l2_causal_episodes": l2_count,
            "l3_personas": l3_count,
            "l4_policies": l4_count,
            "l6_cryptographic_proofs": l6_count,
            "backend": "SQLite-WAL",
            # This is a design target, not a measured latency guarantee.
            "p95_recall_target_ms": "<5.0",
        }

# L0 WAL optimization

# L1 deterministic parser <10ms

# L1 importance 0.0-1.0

# L2 causal links

# hybrid scorer v1

# index on fact, ts

# concurrency fix

# GDPR delete

# L6 api

# user_id fix
