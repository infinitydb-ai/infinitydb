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
import sqlite3
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

    # -------------------------------------------------------------------------
    # Legacy Key-Value Compatibility (Passes existing tests)
    # -------------------------------------------------------------------------
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
    ) -> str:
        """Atomically ingest one conversation across all memory layers."""
        with self.conn:
            return self._add(
                conversation=conversation,
                user_id=user_id,
                session_id=session_id,
                importance=importance,
            )

    def _add(
        self,
        conversation: str,
        user_id: str = "default",
        session_id: str | None = None,
        importance: float = 0.9,
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

        now = time.time()
        self._seq += 1
        log_id = f"L0_{self._sha256(conversation + str(now) + str(self._seq))[:16]}"

        # L0: Persist raw log
        self.conn.execute(
            "INSERT INTO l0_logs VALUES (?, ?, ?, ?, ?, ?)",
            (log_id, user_id, session_id or "", conversation, "{}", now),
        )

        # L1: Deterministic Zero-LLM Fact Extraction (Regex & Semantic Patterns)
        patterns: list[tuple[str, str, float]] = [
            (r"(?:I am|I'm|my name is)\s+([A-Z][a-zA-Z0-9_\-\s]+?)(?:[.,\n]|$)", "identity", 0.95),
            (r"(?:I work as|I am a|my role is)\s+([^.,\n]+)", "role", 0.90),
            (r"(?:I prefer|I like|I love|I use|my preference is)\s+([^.,\n]+)", "preference", 0.92),
            (r"(?:I hate|I dislike|I avoid|never use)\s+([^.,\n]+)", "dislike", 0.88),
            (r"(?:my budget is|budget of|budget:\s*)\s*([$€£¥]?[0-9,]+(?:\s*[kKmMbB])?)", "budget", 0.94),
            (r"(?:I live in|based in|located in)\s+([^.,\n]+)", "location", 0.85),
            (r"(?:our goal is|aiming to|target is|deadline is)\s+([^.,\n]+)", "goal", 0.91),
            (r"(?:must comply with|required to|constraint:\s*)\s+([^.,\n]+)", "constraint", 0.95),
        ]

        extracted_atoms = []
        for pat, cat, weight in patterns:
            for match in re.finditer(pat, conversation, re.IGNORECASE):
                extracted_fact = match.group(0).strip()
                start_idx, end_idx = match.span()
                atom_id = f"L1_{self._sha256(log_id + extracted_fact)[:16]}"
                extracted_atoms.append((atom_id, extracted_fact, cat, weight, start_idx, end_idx))

        # Fallback if no specific regex pattern matched: sentences with key indicators
        if not extracted_atoms:
            # Keep fallback fact offsets aligned with the exact punctuation-free
            # sentence stored in L1 so its L6 source-range proof can verify.
            for sentence_match in re.finditer(r"[^.!?\n]+", conversation):
                raw_sentence = sentence_match.group(0)
                leading_whitespace = len(raw_sentence) - len(raw_sentence.lstrip())
                fact = raw_sentence.strip()
                if len(fact) <= 8:
                    continue
                start_idx = sentence_match.start() + leading_whitespace
                end_idx = start_idx + len(fact)
                atom_id = f"L1_{self._sha256(log_id + fact)[:16]}"
                extracted_atoms.append(
                    (atom_id, fact, "general", 0.75, start_idx, end_idx)
                )
                if len(extracted_atoms) == 3:
                    break

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
            snippet = conversation[max(0, start_idx - 20) : min(len(conversation), end_idx + 20)]
            self.conn.execute(
                """
                INSERT OR REPLACE INTO l6_proof
                (atom_id, hash, source_log_id, raw_snippet, char_start, char_end, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (atom_id, proof_hash, log_id, snippet, start_idx, end_idx, now),
            )

            # L3: Persona extraction with Conflict Auto-Resolution
            if cat in ("identity", "role", "preference", "budget", "location", "dislike"):
                self._update_persona_with_conflict_resolution(
                    user_id=user_id,
                    category=cat,
                    fact=fact,
                    confidence=weight,
                    source_atom_id=atom_id,
                    timestamp=now,
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

    def recall(
        self,
        query: str,
        n: int = 5,
        user_id: str | None = None,
        include_policies: bool = True,
        as_of: float | None = None,
    ) -> dict[str, Any]:
        """Hybrid memory recall combining BM25 keyword matching, causal links,

        importance weighting, and L5 Ebbinghaus time-decay scoring.
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

        if as_of is not None and (
            isinstance(as_of, bool) or not isinstance(as_of, (int, float))
            or not math.isfinite(as_of)
        ):
            raise ValueError("as_of must be a finite Unix timestamp")

        start_time = time.perf_counter()
        q_terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
        now = time.time()

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
            cur = self.conn.execute(sql, params)
            rows = cur.fetchall()

            for row in rows:
                fact = row["fact"]
                fact_lower = fact.lower()
                term_matches = sum(1 for term in q_terms if term in fact_lower)

                # BM25-style term frequency & exact phrase boost.
                term_score = term_matches * 2.5
                if query.lower() in fact_lower:
                    term_score += 4.0

                # L5 Ebbinghaus retention decay.
                delta_days = (now - row["ts"]) / 86400.0
                stability = 1.0 + (row["importance"] * 3.0) + math.log1p(row["access_count"])
                retention_decay = math.exp(-delta_days / max(stability, 0.1))
                total_score = (term_score + row["importance"] * 1.5) * retention_decay

                if term_matches > 0:
                    scored_results.append({
                        "id": row["id"],
                        "fact": fact,
                        "category": row["category"],
                        "source_id": row["source_id"],
                        "score": round(total_score, 4),
                        "retention": round(retention_decay, 3),
                        "ts": row["ts"],
                    })


        # Sort descending by composite score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        top_k = scored_results[:n]

        # Reinforce only the memories actually returned to the caller.
        top_ids = [item["id"] for item in top_k]
        if top_ids:
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

    def get_persona(self, user_id: str = "default") -> dict[str, str]:
        """L3 Dynamic Persona: return current resolved user profile."""
        cur = self.conn.execute(
            "SELECT key, value FROM l3_persona WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        return {row["key"]: row["value"] for row in cur.fetchall()}

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
        return {
            "l0_raw_logs": l0_count,
            "l1_atom_facts": l1_count,
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
