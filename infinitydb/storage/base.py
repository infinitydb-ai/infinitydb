"""The storage contract behind InfinityDB's L0-L6 memory hierarchy.

Every operation is explicitly scoped to one ``(tenant_id, user_id)`` pair: the
backends filter on both columns in SQL, so a forgotten predicate cannot widen a
query across tenants, and the Postgres backend additionally enables Row-Level
Security as a second line of defence.

The default ``sqlite3`` engine is untouched by this module. ``infinitydb.storage``
is the portable persistence layer: ``SqliteStorage`` implements the contract with
the standard library only, and ``PostgresStorage`` implements the same contract
on PostgreSQL + pgvector behind the ``postgres`` extra.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# The hash that starts a user's L6 provenance chain. It is not a real digest:
# it is a fixed sentinel so a chain always has a well-defined first ``prev_hash``.
GENESIS_HASH = "0" * 64

_MAX_EMBEDDING_DIM = 16000
_RELATION_TYPES = frozenset({"CAUSES", "SUPERSEDES", "ELABORATES", "CONTRADICTS", "TEMPORAL"})


def _require_identifier(value: Any, name: str) -> str:
    """Every entry point validates its tenant scope before touching storage."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return value


def normalise_embedding(vector: Sequence[float]) -> list[float]:
    """Scale a vector to unit length.

    Both backends store unit vectors, so cosine similarity is a plain dot
    product (SQLite) and equals ``1 - cosine_distance`` (pgvector). An
    all-zero vector stays all-zero rather than raising: it simply never wins a
    nearest-neighbour query.
    """
    values = [float(value) for value in vector]
    if not values:
        raise ValueError("embedding must not be empty")
    if len(values) > _MAX_EMBEDDING_DIM:
        raise ValueError(f"embedding must have at most {_MAX_EMBEDDING_DIM} dimensions")
    total = 0.0
    for value in values:
        total += value * value
    if total == 0.0:
        return values
    scale = total ** 0.5
    return [value / scale for value in values]


@dataclass(slots=True)
class TurnRecord:
    """L0 append-only raw conversation turn."""

    turn_id: str
    tenant_id: str
    user_id: str
    session_id: str
    turn_index: int
    speaker: str
    content: str
    occurred_at: float
    recorded_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FactAtom:
    """L1 extracted atomic memory + L5 retention state + L6 cryptographic proof.

    ``prev_hash`` and ``payload_hash`` are written by ``StorageBackend.insert_atoms``
    (in place) so the caller can hand the same objects to an audit trail without
    a re-read.
    """

    atom_id: str
    tenant_id: str
    user_id: str
    session_id: str
    source_turn_id: str
    fact_text: str
    subject: str | None = None
    predicate: str | None = None
    object_val: str | None = None
    valid_at: float = 0.0
    invalid_at: float | None = None
    recorded_at: float = 0.0
    confidence: float = 1.0
    # L5 Ebbinghaus retention fields
    stability: float = 1.0
    access_count: int = 0
    last_accessed_at: float = 0.0
    # L6 SHA-256 hash chain fields
    prev_hash: str = ""
    payload_hash: str = ""
    # Optional dense vector (L1/L2 hybrid retrieval)
    embedding: list[float] | None = None
    # Transient scoring fields populated on retrieval; never persisted.
    lexical_score: float = 0.0
    dense_score: float = 0.0


@dataclass(slots=True)
class CausalEdge:
    """L2 directed causal / temporal / supersession edge between atoms."""

    edge_id: str
    tenant_id: str
    user_id: str
    src_atom_id: str
    dst_atom_id: str
    relation_type: str
    weight: float = 1.0
    valid_at: float = 0.0
    recorded_at: float = 0.0


@dataclass(slots=True)
class PolicyRule:
    """L4 tenant-scoped governance / guardrail rule."""

    policy_id: str
    tenant_id: str
    user_id: str | None  # None = applies to the entire tenant
    rule_type: str
    pattern: str
    action: str
    priority: int = 100
    enabled: bool = True
    created_at: float = 0.0


def canonical_payload(atom: FactAtom) -> str:
    """A stable serialisation of the fields an L6 hash must cover.

    Only source-bearing content is included. Retention counters (``access_count``,
    ``stability``) and the invalidated marker are deliberately excluded: they
    mutate after the fact, and a provenance hash that changes when a memory is
    read is a hash nobody can audit.
    """
    body = {
        "atom_id": atom.atom_id,
        "tenant_id": atom.tenant_id,
        "user_id": atom.user_id,
        "session_id": atom.session_id,
        "source_turn_id": atom.source_turn_id,
        "fact_text": atom.fact_text,
        "subject": atom.subject,
        "predicate": atom.predicate,
        "object_val": atom.object_val,
        "valid_at": atom.valid_at,
        "recorded_at": atom.recorded_at,
        "confidence": atom.confidence,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_hash(prev_hash: str, atom: FactAtom) -> str:
    """Hash one atom into its predecessor's chain.

    Tampering with any earlier atom changes the hash of every later one, which is
    what makes ``get_atom_proof_chain`` + ``verify_chain`` meaningful.
    """
    material = f"{prev_hash}\n{canonical_payload(atom)}".encode()
    return hashlib.sha256(material).hexdigest()


def verify_chain(atoms: Sequence[FactAtom], genesis_hash: str | None = None) -> bool:
    """Verify that ``atoms`` form an unbroken hash chain.

    ``get_atom_proof_chain`` returns a prefix of the chain, so when the first
    element is not the genesis atom its own ``prev_hash`` is the anchor.
    """
    if not atoms:
        return True
    previous = genesis_hash if genesis_hash is not None else atoms[0].prev_hash
    for atom in atoms:
        if atom.prev_hash != previous:
            return False
        if payload_hash(previous, atom) != atom.payload_hash:
            return False
        previous = atom.payload_hash
    return True


class StorageBackend(ABC):
    """Persistence contract for OAMS-1.0 (L0-L6) memory.

    Every method MUST enforce strict ``(tenant_id, user_id)`` isolation and MUST
    raise ``ValueError`` for an empty tenant or user identifier.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short label for diagnostics, e.g. ``"sqlite"`` or ``"postgres"``."""

    @abstractmethod
    def initialize(self) -> None:
        """Run idempotent DDL migrations and verify required extensions/indexes."""

    @abstractmethod
    def close(self) -> None:
        """Close connections / release the connection pool."""

    # --- L0: Append-Only Raw Log ---

    @abstractmethod
    def append_turns(self, turns: Sequence[TurnRecord]) -> None:
        """Batch-insert L0 raw turns idempotently."""

    @abstractmethod
    def get_turn_window(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        center_index: int,
        window_radius: int = 1,
    ) -> list[TurnRecord]:
        """Neighbouring turns around a matched turn, for context-window assembly."""

    # --- L1 + L6: Atomic Facts & Cryptographic Provenance Chain ---

    @abstractmethod
    def get_latest_chain_hash(self, tenant_id: str, user_id: str) -> str:
        """Tip ``payload_hash`` for L6 chain continuity (or ``GENESIS_HASH``)."""

    @abstractmethod
    def insert_atoms(self, atoms: Sequence[FactAtom]) -> None:
        """Persist L1 atoms and their L6 chain hashes in one transaction.

        ``prev_hash``/``payload_hash`` are assigned here, serialised per user, so
        two concurrent writers can never fork a user's chain.
        """

    @abstractmethod
    def invalidate_atom(
        self,
        tenant_id: str,
        user_id: str,
        atom_id: str,
        invalid_at: float,
        superseded_by: str | None = None,
    ) -> None:
        """Bi-temporal soft invalidation that preserves the L6 audit history."""

    @abstractmethod
    def get_atom_proof_chain(
        self, tenant_id: str, user_id: str, atom_id: str
    ) -> list[FactAtom]:
        """The ordered L6 hash-chain slice leading up to ``atom_id``."""

    # --- L2: Causal / Relational DAG ---

    @abstractmethod
    def upsert_edges(self, edges: Sequence[CausalEdge]) -> None:
        """Insert or strengthen L2 causal/temporal edges between atoms."""

    @abstractmethod
    def traverse_neighbors(
        self,
        tenant_id: str,
        user_id: str,
        seed_atom_ids: Sequence[str],
        max_hops: int = 1,
        limit: int = 20,
    ) -> list[tuple[FactAtom, CausalEdge]]:
        """Graph expansion from seed atoms along outgoing edges."""

    # --- Hybrid retrieval (lexical + dense) ---

    @abstractmethod
    def search_lexical(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int = 30,
        include_invalidated: bool = False,
    ) -> list[FactAtom]:
        """Candidate narrowing via SQLite FTS5 BM25 or Postgres tsvector ranking."""

    @abstractmethod
    def upsert_embeddings(
        self,
        tenant_id: str,
        user_id: str,
        atom_embeddings: Sequence[tuple[str, list[float]]],
    ) -> None:
        """Attach or replace dense embeddings on existing L1 atoms."""

    @abstractmethod
    def search_dense(
        self,
        tenant_id: str,
        user_id: str,
        query_embedding: Sequence[float],
        limit: int = 30,
        include_invalidated: bool = False,
    ) -> list[FactAtom]:
        """Nearest-neighbour search over active embedded atoms."""

    # --- L3: Dynamic persona summary ---

    @abstractmethod
    def get_persona_state(self, tenant_id: str, user_id: str) -> dict[str, Any]:
        """Load the materialised L3 persona attributes for a user."""

    @abstractmethod
    def upsert_persona_state(
        self, tenant_id: str, user_id: str, state: dict[str, Any], updated_at: float
    ) -> None:
        """Merge new attributes into the materialised L3 persona."""

    # --- L4: Enterprise policy guardrails ---

    @abstractmethod
    def list_active_policies(
        self, tenant_id: str, user_id: str | None = None
    ) -> list[PolicyRule]:
        """Enabled tenant-wide and user-specific policies, lowest priority first."""

    @abstractmethod
    def upsert_policy(self, policy: PolicyRule) -> None:
        """Create or update an L4 governance policy."""

    # --- L5: Ebbinghaus retention & reinforcement ---

    @abstractmethod
    def reinforce_atoms(
        self,
        tenant_id: str,
        user_id: str,
        atom_ids: Sequence[str],
        accessed_at: float,
        stability_multiplier: float = 1.5,
    ) -> None:
        """Bump access counters and stability when a recalled atom is reinforced."""

    # --- Compliance / GDPR right to erasure ---

    @abstractmethod
    def delete_user_data(self, tenant_id: str, user_id: str) -> dict[str, int]:
        """Hard-delete every L0-L6 record for ``(tenant_id, user_id)``."""
