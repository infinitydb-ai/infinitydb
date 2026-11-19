"""The StorageBackend contract, run against SQLite and (when available) Postgres.

The Postgres arm needs the ``pgserver`` test wheel, which bundles PostgreSQL 16
and pgvector; without it those parameters are skipped and CI stays on the
dependency-free SQLite arm.
"""

from __future__ import annotations

import tempfile
import uuid
from types import SimpleNamespace

import pytest

from infinitydb.storage import (
    GENESIS_HASH,
    CausalEdge,
    FactAtom,
    PolicyRule,
    SqliteStorage,
    TurnRecord,
    open_backend,
    verify_chain,
)


@pytest.fixture(scope="session")
def pg_dsn():
    pgserver = pytest.importorskip("pgserver")
    path = tempfile.mkdtemp(prefix="infinitydb-pg-")
    server = pgserver.get_server(path)
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


@pytest.fixture(params=["sqlite", "postgres"])
def storage(request, tmp_path):
    if request.param == "sqlite":
        backend = SqliteStorage(str(tmp_path / "store.db"))
    else:
        pytest.importorskip("psycopg")
        pytest.importorskip("psycopg_pool")
        pytest.importorskip("pgserver")
        from infinitydb.storage.postgres import PostgresStorage

        backend = PostgresStorage(
            request.getfixturevalue("pg_dsn"), table_prefix="oams_test_"
        )
    backend.initialize()
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture
def scope():
    tag = uuid.uuid4().hex[:10]
    return SimpleNamespace(
        tenant=f"tenant-{tag}",
        user=f"user-{tag}",
        other_user=f"other-{tag}",
        session=f"session-{tag}",
    )


def make_atom(scope, atom_id: str, text: str, **overrides) -> FactAtom:
    fields = dict(
        atom_id=atom_id,
        tenant_id=scope.tenant,
        user_id=scope.user,
        session_id=scope.session,
        source_turn_id=f"turn-{atom_id}",
        fact_text=text,
        valid_at=1_000.0,
        recorded_at=1_000.0,
        last_accessed_at=1_000.0,
    )
    fields.update(overrides)
    return FactAtom(**fields)


def make_turn(scope, index: int, **overrides) -> TurnRecord:
    fields = dict(
        turn_id=f"{scope.session}-{index}",
        tenant_id=scope.tenant,
        user_id=scope.user,
        session_id=scope.session,
        turn_index=index,
        speaker="Caroline",
        content=f"line {index}",
        occurred_at=float(index),
        recorded_at=float(index),
    )
    fields.update(overrides)
    return TurnRecord(**fields)


def test_append_turns_is_idempotent_and_windows(storage, scope):
    turns = [make_turn(scope, index) for index in range(5)]
    turns[2].metadata = {"source": "chat", "tags": ["a", "b"]}
    other_session = make_turn(scope, 2, turn_id="elsewhere", session_id="other-session")
    other_user = make_turn(scope, 0, turn_id="someone-else", user_id=scope.other_user)

    storage.append_turns(turns + [other_session, other_user])
    storage.append_turns(turns)  # idempotent

    window = storage.get_turn_window(scope.tenant, scope.user, scope.session, 2, window_radius=1)
    assert [turn.turn_index for turn in window] == [1, 2, 3]
    assert all(turn.session_id == scope.session for turn in window)
    assert window[1].content == "line 2"
    assert window[1].metadata == {"source": "chat", "tags": ["a", "b"]}
    assert storage.get_turn_window(scope.tenant, scope.other_user, scope.session, 2) == []


def test_chain_is_ordered_verifiable_and_per_tenant(storage, scope):
    atoms = [make_atom(scope, f"a{index}", f"fact number {index}") for index in range(3)]
    storage.insert_atoms(atoms)

    assert atoms[0].prev_hash == GENESIS_HASH
    assert atoms[1].prev_hash == atoms[0].payload_hash
    assert storage.get_latest_chain_hash(scope.tenant, scope.user) == atoms[2].payload_hash

    chain = storage.get_atom_proof_chain(scope.tenant, scope.user, "a2")
    assert [atom.atom_id for atom in chain] == ["a0", "a1", "a2"]
    assert verify_chain(chain)
    assert verify_chain(chain[1:])

    later = make_atom(scope, "a3", "fact number 3")
    storage.insert_atoms([later])
    assert later.prev_hash == atoms[2].payload_hash

    # A different user's chain starts at genesis and is invisible to this one.
    stranger = make_atom(scope, "z0", "someone else's fact", user_id=scope.other_user)
    storage.insert_atoms([stranger])
    assert stranger.prev_hash == GENESIS_HASH
    assert storage.get_atom_proof_chain(scope.tenant, scope.user, "z0") == []
    assert storage.get_latest_chain_hash(scope.tenant, scope.other_user) == stranger.payload_hash

    chain[1].fact_text = "tampered"
    assert not verify_chain(chain)


def test_insert_atoms_validates_scope(storage, scope):
    with pytest.raises(ValueError):
        storage.insert_atoms([make_atom(scope, "a0", "x", tenant_id="")])
    with pytest.raises(ValueError):
        storage.insert_atoms(
            [
                make_atom(scope, "a0", "x"),
                make_atom(scope, "a1", "y", user_id=scope.other_user),
            ]
        )
    with pytest.raises(ValueError):
        storage.get_latest_chain_hash(scope.tenant, "")


def test_invalidation_hides_atom_but_keeps_history(storage, scope):
    atoms = [
        make_atom(scope, "old", "Caroline works at the gym in Bristol"),
        make_atom(scope, "new", "Caroline works at the hospital in Bristol"),
    ]
    storage.insert_atoms(atoms)

    before = storage.search_lexical(scope.tenant, scope.user, "gym")
    assert [atom.atom_id for atom in before] == ["old"]

    storage.invalidate_atom(scope.tenant, scope.user, "old", 2_000.0, superseded_by="new")

    assert storage.search_lexical(scope.tenant, scope.user, "gym") == []
    history = storage.search_lexical(
        scope.tenant, scope.user, "gym", include_invalidated=True
    )
    assert [atom.atom_id for atom in history] == ["old"]
    assert history[0].invalid_at == 2_000.0

    chain = storage.get_atom_proof_chain(scope.tenant, scope.user, "new")
    assert chain[0].invalid_at == 2_000.0
    assert chain[0].fact_text == atoms[0].fact_text

    hops = storage.traverse_neighbors(scope.tenant, scope.user, ["new"])
    assert [(atom.atom_id, edge.relation_type) for atom, edge in hops] == [
        ("old", "SUPERSEDES")
    ]


def test_edges_strengthen_and_traverse(storage, scope):
    storage.insert_atoms(
        [make_atom(scope, f"a{index}", f"fact {index}") for index in range(4)]
    )
    storage.upsert_edges(
        [
            CausalEdge(
                edge_id="e01",
                tenant_id=scope.tenant,
                user_id=scope.user,
                src_atom_id="a0",
                dst_atom_id="a1",
                relation_type="CAUSES",
                weight=0.5,
            )
        ]
    )
    storage.upsert_edges(
        [
            CausalEdge(
                edge_id="e01",
                tenant_id=scope.tenant,
                user_id=scope.user,
                src_atom_id="a0",
                dst_atom_id="a1",
                relation_type="CAUSES",
                weight=0.9,
            ),
            CausalEdge(
                edge_id="e12",
                tenant_id=scope.tenant,
                user_id=scope.user,
                src_atom_id="a1",
                dst_atom_id="a2",
                relation_type="ELABORATES",
                weight=1.5,
            ),
        ]
    )

    one_hop = storage.traverse_neighbors(scope.tenant, scope.user, ["a0"])
    assert [(atom.atom_id, edge.weight) for atom, edge in one_hop] == [("a1", 0.9)]

    two_hops = storage.traverse_neighbors(scope.tenant, scope.user, ["a0"], max_hops=2)
    assert [atom.atom_id for atom, _edge in two_hops] == ["a1", "a2"]
    assert storage.traverse_neighbors(scope.tenant, scope.user, ["a3"]) == []
    assert storage.traverse_neighbors(scope.tenant, scope.other_user, ["a0"]) == []
    with pytest.raises(ValueError):
        storage.traverse_neighbors(scope.tenant, scope.user, ["a0"], max_hops=0)


def test_lexical_search_ranks_and_isolates(storage, scope):
    storage.insert_atoms(
        [
            make_atom(scope, "m", "I moved to Madrid last spring"),
            make_atom(scope, "d", "My dog is called Rex"),
            make_atom(scope, "h", "I work at the hospital"),
        ]
    )
    results = storage.search_lexical(scope.tenant, scope.user, "Madrid spring")
    assert [atom.atom_id for atom in results] == ["m"]
    assert results[0].fact_text.startswith("I moved")
    assert storage.search_lexical(scope.tenant, scope.other_user, "Madrid") == []
    assert storage.search_lexical(scope.tenant, scope.user, "   ") == []


def embed(storage, *values: float) -> list[float]:
    """Pad to the backend's configured dimension (pgvector columns are fixed-width)."""
    dim = getattr(storage, "embedding_dim", len(values))
    return list(values) + [0.0] * (dim - len(values))


def test_dense_search_is_cosine_and_isolated(storage, scope):
    storage.insert_atoms(
        [
            make_atom(scope, "x", "east", embedding=embed(storage, 1.0, 0.0)),
            make_atom(scope, "y", "north", embedding=embed(storage, 0.0, 1.0)),
            make_atom(scope, "z", "north-east", embedding=embed(storage, 0.7, 0.7)),
            make_atom(scope, "n", "no vector"),
        ]
    )
    storage.upsert_embeddings(scope.tenant, scope.user, [("y", embed(storage, 0.1, 0.9))])

    results = storage.search_dense(scope.tenant, scope.user, embed(storage, 0.9, 0.1))
    assert [atom.atom_id for atom in results] == ["x", "z", "y"]
    assert results[0].dense_score == pytest.approx(0.99388, abs=1e-4)
    assert results[1].dense_score == pytest.approx(0.7809, abs=1e-3)
    assert storage.search_dense(scope.tenant, scope.other_user, embed(storage, 0.9, 0.1)) == []
    assert len(storage.search_dense(scope.tenant, scope.user, embed(storage, 0.9, 0.1), limit=1)) == 1


def test_persona_state_merges(storage, scope):
    assert storage.get_persona_state(scope.tenant, scope.user) == {}
    storage.upsert_persona_state(scope.tenant, scope.user, {"name": "Ada"}, 10.0)
    storage.upsert_persona_state(scope.tenant, scope.user, {"city": "London"}, 20.0)
    assert storage.get_persona_state(scope.tenant, scope.user) == {
        "name": "Ada",
        "city": "London",
    }
    assert storage.get_persona_state(scope.tenant, scope.other_user) == {}


def test_policies_filter_and_order(storage, scope):
    storage.upsert_policy(
        PolicyRule(
            policy_id="deny-ssn",
            tenant_id=scope.tenant,
            user_id=None,
            rule_type="DENY_PATTERN",
            pattern=r"\b\d{3}-\d{2}-\d{4}\b",
            action="BLOCK_WRITE",
            priority=50,
            created_at=1.0,
        )
    )
    storage.upsert_policy(
        PolicyRule(
            policy_id="retention",
            tenant_id=scope.tenant,
            user_id=None,
            rule_type="MAX_RETENTION_DAYS",
            pattern="365",
            action="DROP_EXPIRED",
            priority=10,
            enabled=False,
            created_at=1.0,
        )
    )
    storage.upsert_policy(
        PolicyRule(
            policy_id="user-lock",
            tenant_id=scope.tenant,
            user_id=scope.user,
            rule_type="SCOPE_LOCK",
            pattern="*",
            action="MASK_READ",
            priority=20,
            created_at=1.0,
        )
    )

    tenant_wide = storage.list_active_policies(scope.tenant)
    assert [policy.policy_id for policy in tenant_wide] == ["deny-ssn"]

    scoped = storage.list_active_policies(scope.tenant, scope.user)
    assert [policy.policy_id for policy in scoped] == ["user-lock", "deny-ssn"]
    assert scoped[0].user_id == scope.user
    assert storage.list_active_policies(scope.tenant, scope.other_user)[0].policy_id == "deny-ssn"
    assert storage.list_active_policies(f"{scope.tenant}-elsewhere") == []

    storage.upsert_policy(
        PolicyRule(
            policy_id="deny-ssn",
            tenant_id=scope.tenant,
            user_id=None,
            rule_type="DENY_PATTERN",
            pattern=r"\b\d{3}-\d{2}-\d{4}\b",
            action="REDACT",
            priority=5,
            created_at=1.0,
        )
    )
    updated = storage.list_active_policies(scope.tenant)
    assert (updated[0].action, updated[0].priority) == ("REDACT", 5)


def test_reinforcement_updates_retention(storage, scope):
    storage.insert_atoms([make_atom(scope, "r0", "Caroline likes tea")])
    storage.reinforce_atoms(scope.tenant, scope.user, ["r0"], accessed_at=1_500.0, stability_multiplier=2.0)
    atom = storage.get_atom_proof_chain(scope.tenant, scope.user, "r0")[0]
    assert (atom.access_count, atom.stability, atom.last_accessed_at) == (1, 2.0, 1_500.0)

    storage.reinforce_atoms(scope.tenant, scope.user, ["r0"], accessed_at=1_600.0)
    atom = storage.get_atom_proof_chain(scope.tenant, scope.user, "r0")[0]
    assert (atom.access_count, atom.stability, atom.last_accessed_at) == (2, 3.0, 1_600.0)
    assert storage.get_atom_proof_chain(scope.tenant, scope.other_user, "r0") == []


def test_delete_user_data_erases_only_that_user(storage, scope):
    storage.append_turns([make_turn(scope, 0), make_turn(scope, 1)])
    storage.insert_atoms(
        [
            make_atom(scope, "a0", "fact zero"),
            make_atom(scope, "a1", "fact one"),
        ]
    )
    storage.upsert_edges(
        [
            CausalEdge(
                edge_id="e01",
                tenant_id=scope.tenant,
                user_id=scope.user,
                src_atom_id="a0",
                dst_atom_id="a1",
                relation_type="CAUSES",
            )
        ]
    )
    storage.upsert_persona_state(scope.tenant, scope.user, {"name": "Ada"}, 1.0)
    storage.upsert_policy(
        PolicyRule(
            policy_id="p0",
            tenant_id=scope.tenant,
            user_id=scope.user,
            rule_type="SCOPE_LOCK",
            pattern="*",
            action="BLOCK_WRITE",
            created_at=1.0,
        )
    )
    storage.insert_atoms([make_atom(scope, "keep", "other user fact", user_id=scope.other_user)])
    storage.append_turns([make_turn(scope, 0, turn_id="keep-turn", user_id=scope.other_user)])

    counts = storage.delete_user_data(scope.tenant, scope.user)
    assert counts == {"turns": 2, "atoms": 2, "edges": 1, "personas": 1, "policies": 1}
    assert storage.get_latest_chain_hash(scope.tenant, scope.user) == GENESIS_HASH
    assert storage.get_persona_state(scope.tenant, scope.user) == {}
    assert storage.search_lexical(scope.tenant, scope.user, "fact") == []

    survivor = storage.get_atom_proof_chain(scope.tenant, scope.other_user, "keep")
    assert [atom.atom_id for atom in survivor] == ["keep"]
    assert storage.get_turn_window(scope.tenant, scope.other_user, scope.session, 0)[0].turn_id == "keep-turn"


def test_open_backend_factory(tmp_path):
    backend = open_backend(db_path=str(tmp_path / "factory.db"))
    try:
        assert isinstance(backend, SqliteStorage)
        assert backend.backend_name == "sqlite"
    finally:
        backend.close()
    with pytest.raises(ValueError):
        open_backend()
    with pytest.raises(ValueError):
        open_backend(db_path="x.db", dsn="postgresql://localhost/x")


def test_sqlite_storage_shares_a_file_with_the_core_engine(tmp_path):
    from infinitydb.core.engine import InfinityDB

    path = str(tmp_path / "shared.db")
    db = InfinityDB(path)
    try:
        db.add("Caroline: I moved to Madrid last spring.", user_id="u", ts=1_000.0)
        store = SqliteStorage(path)
        store.initialize()
        try:
            store.insert_atoms([make_atom(SimpleNamespace(tenant="t", user="u", session="s"), "a0", "Caroline lives in Madrid")])
            hits = db.recall("where does Caroline live", n=3, user_id="u", reinforce=False)["results"]
            assert hits and "Madrid" in hits[0]["fact"]
            assert store.search_lexical("t", "u", "Madrid")[0].atom_id == "a0"
        finally:
            store.close()
    finally:
        db.close()
