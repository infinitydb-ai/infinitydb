"""LangGraph ``BaseStore`` contract tests for :class:`InfinityDBStore`."""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("langgraph.store.base")

from infinitydb.integrations.langgraph import InfinityDBStore  # noqa: E402


def test_langgraph_store_crud_and_search(tmp_path):
    store = InfinityDBStore(db_path=tmp_path / "lg_test.db")
    try:
        ns_alice = ("acme", "user_alice", "preferences")
        ns_bob = ("acme", "user_bob", "preferences")

        store.put(ns_alice, "diet", {"rule": "Vegetarian, allergic to peanuts", "priority": 1})
        store.put(ns_alice, "editor", {"rule": "Uses Neovim with Rust analyzer", "priority": 2})
        store.put(ns_bob, "diet", {"rule": "Eats steak every Friday", "priority": 1})

        item = store.get(ns_alice, "diet")
        assert item is not None
        assert item.value["rule"] == "Vegetarian, allergic to peanuts"
        assert item.namespace == ns_alice

        # Search is scoped to the namespace prefix: nothing of Bob's leaks in.
        hits = store.search(
            ("acme", "user_alice"), query="peanut allergy dietary restriction", limit=5
        )
        assert len(hits) >= 1
        assert hits[0].key == "diet"
        assert all(hit.namespace == ns_alice for hit in hits)

        # Metadata filters run before ranking.
        filtered = store.search(
            ("acme", "user_alice"),
            query="Neovim Rust",
            filter={"priority": {"$gte": 2}},
        )
        assert [hit.key for hit in filtered] == ["editor"]

        # List namespaces with max_depth collapses the leaves.
        top_ns = store.list_namespaces(prefix=("acme",), max_depth=2)
        assert top_ns == [("acme", "user_alice"), ("acme", "user_bob")]

        # LangGraph's delete convention: put with value=None.
        store.put(ns_alice, "editor", None)
        assert store.get(ns_alice, "editor") is None
        remaining = store.search(("acme", "user_alice"), query="Neovim Rust")
        assert [hit.key for hit in remaining] == []
    finally:
        store.close()


def test_langgraph_store_async(tmp_path):
    store = InfinityDBStore(db_path=tmp_path / "lg_async.db")
    try:
        ns = ("org_1", "agent_memory")

        async def exercise() -> tuple[object, list]:
            await store.aput(
                ns,
                "incident_42",
                {"summary": "Postgres connection pool exhaustion in us-east-1"},
            )
            fetched = await store.aget(ns, "incident_42")
            results = await store.asearch(ns, query="Postgres pool us-east-1")
            return fetched, results

        fetched, results = asyncio.run(exercise())
        assert fetched is not None
        assert "us-east-1" in fetched.value["summary"]
        assert len(results) == 1
        assert results[0].key == "incident_42"
    finally:
        store.close()


def test_langgraph_store_ttl_and_refresh(tmp_path):
    store = InfinityDBStore(db_path=tmp_path / "lg_ttl.db")
    try:
        ns = ("org", "ttl")
        store.put(ns, "short", {"v": 1}, ttl=0.002)  # 0.12 s
        assert store.get(ns, "short") is not None
        assert store.get(ns, "short", refresh_ttl=True) is not None
        time.sleep(0.2)
        assert store.get(ns, "short") is None
        assert store.list_namespaces() == []
    finally:
        store.close()
