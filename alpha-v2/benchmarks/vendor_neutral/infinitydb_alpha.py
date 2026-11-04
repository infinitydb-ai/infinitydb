"""Reference adapter for the InfinityDB v2 alpha exact-vector baseline."""
from __future__ import annotations

from typing import Any

from infinitydb_next import MemoryStore
from benchmarks.vendor_neutral.runner import vector_for


class InfinityDBAlphaAdapter:
    def __init__(self) -> None:
        self.store = MemoryStore(":memory:")

    def add(self, memory_id: str, tenant_id: str, content: str, vector: list[float]) -> None:
        self.store.add_memory(content, tenant_id=tenant_id, memory_id=memory_id)
        self.store.set_embedding(
            memory_id, vector, tenant_id=tenant_id,
            provider="vendor-neutral-benchmark", model="feature-hash-v1",
        )

    def search(self, tenant_id: str, query: str, vector: list[float], limit: int) -> list[str]:
        hits = self.store.search_vector(
            vector, tenant_id=tenant_id, limit=limit,
            provider="vendor-neutral-benchmark", model="feature-hash-v1",
        )
        return [hit.memory.id for hit in hits]

    def metadata(self) -> dict[str, Any]:
        return {
            "system": "InfinityDB v2 alpha",
            "retrieval": "exact cosine",
            "database": "SQLite in-memory",
            "index": "exact scan",
            "embedding": "benchmark feature-hash-v1 (not a production embedding model)",
        }

    def close(self) -> None:
        self.store.close()


def build() -> InfinityDBAlphaAdapter:
    return InfinityDBAlphaAdapter()
