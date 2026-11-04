"""Mem0 OSS adapter (local/self-hosted Memory store)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class Mem0LocalAdapter:
    def __init__(self) -> None:
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise RuntimeError("Install adapter dependency: pip install 'mem0ai'") from exc
        path = Path(os.environ.get("MEM0_DATA_DIR", ".benchmark-data/mem0")).resolve()
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "version": "v1.1",
            "embedder": {
                "provider": "huggingface",
                "config": {"model": os.environ.get("MEM0_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {"collection_name": "infinitydb_benchmark", "path": str(path / "qdrant")},
            },
        }
        self.memory = Memory.from_config(config)
        self._ids: dict[str, str] = {}

    def add(self, memory_id: str, tenant_id: str, content: str, vector: list[float]) -> None:
        result = self.memory.add(content, user_id=tenant_id, infer=False)
        rows = result.get("results", []) if isinstance(result, dict) else []
        if not rows:
            raise RuntimeError(f"Mem0 did not return a stored memory ID: {result!r}")
        self._ids[memory_id] = str(rows[0]["id"])

    def search(self, tenant_id: str, query: str, vector: list[float], limit: int) -> list[str]:
        result = self.memory.search(query, user_id=tenant_id, limit=limit)
        rows = result.get("results", []) if isinstance(result, dict) else []
        reverse = {value: key for key, value in self._ids.items()}
        return [reverse[str(row["id"])] for row in rows if str(row.get("id")) in reverse]

    def metadata(self) -> dict[str, Any]:
        return {
            "system": "Mem0 OSS",
            "deployment": "local",
            "embedding_model": os.environ.get("MEM0_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            "vector_store": "Qdrant local",
            "inference": "disabled (infer=False)",
        }

    def close(self) -> None:
        close = getattr(self.memory, "close", None)
        if callable(close):
            close()


def build() -> Mem0LocalAdapter:
    return Mem0LocalAdapter()
