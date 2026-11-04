"""Zep Cloud adapter using graph episode ingestion and episode search."""
from __future__ import annotations

import os
from typing import Any


class ZepCloudAdapter:
    def __init__(self) -> None:
        api_key = os.environ.get("ZEP_API_KEY")
        if not api_key:
            raise RuntimeError("Set ZEP_API_KEY to run the Zep Cloud comparison")
        try:
            from zep_cloud.client import Zep
        except ImportError as exc:
            raise RuntimeError("Install adapter dependency: pip install zep-cloud") from exc
        self.client = Zep(api_key=api_key)
        self._ids: dict[str, str] = {}

    def add(self, memory_id: str, tenant_id: str, content: str, vector: list[float]) -> None:
        result = self.client.graph.episode.add(
            user_id=tenant_id,
            type="text",
            data=content,
            reference_time=None,
        )
        episode = getattr(result, "episode", None)
        remote_id = getattr(episode, "uuid_", None) or getattr(episode, "uuid", None)
        if not remote_id:
            raise RuntimeError(f"Zep episode ingestion returned no episode UUID: {result!r}")
        self._ids[memory_id] = str(remote_id)

    def search(self, tenant_id: str, query: str, vector: list[float], limit: int) -> list[str]:
        result = self.client.graph.search_episodes(
            user_id=tenant_id,
            query=query[:256],
            limit=limit,
        )
        rows = getattr(result, "episodes", None) or getattr(result, "results", None) or []
        reverse = {value: key for key, value in self._ids.items()}
        found: list[str] = []
        for row in rows:
            remote_id = getattr(row, "uuid_", None) or getattr(row, "uuid", None)
            if remote_id and str(remote_id) in reverse:
                found.append(reverse[str(remote_id)])
        return found

    def metadata(self) -> dict[str, Any]:
        return {
            "system": "Zep Cloud",
            "deployment": "managed cloud",
            "ingestion": "graph episode",
            "retrieval": "episode search",
            "embedding": "managed by Zep",
        }

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def build() -> ZepCloudAdapter:
    return ZepCloudAdapter()
