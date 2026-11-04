"""Zep Cloud v4 adapter using user graphs and episode search."""
from __future__ import annotations

import os
import time
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
        self._users: dict[str, str] = {}
        self._ids: dict[str, str] = {}

    def _user_uuid(self, tenant_id: str) -> str:
        if tenant_id not in self._users:
            user = self.client.user.create(first_name="Benchmark", last_name=tenant_id)
            self._users[tenant_id] = str(user.uuid_)
        return self._users[tenant_id]

    def add(self, memory_id: str, tenant_id: str, content: str, vector: list[float]) -> None:
        result = self.client.graph.episode.add(
            self._user_uuid(tenant_id),
            type="text",
            data=content,
        )
        episode = getattr(result, "episode", None)
        remote_id = getattr(episode, "uuid_", None) or getattr(episode, "uuid", None)
        if not remote_id:
            raise RuntimeError(f"Zep episode ingestion returned no episode UUID: {result!r}")
        self._ids[memory_id] = str(remote_id)
        task = getattr(result, "task", None)
        task_uuid = getattr(task, "uuid_", None) if task is not None else None
        if task_uuid:
            deadline = time.monotonic() + 600
            while True:
                current = self.client.task.get(str(task_uuid))
                status = str(getattr(current, "status", "")).lower()
                if status == "succeeded":
                    break
                if status in {"failed", "partial", "canceled", "invalid"}:
                    raise RuntimeError(f"Zep episode indexing task ended with status {status}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Zep indexing task {task_uuid} did not finish within 600 seconds")
                time.sleep(1)

    def search(self, tenant_id: str, query: str, vector: list[float], limit: int) -> list[str]:
        if tenant_id not in self._users:
            return []
        result = self.client.graph.search_episodes(
            self._users[tenant_id],
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
            "api": "v4",
            "deployment": "managed cloud",
            "ingestion": "user graph episodes",
            "retrieval": "episode search",
            "embedding": "managed by Zep",
        }

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def build() -> ZepCloudAdapter:
    return ZepCloudAdapter()
