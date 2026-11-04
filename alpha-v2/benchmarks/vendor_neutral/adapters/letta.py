"""Letta archival-memory adapter.

Provide a pre-created agent ID per benchmark tenant in LETTA_AGENT_IDS_JSON.
This avoids silently sharing a single agent across tenant boundaries.
"""
from __future__ import annotations

import json
import os
from typing import Any


class LettaAdapter:
    def __init__(self) -> None:
        token = os.environ.get("LETTA_API_KEY")
        mapping = os.environ.get("LETTA_AGENT_IDS_JSON")
        if not token or not mapping:
            raise RuntimeError("Set LETTA_API_KEY and LETTA_AGENT_IDS_JSON (tenant-to-agent ID map)")
        try:
            from letta_client import Letta
        except ImportError as exc:
            raise RuntimeError("Install adapter dependency: pip install letta-client") from exc
        self.client = Letta(api_key=token, environment=os.environ.get("LETTA_ENVIRONMENT", "cloud"))
        self.agent_ids = json.loads(mapping)
        if not isinstance(self.agent_ids, dict):
            raise ValueError("LETTA_AGENT_IDS_JSON must be a JSON object")
        self._ids: dict[str, str] = {}

    def _agent(self, tenant_id: str) -> str:
        try:
            return str(self.agent_ids[tenant_id])
        except KeyError as exc:
            raise ValueError(f"No Letta agent configured for tenant {tenant_id!r}") from exc

    def add(self, memory_id: str, tenant_id: str, content: str, vector: list[float]) -> None:
        result = self.client.agents.passages.create(
            self._agent(tenant_id),
            text=content,
            tags=["infinitydb-benchmark", memory_id],
        )
        remote_id = getattr(result, "id", None)
        if not remote_id:
            raise RuntimeError(f"Letta passage creation returned no ID: {result!r}")
        self._ids[memory_id] = str(remote_id)

    def search(self, tenant_id: str, query: str, vector: list[float], limit: int) -> list[str]:
        result = self.client.agents.passages.search(
            self._agent(tenant_id),
            query=query,
            limit=limit,
        )
        rows = getattr(result, "results", []) or []
        reverse = {value: key for key, value in self._ids.items()}
        return [reverse[str(row.id)] for row in rows if str(row.id) in reverse]

    def metadata(self) -> dict[str, Any]:
        return {
            "system": "Letta archival memory",
            "deployment": os.environ.get("LETTA_ENVIRONMENT", "cloud"),
            "embedding": "configured by Letta agent",
            "tenant_isolation": "one pre-created agent per tenant",
        }

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def build() -> LettaAdapter:
    return LettaAdapter()
