"""Adapter contract for vendor-neutral memory benchmarks.

Implement this small interface for each system under test. Keep provider/model
configuration inside the adapter and report it in metadata.
"""
from __future__ import annotations

from typing import Any, Protocol


class MemoryBenchmarkAdapter(Protocol):
    def add(self, memory_id: str, tenant_id: str, content: str, vector: list[float]) -> None: ...
    def search(self, tenant_id: str, query: str, vector: list[float], limit: int) -> list[str]: ...
    def metadata(self) -> dict[str, Any]: ...
    def close(self) -> None: ...
