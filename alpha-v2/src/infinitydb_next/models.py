from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Memory:
    id: str
    tenant_id: str
    kind: str
    content: str
    created_at: float
    valid_from: float
    valid_to: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_id: str | None = None
    confidence: float = 1.0


@dataclass(slots=True)
class SearchHit:
    memory: Memory
    score: float
    matched_by: list[str]


@dataclass(slots=True)
class ContextPack:
    query: str
    memories: list[Memory]
    evidence_ids: list[str]
    estimated_tokens: int
    generated_at: float


@dataclass(slots=True)
class GraphHit:
    memory: Memory
    depth: int
    path: list[str]
    relations: list[str]
    score: float
