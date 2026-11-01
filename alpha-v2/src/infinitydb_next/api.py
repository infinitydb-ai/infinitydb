"""Development-only REST adapter. Bind to localhost; authentication is not included."""
from __future__ import annotations
import os
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .store import MemoryStore

class AddMemory(BaseModel):
    content: str
    tenant_id: str = "default"
    kind: str = "fact"
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_id: str | None = None
    raw_evidence: str | None = None
    confidence: float = 1.0

class SearchRequest(BaseModel):
    query: str
    tenant_id: str = "default"
    limit: int = Field(default=10, ge=1, le=100)

class VectorSearchRequest(BaseModel):
    embedding: list[float]
    tenant_id: str = "default"
    limit: int = Field(default=10, ge=1, le=100)
    kinds: list[str] = Field(default_factory=list)

store = MemoryStore(os.environ.get("INFINITYDB_NEXT_PATH", "infinitydb-next.sqlite3"))
app = FastAPI(title="InfinityDB Next Alpha API", version="2.0.0a1")

@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "stats": store.stats()}

@app.post("/v1/memories")
def add_memory(body: AddMemory) -> dict[str, str]:
    try:
        return {"id": store.add_memory(**body.model_dump())}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.post("/v1/search")
def search(body: SearchRequest) -> list[dict[str, Any]]:
    return [{"id": hit.memory.id, "content": hit.memory.content, "kind": hit.memory.kind,
             "score": hit.score, "matched_by": hit.matched_by}
            for hit in store.search(body.query, tenant_id=body.tenant_id, limit=body.limit)]

@app.post("/v1/search/vector")
def vector_search(body: VectorSearchRequest) -> list[dict[str, Any]]:
    try:
        hits = store.search_vector(
            body.embedding, tenant_id=body.tenant_id,
            limit=body.limit, kinds=body.kinds or None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return [{"id": hit.memory.id, "content": hit.memory.content,
             "kind": hit.memory.kind, "score": hit.score,
             "matched_by": hit.matched_by} for hit in hits]


@app.post("/v1/context")
def context(body: SearchRequest) -> dict[str, Any]:
    pack = store.context_pack(body.query, tenant_id=body.tenant_id, limit=body.limit)
    return {"query": pack.query,
            "memories": [{"id": item.id, "content": item.content, "kind": item.kind} for item in pack.memories],
            "evidence_ids": pack.evidence_ids, "estimated_tokens": pack.estimated_tokens}
