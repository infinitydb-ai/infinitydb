"""Tenant-authenticated REST adapter for local and private deployments.

Configure INFINITYDB_API_KEYS as a JSON object mapping API tokens to tenant IDs.
The unauthenticated health endpoint intentionally exposes liveness only.
"""
from __future__ import annotations

import json
import os
import secrets
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .store import MemoryStore


class AddMemory(BaseModel):
    content: str
    kind: str = "fact"
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_id: str | None = None
    raw_evidence: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)


class VectorSearchRequest(BaseModel):
    embedding: list[float]
    limit: int = Field(default=10, ge=1, le=100)
    kinds: list[str] = Field(default_factory=list)


class HybridSearchRequest(BaseModel):
    query: str
    embedding: list[float]
    limit: int = Field(default=10, ge=1, le=100)
    kinds: list[str] = Field(default_factory=list)
    rrf_k: int = Field(default=60, ge=1, le=1000)


class SetEmbeddingRequest(BaseModel):
    embedding: list[float]


store = MemoryStore(os.environ.get("INFINITYDB_NEXT_PATH", "infinitydb-next.sqlite3"))
app = FastAPI(title="InfinityDB Next API", version="2.0.0a1")


def authenticated_tenant(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    """Resolve an API token to exactly one tenant; never trust tenant IDs in payloads."""
    raw_keys = os.environ.get("INFINITYDB_API_KEYS", "")
    if not raw_keys:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured",
        )
    try:
        credentials = json.loads(raw_keys)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=503, detail="API authentication configuration is invalid") from exc
    if not isinstance(credentials, dict) or not credentials:
        raise HTTPException(status_code=503, detail="API authentication configuration is invalid")
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key",
                            headers={"WWW-Authenticate": "ApiKey"})
    tenant: str | None = None
    # Compare every configured token to avoid returning early on a match.
    for token, tenant_id in credentials.items():
        if isinstance(token, str) and secrets.compare_digest(token.encode("utf-8"), x_api_key.encode("utf-8")):
            if isinstance(tenant_id, str) and tenant_id.strip():
                tenant = tenant_id
    if tenant is None:
        raise HTTPException(status_code=401, detail="Invalid API key",
                            headers={"WWW-Authenticate": "ApiKey"})
    return tenant


@app.get("/health")
def health() -> dict[str, str]:
    """Public liveness probe; intentionally returns no tenant or database statistics."""
    return {"status": "ok"}


@app.get("/v1/stats")
def stats(tenant_id: str = Depends(authenticated_tenant)) -> dict[str, int]:
    return store.stats(tenant_id=tenant_id)


@app.post("/v1/memories")
def add_memory(body: AddMemory, tenant_id: str = Depends(authenticated_tenant)) -> dict[str, str]:
    try:
        values = body.model_dump()
        values["tenant_id"] = tenant_id
        return {"id": store.add_memory(**values)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/search")
def search(body: SearchRequest, tenant_id: str = Depends(authenticated_tenant)) -> list[dict[str, Any]]:
    return [{"id": hit.memory.id, "content": hit.memory.content, "kind": hit.memory.kind,
             "score": hit.score, "matched_by": hit.matched_by}
            for hit in store.search(body.query, tenant_id=tenant_id, limit=body.limit)]


@app.put("/v1/memories/{memory_id}/embedding")
def set_embedding(memory_id: str, body: SetEmbeddingRequest,
                  tenant_id: str = Depends(authenticated_tenant)) -> dict[str, bool]:
    try:
        store.set_embedding(memory_id, body.embedding, tenant_id=tenant_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"stored": True}


@app.post("/v1/search/vector")
def vector_search(body: VectorSearchRequest,
                  tenant_id: str = Depends(authenticated_tenant)) -> list[dict[str, Any]]:
    try:
        hits = store.search_vector(body.embedding, tenant_id=tenant_id,
                                   limit=body.limit, kinds=body.kinds or None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return [{"id": hit.memory.id, "content": hit.memory.content,
             "kind": hit.memory.kind, "score": hit.score,
             "matched_by": hit.matched_by} for hit in hits]


@app.post("/v1/search/hybrid")
def hybrid_search(body: HybridSearchRequest,
                  tenant_id: str = Depends(authenticated_tenant)) -> list[dict[str, Any]]:
    try:
        hits = store.search_hybrid(
            body.query, body.embedding, tenant_id=tenant_id,
            limit=body.limit, kinds=body.kinds or None, rrf_k=body.rrf_k,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return [{"id": hit.memory.id, "content": hit.memory.content,
             "kind": hit.memory.kind, "score": hit.score,
             "matched_by": hit.matched_by} for hit in hits]


@app.post("/v1/context")
def context(body: SearchRequest,
            tenant_id: str = Depends(authenticated_tenant)) -> dict[str, Any]:
    pack = store.context_pack(body.query, tenant_id=tenant_id, limit=body.limit)
    return {"query": pack.query,
            "memories": [{"id": item.id, "content": item.content, "kind": item.kind}
                         for item in pack.memories],
            "evidence_ids": pack.evidence_ids, "estimated_tokens": pack.estimated_tokens}
