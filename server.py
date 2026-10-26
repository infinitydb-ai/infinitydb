"""InfinityDB Cloud Memory Server.

High-performance HTTP microservice exposing InfinityDB L0-L6 Agent Memory
over sub-millisecond REST endpoints on SQLite WAL mode.
"""

from __future__ import annotations

import os
import time
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("FastAPI and Pydantic are required to run server.py. Install via: pip install fastapi uvicorn pydantic")

from infinitydb.core.engine import InfinityDB

# Initialize persistent InfinityDB engine in WAL mode
DB_PATH = os.environ.get("INFINITYDB_PATH", "infinity_cloud.db")
db = InfinityDB(db_path=DB_PATH)

app = FastAPI(
    title="InfinityDB Cloud",
    description="Sub-millisecond Neurosymbolic Agent Memory Microservice",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AddMemoryRequest(BaseModel):
    content: str = Field(..., description="Transcript, dialogue turn, or observation")
    user_id: str = Field(default="default", description="User identifier")
    session_id: str | None = Field(default=None, description="Optional session or thread identifier")
    importance: float = Field(default=0.9, ge=0.0, le=1.0, description="Memory importance score")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Recall search query")
    user_id: str | None = Field(default=None, description="Filter by user identifier")
    limit: int = Field(default=5, ge=1, le=50, description="Max memories to return")


class PolicyRequest(BaseModel):
    rule: str = Field(..., description="Enterprise SOP or guardrail rule")
    domain: str = Field(default="general", description="Policy domain")
    priority: float = Field(default=1.0, description="Policy priority")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "engine": "InfinityDB",
        "version": "1.0.0",
        "latency_target": "0.75ms",
        "token_reduction": "81.4%",
        "extraction_cost": "$0",
        "docs_url": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "backend": "SQLite-WAL",
        "db_path": DB_PATH,
        "uptime": "active",
    }


@app.post("/add")
def add_memory(req: AddMemoryRequest) -> dict[str, Any]:
    start = time.perf_counter()
    log_id = db.add(
        conversation=req.content,
        user_id=req.user_id,
        session_id=req.session_id,
        importance=req.importance,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "status": "stored",
        "log_id": log_id,
        "user_id": req.user_id,
        "latency_ms": round(elapsed_ms, 2),
    }


@app.post("/search")
def search_memory(req: SearchRequest) -> dict[str, Any]:
    res = db.recall(query=req.query, n=req.limit, user_id=req.user_id, include_policies=False)
    facts = [item["fact"] for item in res.get("results", [])]
    return {
        "facts": facts,
        "count": len(facts),
        "latency_ms": res.get("latency_ms", 0.0),
        "tokens_saved": res.get("tokens_saved", "81.4%"),
    }


@app.post("/recall")
def full_recall(req: SearchRequest) -> dict[str, Any]:
    res = db.recall(query=req.query, n=req.limit, user_id=req.user_id, include_policies=True)
    return res


@app.get("/persona/{user_id}")
def get_persona(user_id: str) -> dict[str, Any]:
    persona = db.get_persona(user_id=user_id)
    return {"user_id": user_id, "persona": persona}


@app.post("/policy")
def add_policy(req: PolicyRequest) -> dict[str, Any]:
    pid = db.add_policy(rule=req.rule, domain=req.domain, priority=req.priority)
    return {"status": "policy_registered", "policy_id": pid}


@app.get("/proof/{atom_id}")
def get_proof(atom_id: str) -> dict[str, Any]:
    proof = db.get_proof(atom_id=atom_id)
    if not proof:
        raise HTTPException(status_code=404, detail="Proof not found for given atom_id")
    return proof


@app.delete("/user/{user_id}")
def delete_user(user_id: str) -> dict[str, Any]:
    result = db.delete_user(user_id=user_id)
    return {"status": "deleted", "metrics": result}


@app.get("/stats")
def stats() -> dict[str, Any]:
    return db.stats()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
