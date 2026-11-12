"""InfinityDB Model Context Protocol (MCP) Server.

Enables Claude Desktop, Claude Code, Cursor, Windsurf, and other MCP hosts to use
InfinityDB as long-term memory over JSON-RPC 2.0 stdio.

Zero external dependencies: the transport is stdin/stdout, the store is SQLite.

Tools:
    recall_memory   scoped recall (session / entity / point-in-time, token budget)
    add_memory      ingest a conversation turn or note
    get_persona     resolved preference profile for a user
    correct_fact    mark one fact as superseded by another (bi-temporal correction)
    forget          erase everything stored for a user
    add_policy      governance rule surfaced during recall
    verify_proof    check the SHA-256 provenance chain of a fact
    stats           layer counts for the active database
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

from infinitydb.core.engine import InfinityDB

SERVER_NAME = "infinitydb"
SERVER_VERSION = "1.3.0"

DB_PATH = os.environ.get("INFINITYDB_PATH", "infinity_mcp.db")
db = InfinityDB(DB_PATH)

TOOLS = [
    {
        "name": "recall_memory",
        "description": (
            "Recall long-term memory. Scope it with session_id (one conversation or "
            "project) and/or entities (names, products) — scoped recall matches a "
            "single-conversation store instead of competing with every other chat."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to recall."},
                "user_id": {"type": "string", "default": "default"},
                "session_id": {
                    "type": "string",
                    "description": "Restrict to one session, project, or conversation.",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only recall memories mentioning every one of these.",
                },
                "as_of": {
                    "type": "number",
                    "description": "Unix time: answer as the store looked then (excludes later facts).",
                },
                "top_k": {"type": "integer", "default": 8},
                "budget_tokens": {
                    "type": "integer",
                    "description": "Hard cap on returned words.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_memory",
        "description": (
            "Store a conversation turn, decision, or note. Facts are extracted "
            "deterministically on write; no model call is made."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "user_id": {"type": "string", "default": "default"},
                "session_id": {"type": "string", "description": "Conversation or project id."},
                "timestamp": {"type": "number", "description": "Unix event time (default: now)."},
                "importance": {"type": "number", "default": 0.9},
            },
            "required": ["content"],
        },
    },
    {
        "name": "get_persona",
        "description": "Resolved preference profile (identity, role, employer, preferences) for a user.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "default": "default"},
                "summary": {
                    "type": "boolean",
                    "default": False,
                    "description": "Return a one-paragraph profile instead of the key/value map.",
                },
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "correct_fact",
        "description": (
            "Mark a stored fact as superseded by a newer one. The old fact stays "
            "readable at earlier as_of times; later reads no longer return it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "old_atom_id": {"type": "string"},
                "new_atom_id": {"type": "string"},
                "user_id": {"type": "string", "default": "default"},
            },
            "required": ["old_atom_id", "new_atom_id"],
        },
    },
    {
        "name": "forget",
        "description": "Erase all memory for one user. Returns per-layer deletion counts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true; guards against accidental erasure.",
                },
            },
            "required": ["user_id", "confirm"],
        },
    },
    {
        "name": "add_policy",
        "description": "Add a governance rule or SOP that recall surfaces to the agent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule": {"type": "string"},
                "domain": {"type": "string", "default": "security"},
                "priority": {"type": "number", "default": 1.0},
            },
            "required": ["rule"],
        },
    },
    {
        "name": "verify_proof",
        "description": "Return the SHA-256 provenance record for an atomic fact.",
        "inputSchema": {
            "type": "object",
            "properties": {"atom_id": {"type": "string"}},
            "required": ["atom_id"],
        },
    },
    {
        "name": "entity_graph",
        "description": (
            "Follow typed relation edges out of an entity (works_at, lives_in, "
            "studied_at, likes, dislikes). hops=2 also follows entities that "
            "share an object, which is how two people in one memory connect."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "user_id": {"type": "string", "default": "default"},
                "hops": {"type": "integer", "minimum": 1, "maximum": 3, "default": 1},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["entity"],
        },
    },
    {
        "name": "contradictions",
        "description": (
            "Audit trail of superseded facts: what was replaced, by what, and "
            "when. Nothing is deleted, so as_of recall still sees the history."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "default": "default"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "stats",
        "description": "Counts per memory layer for the active database.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _text(payload: Any) -> Dict[str, Any]:
    body = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    return {"content": [{"type": "text", "text": body}]}


def _render(results: list[dict[str, Any]]) -> str:
    """One memory per line, with its event date and id, for an agent to cite."""
    from datetime import datetime

    lines = []
    for item in results:
        day = datetime.fromtimestamp(item["ts"]).strftime("%d %B %Y")
        lines.append(f"[{day}] ({item['id']}) {item['fact']}")
    return "\n".join(lines) if lines else "No matching memories."


def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == "recall_memory":
        result = db.recall(
            str(arguments.get("query", "")),
            n=int(arguments.get("top_k", 8)),
            user_id=arguments.get("user_id", "default"),
            as_of=arguments.get("as_of"),
            reinforce=False,
            session_id=arguments.get("session_id"),
            entities=arguments.get("entities"),
            budget_tokens=arguments.get("budget_tokens"),
        )
        return _text(
            {
                "memories": _render(result["results"]),
                "count": len(result["results"]),
                "policies": [p["rule"] for p in result.get("policies", [])],
            }
        )

    if name == "add_memory":
        ts = arguments.get("timestamp")
        log_id = db.add(
            str(arguments.get("content", "")),
            user_id=arguments.get("user_id", "default"),
            session_id=arguments.get("session_id"),
            importance=float(arguments.get("importance", 0.9)),
            ts=ts,
        )
        return _text({"status": "ok", "log_id": log_id})

    if name == "get_persona":
        user_id = arguments.get("user_id", "default")
        if arguments.get("summary"):
            return _text({"summary": db.persona_summary(user_id)})
        return _text(db.get_persona(user_id=user_id))

    if name == "correct_fact":
        relation = db.supersede_fact(
            str(arguments["old_atom_id"]),
            str(arguments["new_atom_id"]),
            arguments.get("user_id", "default"),
        )
        return _text({"status": "ok", "relation": relation})

    if name == "forget":
        if arguments.get("confirm") is not True:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "forget requires confirm: true"}],
            }
        return _text(db.delete_user(str(arguments["user_id"])))

    if name == "add_policy":
        policy_id = db.add_policy(
            str(arguments.get("rule", "")),
            domain=arguments.get("domain", "security"),
            priority=float(arguments.get("priority", 1.0)),
        )
        return _text({"status": "ok", "policy_id": policy_id})

    if name == "verify_proof":
        proof = db.get_proof(str(arguments.get("atom_id", "")))
        if proof is None:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "no proof record for that atom"}],
            }
        return _text(proof)

    if name == "entity_graph":
        paths = db.relations(
            str(arguments.get("entity", "")),
            user_id=arguments.get("user_id", "default"),
            hops=int(arguments.get("hops", 1)),
            limit=int(arguments.get("limit", 50)),
        )
        rendered = "\n".join(
            f"hop {p['hop']} ({p['direction']}): {p['subject']} —{p['predicate']}→ {p['object']}"
            for p in paths
        )
        return _text({"paths": rendered or "No typed relations.", "count": len(paths)})

    if name == "contradictions":
        conflicts = db.contradictions(
            arguments.get("user_id", "default"),
            limit=int(arguments.get("limit", 20)),
        )
        return _text({"contradictions": conflicts, "count": len(conflicts)})

    if name == "stats":
        return _text(db.stats())

    return {"isError": True, "content": [{"type": "text", "text": f"Unknown tool: {name}"}]}


def run_server() -> None:
    """Run standard stdio loop for MCP clients."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    },
                }
            elif method and method.startswith("notifications/"):
                continue
            elif method == "tools/list":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
            elif method == "tools/call":
                tool_res = handle_call_tool(
                    params.get("name", ""), params.get("arguments", {})
                )
                resp = {"jsonrpc": "2.0", "id": req_id, "result": tool_res}
            elif method == "ping":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method '{method}' not found"},
                }

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(e)},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_server()
