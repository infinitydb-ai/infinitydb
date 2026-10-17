"""InfinityDB Model Context Protocol (MCP) Server.

Enables Claude Desktop, Cursor, Windsurf, and LLM agents to seamlessly connect to
InfinityDB long-term memory over standard JSON-RPC 2.0 stdio.
Zero external dependencies required.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

from infinitydb.core.engine import InfinityDB

SERVER_NAME = "infinitydb"
SERVER_VERSION = "1.0.0"

DB_PATH = os.environ.get("INFINITYDB_PATH", "infinity_mcp.db")
db = InfinityDB(DB_PATH)

TOOLS = [
    {
        "name": "recall_memory",
        "description": "Recall long-term neurosymbolic memory, atomic facts, and causal relations for a given query and user.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query or contextual question to recall memories for."},
                "user_id": {"type": "string", "description": "User or agent identifier (defaults to 'default').", "default": "default"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_memory",
        "description": "Ingest conversational interactions into InfinityDB. Automatically generates L0 logs, L1 atomic facts, and L2 causal graphs with sub-millisecond writes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The dialogue, interaction transcript, or statement to commit to long-term memory."},
                "user_id": {"type": "string", "description": "User or agent identifier.", "default": "default"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "get_persona",
        "description": "Retrieve the dynamic persona and conflict-resolved preference profile of a user or agent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User or agent identifier.", "default": "default"},
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "add_policy",
        "description": "Add an enterprise governance rule, safety policy, or operational SOP that is strictly enforced during memory recall.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule": {"type": "string", "description": "The policy statement to enforce (e.g. 'Never disclose internal API keys')."},
                "domain": {"type": "string", "description": "Policy category (security, privacy, compliance).", "default": "security"},
                "priority": {"type": "number", "description": "Enforcement priority weighting.", "default": 1.0},
            },
            "required": ["rule"],
        },
    },
    {
        "name": "verify_proof",
        "description": "Verify the tamper-evident SHA-256 cryptographic proof chain of an atomic fact.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "atom_id": {"type": "string", "description": "The atomic fact identifier to verify."},
            },
            "required": ["atom_id"],
        },
    },
]


def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == "recall_memory":
        query = arguments.get("query", "")
        user_id = arguments.get("user_id", "default")
        res = db.recall(query, user_id=user_id)
        return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}

    elif name == "add_memory":
        content = arguments.get("content", "")
        user_id = arguments.get("user_id", "default")
        log_id = db.add(content, user_id=user_id)
        return {"content": [{"type": "text", "text": json.dumps({"status": "success", "log_id": log_id, "user_id": user_id})}]}

    elif name == "get_persona":
        user_id = arguments.get("user_id", "default")
        persona = db.get_persona(user_id=user_id)
        return {"content": [{"type": "text", "text": json.dumps(persona, indent=2)}]}

    elif name == "add_policy":
        rule = arguments.get("rule", "")
        domain = arguments.get("domain", "security")
        priority = float(arguments.get("priority", 1.0))
        pol_id = db.add_policy(rule, domain=domain, priority=priority)
        return {"content": [{"type": "text", "text": json.dumps({"status": "success", "policy_id": pol_id, "rule": rule})}]}

    elif name == "verify_proof":
        atom_id = arguments.get("atom_id", "")
        proof = db.get_proof(atom_id)
        return {"content": [{"type": "text", "text": json.dumps(proof, indent=2)}]}

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
                        "capabilities": {
                            "tools": {},
                        },
                        "serverInfo": {
                            "name": SERVER_NAME,
                            "version": SERVER_VERSION,
                        },
                    },
                }
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": TOOLS,
                    },
                }
            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                tool_res = handle_call_tool(tool_name, arguments)
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": tool_res,
                }
            elif method == "ping":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method '{method}' not found",
                    },
                }

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": str(e),
                },
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_server()
