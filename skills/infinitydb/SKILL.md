---
name: infinitydb
description: High-performance neurosymbolic long-term agent memory engine. Provides sub-5ms deterministic memory recall, causal episode tracking, dynamic persona conflict resolution, and SHA-256 cryptographic audit proofs.
version: 1.0.0
author: InfinityDB AI
license: Apache-2.0
repository: https://github.com/infinitydb-ai/infinitydb
---

# InfinityDB Agent Memory Skill

InfinityDB gives autonomous AI agents persistent, verifiable long-term memory with zero external LLM extraction API costs and sub-millisecond local retrieval on SQLite WAL mode.

---

## Capabilities

1. **Deterministic Fact Extraction (L1):**  
   Extracts atomic facts from dialogue and tool transcripts in <10ms with zero extra LLM token calls.

2. **Causal Episode Tracking (L2):**  
   Builds dependency edges between memory events to understand causal chains.

3. **Dynamic Persona & Conflict Auto-Resolution (L3):**  
   Automatically reconciles outdated user or agent preferences based on chronological provenance.

4. **Enterprise Policy Injection (L4):**  
   Guarantees corporate SOPs, compliance rules, and safety guardrails govern every recall query.

5. **Cryptographic Proofs (L6):**  
   Verifies an immutable SHA-256 hash linking each fact directly back to its raw interaction transcript.

---

## Quick Usage

### Python
```bash
pip install infinitydb-ai
```

```python
from infinitydb import InfinityDB

db = InfinityDB("agent_memory.db")

# Ingest dialogue
db.add("Dr. Meyer prefers Rust for distributed systems.", user_id="user_1")

# Sub-5ms hybrid recall
memory = db.recall("what language does the user prefer?", user_id="user_1")
print(memory["results"])
```

### Model Context Protocol (MCP) Server
To run InfinityDB as an MCP server for Claude Desktop and Cursor:
```bash
python -m infinitydb.mcp_server
```

---

## Tools Specification

- `recall_memory(query: str, user_id: str)`: Recalls relevant facts and causal relations.
- `add_memory(content: str, user_id: str)`: Commits interaction to L0-L2 memory.
- `get_persona(user_id: str)`: Retrieves conflict-resolved preference profile.
- `add_policy(rule: str, domain: str, priority: float)`: Injects governance constraints.
- `verify_proof(atom_id: str)`: Validates SHA-256 audit proof.
