# InfinityDB MCP Server

> Plug-and-play Model Context Protocol (MCP) server for Claude Desktop, Cursor, Windsurf, and AI Agents.

The InfinityDB MCP server gives any MCP-compliant AI assistant long-term neurosymbolic memory with deterministic recall, sub-millisecond lookups, causal relation tracking, and tamper-evident SHA-256 audit proofs.

---

## Quick Setup

### 1. Claude Desktop Configuration

Add the following to your `claude_desktop_config.json`:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "infinitydb": {
      "command": "python",
      "args": ["-m", "infinitydb.mcp_server"],
      "env": {
        "INFINITYDB_PATH": "infinity_memory.db"
      }
    }
  }
}
```

---

### 2. Cursor / Windsurf Configuration

Add InfinityDB to your project's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "infinitydb": {
      "command": "python",
      "args": ["-m", "infinitydb.mcp_server"]
    }
  }
}
```

---

## Supported Tools

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `recall_memory` | `query`, `user_id` | Recalls relevant long-term memories, facts, and causal relations in sub-5ms. |
| `add_memory` | `content`, `user_id` | Commits raw dialogue into L0 logs, extracts L1 facts, and updates L2 causal graphs. |
| `get_persona` | `user_id` | Retrieves dynamic user/agent persona with automatic conflict resolution. |
| `add_policy` | `rule`, `domain`, `priority` | Injects corporate SOPs or safety guardrails that govern every recall. |
| `verify_proof` | `atom_id` | Verifies SHA-256 cryptographic audit trail linking facts to source transcripts. |
