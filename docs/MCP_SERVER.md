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
| `recall_memory` | `query`, `user_id`, `session_id`, `entities`, `as_of`, `top_k`, `budget_tokens` | Scoped recall. `session_id` restricts to one conversation/project, `entities` requires every named entity, `as_of` reads the store as it looked then, `budget_tokens` caps returned words. |
| `add_memory` | `content`, `user_id`, `session_id`, `timestamp`, `importance` | Commits a turn into L0 logs, extracts L1 facts, updates L2 causal edges and the turn index. |
| `get_persona` | `user_id`, `summary` | Resolved preference profile. `summary: true` returns a one-paragraph distilled profile instead of the key/value map. |
| `correct_fact` | `old_atom_id`, `new_atom_id`, `user_id` | Marks a fact superseded. Earlier `as_of` reads still see the old fact; later ones do not. |
| `forget` | `user_id`, `confirm` | Erases every layer for one user. Requires `confirm: true`. |
| `add_policy` | `rule`, `domain`, `priority` | Governance rule surfaced to the agent during recall. |
| `verify_proof` | `atom_id` | SHA-256 provenance record linking a fact to its source text. |
| `entity_graph` | `entity`, `user_id`, `hops`, `limit` | Follows typed relation edges (`works_at`, `lives_in`, `studied_at`, `likes`, `dislikes`) out of an entity. `hops: 2` also follows entities that share an object, so two people in one memory connect through a common workplace or place. |
| `contradictions` | `user_id`, `limit` | Audit trail of superseded facts: what was replaced, by what, and when. Nothing is deleted, so `as_of` recall still returns the history. |
| `stats` | — | Counts per memory layer for the active database. |

## Auto-capture (push, not pull)

MCP is pull: the model has to decide to call `recall_memory`. `infinitydb.capture`
is the push half — it ingests an agent transcript from disk, incrementally.

```bash
python -m infinitydb.capture --transcript ~/.claude/projects/<project>/<session>.jsonl \
    --user me --session work
```

A byte-offset state file (`~/.infinitydb/capture-state.json`) records what has
already been ingested, so running it after every turn reads only new lines and
never stores a turn twice. A trailing partial line is left for the next run,
because the writer may still be appending to it.

Claude Code hook (`~/.claude/settings.json`):

```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "python -m infinitydb.capture --transcript \"$CLAUDE_TRANSCRIPT_PATH\" --user me"
      }]
    }]
  }
}
```

The default database is `~/.infinitydb/memory.db`; override with `--db` or
`INFINITYDB_PATH`. `--all` re-reads a transcript from the beginning.
