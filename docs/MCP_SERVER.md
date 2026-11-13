# InfinityDB MCP Server

> Plug-and-play Model Context Protocol (MCP) server for Claude Desktop, Cursor, Windsurf, and AI Agents.

The InfinityDB MCP server gives any MCP-compliant AI assistant long-term neurosymbolic memory with deterministic recall, causal relation tracking, and tamper-evident SHA-256 audit proofs. Measured cost of adoption is below — the round trip through the transport is **~1.3 ms to write and ~2 ms to recall** on the machine described there, not sub-millisecond.

---

## Measured friction

Numbers instead of adjectives, reproducible with `python scripts/measure_mcp_friction.py`
(raw output committed as [`mcp_friction.json`](mcp_friction.json)). Linux 6.1 x86_64,
Python 3.12.3, three runs on the same machine.

| What it costs | Measured |
|---|---|
| third-party packages pulled in | **0** |
| wheel size | **60.6 KB** (`infinitydb_ai-1.4.1-py3-none-any.whl`) |
| cold start, spawn to valid `initialize` reply | **65–82 ms** |
| `tools/list` | 0.1–0.2 ms, 4,099 bytes, 10 tools |
| `add_memory` | p50 **1.3–1.8 ms**, max 8.1 ms |
| `recall_memory` | p50 **1.9–2.5 ms**, max 3.3 ms |
| peak RSS of the server process | **19.8 MB** |
| configuration | one `mcpServers` entry, 7 lines of JSON |

Read the cold start as the host's per-session tax, not per-call: a host starts the
process once and keeps it. The ~70 ms is Python plus SQLite import, so it is the
floor for any Python MCP server, not something this implementation adds.

### Friction that the numbers do not capture

- **Working-directory dependence is fixed, not gone.** `INFINITYDB_PATH` now
  defaults to an absolute `~/.infinitydb/memory.db` (see *Where memory lives*), so
  a host cannot move the store by changing cwd. A *relative* value you supply
  yourself still resolves against cwd — that is your choice, not a default.
- **Single-user, no authentication.** The store is a local SQLite file with file
  permissions as its only access control. It is not safe as a shared or networked
  memory service; there is no multi-tenant scoping beyond `user_id` strings.
- **`initialize` always answers protocol version `2024-11-05`.** Hosts built
  against newer revisions accept it, but this server does not negotiate.
- **Host coverage is untested.** The transport is verified end to end by
  `tests/test_mcp_protocol.py` (spawn, handshake, `tools/list`, a write, a recall,
  and an unknown-method error), but no Claude Desktop, Cursor or Windsurf install
  was driven here — those runs need a desktop host this environment does not have.

## Quick Setup

### 0. One command, no clone (recommended)

```bash
# Claude Code
claude mcp add infinitydb -- uvx --from infinitydb-ai infinitydb-mcp

# or run it directly to check it starts
uvx --from infinitydb-ai infinitydb-mcp
```

`uvx` fetches the package into an ephemeral environment and runs the console
script, so there is no virtualenv to create and nothing to keep in sync. `pip
install infinitydb-ai` then `infinitydb-mcp` is equivalent if you already manage an
environment. `npx` is not needed: the npm package is the JS client, not this server.

A ready-to-copy [`.mcp.json`](../.mcp.json) lives in the repository root; commit it
into a project and a host that reads project config will offer the tools with no
editing.

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
        "INFINITYDB_PATH": "/Users/you/.infinitydb/memory.db"
      }
    }
  }
}
```

---

### 2. Cursor / Windsurf Configuration

Add InfinityDB to your project's `.cursor/mcp.json`, pointing at a store inside
the project so this workspace's memory stays separate:

```json
{
  "mcpServers": {
    "infinitydb": {
      "command": "python",
      "args": ["-m", "infinitydb.mcp_server"],
      "env": {
        "INFINITYDB_PATH": "${workspaceFolder}/.infinitydb/memory.db"
      }
    }
  }
}
```

If `${workspaceFolder}` is not expanded by your host, write the absolute path.

---

## Where memory lives (scopes)

The store is chosen by `INFINITYDB_PATH`. Without it, both the MCP server and
`infinitydb.capture` use **`~/.infinitydb/memory.db`** — absolute, so a host's
working directory can never move it, and auto-captured turns show up over MCP.

| Scope | `INFINITYDB_PATH` | Use it when |
|---|---|---|
| **Global (default)** | unset → `~/.infinitydb/memory.db` | One memory across every chat and project, shared with auto-capture |
| **Per project** | `/abs/path/to/project/.infinitydb/memory.db` | Work chats must not see a client's or another project's facts |
| **Per host/profile** | `/abs/path/to/work.db` | Separate work and personal assistants on one machine |
| **Throwaway** | `:memory:` | Trying the tools out; nothing is written to disk |

Rules of thumb:

- **Always use an absolute path** in a host config. A relative path is resolved
  against the process's working directory, which a GUI host does not show you.
- **One store per project** is the safest default for anything client-facing. Two
  projects pointed at one global file is a data-boundary bug, not a feature.
- **`user_id` is not a security boundary.** It separates memories inside one file;
  it does not stop another tool that can read the file.

### Migrating from the old relative default

Servers started before 1.4.1 without `INFINITYDB_PATH` wrote `infinity_mcp.db`
into whatever directory the host used. Nothing was deleted: the file is still
there, and the new default simply does not pick it up. To keep that memory, point
the config at it, or fold it into the new store:

```bash
# keep using the old file, explicitly
INFINITYDB_PATH=/abs/path/where/infinity_mcp.db python -m infinitydb.mcp_server
```

To check which store a running server actually opened, call the `stats` tool: the
row counts tell you immediately whether you are looking at the expected database.

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
