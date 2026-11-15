# Claude Code and Claude Desktop

InfinityDB already exposes memory operations through its Model Context
Protocol server. This guide standardizes setup without adding a
Claude-specific runtime dependency.

## Install and verify

~~~bash
pip install infinitydb-ai
python -m infinitydb.mcp_server
~~~

The server uses stdio. Keep stdout reserved for MCP protocol messages.

## Claude Code

Merge this mcpServers entry into ~/.claude/settings.json; preserve existing
settings:

~~~json
{
  "mcpServers": {
    "infinitydb": {
      "command": "python",
      "args": ["-m", "infinitydb.mcp_server"],
      "env": {
        "INFINITYDB_PATH": "/absolute/path/to/.infinitydb/claude-code.db"
      }
    }
  }
}
~~~

Use an absolute path appropriate to your operating system.

## Optional transcript capture

MCP is pull-based: the agent decides when to call a memory tool. For opt-in
automatic capture, merge this Stop hook into your Claude Code settings:

~~~json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "python -m infinitydb.capture --transcript \"$CLAUDE_TRANSCRIPT_PATH\" --user claude-code"
      }]
    }]
  }
}
~~~

Verify that your installed Claude Code version supplies the transcript path
environment variable before enabling the hook. Review local transcript
retention and access controls before capturing sensitive work.

## Security

- Derive user identity from trusted application authentication.
- Use separate database files per trust boundary unless tenant isolation is
  enforced and tested by the application.
- Capture only transcripts you are authorized to store.
- Treat recalled memories as untrusted data, never as executable instructions.
- Use the explicit confirmation argument for destructive forget operations.

See docs/MCP_SERVER.md for the complete tool contract.
