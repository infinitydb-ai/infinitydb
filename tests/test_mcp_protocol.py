"""End-to-end MCP protocol test: real subprocess, real stdin/stdout.

`tests/test_integrations.py` calls `handle_call_tool` directly, which does not
prove the transport works. This spawns the server the way a host does and speaks
JSON-RPC 2.0 over pipes, so a broken handshake, a missing `Content-Length`, or a
non-flushing write fails here instead of in Claude Desktop.
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _spawn(db_path: Path | None, home: Path) -> subprocess.Popen:
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),  # never touch the developer's real ~/.infinitydb
        "PYTHONPATH": str(ROOT),
    }
    if db_path is not None:
        env["INFINITYDB_PATH"] = str(db_path)
    return subprocess.Popen(
        [sys.executable, "-m", "infinitydb.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=env,
    )


class _Client:
    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self.lines: queue.Queue[str] = queue.Queue()
        self.next_id = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put("")

    def send(self, method: str, params: dict | None = None, notify: bool = False) -> dict | None:
        assert self.proc.stdin is not None
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            self.next_id += 1
            msg["id"] = self.next_id
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        if notify:
            return None
        line = self.lines.get(timeout=10)
        assert line, "server closed stdout before answering"
        return json.loads(line)


def test_mcp_stdio_session(tmp_path):
    proc = _spawn(tmp_path / "mcp.db", tmp_path)
    client = _Client(proc)
    try:
        init = client.send("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
        assert init["result"]["serverInfo"]["name"] == "infinitydb"
        assert re.fullmatch(r"\d+\.\d+\.\d+", init["result"]["serverInfo"]["version"])
        client.send("notifications/initialized", notify=True)

        listed = client.send("tools/list")
        names = {tool["name"] for tool in listed["result"]["tools"]}
        assert {"add_memory", "recall_memory", "get_persona"} <= names
        for tool in listed["result"]["tools"]:
            assert tool["inputSchema"]["type"] == "object"

        added = client.send("tools/call", {
            "name": "add_memory",
            "arguments": {"content": "User: I moved to Porto and work at a ceramic studio.",
                          "user_id": "u1", "session_id": "s1"},
        })
        assert added["result"].get("isError") is not True

        recalled = client.send("tools/call", {
            "name": "recall_memory",
            "arguments": {"query": "where did I move", "user_id": "u1", "session_id": "s1"},
        })
        text = json.dumps(recalled["result"])
        assert "Porto" in text, "a fact written over the transport must come back over it"

        unknown = client.send("no/such/method")
        assert unknown["error"]["code"] == -32601
    finally:
        proc.stdin.close()  # type: ignore[union-attr]
        proc.wait(timeout=10)
        assert proc.returncode == 0


def test_default_store_is_absolute_and_cwd_independent(tmp_path):
    """A host's working directory must not decide where memory lives."""
    proc = _spawn(None, tmp_path)
    client = _Client(proc)
    try:
        init = client.send("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
        assert init["result"]["serverInfo"]["name"] == "infinitydb"
        client.send("tools/call", {"name": "add_memory",
                                  "arguments": {"content": "User: I keep bees.", "user_id": "u1"}})
    finally:
        proc.stdin.close()  # type: ignore[union-attr]
        proc.wait(timeout=10)
    created = tmp_path / ".infinitydb" / "memory.db"
    assert created.exists(), f"expected the default store at {created}"


def test_env_var_still_overrides_the_default(tmp_path, monkeypatch):
    from infinitydb import mcp_server

    monkeypatch.setenv("INFINITYDB_PATH", str(tmp_path / "custom.db"))
    assert mcp_server.default_db_path() == str(tmp_path / "custom.db")

    monkeypatch.delenv("INFINITYDB_PATH")
    fallback = mcp_server.default_db_path()
    assert os.path.isabs(fallback)
    assert fallback.endswith(os.path.join(".infinitydb", "memory.db"))


def test_mcp_server_version_matches_package():
    """A stale version makes an installed server look older than its package."""
    from infinitydb import mcp_server

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert declared, "pyproject.toml has no version line"
    assert mcp_server.SERVER_VERSION == declared.group(1)
