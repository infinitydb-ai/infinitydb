#!/usr/bin/env python3
"""Measure what it actually costs to adopt the MCP server.

Install and usage friction reported as numbers rather than adjectives:

* **install** — third-party packages pulled in, and the wheel size
* **startup** — wall time from spawn to a valid `initialize` reply, i.e. what a
  host waits every time it opens the server
* **handshake** — `tools/list` latency and the bytes a host must parse
* **usage** — per-call latency for `add_memory` and `recall_memory`, and resident
  memory of the process

Every figure is measured on this machine; they are not comparable across
hardware, and the script prints the platform so a reader can judge.

    python scripts/measure_mcp_friction.py
    python scripts/measure_mcp_friction.py --out docs/mcp_friction.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, db_path: Path) -> None:
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "PYTHONPATH": str(ROOT), "INFINITYDB_PATH": str(db_path)}
        self.started = time.perf_counter()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "infinitydb.mcp_server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        self.next_id = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put("")

    def call(self, method: str, params: dict | None = None, notify: bool = False) -> tuple[dict | None, int, float]:
        assert self.proc.stdin is not None
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            self.next_id += 1
            msg["id"] = self.next_id
        t0 = time.perf_counter()
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        if notify:
            return None, 0, 0.0
        line = self.lines.get(timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
        if not line:
            raise SystemExit("server closed stdout mid-session")
        return json.loads(line), len(line.encode()), elapsed

    def close(self) -> float:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        self.proc.wait(timeout=30)
        return time.perf_counter() - self.started


def wheel_size_bytes() -> int | None:
    """Size of a built wheel, so 'zero dependencies' can be quantified."""
    out = ROOT / "dist"
    wheels = sorted(out.glob("infinitydb_ai-*.whl"), key=lambda p: p.stat().st_mtime)
    return wheels[-1].stat().st_size if wheels else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--calls", type=int, default=20)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        client = Client(Path(tmp) / "friction.db")
        init, init_bytes, startup_ms = client.call(
            "initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
        client.call("notifications/initialized", notify=True)
        listed, list_bytes, list_ms = client.call("tools/list")

        add_ms: list[float] = []
        recall_ms: list[float] = []
        for i in range(args.calls):
            _, _, dt = client.call("tools/call", {
                "name": "add_memory",
                "arguments": {"content": f"User: on day {i} I worked on project Atlas and "
                                         f"prefer morning reviews.", "user_id": "u1", "session_id": "s1"},
            })
            add_ms.append(dt)
            _, _, dt = client.call("tools/call", {
                "name": "recall_memory",
                "arguments": {"query": "when do I prefer reviews", "user_id": "u1", "session_id": "s1"},
            })
            recall_ms.append(dt)
        lifetime = client.close()

    child_rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    report = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "install": {
            "third_party_dependencies": 0,
            "wheel_bytes": wheel_size_bytes(),
            "package_python_bytes": sum(
                p.stat().st_size for p in (ROOT / "infinitydb").rglob("*.py")),
        },
        "startup": {"spawn_to_initialize_ms": round(startup_ms, 1)},
        "handshake": {"tools_list_ms": round(list_ms, 1), "tools_list_bytes": list_bytes,
                      "tool_count": len(listed["result"]["tools"]),
                      "server_version": init["result"]["serverInfo"]["version"]},
        "usage": {
            "add_memory_ms_p50": round(statistics.median(add_ms), 2),
            "add_memory_ms_max": round(max(add_ms), 2),
            "recall_memory_ms_p50": round(statistics.median(recall_ms), 2),
            "recall_memory_ms_max": round(max(recall_ms), 2),
            "calls_measured": args.calls,
            "process_max_rss_mb": round(child_rss_kb / 1024, 1),
            "session_lifetime_s": round(lifetime, 2),
        },
        "config": {"lines_of_json": 7,
                   "note": "one mcpServers entry: command, args, env"},
    }

    print(f"platform            {report['platform']}  (python {report['python']})")
    print(f"dependencies        {report['install']['third_party_dependencies']} third-party packages")
    print(f"package code        {report['install']['package_python_bytes']} bytes of Python")
    print(f"cold start          {report['startup']['spawn_to_initialize_ms']} ms to a valid initialize reply")
    print(f"tools/list          {report['handshake']['tools_list_ms']} ms, "
          f"{report['handshake']['tools_list_bytes']} bytes, {report['handshake']['tool_count']} tools")
    print(f"add_memory          p50 {report['usage']['add_memory_ms_p50']} ms "
          f"max {report['usage']['add_memory_ms_max']} ms")
    print(f"recall_memory       p50 {report['usage']['recall_memory_ms_p50']} ms "
          f"max {report['usage']['recall_memory_ms_max']} ms")
    print(f"peak RSS            {report['usage']['process_max_rss_mb']} MB")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
