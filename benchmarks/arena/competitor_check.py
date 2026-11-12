"""Report whether each competitor can actually be benchmarked here.

An adapter is only worth writing when the service behind it will answer, so this
checks the prerequisites first — runtime, SDK, graph database, LLM key, local
gateway — and prints the exact missing variable. Nothing is downloaded, nothing
is installed, and with ``--ping`` it only makes the connectivity probes you ask
for.

    python3 benchmarks/arena/competitor_check.py
    python3 benchmarks/arena/competitor_check.py --json
    python3 benchmarks/arena/competitor_check.py --ping

Exit code is 0 when at least one competitor is ready, 1 otherwise. Baselines
that need no credentials (mem0, the BM25+dense hybrid) are reported separately:
they are what the published tables already contain.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PLAYBOOK = "docs/COMPETITOR_BENCHMARK_PLAYBOOK.md"


def _node_major() -> int | None:
    node = shutil.which("node")
    if not node:
        return None
    try:
        raw = subprocess.run(
            [node, "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return int(raw.lstrip("v").split(".")[0])
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def _port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _get(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError) as exc:
        return False, str(getattr(exc, "reason", exc))


def check_graphiti(ping: bool) -> dict:
    missing: list[str] = []
    notes: list[str] = []
    if importlib.util.find_spec("graphiti_core") is None:
        missing.append("graphiti-core (pip install graphiti-core)")
    host = os.environ.get("FALKORDB_HOST") or os.environ.get("NEO4J_URI")
    if not host:
        missing.append("FALKORDB_HOST=... or NEO4J_URI=... (graph database)")
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("GRAPHITI_LLM_API_KEY")):
        missing.append("OPENAI_API_KEY (entity/edge extraction + embeddings)")
    backend = "up"
    if host and ping:
        if os.environ.get("NEO4J_URI"):
            from urllib.parse import urlparse

            parsed = urlparse(os.environ["NEO4J_URI"])
            backend = "up" if _port_open(parsed.hostname or "127.0.0.1", parsed.port or 7687) else "down"
        else:
            backend = "up" if _port_open(os.environ.get("FALKORDB_HOST", "127.0.0.1"), 6379) else "down"
        notes.append(f"graph database: {backend}")
        if backend == "down":
            missing.append("running graph database (FalkorDB :6379 / Neo4j :7687)")
    return {
        "name": "graphiti",
        "what": "the open-source engine behind Zep",
        "ready": not missing,
        "missing": missing,
        "notes": notes,
    }


def check_tencentdb(ping: bool) -> dict:
    missing: list[str] = []
    notes: list[str] = []
    major = _node_major()
    if major is None:
        missing.append("node (>= 22)")
    elif major < 22:
        missing.append(f"node >= 22 (found v{major})")
    else:
        notes.append(f"node v{major}")
    plugin = Path.home() / ".memory-tencentdb" / "tdai-memory-openclaw-plugin"
    if not plugin.exists():
        missing.append(
            "plugin (~/.memory-tencentdb/tdai-memory-openclaw-plugin, npm install @tencentdb-agent-memory/memory-tencentdb)"
        )
    if not (os.environ.get("TDAI_LLM_API_KEY") or os.environ.get("MODEL_API_KEY")):
        missing.append("TDAI_LLM_API_KEY (any OpenAI-compatible LLM endpoint)")
    if ping:
        ok, detail = _get("http://127.0.0.1:8420/health")
        notes.append(f"gateway :8420: {'up' if ok else detail}")
        if not ok:
            missing.append("running gateway on :8420 (npx tsx src/gateway/server.ts)")
    harness = Path(__file__).resolve().parent / "systems" / "tencentdb_keyword"
    keyword_ready = (major is not None and major >= 22) and harness.exists()
    full_ready = not missing
    if keyword_ready:
        notes.append(
            "keyword/FTS mode is runnable without a key: measured 0.5254 evidence "
            "recall@10 on 1,977 LoCoMo questions (harness: systems/tencentdb_keyword)"
        )
    missing_full = list(missing)
    if keyword_ready:
        missing = []
    return {
        "name": "tencentdb-agent-memory",
        "what": "not TencentDB for MySQL; a Node memory plugin with a sqlite-vec backend",
        "ready": full_ready or keyword_ready,
        "mode": "full pipeline" if full_ready else ("keyword mode only" if keyword_ready else None),
        "full_pipeline_missing": missing_full,
        "missing": missing,
        "notes": notes,
    }


def check_zep_cloud(ping: bool) -> dict:
    missing: list[str] = []
    notes: list[str] = []
    if not os.environ.get("ZEP_API_KEY"):
        missing.append("ZEP_API_KEY (Zep Cloud console -> Project Settings -> API Keys)")
    if ping:
        ok, detail = _get("https://api.getzep.com")
        # 401/403 still proves the endpoint is reachable.
        notes.append(f"api.getzep.com: {detail}")
    notes.append("latency includes your network round-trip; label the row with a region")
    return {
        "name": "zep-cloud",
        "what": "managed product (Community Edition was retired in April 2025)",
        "ready": not missing,
        "missing": missing,
        "notes": notes,
    }


BASELINES = [
    ("mem0 2.2.1", "pip install mem0ai; runs with infer=False, no key", "already in the tables"),
    ("BM25 + FastEmbed hybrid", "pip install fastembed", "already in the tables"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--ping", action="store_true", help="also probe endpoints and ports")
    args = parser.parse_args(argv)

    reports = [check_graphiti(args.ping), check_tencentdb(args.ping), check_zep_cloud(args.ping)]
    ready = [r for r in reports if r["ready"]]

    if args.json:
        print(json.dumps(
            {
                "competitors": reports,
                "baselines": [{"name": n, "setup": s, "status": t} for n, s, t in BASELINES],
                "any_ready": bool(ready),
                "playbook": PLAYBOOK,
            },
            indent=2,
        ))
    else:
        for report in reports:
            status = "READY" if report["ready"] else "BLOCKED"
            print(f"[{status}] {report['name']} — {report['what']}")
            for note in report["notes"]:
                print(f"    · {note}")
            for item in report["missing"]:
                print(f"    ✗ {item}")
        print()
        print("Baselines that need no credentials (already measured):")
        for name, setup, status in BASELINES:
            print(f"    · {name}: {status} ({setup})")
        print()
        if ready:
            print(f"{len(ready)} competitor(s) ready. Write the adapter against the live endpoint, not a guess.")
        else:
            print("Nothing is runnable yet. Do not publish competitor numbers until it is.")
            print(f"See {PLAYBOOK} for the exact steps and the cost of each run.")

    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
