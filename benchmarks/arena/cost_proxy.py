"""Transparent Ollama proxy that counts LLM calls and tokens per system.

Every competitor adapter points its OpenAI-compatible base_url at this proxy, so
"LLM calls" and "tokens processed" are measured on the wire instead of being
estimated or self-reported. The proxy forwards to the real Ollama server and
appends one JSON line per request to the current run's ledger.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM = os.environ.get("ARENA_OLLAMA_UPSTREAM", "http://127.0.0.1:11434")
LEDGER = os.environ.get("ARENA_LEDGER", "")
LOCK = threading.Lock()


def _count_from_response(body: bytes) -> dict:
    """Pull Ollama's own token accounting out of a chat/embeddings response."""
    try:
        data = json.loads(body)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for key in ("prompt_eval_count", "eval_count"):
        if isinstance(data.get(key), int):
            out[key] = data[key]
    usage = data.get("usage")
    if isinstance(usage, dict):
        if isinstance(usage.get("prompt_tokens"), int):
            out.setdefault("prompt_eval_count", usage["prompt_tokens"])
        if isinstance(usage.get("completion_tokens"), int):
            out.setdefault("eval_count", usage["completion_tokens"])
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence stderr noise
        pass

    def _proxy(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        url = UPSTREAM.rstrip("/") + self.path
        req = urllib.request.Request(url, data=body or None, method=method)
        for header in ("Content-Type", "Accept"):
            if self.headers.get(header):
                req.add_header(header, self.headers[header])
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                payload = resp.read()
                status = resp.status
                ctype = resp.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            status = exc.code
            ctype = exc.headers.get("Content-Type", "application/json")
        except Exception as exc:  # upstream down
            payload = json.dumps({"error": str(exc)}).encode()
            status = 502
            ctype = "application/json"

        if status < 400 and "stream" not in self.path:
            record = {"t": round(time.time() - started, 3), "path": self.path,
                      "req_bytes": len(body)}
            record.update(_count_from_response(payload))
            if LEDGER:
                with LOCK, open(LEDGER, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")

        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802
        self._proxy("POST")

    def do_GET(self):  # noqa: N802
        self._proxy("GET")


def main() -> None:
    host = os.environ.get("ARENA_PROXY_HOST", "127.0.0.1")
    port = int(os.environ.get("ARENA_PROXY_PORT", "11435"))
    if LEDGER:
        Path(LEDGER).parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"arena cost proxy on http://{host}:{port} -> {UPSTREAM}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
