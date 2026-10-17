"""InfinityDB Local Dashboard Server.

Serves the interactive web dashboard on http://localhost:8080
Provides live REST API endpoints connected directly to the Python InfinityDB engine.
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import urllib.parse
from typing import Any

from infinitydb import InfinityDB

PORT = int(os.environ.get("PORT", "8080"))
DB_PATH = os.environ.get("INFINITYDB_PATH", "infinity_dashboard.db")
db = InfinityDB(DB_PATH)

# Seed initial demonstration memories
db.add("I am Dr. Robert Meyer. I work as a Principal Distributed Systems Engineer in Zurich.", user_id="demo_user")
db.add("I prefer Rust for mission-critical core memory systems because it prevents concurrency races.", user_id="demo_user")
db.add("My budget is $50k. We must comply with strict medical HIPAA and SOC2 regulations.", user_id="demo_user")
db.add_policy("Always verify cryptographic proof before serving memory to agent", domain="security", priority=2.0)
db.add_policy("Offload raw transcripts from LLM context to maintain 80%+ token saving", domain="optimization", priority=1.5)


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=dashboard_dir, **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/recall":
            params = urllib.parse.parse_qs(parsed.query)
            q = params.get("q", ["what does the user prefer?"])[0]
            res = db.recall(q, user_id="demo_user")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        if parsed.path == "/api/stats":
            st = db.stats()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(st).encode("utf-8"))
            return

        return super().do_GET()


def main() -> None:
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"[*] InfinityDB Dashboard running at http://localhost:{PORT}")
        print(f"[*] Open http://localhost:{PORT} in your browser to inspect L0-L6 layers.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Shutting down dashboard.")


if __name__ == "__main__":
    main()
