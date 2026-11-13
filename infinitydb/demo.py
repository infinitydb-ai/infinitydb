#!/usr/bin/env python3
"""`python -m infinitydb.demo` — see memory work in one command.

No database file, no configuration, no model call, no network. This is the
shortest path from `pip install infinitydb-ai` to seeing a fact come back out,
because the alternative (clone the repo, run an example) is where people stop.

    python -m infinitydb.demo
    python -m infinitydb.demo --query "what is the budget"
"""
from __future__ import annotations

import argparse
import time

from infinitydb import InfinityDB

# Event times are fixed, not "now", so the output is identical on every machine
# and the dates are checkable rather than relative.
CONVERSATION = [
    ("I moved to Lisbon in March 2024 and I work as a data engineer at a ceramics studio.",
     1709251200.0),  # 2024-03-01
    ("My budget for the new cluster is forty thousand euros.", 1711929600.0),  # 2024-04-01
    ("I prefer morning calls and short, direct answers.", 1714521600.0),  # 2024-05-01
]
QUERIES = ["where does the user live", "what do they do for work", "what is the budget"]
NOW = 1714521600.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A zero-config InfinityDB demonstration.")
    ap.add_argument("--query", action="append", default=None,
                    help="recall query; repeatable (default: three fixed questions)")
    ap.add_argument("-n", type=int, default=2, help="memories to recall per question")
    args = ap.parse_args(argv)
    queries = args.query or QUERIES

    started = time.perf_counter()
    db = InfinityDB(":memory:")
    opened_ms = (time.perf_counter() - started) * 1000

    ingested = time.perf_counter()
    for text, ts in CONVERSATION:
        db.add(text, user_id="demo", ts=ts, auto_supersede=False)
    ingest_ms = (time.perf_counter() - ingested) * 1000

    print("InfinityDB demo — in-memory, no model calls, no network")
    print(f"  open    {opened_ms:6.1f} ms")
    print(f"  ingest  {ingest_ms:6.1f} ms  ({len(CONVERSATION)} turns)")
    print()

    total_recall = 0.0
    for query in queries:
        recalled = time.perf_counter()
        result = db.recall(query, n=args.n, user_id="demo", now=NOW)
        elapsed = (time.perf_counter() - recalled) * 1000
        total_recall += elapsed
        print(f"query: {query!r}")
        hits = result.get("results", [])
        for item in hits:
            fact = " ".join(str(item.get("fact", "")).split())
            print(f"  [{item.get('score', 0):.3f}] {fact}")
        if not hits:
            print("  (nothing above threshold)")
        print(f"  {len(hits)} memories in {elapsed:.1f} ms")
        print()

    stats = db.stats()
    print(f"  stored  facts={stats.get('l1_atom_facts', 0)} "
          f"proofs={stats.get('l6_cryptographic_proofs', 0)} "
          f"backend={stats.get('backend', '?')}")
    print(f"  recall  {total_recall:.1f} ms over {len(queries)} questions")
    print()
    print("Next: give this to an agent over MCP —")
    print("  claude mcp add infinitydb -- uvx --from infinitydb-ai infinitydb-mcp")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
