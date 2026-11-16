#!/usr/bin/env python3
"""Measure what an audit bundle costs: export, verify, and bundle size.

Verification is the number that matters — a third party runs it on their own
machine, offline, with no access to the store. If that step were quadratic it
would not be usable for a real tenant, so this prints the growth of both steps
against the fact count.

    python scripts/measure_audit.py
    python scripts/measure_audit.py --sizes 100 1000 5000 --out docs/audit_cost.json
"""
from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

from infinitydb import audit
from infinitydb.core.engine import InfinityDB


def measure(n: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        db = InfinityDB(str(Path(tmp) / "audit.db"))
        try:
            started = time.perf_counter()
            for i in range(n):
                db.add(f"User note {i}: project Atlas milestone moved to week {i % 52}.",
                       user_id="bench", ts=1709251200.0 + i, auto_supersede=False)
            ingest = time.perf_counter() - started

            started = time.perf_counter()
            bundle = audit.export_bundle(db, "bench")
            exported = time.perf_counter() - started

            blob = json.dumps(bundle)
            started = time.perf_counter()
            report = audit.verify_bundle(bundle)
            verified = time.perf_counter() - started
        finally:
            db.close()

    return {
        "facts_requested": n,
        "facts_in_bundle": bundle["count"],
        "ingest_s": round(ingest, 3),
        "export_ms": round(exported * 1000, 1),
        "verify_ms": round(verified * 1000, 1),
        "verify_ok": report["ok"],
        "bundle_bytes": len(blob.encode("utf-8")),
        "bytes_per_fact": round(len(blob.encode("utf-8")) / max(1, bundle["count"]), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[100, 1000, 5000])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [measure(n) for n in args.sizes]
    print(f"platform: {platform.system()} {platform.release()} {platform.machine()}, "
          f"python {platform.python_version()}")
    print(f"{'facts':>7} {'export ms':>10} {'verify ms':>10} {'bundle KB':>10} "
          f"{'B/fact':>7} ok")
    for row in rows:
        print(f"{row['facts_in_bundle']:>7} {row['export_ms']:>10} {row['verify_ms']:>10} "
              f"{row['bundle_bytes'] / 1024:>10.1f} {row['bytes_per_fact']:>7} "
              f"{row['verify_ok']}")

    report = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "rows": rows,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0 if all(row["verify_ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
