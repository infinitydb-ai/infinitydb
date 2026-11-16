#!/usr/bin/env python3
"""Bring existing memory into InfinityDB in one command.

Switching cost is the reason people stay with a memory layer they already have,
so this removes the one real cost: re-ingestion. Exports are accepted as files —
no competitor SDK is installed, no network call is made, nothing is emulated.

    infinitydb-migrate mem0-export.json --db memory.db --user alice
    infinitydb-migrate chat.jsonl --db memory.db --user alice --format jsonl
    infinitydb-migrate notes.txt --db memory.db --user alice --dry-run

Shapes understood:

* **mem0** — the JSON from `client.get_all()`: `{"results": [{"memory": "...",
  "user_id": "...", "created_at": "2023-05-08T10:00:00Z"}]}`
* **generic JSON** — a list of objects with `content`/`text`/`fact`/`memory`
* **JSONL** — one object per line, same fields, plus optional `role`/`speaker`
* **plain text** — one memory per line, blank lines ignored

Nothing is inferred that the export does not state. A missing timestamp becomes
"now" and is reported as such, because silently inventing event times would
corrupt `as_of` recall and every temporal answer built on it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TEXT_KEYS = ("memory", "content", "text", "fact", "message", "value")


@dataclass
class Record:
    user_id: str
    content: str
    ts: float
    session_id: str | None
    ts_inferred: bool


def parse_ts(value: Any) -> tuple[float, bool]:
    """Return (epoch seconds, whether it had to be invented)."""
    if value is None or value == "":
        return time.time(), True
    if isinstance(value, (int, float)):
        number = float(value)
        # Milliseconds are the common mistake; 1e11 seconds is year 5138.
        if number > 1e11:
            number /= 1000.0
        return number, False
    text = str(value).strip()
    if text.isdigit():
        return parse_ts(int(text))
    cleaned = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return time.time(), True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp(), False


def _content_of(row: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _from_object(row: dict[str, Any], default_user: str) -> Record | None:
    content = _content_of(row)
    if not content:
        return None
    # Only "user" is left bare: an assistant's or system's words carry different
    # weight in memory, so the speaker is kept in the text rather than dropped.
    role = row.get("role") or row.get("speaker") or row.get("author")
    if isinstance(role, str) and role.strip() and role.strip().lower() != "user":
        content = f"{role.strip()}: {content}"
    ts, inferred = parse_ts(row.get("ts") or row.get("timestamp") or row.get("created_at")
                            or row.get("createdAt") or row.get("time"))
    return Record(
        user_id=str(row.get("user_id") or row.get("userId") or default_user),
        content=content,
        ts=ts,
        session_id=(row.get("session_id") or row.get("thread_id") or row.get("conversation_id")),
        ts_inferred=inferred,
    )


def detect_format(path: Path, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix in (".txt", ".md", ".log"):
        return "text"
    if suffix == ".json":
        head = path.read_text(encoding="utf-8", errors="replace")[:2000].lstrip()
        return "json"
    return "jsonl"


def read_records(path: Path, fmt: str, default_user: str) -> Iterable[Record]:
    if fmt == "text":
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                yield Record(default_user, line.strip(), time.time(), None, True)
        return

    if fmt == "jsonl":
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                record = _from_object(row, default_user)
                if record:
                    yield record
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]]
    if isinstance(payload, dict):
        # mem0 get_all() and several SDKs wrap the list in "results"/"memories"/"data".
        rows = next((payload[key] for key in ("results", "memories", "data", "items")
                     if isinstance(payload.get(key), list)), [payload])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    for row in rows:
        if isinstance(row, dict):
            record = _from_object(row, default_user)
            if record:
                yield record


def import_records(db, records: Iterable[Record], dry_run: bool = False) -> dict[str, Any]:
    imported = skipped = inferred = 0
    seen: set[str] = set()
    sample = ""
    for record in records:
        fingerprint = hashlib.sha256(f"{record.user_id}\n{record.content}".encode("utf-8")).hexdigest()
        if fingerprint in seen:
            skipped += 1
            continue
        seen.add(fingerprint)
        already = db.conn.execute(
            "SELECT 1 FROM l0_logs WHERE user_id = ? AND raw = ? LIMIT 1",
            (record.user_id, record.content),
        ).fetchone()
        if already:
            # Re-running an import must not double a user's memory.
            skipped += 1
            continue
        if record.ts_inferred:
            inferred += 1
        if dry_run:
            imported += 1
            sample = sample or record.content
            continue
        db.add(record.content, user_id=record.user_id, session_id=record.session_id,
               ts=record.ts, auto_supersede=False)
        imported += 1
        sample = sample or record.content
    return {"imported": imported, "skipped_existing": skipped,
            "timestamps_inferred": inferred, "sample": sample, "dry_run": dry_run}


def verification(db, stats: dict[str, Any]) -> dict[str, Any]:
    """Prove the import landed: count the atoms and recall the first record."""
    atoms = db.stats().get("l1_atom_facts", 0)
    sample = stats.get("sample") or ""
    query = " ".join(sample.split()[:6])
    hits = []
    if query and not stats["dry_run"]:
        result = db.recall(query, n=3, user_id="default", reinforce=False,
                           include_policies=False)
        if not result.get("results"):
            result = db.recall(query, n=3, reinforce=False, include_policies=False)
        hits = [item.get("fact", "") for item in result.get("results", [])]
    return {"atoms_after_import": atoms, "recall_query": query, "recall_hits": hits}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="infinitydb-migrate",
        description="Import memory from mem0 exports, JSONL, JSON or plain text.",
    )
    ap.add_argument("source", help="file to import")
    ap.add_argument("--db", default="memory.db", help="target SQLite file")
    ap.add_argument("--user", default="default", help="user id when the export has none")
    ap.add_argument("--format", default="auto",
                    choices=("auto", "json", "jsonl", "text"))
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    args = ap.parse_args(argv)

    path = Path(args.source)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    fmt = detect_format(path, args.format)
    from infinitydb.core.engine import InfinityDB

    db = InfinityDB(args.db)
    try:
        stats = import_records(db, read_records(path, fmt, args.user), dry_run=args.dry_run)
        report = verification(db, stats)
    finally:
        db.close()

    summary = {"source": str(path), "format": fmt, **stats, **report}
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        verb = "would import" if args.dry_run else "imported"
        print(f"{verb} {stats['imported']} memories from {path.name} ({fmt}) -> {args.db}")
        if stats["skipped_existing"]:
            print(f"  skipped {stats['skipped_existing']} memories already in the store (re-runs are safe)")
        if stats["timestamps_inferred"]:
            print(f"  {stats['timestamps_inferred']} records had no timestamp and were stored at import time")
        if not args.dry_run:
            print(f"  atoms now in store: {report['atoms_after_import']}")
            if report["recall_hits"]:
                print(f"  recall check for {report['recall_query']!r}: "
                      f"{len(report['recall_hits'])} memories")
    return 0 if stats["imported"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
