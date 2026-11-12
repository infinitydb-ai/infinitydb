"""Capture an agent transcript into InfinityDB.

MCP is pull: the agent has to decide to call ``recall_memory``. This module is
the push half, so a session can be remembered without the model cooperating —
run it from a Claude Code ``Stop``/``SessionEnd`` hook, a cron job, or a plain
shell alias, and the turns land in the store.

It is incremental. A small state file records the byte offset already ingested
per transcript, so re-running the hook after every turn only reads what is new
and never duplicates memories.

Usage:
    python -m infinitydb.capture --transcript ~/.claude/projects/x/<id>.jsonl
    python -m infinitydb.capture --transcript t.jsonl --user me --session work
    python -m infinitydb.capture --transcript t.jsonl --all

Claude Code hook (settings.json), capturing after each assistant turn::

    {
      "hooks": {
        "Stop": [{
          "hooks": [{
            "type": "command",
            "command": "python -m infinitydb.capture --transcript \"$CLAUDE_TRANSCRIPT_PATH\""
          }]
        }]
      }
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infinitydb.core.engine import InfinityDB

DEFAULT_STATE = os.path.expanduser("~/.infinitydb/capture-state.json")


def _message_texts(entry: dict[str, Any]) -> list[str]:
    """Extract plain text from one transcript line (string or block content)."""
    message = entry.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
    return texts


def _timestamp(entry: dict[str, Any]) -> float | None:
    raw = entry.get("timestamp")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def read_transcript(path: Path, offset: int) -> tuple[list[tuple[str, str, float | None]], int]:
    """Return ((role, text, ts), ...) for complete lines after ``offset``.

    A trailing partial line is left unconsumed: the writer may still be
    appending to it, and a half-written turn would corrupt the transcript.
    """
    turns: list[tuple[str, str, float | None]] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                offset = position
                break
            offset = handle.tell()
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            role = entry.get("type")
            if role not in ("user", "assistant"):
                continue
            for text in _message_texts(entry):
                text = text.strip()
                if text:
                    speaker = "User" if role == "user" else "Assistant"
                    turns.append((speaker, text, _timestamp(entry)))
    return turns, offset


def capture(
    transcript: Path,
    db_path: str,
    user_id: str,
    session_id: str | None,
    state_path: Path,
    all_turns: bool = False,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
    key = str(transcript.resolve())
    offset = 0 if all_turns else int(state.get(key, {}).get("offset", 0))

    turns, new_offset = read_transcript(transcript, offset)
    added = 0
    if turns:
        db = InfinityDB(db_path)
        try:
            for speaker, text, ts in turns:
                db.add(
                    f"{speaker}: {text}",
                    user_id=user_id,
                    session_id=session_id,
                    ts=ts,
                    auto_supersede=False,
                )
                added += 1
        finally:
            db.close()

    state[key] = {"offset": new_offset, "session_id": session_id, "user_id": user_id}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return {"turns": added, "offset": new_offset, "transcript": key}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture an agent transcript into InfinityDB.")
    parser.add_argument("--transcript", required=True, help="Path to the JSONL transcript.")
    parser.add_argument(
        "--db",
        default=os.environ.get("INFINITYDB_PATH", os.path.expanduser("~/.infinitydb/memory.db")),
    )
    parser.add_argument("--user", default="default")
    parser.add_argument("--session", default=None)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument(
        "--all", action="store_true", help="Re-read the whole transcript instead of resuming."
    )
    args = parser.parse_args(argv)

    transcript = Path(os.path.expanduser(args.transcript))
    if not transcript.exists():
        print(json.dumps({"error": f"transcript not found: {transcript}"}), file=sys.stderr)
        return 1

    summary = capture(
        transcript,
        args.db,
        args.user,
        args.session,
        Path(os.path.expanduser(args.state)),
        all_turns=args.all,
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
