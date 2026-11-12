"""LongMemEval (ICLR 2025) loader for the arena.

MemPalace publishes "96.6% R@5 on LongMemEval, zero API calls". LoCoMo numbers
are not comparable to that, so this harness runs the same dataset with the same
retrieval-only setting and reports session-level recall: a question is a hit if
any returned memory comes from one of its annotated answer sessions.

Data: https://huggingface.co/datasets/xiaowu0162/longmemeval  (``longmemeval_s``)

Place the file at ``benchmarks/arena/data/longmemeval_s.json`` (or pass
``--path``); it is 278 MB and is not committed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "longmemeval_s.json"

# "2023/05/30 (Tue) 23:40"
_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})")


def parse_date(raw: str) -> float:
    match = _DATE_RE.search(raw or "")
    if not match:
        return 0.0
    year, month, day, hour, minute = (int(g) for g in match.groups())
    return datetime(year, month, day, hour, minute).timestamp()


@dataclass
class Entry:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_ts: float
    session_ids: list[str]
    sessions: list[list[dict]]
    session_ts: list[float]
    answer_session_ids: list[str]

    def turns(self):
        for sid, session, ts in zip(self.session_ids, self.sessions, self.session_ts):
            for turn in session:
                yield sid, ts, str(turn.get("role", "user")), str(turn.get("content", ""))


def load(path: Path | str = DATA_PATH) -> list[Entry]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries: list[Entry] = []
    for item in raw:
        entries.append(
            Entry(
                question_id=item["question_id"],
                question_type=item.get("question_type", "unknown"),
                question=item["question"],
                answer=str(item.get("answer", "")),
                question_ts=parse_date(item.get("question_date", "")),
                session_ids=list(item["haystack_session_ids"]),
                sessions=list(item["haystack_sessions"]),
                session_ts=[parse_date(d) for d in item.get("haystack_dates", [])],
                answer_session_ids=list(item.get("answer_session_ids", [])),
            )
        )
    return entries
