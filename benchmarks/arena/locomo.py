"""Load the public LoCoMo dataset into a system-agnostic shape.

LoCoMo (Maharana et al., 2024; snap-research/locomo) ships 10 long multi-session
conversations, each with QA pairs. `qa[].evidence` lists dialogue ids (``D1:3``)
that justify the answer. Categories: 1 multi-hop, 2 temporal, 3 open-domain /
commonsense, 4 single-hop, 5 adversarial (unanswerable).

Only dialogue turns are exposed here. Event summaries and session summaries in
the raw file are derived annotations, not part of the conversation a memory
system would see, so they are deliberately dropped.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "locomo10.json"

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

_DATE_RE = re.compile(r"^\s*(\d{1,2}:\d{2}\s*[ap]m)\s+on\s+(\d{1,2}\s+\w+,\s*\d{4})\s*$")


def parse_session_time(raw: str) -> float:
    """Parse LoCoMo's ``1:56 pm on 8 May, 2023`` into a Unix timestamp."""
    m = _DATE_RE.match(raw or "")
    if not m:
        raise ValueError(f"unrecognised LoCoMo session timestamp: {raw!r}")
    clock, date = m.group(1).upper().replace(" ", ""), m.group(2).replace(" ", "")
    return datetime.strptime(f"{clock} {date}", "%I:%M%p %d%B,%Y").timestamp()


@dataclass
class Turn:
    speaker: str
    text: str
    dia_id: str
    ts: float


@dataclass
class Session:
    index: int
    ts: float
    turns: list[Turn] = field(default_factory=list)


@dataclass
class QA:
    qa_id: str
    question: str
    answer: str
    category: int
    evidence: list[str]
    adversarial_answer: str | None = None

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, f"cat-{self.category}")

    @property
    def is_adversarial(self) -> bool:
        return self.category == 5


@dataclass
class Conversation:
    sample_id: str
    speakers: tuple[str, str]
    sessions: list[Session]
    qa: list[QA]

    @property
    def turn_count(self) -> int:
        return sum(len(s.turns) for s in self.sessions)

    def all_turns(self) -> list[Turn]:
        return [t for s in self.sessions for t in s.turns]

    def turn_text_by_dia(self) -> dict[str, str]:
        return {t.dia_id: f"{t.speaker}: {t.text}" for t in self.all_turns()}


def load(path: str | Path = DATA_PATH) -> list[Conversation]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    conversations: list[Conversation] = []
    for sample in raw:
        convo = sample["conversation"]
        session_keys = sorted(
            (k for k in convo if re.fullmatch(r"session_\d+", k)),
            key=lambda k: int(k.split("_")[1]),
        )
        sessions: list[Session] = []
        for key in session_keys:
            idx = int(key.split("_")[1])
            ts = parse_session_time(convo[f"{key}_date_time"])
            turns = [
                Turn(
                    speaker=t["speaker"],
                    text=t["text"],
                    dia_id=t["dia_id"],
                    ts=ts,
                )
                for t in convo[key]
            ]
            sessions.append(Session(index=idx, ts=ts, turns=turns))

        qa_items: list[QA] = []
        for i, q in enumerate(sample.get("qa", [])):
            qa_items.append(
                QA(
                    qa_id=f"{sample['sample_id']}-q{i:03d}",
                    question=q["question"],
                    answer=str(q.get("answer", "")),
                    category=int(q["category"]),
                    evidence=list(q.get("evidence", [])),
                    adversarial_answer=q.get("adversarial_answer"),
                )
            )

        speakers = (convo["speaker_a"], convo["speaker_b"])
        conversations.append(
            Conversation(
                sample_id=sample["sample_id"],
                speakers=speakers,
                sessions=sessions,
                qa=qa_items,
            )
        )
    return conversations


def by_id(conversations: list[Conversation], sample_id: str) -> Conversation:
    for c in conversations:
        if c.sample_id == sample_id:
            return c
    raise KeyError(sample_id)


if __name__ == "__main__":
    convs = load()
    for c in convs:
        print(
            f"{c.sample_id}: {len(c.sessions)} sessions, {c.turn_count} turns, "
            f"{len(c.qa)} QA, categories="
            f"{sorted({q.category for q in c.qa})}"
        )
