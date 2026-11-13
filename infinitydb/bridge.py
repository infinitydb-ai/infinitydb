"""Bridge selection for multi-hop retrieval.

The engine writes typed edges at ingest (``works_at``, ``lives_in``,
``studied_at``, ``likes``) and :meth:`InfinityDB.relations` walks them, including
inbound hops through a shared object. Two earlier experiments fed the reached
terms back into the ranking and measured *worse* multi-hop recall — 0.4484 with
traversal terms OR-ed into the query, 0.4863 with a compositional second pass,
against 0.4939 — because OR-ing more terms into a BM25 query imports distractors
faster than it finds the bridge. Inspection showed why: the deterministic
patterns produce phrases like "all the yellow leaves" for a question about where
someone works, so a second pass spends real evidence slots on noise.

The teardown's conclusion was that the decision *"is this edge a real bridge for
this question"* is a classification problem, not a pattern-matching one. This
module is that seam, and nothing more: a selector receives candidate edges and
may veto them. It does not rank, it does not rewrite the query, and it is not
called anywhere in the default path.

A selector's job is to be **beaten**. ``PredicateBridge`` is the strongest gate
that can be written without a parser; a learned selector has to outperform it on
held-out multi-hop recall to justify its inference cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Bridge:
    """One typed edge a selector may accept or veto."""

    subject: str
    predicate: str
    object: str
    atom_id: str = ""
    hops: int = 1
    # "out" when the named entity is the subject, "in" when it is the object
    # and the edge arrives from someone else. Second-hop witnesses only exist
    # on the inbound side: "Tom also works at the gym".
    direction: str = "out"


class BridgeSelector(Protocol):
    """Contract for anything that decides which edges are worth using."""

    name: str

    def select(self, question: str, candidates: Sequence[Bridge]) -> list[Bridge]:
        """Return the bridges to use for this question (possibly none)."""
        ...


class NullBridge:
    """Accept nothing. The default, and the baseline a selector must beat."""

    name = "null"

    def select(self, question: str, candidates: Sequence[Bridge]) -> list[Bridge]:
        return []


class AllBridge:
    """Accept every candidate edge, up to a cap.

    This is the **upper bound** arm, not a recommendation: it answers "would
    using these bridge objects help at all, if we knew they were all relevant?"
    If this arm does not beat the baseline, no selector can, and the bridge
    approach is finished for this ranking. The teardown's two rejected
    experiments suggest it will not — OR-ing reached terms imported distractors
    faster than they found bridges — and this arm re-tests that at the object
    level rather than at the term level.
    """

    name = "all"

    def __init__(self, max_bridges: int = 3) -> None:
        self.max_bridges = max_bridges

    def select(self, question: str, candidates: Sequence[Bridge]) -> list[Bridge]:
        chosen: list[Bridge] = []
        seen: set[str] = set()
        for bridge in candidates:
            key = bridge.object.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            chosen.append(bridge)
            if len(chosen) >= self.max_bridges:
                break
        return chosen


class SubjectBridge:
    """Accept edges belonging to a person the question actually names.

    The middle ground between the strict aspect gate (fires on 7.8% of
    multi-hop questions) and accepting everything (halves recall). A question
    that names a person is asking about *that person's* facts, whatever the
    predicate, so the subject itself is the relevance signal. It still refuses
    an object the question already mentions, because that adds nothing.
    """

    name = "subject"

    def __init__(self, max_bridges: int = 2, min_object_chars: int = 3) -> None:
        self.max_bridges = max_bridges
        self.min_object_chars = min_object_chars

    def select(self, question: str, candidates: Sequence[Bridge]) -> list[Bridge]:
        words = {word.strip(".,?!'\"").lower() for word in question.split()}
        chosen: list[Bridge] = []
        seen: set[str] = set()
        for bridge in candidates:
            named = bridge.subject.strip().lower() in words
            # An inbound edge is one `relations()` reached *through* the named
            # entity's edge: its object is the shared node, and its subject is
            # the other party. Requiring the object in the question would make
            # second-hop witnesses unreachable — the shared node is precisely
            # what the question does not mention.
            inbound = bridge.direction == "in"
            if not (named or inbound):
                continue
            obj = bridge.object.strip()
            if inbound:
                # The inbound edge's useful string is the other party, not the
                # object the question already names.
                obj = bridge.subject.strip()
            if len(obj) < self.min_object_chars or obj.lower() in words:
                continue
            if obj.lower() in seen:
                continue
            seen.add(obj.lower())
            chosen.append(bridge)
            if len(chosen) >= self.max_bridges:
                break
        return chosen


class PredicateBridge:
    """Deterministic gate: question aspect must match the edge predicate.

    Deliberately strict:

    * only predicates implied by the question are accepted (``where`` →
      ``lives_in``, work words → ``works_at``, study words → ``studied_at``,
      preference words → ``likes``); an unmatched question accepts nothing,
    * the object must not already appear in the question — a bridge that says
      what the question already says adds nothing,
    * objects are deduplicated and capped, because evidence slots are scarce.

    This exists to be a fair, strong baseline for a learned selector: it uses the
    same typed edges and no more information than a question contains.
    """

    name = "predicate"

    _ASPECTS: dict[str, set[str]] = {
        "lives_in": {"where", "live", "lives", "living", "lived", "reside", "resides",
                     "based", "moved", "relocated", "city", "town", "home", "address"},
        "works_at": {"work", "works", "working", "job", "employer", "company",
                     "employed", "workplace", "joined"},
        "studied_at": {"study", "studied", "school", "college", "university",
                       "degree", "graduated", "class"},
        "likes": {"like", "likes", "love", "loves", "enjoy", "enjoys", "prefer",
                  "favourite", "favorite", "hobby"},
    }

    def __init__(self, max_bridges: int = 3, min_object_chars: int = 3) -> None:
        self.max_bridges = max_bridges
        self.min_object_chars = min_object_chars

    def select(self, question: str, candidates: Sequence[Bridge]) -> list[Bridge]:
        words = {word.strip(".,?!'\"").lower() for word in question.split()}
        allowed = {predicate for predicate, hints in self._ASPECTS.items() if hints & words}
        if not allowed:
            return []
        chosen: list[Bridge] = []
        seen: set[str] = set()
        for bridge in candidates:
            if bridge.predicate not in allowed:
                continue
            obj = bridge.object.strip()
            if len(obj) < self.min_object_chars:
                continue
            if obj.lower() in {word.strip(".,?!'\"").lower() for word in question.split()}:
                continue
            key = obj.lower()
            if key in seen:
                continue
            seen.add(key)
            chosen.append(bridge)
            if len(chosen) >= self.max_bridges:
                break
        return chosen


__all__ = ["AllBridge", "Bridge", "BridgeSelector", "NullBridge", "PredicateBridge",
           "SubjectBridge"]
