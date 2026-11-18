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

import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


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


class BroadPredicateBridge(PredicateBridge):
    """`PredicateBridge` with the aspect vocabulary widened for multi-hop phrasing.

    The strict gate fires on 7.8% of multi-hop questions because LoCoMo asks
    "What did Caroline research?" rather than "Where does Caroline study?".
    Widening only the *aspect* words — never the subject matching — keeps the
    property that gave the strict gate its precision: an edge is used only when
    the question asks for that kind of fact. The question this measures is
    whether vocabulary, not selectivity, was the limit.
    """

    name = "broad"

    _ASPECTS: dict[str, set[str]] = {
        "lives_in": {"where", "live", "lives", "living", "lived", "reside", "resides",
                     "based", "moved", "move", "moves", "moving", "relocated",
                     "city", "town", "home", "address", "country", "originally",
                     "grew", "birthplace", "apartment", "neighborhood", "neighbourhood"},
        "works_at": {"work", "works", "working", "worked", "job", "jobs", "employer",
                     "company", "employed", "workplace", "joined", "career", "profession",
                     "occupation", "position", "role", "industry", "internship", "business",
                     "manager", "colleague", "team", "office"},
        "studied_at": {"study", "studied", "studies", "studying", "school", "college",
                       "university", "degree", "graduated", "class", "research", "researched",
                       "major", "field", "course", "learning", "training", "thesis",
                       "education"},
        "likes": {"like", "likes", "loved", "love", "loves", "enjoy", "enjoys", "prefer",
                  "favourite", "favorite", "hobby", "hobbies", "activity", "activities",
                  "interest", "interests", "passion", "destination", "trip", "travel"},
    }


def entities_in_question(db: Any, question: str, user_id: str = "default") -> list[str]:
    """Capitalised words in the question that are actually edge subjects here."""
    words = {re.sub(r"'s$", "", word) for word in re.findall(r"[A-Za-z][A-Za-z'\-]+", question)
             if word[:1].isupper()}
    if not words:
        return []
    subjects = {row["subject"] for row in db.conn.execute(
        "SELECT DISTINCT subject FROM l1_entity_edges WHERE user_id = ?", (user_id,))}
    return sorted(word for word in words if word.lower() in subjects)


def candidates(db: Any, question: str, user_id: str = "default", hops: int = 2) -> list[Bridge]:
    """Every edge reachable from the people the question names."""
    out: list[Bridge] = []
    seen: set[tuple[str, str, str]] = set()
    for entity in entities_in_question(db, question, user_id):
        for relation in db.relations(entity, user_id=user_id, hops=hops):
            key = (relation["subject"], relation["predicate"], relation["object"])
            if key in seen:
                continue
            seen.add(key)
            out.append(Bridge(
                subject=relation["subject"],
                predicate=relation["predicate"],
                object=relation["object"],
                atom_id=relation.get("atom_id", ""),
                hops=relation.get("hop", 1),
                direction=relation.get("direction", "out"),
            ))
    return out


def witness_texts(db: Any, bridges: Sequence[Bridge], limit: int = 8) -> list[str]:
    """The turns that asserted each edge, with the event date when we have it."""
    out: list[str] = []
    seen: set[str] = set()
    for bridge in bridges:
        if not bridge.atom_id or bridge.atom_id in seen:
            continue
        seen.add(bridge.atom_id)
        row = db.conn.execute("SELECT fact FROM l1_atoms WHERE id = ?", (bridge.atom_id,)).fetchone()
        if row is None:
            continue
        text = " ".join(str(row["fact"]).split())
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def recall_with_bridges(
    db: Any,
    question: str,
    *,
    user_id: str = "default",
    n: int = 10,
    selector: BridgeSelector | None = None,
    max_bridges: int = 8,
    hops: int = 2,
    **recall_kwargs: Any,
) -> dict[str, Any]:
    """Recall, then **append** bridge witnesses instead of spending slots on them.

    Phase 2f measured what happens when witnesses replace retrieved turns: at 89%
    coverage the same selector went from −0.0254 to +0.0228 once they were added
    on top of `top_k` rather than swapped in. A witness is a sentence the retriever
    missed; its damage is a sentence it had already found.

    Opt-in by construction. Multi-hop @10 is 0.5167 against a 0.55 gate, so this is
    not the default path and must not become one until the gate clears.
    """
    from infinitydb.bridge import SubjectBridge

    chosen_selector = selector or SubjectBridge(max_bridges=max_bridges)
    result = db.recall(question, n=n, user_id=user_id, **recall_kwargs)
    picked = chosen_selector.select(
        question, candidates(db, question, user_id=user_id, hops=hops))[:max_bridges]
    witnesses = witness_texts(db, picked, limit=max_bridges)
    extra = [{"fact": text, "score": 0.0, "source": "bridge"} for text in witnesses]
    return {
        **result,
        "results": list(result.get("results", [])) + extra,
        "bridges": [{"subject": b.subject, "predicate": b.predicate, "object": b.object,
                     "hop": b.hops, "direction": b.direction} for b in picked[:len(witnesses)]],
        "bridge_witnesses": witnesses,
        "selector": chosen_selector.name,
    }


__all__ = ["AllBridge", "Bridge", "BridgeSelector", "BroadPredicateBridge", "NullBridge",
           "PredicateBridge", "SubjectBridge", "candidates", "entities_in_question",
           "recall_with_bridges", "witness_texts"]
