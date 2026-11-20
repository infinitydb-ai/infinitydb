"""Calibrated retrieval confidence and abstention gating for InfinityDB.

Zero third-party dependencies. Maps unbounded lexical scores and cross-encoder
logits onto a bounded [0, 1] confidence scale so ``recall()`` can emit calibrated
``confidence`` / ``abstain_recommended`` signals and optionally filter results
with ``min_score``.

Two independent signals are produced:

* Lexical confidence is the IDF-weighted fraction of the query's content terms
  that the candidate actually contains, so it does not inherit the unbounded
  scale of a BM25 score. Query terms that appear nowhere in the tenant corpus
  are left out of the denominator: they cannot be matched by any candidate, and
  counting them at maximum IDF would mark every paraphrase as low confidence.
  When *no* query term exists anywhere in the corpus the confidence is 0, which
  is the honest reading of "this store has never mentioned the question".
* A candidate that misses the query's highest-IDF term (the entity or topic the
  question is really about) is penalised, because unanswerable questions tend to
  share generic verbs with the store while their specific anchor is absent.

Reranker confidence maps the cross-encoder logit ``z`` through
``sigmoid((z - shift) / temperature)`` and blends it with the lexical score, so
a high-logit false friend cannot carry a whole query on its own.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
    "to", "in", "on", "at", "for", "of", "with", "by", "from", "about", "as",
    "into", "like", "through", "after", "over", "between", "out", "against",
    "during", "without", "before", "under", "around", "among",
    "and", "or", "but", "if", "because", "until", "while",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "what", "where", "when", "who", "whom", "whose", "which", "why", "how",
    "that", "this", "these", "those", "there", "here", "tell", "know", "remember",
})

DEFAULT_LEXICAL_ABSTAIN_THRESHOLD = 0.42
DEFAULT_RERANK_ABSTAIN_THRESHOLD = 0.45

# Threshold for the shipped post-packing gate (``score_query_confidence``).
DEFAULT_ABSTAIN_THRESHOLD = 0.45


def tokenize_content(text: str) -> list[str]:
    """Extract lowercase alphanumeric content tokens excluding stopwords."""
    raw = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [tok for tok in raw if len(tok) > 1 and tok not in _STOPWORDS]


def compute_corpus_idf(documents: list[str]) -> dict[str, float]:
    """Smoothed BM25-style IDF weights across the tenant's stored facts.

    ``idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))``
    """
    n_docs = max(1, len(documents))
    df: Counter[str] = Counter()
    for doc in documents:
        for tok in set(tokenize_content(doc)):
            df[tok] += 1

    idf: dict[str, float] = {}
    for tok, freq in df.items():
        idf[tok] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
    return idf


def unseen_token_idf(n_docs: int) -> float:
    """Maximum smoothed IDF assigned to a term completely absent from the corpus."""
    n = max(1, n_docs)
    return math.log(1.0 + (n + 0.5) / 0.5)


def compute_lexical_confidence(
    query: str,
    candidate_text: str,
    corpus_idf: dict[str, float],
    n_docs: int,
    *,
    raw_bm25_score: float | None = None,
) -> float:
    """Bounded [0, 1] lexical match confidence for one candidate.

    Coverage is measured only over query terms the corpus could ever match
    (present in the corpus vocabulary, or already present in the candidate).
    The rarest of those terms is the query's anchor; a candidate that lacks it
    keeps at most 55% of its coverage.
    """
    q_tokens = list(dict.fromkeys(tokenize_content(query)))
    if not q_tokens:
        return 0.0

    cand_tokens = set(tokenize_content(candidate_text))
    if not cand_tokens:
        return 0.0

    missing_weight = unseen_token_idf(n_docs)
    q_weights: dict[str, float] = {}
    for tok in q_tokens:
        if tok in corpus_idf:
            q_weights[tok] = corpus_idf[tok]
        elif tok in cand_tokens:
            # Outside the sampled corpus but present here: still worth credit.
            q_weights[tok] = missing_weight
    if not q_weights:
        # The store has never used any of these words; no candidate can answer.
        return 0.0

    total_weight = sum(q_weights.values())
    matched_weight = sum(w for t, w in q_weights.items() if t in cand_tokens)
    idf_coverage = matched_weight / total_weight

    # Anchor check: unanswerable questions share generic verbs and nouns with the
    # store while the specific entity/object (highest-IDF term) never appears.
    anchor_tok = max(q_weights, key=lambda t: q_weights[t])
    anchor_multiplier = 1.0 if anchor_tok in cand_tokens else 0.55

    conf = idf_coverage * anchor_multiplier
    if raw_bm25_score is not None and raw_bm25_score <= 0.0:
        conf *= 0.25

    return round(max(0.0, min(1.0, conf)), 6)


def calibrate_reranker_logit(
    logit: float,
    *,
    shift: float = -1.5,
    temperature: float = 2.0,
) -> float:
    """Map a raw cross-encoder logit (e.g. MS-MARCO MiniLM) onto [0, 1].

    The defaults put logit -1.5 at 0.50, -5.0 at 0.15 and +3.0 at 0.91 for a
    model whose logits are already moderately positive at 0.
    """
    scaled = max(-20.0, min(20.0, (float(logit) - shift) / max(1e-6, temperature)))
    return round(1.0 / (1.0 + math.exp(-scaled)), 6)


def _candidate_text(candidate: dict[str, Any]) -> str:
    return str(candidate.get("fact", ""))


def _idf_weights(
    query: str,
    token_sets: list[set[str]],
    corpus_idf: dict[str, float],
    n_docs: int,
) -> dict[str, float]:
    """Query tokens that the store or the retrieved items could ever match.

    Tokens that appear nowhere are excluded rather than weighted at maximum IDF:
    they cannot be matched by any candidate, and counting them would mark every
    paraphrase as low confidence.
    """
    everything = set().union(*token_sets) if token_sets else set()
    missing = unseen_token_idf(n_docs)
    weights: dict[str, float] = {}
    for tok in dict.fromkeys(tokenize_content(query)):
        if tok in corpus_idf:
            weights[tok] = corpus_idf[tok]
        elif tok in everything:
            weights[tok] = missing
    return weights


def _coverage(tokens: set[str], q_weights: dict[str, float]) -> float:
    total = sum(q_weights.values())
    if total <= 0.0:
        return 0.0
    return sum(weight for tok, weight in q_weights.items() if tok in tokens) / total


def score_query_confidence(
    query: str,
    results: list[dict[str, Any]],
    tenant_facts: list[str] | None = None,
    threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
    *,
    corpus_statistics: tuple[dict[str, float], int] | None = None,
    has_reranker: bool = False,
    rerank_shift: float = -1.5,
    rerank_temperature: float = 2.0,
    best_weight: float = 0.65,
    union_weight: float = 0.35,
    anchor_penalty: float = 0.50,
) -> dict[str, Any]:
    """Confidence for an already-packed result set, and whether to abstain.

    Coverage is ``0.65 * best single item + 0.35 * top-3 union``, so a question
    whose two evidence items landed in different slots is still counted as
    covered. The query's highest-IDF term is the semantic anchor: when it is
    absent from the top three, coverage is halved, which is what separates an
    unanswerable question that shares generic verbs from a real hit.

    This runs after budget packing and after the recency tie-break, so it can
    only empty the result set; it cannot move a retrieval rank. ``threshold`` is
    the confidence below which the gate abstains.
    """
    threshold = float(threshold)
    if not results:
        return {
            "confidence": 0.0,
            "abstain_recommended": True,
            "confidence_threshold": round(threshold, 4),
        }

    if corpus_statistics is not None:
        corpus_idf, n_docs = corpus_statistics
    else:
        docs = tenant_facts if tenant_facts is not None else [_candidate_text(r) for r in results]
        corpus_idf = compute_corpus_idf(docs)
        n_docs = max(1, len(docs))

    item_tokens = [set(tokenize_content(_candidate_text(item))) for item in results]
    head = item_tokens[:3]
    q_weights = _idf_weights(query, item_tokens, corpus_idf, n_docs)
    if not q_weights:
        # No token of the question exists in the store or in what came back.
        return {
            "confidence": 0.0,
            "abstain_recommended": True,
            "confidence_threshold": round(threshold, 4),
        }

    best = max((_coverage(tokens, q_weights) for tokens in item_tokens), default=0.0)
    union = _coverage(set().union(*head), q_weights) if head else 0.0
    coverage = best_weight * best + union_weight * union

    anchor = max(q_weights, key=lambda tok: q_weights[tok])
    if anchor not in set().union(*head):
        coverage *= anchor_penalty

    confidence = max(0.0, min(1.0, coverage))
    if has_reranker:
        logits = [_rerank_logit(item) for item in results]
        available = [logit for logit in logits if logit is not None]
        if available:
            ce_prob = calibrate_reranker_logit(
                max(available), shift=rerank_shift, temperature=rerank_temperature
            )
            confidence = 0.80 * ce_prob + 0.20 * confidence

    confidence = round(max(0.0, min(1.0, confidence)), 6)
    return {
        "confidence": round(confidence, 4),
        "abstain_recommended": bool(confidence < threshold),
        "confidence_threshold": round(threshold, 4),
    }


def _lexical_score(candidate: dict[str, Any]) -> float:
    for key in ("_raw", "raw_score", "score"):
        value = candidate.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _rerank_logit(candidate: dict[str, Any]) -> float | None:
    for key in ("_ce", "ce_score"):
        value = candidate.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def score_candidates_confidence(
    query: str,
    candidates: list[dict[str, Any]],
    corpus_documents: list[str] | None = None,
    *,
    has_reranker: bool = False,
    rerank_shift: float = -1.5,
    rerank_temperature: float = 2.0,
    min_score: float | None = None,
    abstain_on_low_confidence: bool = False,
    corpus_statistics: tuple[dict[str, float], int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach calibrated ``confidence`` in [0, 1] to every candidate.

    Returns the (possibly filtered) candidates plus query-level metadata:
    ``confidence`` (best candidate), ``abstain_recommended`` and
    ``confidence_threshold``. Nothing is filtered unless ``min_score`` is given
    or ``abstain_on_low_confidence`` is true, so existing callers keep every
    result they used to get. ``corpus_statistics`` is an optional precomputed
    ``(idf, n_docs)`` pair, letting a caller cache the corpus statistics
    instead of re-tokenizing the tenant on every query.
    """
    if corpus_statistics is not None:
        corpus_idf, n_docs = corpus_statistics
    else:
        docs = (
            corpus_documents
            if corpus_documents is not None
            else [_candidate_text(c) for c in candidates]
        )
        corpus_idf = compute_corpus_idf(docs)
        n_docs = max(1, len(docs))

    default_threshold = (
        DEFAULT_RERANK_ABSTAIN_THRESHOLD if has_reranker else DEFAULT_LEXICAL_ABSTAIN_THRESHOLD
    )
    active_threshold = float(min_score) if min_score is not None else default_threshold

    for cand in candidates:
        lex_conf = compute_lexical_confidence(
            query=query,
            candidate_text=_candidate_text(cand),
            corpus_idf=corpus_idf,
            n_docs=n_docs,
            raw_bm25_score=None if has_reranker else _lexical_score(cand),
        )
        logit = _rerank_logit(cand) if has_reranker else None
        if logit is not None:
            # Blend the calibrated cross-encoder probability with lexical entity
            # coverage so high-logit false friends cannot answer alone.
            ce_prob = calibrate_reranker_logit(
                logit,
                shift=rerank_shift,
                temperature=rerank_temperature,
            )
            conf = 0.80 * ce_prob + 0.20 * lex_conf
        else:
            conf = lex_conf
        cand["confidence"] = round(max(0.0, min(1.0, conf)), 6)

    top_confidence = max((float(c["confidence"]) for c in candidates), default=0.0)
    abstain_recommended = (len(candidates) == 0) or (top_confidence < active_threshold)

    if min_score is not None or abstain_on_low_confidence:
        candidates = [c for c in candidates if float(c["confidence"]) >= active_threshold]
        if not candidates:
            abstain_recommended = True

    meta = {
        "confidence": round(top_confidence, 4),
        "abstain_recommended": bool(abstain_recommended),
        "confidence_threshold": round(active_threshold, 4),
    }
    return candidates, meta
