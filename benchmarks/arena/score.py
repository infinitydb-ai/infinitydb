"""Score arena runs: retrieval quality, reader-LLM QA accuracy, latency.

Two independent layers are reported, because they fail for different reasons:

1. Retrieval metrics (no LLM): does the recalled context contain the gold
   answer, and does it contain the dialogue turns annotated as evidence?
2. End-to-end QA (one fixed local reader model for every system): token F1
   against the gold answer, plus a judge-free "gold answer contained in the
   generated answer" check. Category 5 (adversarial) is scored separately:
   the correct behaviour there is to say the answer is not in memory.

Every generated answer is cached by (system, qa_id, context) hash so reruns and
rescoring never pay for the same generation twice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parent
ROOT = ARENA_DIR.parents[1]
for p in (str(ROOT), str(ARENA_DIR.parents[0])):
    if p not in sys.path:
        sys.path.insert(0, p)

from arena import locomo  # noqa: E402

OLLAMA = os.environ.get("ARENA_OLLAMA", "http://127.0.0.1:11434")
READER_MODEL = os.environ.get("ARENA_READER_MODEL", "llama3.1:8b")

PROMPT = """You are answering questions about a long two-person conversation.

Rules:
- Use only the memories below.
- Answer in a few words (a name, date, place, or short phrase).
- If the memories do not contain the answer, reply exactly: Not mentioned

Memories:
{context}

Question: {question}
Answer:"""

REFUSAL = re.compile(r"not mentioned|no information|not (?:stated|specified|available)|unknown|cannot", re.I)

STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on",
    "at", "for", "and", "or", "does", "do", "did", "what", "which", "who", "whom",
    "where", "when", "how", "why", "my", "me", "i", "it", "its", "this", "that",
    "with", "as", "by", "from", "one", "they", "their", "them", "he", "she", "his", "her",
}

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in STOP]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if len(t) > 2]


def normalize_answer(text: str) -> str:
    """Lowercase, expand month names to numbers, drop punctuation and articles."""
    t = (text or "").lower()
    for name, num in MONTHS.items():
        t = re.sub(rf"\b{name}\b", str(num), t)
    return " ".join(tokens(t))


def f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return 0.0
    common = {}
    for tok in p:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in g:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def gold_contained(pred: str, gold: str) -> bool:
    """Judge-free proxy for semantic correctness: all gold content words present."""
    gold_toks = content_tokens(gold)
    if not gold_toks:
        return False
    pred_toks = set(tokens(pred))
    return all(t in pred_toks for t in gold_toks)


def evidence_hit(retrieved: list[str], evidence_texts: list[str]) -> float:
    """Fraction of annotated evidence turns that appear in the recalled context."""
    if not evidence_texts:
        return float("nan")
    blob = " ".join(retrieved).lower()
    hits = 0
    for ev in evidence_texts:
        ev_toks = content_tokens(ev)
        if not ev_toks:
            continue
        ev_set = set(ev_toks)
        best = 0.0
        for chunk in retrieved:
            chunk_set = set(content_tokens(chunk))
            if not chunk_set:
                continue
            best = max(best, len(ev_set & chunk_set) / len(ev_set))
        if best >= 0.6 or len(ev_set & set(tokens(blob))) / len(ev_set) >= 0.9:
            hits += 1
    return hits / len(evidence_texts)


def answer_in_context(retrieved: list[str], gold: str) -> bool:
    gold_toks = content_tokens(gold)
    if not gold_toks:
        return False
    blob = set(tokens(" ".join(retrieved)))
    return all(t in blob for t in gold_toks)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[idx]


def generate(question: str, retrieved: list[str], top_k: int = 10) -> str:
    context = "\n".join(f"- {m}" for m in retrieved[:top_k]) or "- (no memories)"
    body = {
        "model": READER_MODEL,
        "prompt": PROMPT.format(context=context, question=question),
        "stream": False,
        "options": {"temperature": 0, "num_predict": 40},
    }
    req = urllib.request.Request(
        OLLAMA.rstrip("/") + "/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    return (data.get("response") or "").strip()


def _hash_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


def score_run(
    result_path: Path,
    conversation: locomo.Conversation,
    cache_path: Path,
    readers: int,
    use_reader: bool = True,
    limit: int = 0,
) -> dict:
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    qa_rows = [r for r in rows if r.get("type") == "qa"]
    # A deterministic sample (first N questions per conversation) lets a CPU-only
    # reader score a slice without touching the retrieval rows. Generations are
    # cached per question, so a later full run reuses whatever a sample already
    # produced.
    sampled = 0
    if limit:
        qa_rows = qa_rows[:limit]
        sampled = len(qa_rows)
    meta = next((r for r in rows if r.get("type") == "meta"), {})
    ingest = next((r for r in rows if r.get("type") == "ingest_done"), {})
    evidence_text = conversation.turn_text_by_dia()
    # Enforce the configured top-k for every system. A system that returns more
    # than k (mem0 ignores ``limit`` and defaults to 20) must not be scored on
    # context the reader would never have received.
    k = int(meta.get("top_k") or 0) or None

    cache: dict[str, str] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                cache[item["key"]] = item["answer"]

    pending = []
    for row in qa_rows:
        key = _hash_key(READER_MODEL, row["qa_id"], "\u241f".join(row["retrieved"]))
        row["_key"] = key
        if use_reader and key not in cache:
            pending.append(row)

    def work(row: dict) -> tuple[str, str]:
        answer = generate(row["question"], row["retrieved"])
        return row["_key"], answer

    if pending:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=readers) as pool, cache_path.open("a", encoding="utf-8") as fh:
            for key, answer in pool.map(work, pending):
                cache[key] = answer
                fh.write(json.dumps({"key": key, "answer": answer}) + "\n")
                fh.flush()

    per_category: dict[str, list[dict]] = {}
    for row in qa_rows:
        prediction = cache.get(row["_key"], "") if use_reader else ""
        gold = row["answer"]
        adversarial = row["category"] == 5
        ev_texts = [evidence_text[e] for e in row["evidence"] if e in evidence_text]
        retrieved = row["retrieved"][:k] if k else row["retrieved"]
        rec = {
            "qa_id": row["qa_id"],
            "category": row["category"],
            "ms": row["ms"],
            "error": row.get("error"),
            "f1": 0.0 if adversarial else f1(prediction, gold),
            "gold_in_answer": False if adversarial else gold_contained(prediction, gold),
            "answer_in_context": answer_in_context(retrieved, gold),
            "evidence_recall": evidence_hit(retrieved, ev_texts),
            "context_tokens": len(re.findall(r"\w+", " ".join(retrieved))),
            "refused": bool(REFUSAL.search(prediction)),
            "adversarial": adversarial,
            "prediction": prediction,
            "gold": gold,
            "question": row["question"],
            "retrieved": retrieved,
        }
        per_category.setdefault(row["category_name"], []).append(rec)

    answerable = [r for r in qa_rows if r["category"] != 5]
    answered = [rec for rec in (r for cat in per_category.values() for r in cat) if not rec["adversarial"]]
    adversarial = [rec for cat in per_category.values() for rec in cat if rec["adversarial"]]
    latencies = [r["ms"] for r in qa_rows if r.get("ms") is not None]

    summary = {
        "system": meta.get("system"),
        "conv": meta.get("conv"),
        "questions": len(qa_rows),
        "answerable_questions": len(answerable),
        "ingest_seconds": ingest.get("seconds"),
        "qa_f1": round(sum(r["f1"] for r in answered) / len(answered), 4) if answered else None,
        "gold_in_answer": round(sum(r["gold_in_answer"] for r in answered) / len(answered), 4) if answered else None,
        "answer_in_context": round(
            sum(r["answer_in_context"] for r in answered) / len(answered), 4) if answered else None,
        "evidence_recall@k": round(
            sum(r["evidence_recall"] for r in answered if not math.isnan(r["evidence_recall"]))
            / max(1, len([r for r in answered if not math.isnan(r["evidence_recall"])])), 4),
        "adversarial_refusal_rate": round(
            sum(r["refused"] for r in adversarial) / len(adversarial), 4) if adversarial else None,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "context_tokens_median": percentile(
            [r["context_tokens"] for r in answered], 50) if answered else None,
        "errors": sum(1 for r in qa_rows if r.get("error")),
        "by_category": {
            name: {
                "n": len(recs),
                "f1": round(sum(r["f1"] for r in recs) / len(recs), 4),
                "gold_in_answer": round(sum(r["gold_in_answer"] for r in recs) / len(recs), 4),
                "answer_in_context": round(
                    sum(r["answer_in_context"] for r in recs) / len(recs), 4),
                "evidence_recall@k": round(
                    sum(r["evidence_recall"] for r in recs if not math.isnan(r["evidence_recall"]))
                    / max(1, len([r for r in recs if not math.isnan(r["evidence_recall"])])), 4),
            }
            for name, recs in sorted(per_category.items())
        },
        "reader_model": READER_MODEL if use_reader else None,
        "sampled_questions": sampled,
        "details": per_category,
    }
    return summary


CORE_KEYS = (
    "system", "conv", "questions", "answerable_questions", "ingest_seconds",
    "qa_f1", "gold_in_answer", "answer_in_context", "evidence_recall@k",
    "adversarial_refusal_rate", "latency_ms", "context_tokens_median", "errors",
    "reader_model",
)


def aggregate(summaries: list[dict]) -> dict:
    """Question-weighted mean across conversations, with overall latencies."""
    answered = [s for s in summaries if s.get("answerable_questions")]
    latencies = [
        r["ms"] for s in summaries for recs in s["details"].values()
        for r in recs if r.get("ms") is not None
    ]
    contexts = [
        rec["context_tokens"]
        for s in summaries
        for recs in s["details"].values()
        for rec in recs
        if rec.get("context_tokens") is not None
    ]

    def weighted(key: str) -> float | None:
        den = sum(s["answerable_questions"] for s in answered)
        if not den:
            return None
        num = sum((s.get(key) or 0.0) * s["answerable_questions"] for s in answered)
        return round(num / den, 4)

    return {
        "conversations": len(summaries),
        "questions": sum(s["questions"] for s in summaries),
        "answerable_questions": sum(s["answerable_questions"] for s in summaries),
        "ingest_seconds": round(sum(s.get("ingest_seconds") or 0.0 for s in summaries), 3),
        "qa_f1": weighted("qa_f1"),
        "gold_in_answer": weighted("gold_in_answer"),
        "answer_in_context": weighted("answer_in_context"),
        "evidence_recall@k": weighted("evidence_recall@k"),
        "adversarial_refusal_rate": weighted("adversarial_refusal_rate"),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "context_tokens_median": percentile(contexts, 50) if contexts else None,
        "by_category": {
            name: {
                "n": round(sum(s["by_category"].get(name, {}).get("n", 0) for s in answered), 4),
                "answer_in_context": round(
                    sum(s["by_category"].get(name, {}).get("answer_in_context", 0.0)
                        * s["by_category"].get(name, {}).get("n", 0) for s in answered)
                    / max(1, sum(s["by_category"].get(name, {}).get("n", 0) for s in answered)), 4),
                "evidence_recall@k": round(
                    sum(s["by_category"].get(name, {}).get("evidence_recall@k", 0.0)
                        * s["by_category"].get(name, {}).get("n", 0) for s in answered)
                    / max(1, sum(s["by_category"].get(name, {}).get("n", 0) for s in answered)), 4),
            }
            for name in sorted({n for s in answered for n in s["by_category"]})
        },
        "errors": sum(s["errors"] for s in summaries),
        "reader_model": summaries[0].get("reader_model") if summaries else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ARENA_DIR / "results"))
    ap.add_argument("--out", default=str(ARENA_DIR / "results" / "scores"))
    ap.add_argument("--systems", default="", help="comma separated; default all found")
    ap.add_argument("--convs", default="")
    ap.add_argument("--readers", type=int, default=4)
    ap.add_argument("--no-reader", action="store_true", help="skip LLM answers, retrieval metrics only")
    ap.add_argument("--limit", type=int, default=0,
                    help="score the first N questions per conversation (deterministic sample; 0 = all)")
    args = ap.parse_args()

    convo_map = {c.sample_id: c for c in locomo.load()}
    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    systems = [s for s in args.systems.split(",") if s] or sorted(
        p.name for p in results_dir.iterdir() if p.is_dir() and p.name != "scores"
    )
    wanted = [c for c in args.convs.split(",") if c]
    aggregate_summaries: dict[str, list[dict]] = {}

    for system in systems:
        aggregate_summaries[system] = []
        for path in sorted((results_dir / system).glob("*.jsonl")):
            conv = path.stem
            if wanted and conv not in wanted:
                continue
            if conv not in convo_map:
                continue
            cache = out_dir / f"{system}-{conv}-answers.jsonl"
            summary = score_run(path, convo_map[conv], cache, args.readers, not args.no_reader, args.limit)
            aggregate_summaries[system].append(summary)
            target = out_dir / f"{system}-{conv}.json"
            # The per-question details repeat the retrieved memories that are
            # already in results/<system>/<conv>.jsonl, and they dominate the
            # size of the committed artifacts; keep only the summary here.
            target.write_text(
                json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            print(
                f"{system:11s} {conv:8s} "
                f"F1={summary['qa_f1']} gold={summary['gold_in_answer']} "
                f"inCtx={summary['answer_in_context']} evRec={summary['evidence_recall@k']} "
                f"advRefusal={summary['adversarial_refusal_rate']} "
                f"p50={summary['latency_ms']['p50']}ms p95={summary['latency_ms']['p95']}ms "
                f"ingest={summary['ingest_seconds']}s"
            )

    if not aggregate_summaries:
        return
    combined = {
        system: {
            "aggregate": aggregate(summaries),
            "per_conversation": {
                s["conv"]: {k: s.get(k) for k in CORE_KEYS} for s in summaries
            },
        }
        for system, summaries in aggregate_summaries.items() if summaries
    }
    (out_dir / "summary.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    for system, data in combined.items():
        a = data["aggregate"]
        print(
            f"{system:11s} ALL      questions={a['questions']} "
            f"inCtx={a['answer_in_context']} evRec={a['evidence_recall@k']} "
            f"F1={a['qa_f1']} ctxTok={a['context_tokens_median']} "
            f"p50={a['latency_ms']['p50']}ms p95={a['latency_ms']['p95']}ms "
            f"ingest={a['ingest_seconds']}s"
        )


if __name__ == "__main__":
    main()
