# Self-improvement: what can be built, and what cannot be claimed

This document exists because "self-improving superintelligent memory" is a phrase
that describes a research programme, not a feature, and because this repository
has already retracted one claim it could not support
([audit](BENCHMARK_INTEGRITY_AUDIT.md)). Every phase below has a **gate**: a
number that must move, or the phase is published as a negative result and the
code does not ship on by default.

## What "self-improving" can honestly mean here

Three mechanisms are real and implementable without a foundation model:

1. **Learning a retrieval prior from outcomes.** A caller observes whether a
   recalled memory was used. That is a label the engine cannot compute itself.
2. **Learning which bridge is real** (multi-hop), which is the one place the
   teardown concluded a model is required.
3. **Consolidating memory over time** — merging, expiring and rewriting facts
   as evidence accumulates, measured against a fixed token budget.

What it cannot mean: the engine rewriting its own code, choosing its own
objectives, or improving without measurement. Anything in that list is a
different project with a different safety story, and calling it "superintelligent"
would be marketing.

## Phase 0 — feedback substrate (shipped, opt-in, unmeasured)

Shipped in this repository:

```python
db.record_feedback("how do deploys run", atom_id, user_id="u", relevant=True)
db.feedback_summary("u")          # events, relevant, irrelevant, precision
db.learned_term_weights("u")      # inspectable term prior, needs 2 net votes
db.learned_atom_weights("u")      # direct per-memory evidence, 1 vote is enough
db.recall(q, user_id="u", learned_boost=1.0)
db.clear_feedback("u")            # drops the signal, keeps every memory
```

Design constraints, each of which is a test in `tests/test_learning.py`:

- **Append-only labels.** Feedback never edits a memory; a wrong label is offset
  by later ones and the whole signal is removable.
- **Bounded influence.** A prior is capped at `+3.0` on the term score and added
  per matched atom, so a well-liked word cannot outrank a fact that answers the
  question.
- **Cannot rescue unrelated facts.** A memory sharing no content term with the
  query stays unretrievable no matter how much feedback it has.
- **Off by default.** `learned_boost=0` is the default and the arena never sets
  it, so the published baseline is unchanged at 0.8101 (`dense=off n=1977`,
  verified by `./scripts/verify.sh`).

**Gate 0 (open):** there is no retrieval-quality number for this loop yet,
because there is no real feedback corpus. It is a mechanism with tests, and the
changelog says exactly that. Claiming a gain before Phase 1 runs would repeat the
mistake this repository already paid for.

## Phase 1 — offline policy fit, with an honest label simulation

**Question:** does feedback-driven reweighting beat the fixed policy?

**Method:** split LoCoMo by conversation (never by question) into fit and
held-out halves. Simulate labels two ways and report both, because the first is
an upper bound and the second is what a real caller can supply:

- *Oracle labels* — retrieved evidence turned out to be gold evidence. This
  cannot be obtained without gold answers, so it is an upper bound only.
- *Heuristic labels* — answer-token overlap between the recalled memory and the
  agent's response, no gold answer used. This is what a deployment can produce.

Fit `learned_boost`, `min_support` and the term-prior cap on the fit half, then
report held-out recall.

**Gate 1 (as originally written, and wrong):** held-out `evidence_recall@10`
must improve by ≥0.01 absolute. Running the experiment exposed the flaw: a
re-ranker acting on a fixed candidate list **cannot change recall@10 at all**, by
construction — it is the same ten memories in a different order. Both halves
measured @10 identically, exactly as the set argument predicts. The gate was
corrected after seeing the data, which is stated here rather than quietly
edited: the metric a re-ranker can move is `recall@1` and `recall@3`.

**Corrected gate 1:** held-out `recall@1` or `recall@3` improves, with no metric
down; oracle labels alone do not clear it.

### Result, measured 2026-11-13 — **gate failed, negative result**

`benchmarks/arena/feedback_probe.py`, `infinitydb` rows, hyperparameters chosen
on the fit half only.

| split | labels | fit | recall@1 | recall@3 | recall@10 |
|---|---|---|---|---|---|
| by conversation | oracle | 0.6075 | **0.5971** (−0.0104) | **0.7373** (−0.0187) | 0.8546 (=) |
| by conversation | answer | 0.6075 | 0.6075 (=) | 0.7560 (=) | 0.8546 (=) |
| within conversation | oracle | 0.6930 | **0.6604** (−0.0326) | **0.8346** (−0.0075) | 0.9198 (=) |
| within conversation | answer | 0.6930 | 0.6930 (=) | 0.8434 (+0.0013) | 0.9198 (=) |

Recall@10 is identical in every row, confirming the set-invariance argument.
These are subset numbers: questions that carry annotated evidence, split by
conversation or by position, so they are **not** comparable to the published
0.8101 overall figure.

The learned prior made the fit half look better each time and then degraded
held-out performance — textbook overfitting with a clear mechanism: the term
prior memorises conversation-specific vocabulary. In one conversation's
questions the names and places dominate the statistics, and those terms are
useless in another conversation. Oracle labels hurt more than answer labels
because a memory that contains evidence is not necessarily the memory a ranker
should put first, so the labels are both noisy and strongly correlated with the
current ranking.

**Decision:** `learned_boost` stays opt-in and is not used anywhere in the
default path or in any published number. Phase 2 (the learned bridge selector)
remains the only change with a plausible path past the multi-hop 0.4939 number.

**What would be tried next, if anyone continues this:** replace the pointwise
prior with a learned pairwise ranker over features that already exist (BM25
score, recency, category agreement, entity overlap, graph adjacency), trained on
list-level feedback rather than per-memory labels, and evaluated with this exact
protocol. A pointwise term prior is the weakest possible instance of the idea;
its failure does not close the idea, but it does rule out shipping it.

### Phase 1b — pairwise ranker, measured 2026-11-13 — **positive but not gate-clearing**

`benchmarks/arena/pairwise_probe.py`: a perceptron over eight interpretable
features (bias, own rank, question/memory stem overlap, question-type
agreement for dates, places and names, recency within the conversation, length),
trained on candidate **lists** rather than per-memory labels, same splits and
label schemes as Phase 1, three folds.

Held-out change versus the existing ranking:

| fold | labels | recall@1 | recall@3 | recall@10 |
|---|---|---|---|---|
| conversations A→B | oracle | +0.0218 | +0.0135 | 0 |
| conversations A→B | answer | 0.0000 | +0.0145 | 0 |
| conversations B→A | oracle | +0.0286 | +0.0188 | 0 |
| conversations B→A | answer | +0.0069 | +0.0188 | 0 |
| within conversation | oracle | +0.0263 | +0.0050 | 0 |
| within conversation | answer | −0.0025 | +0.0138 | 0 |

**Verdict: the corrected gate is not cleared, though the direction is
consistent.** `recall@3` improves in all six configurations under both label
schemes; `recall@1` is positive in five and −0.0025 in the within-conversation
answer-label run, which fails the "no metric down" clause. The effect is small,
stable in sign, and bounded: **recall@10 cannot move** because this re-ranks a
fixed candidate list.

Two limits keep this out of the engine:

1. The labels still need gold data. A deployable version needs a reader or a
   user thumb, so the measured gain is an upper bound on what feedback could buy.
2. Most of the learned weight lands on lexical overlap and the candidate's own
   rank — the perceptron is largely rediscovering what the BM25 step already
   does, which is why the gain is a point or two rather than ten.

The honest reading: there is roughly **1.4–1.9 points of recall@3 headroom** in
re-ranking these candidates with cheap features, and none at all at k=10. That is
worth knowing and not worth shipping on its own. The next step is not more
features; it is a deployable label source, so the training signal matches what a
deployment can actually observe.

## Phase 2 — a learned bridge selector for multi-hop

**Question:** can multi-hop move past 0.4939?

The teardown records two deterministic failures (0.4484 with traversal terms,
0.4863 with a compositional second pass) and names the cause: deciding whether a
bridge is real is a classification problem, not a pattern-matching one.

**Method:** a small model, called only to answer *"is entity X related to
object Y in a way that answers this question"*, over the typed edges that
already exist. Not a generative rewrite, not an LLM in the retrieval path — one
bounded classifier with a cached verdict per (question, edge) pair.

**Gate 2:** multi-hop ≥ 0.55 at `top_k=10` with single-hop down ≤ 0.005, and
inference cost recorded per question. Below the gate, the model stays out of the
default path and the adapter ships behind a flag, if at all.

## Phase 3 — pronoun coreference without a model

**Question:** can "she moved to Lisbon" attach to the right person?

Deterministic approach only: track speakers per session, and treat a pronoun as
bound when a single same-session candidate of that gender exists, otherwise
abstain. Precision over recall — a wrong supersession deletes a true fact, which
is worse than missing one.

**Gate 3:** on a hand-labelled fixture set committed with the change,
precision ≥ 0.95 and abstention reported alongside accuracy. No LoCoMo claim is
expected, since LoCoMo is first-person.

## Phase 4 — consolidation under a token budget

**Question:** can memory shrink without losing answers?

Merge near-duplicate atoms, expire facts that were never recalled and never
confirmed, and re-check that `budget_tokens=320` recall does not fall.
**Gate 4:** ≤ 0.005 loss at `budget_tokens=320` (0.6958) while reducing stored
atoms by ≥15%.

## Rules that apply to every phase

1. **Opt-in.** No phase changes a default without clearing its gate.
2. **Measured or unclaimed.** If the number is not in an artifact, it does not
   go in the README, the release notes or a pitch.
3. **Negative results are published** with the same care as positive ones. The
   experiment log in [`COMPETITIVE_TEARDOWN.md`](COMPETITIVE_TEARDOWN.md) is the
   precedent: three measured-and-rejected ranking changes already cost nothing
   in credibility and bought a lot.
4. **Learning is data, not code.** Every learned artifact is a row you can read,
   export and delete. Nothing self-modifies the engine.
5. **Re-run the arena and `arena.py verify`** before quoting any number that a
   phase could influence.
