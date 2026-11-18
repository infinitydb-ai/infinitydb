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

### Phase 1c — a deployable label source, measured 2026-11-13 — **transfers about 40% of the gain**

The question phase 1b raised: if gold labels buy 1.4–1.9 points, how much of
that survives when the labels must be produced in production, with no gold
answers and no reader model?

**The label.** Agreement between two independent rankers, which is observable at
runtime and reads neither the answer nor the evidence ids:

- positive — a memory in the top 3 of *both* the existing ranking and an
  independent overlap/recency ranking,
- negative — in the bottom half of both,
- **abstain** otherwise, because a weak label should say nothing rather than guess.

Held-out, same three folds, same features:

| fold | deployable labels, recall@1 | recall@3 | gold labels, recall@3 (1b) |
|---|---|---|---|
| conversations A→B | 0.0000 | +0.0052 | +0.0145 |
| conversations B→A | 0.0000 | +0.0069 | +0.0188 |
| within conversation | 0.0000 | +0.0075 | +0.0138 |

**Deployable labels keep roughly 35–55% of the recall@3 gain and none of the
recall@1 gain.** The learned weights say why: `rank=+6.04`, an order of
magnitude above every other feature, against `rank=+0.68` under gold labels. The
agreement label is *partly a function of the base ranking* — "top 3 of both"
means "top 3 of the existing order" — so the model mostly learns "do not disturb
the ranking", plus a small recency/overlap correction. That circularity is the
ceiling of free supervision, and it is now measured rather than assumed.

**What this settles.** The binding constraint in this whole family is label
quality, not feature count or model capacity. Capturing the remaining ~1 point
requires a label a deployment cannot fake: a reader's answer, or a user's thumb
on a cited memory. It also re-confirms the outer bound: this family re-ranks a
fixed candidate list, so it can never move recall@10 or any published headline
number. Only candidate *generation* — Phase 2's bridge selector — can do that.

## Phase 2 — a learned bridge selector for multi-hop

**Question:** can multi-hop move past 0.4939?

The teardown records two deterministic failures (0.4484 with traversal terms,
0.4863 with a compositional second pass) and names the cause: deciding whether a
bridge is real is a classification problem, not a pattern-matching one.

**Method:** a small model, called only to answer *"is entity X related to
object Y in a way that answers this question"*, over the typed edges that
already exist. Not a generative rewrite, not an LLM in the retrieval path — one
bounded classifier with a cached verdict per (question, edge) pair.

### Phase 2 scaffold, measured 2026-11-13 — **the mechanism, not just the gate, may be wrong**

`infinitydb/bridge.py` defines the veto interface (`NullBridge`, `AllBridge`,
`PredicateBridge`); `benchmarks/arena/bridge_probe.py` measures it. The harness
first had to prove it measures the published quantity, and does: the null
selector reproduces **multi-hop 0.4939 on n=281** exactly, after two corrections
that were both my errors — a binary any-hit instead of the scorer's fractional
evidence recall (which read 0.7544), and missing date rendering (which is what
the adapter feeds a reader).

With the harness validated, scoping recall to the selected bridge objects:

| selector | multi-hop recall@10 | delta | questions with bridges |
|---|---|---|---|
| `null` (control) | 0.4939 | 0 | 0 |
| `all` (accept every edge — the upper bound) | **0.2468** | −0.2471 | 281 of 281 |
| `predicate` (aspect-matched, strict) | 0.4778 | −0.0161 | 1 of 31 on conv-26 |

Two findings, both negative and both useful:

1. **Accepting every bridge halves recall.** The upper bound of "use the bridge
   objects as an entity scope" is not a ceiling above the baseline — it is far
   below it. This reproduces the teardown's OR-the-terms failure at the object
   level (0.4484 before), and rules out the mechanism as a win, not just the
   deterministic gate.
2. **The strict gate fires almost never.** On conv-26 it fired once in 31
   questions; its apparent +0.032 there was a single question, which is why
   running all ten conversations mattered. Over the full set it is −0.0161.

**Consequence for Phase 2:** a learned selector must not merely be more accurate
than `PredicateBridge`; it must be accurate enough to approach the `null` result
from a mechanism that loses 0.25 when its input is noisy. The next attempt should
change *candidate generation* rather than filter bridges — e.g. retrieve the turn
that witnessed the edge and quote it, instead of scoping by the object string —
and this probe already provides the baseline and the controls to test that.

### Phase 2b — quote the witnessing turn, measured 2026-11-13 — **first positive multi-hop signal, at low coverage**

Instead of filtering by bridge object, append the turn that asserted the edge
(`l1_entity_edges.atom_id` → `l1_atoms.fact`), rendered with its date, and give
up that many recall slots. Same harness, same 281 multi-hop questions.

| arm | overall multi-hop | delta | fired subset | subset delta |
|---|---|---|---|---|
| scope / all | 0.2468 | −0.2471 | n=281 | — |
| scope / predicate | 0.4778 | −0.0161 | n=22 | — |
| witness / all | 0.4524 | −0.0415 | n=277 | −0.0421 |
| **witness / predicate (cap 2)** | **0.4984** | **+0.0045** | **n=22** | **+0.0568** |
| witness / predicate (cap 1) | 0.4966 | +0.0027 | n=22 | +0.0341 |

Finding the witness is a much better use of a bridge than scoping by its object:
quoting every witness costs 4 points instead of 25, and with the strict gate it is
the only arm in this entire document that lands above the baseline. On the 22
questions where the gate fires, the gain is **+0.0568** (0.4583 → 0.5152).

**Coverage is now the binding constraint, not accuracy.** The gate fires on 7.8%
of multi-hop questions. If the same +0.057 held at full coverage, multi-hop would
be ≈0.55 — the Phase 2 gate. With n=22 the confidence interval is wide, so this
is a signal to chase, not a result to quote. Two witnesses beat one, which is
consistent with the mechanism: one witness often quotes an unrelated assertion.

The next step is therefore to raise coverage without spending precision — better
entity linking into the question, aspect coverage for the multi-hop phrasings that
currently match nothing ("What did X research after…"), and second-hop witnesses.
That is what a learned selector should be asked to do: fire on many more questions
with the same precision, at which point its inference cost is justified.

### Phase 2c — raising coverage by naming the person, measured 2026-11-13 — **coverage bought, precision lost**

`SubjectBridge` accepts any edge whose **subject is named in the question**, plus
inbound edges as second-hop witnesses ("Melanie also works at the gym"), still
refusing objects the question already mentions. The hypothesis was that the
person, not the question aspect, is the relevance signal.

| selector | coverage (fired / 281) | overall | delta | fired-subset delta |
|---|---|---|---|---|
| witness / predicate (2b) | 22 (7.8%) | 0.4984 | **+0.0045** | **+0.0568** |
| witness / subject, cap 2 | 249 (89%) | 0.4685 | −0.0254 | −0.0287 |
| witness / subject, cap 3 | 249 (89%) | 0.4575 | −0.0364 | −0.0411 |

One bug was caught by a test rather than by the numbers: the first version of
`SubjectBridge` required an inbound edge's object to appear in the question,
which made second-hop witnesses unreachable — the shared node is exactly what
the question does not name. Fixing it raised coverage from 237 to 249 of 281 and
left the deltas essentially unchanged, which is itself informative: the extra
reach did not contain the missing signal.

**Coverage rose tenfold and the gain disappeared.** Naming a person is not enough
to know which of their sentences answers the question: their facts include
hobbies, meetings and passing remarks, and spending two evidence slots on them
costs more than the occasional bridge gains. Even on the 237 questions the
selector fires on, the arm is worse than the baseline.

This is a real narrowing of the search space, and it sharpens what Phase 2 must
be: the selector has to bind **question aspect to edge predicate**, not link
persons. That is exactly the classification the teardown named, and the two
cheap signals available without a model — person matching and predicate aspect —
now have measured upper bounds. A learned selector must beat both, not just fire
more often.

Still open, and untested here: expanding aspect vocabulary ("research",
"career", "activities") rather than expanding subject matching, which would keep
aspect precision while raising coverage. That is the cheapest next experiment.

### Phase 2d — wider aspect vocabulary, measured 2026-11-13 — **best arm so far, gate still missed**

`BroadPredicateBridge` keeps the strict gate's structure (an edge is used only
when the question asks for that kind of fact) but widens the aspect words:
`research`, `career`, `activities`, `hobby`, `neighbourhood`, `thesis` and so on.
The narrow gate is left untouched so every earlier row stays reproducible.

| selector | coverage | overall | delta | fired-subset delta |
|---|---|---|---|---|
| predicate, cap 2 (2b) | 22 (7.8%) | 0.4984 | +0.0045 | +0.0568 |
| broad, cap 2 | 49 (17.4%) | 0.4982 | +0.0043 | +0.0245 |
| **broad, cap 3** | **49 (17.4%)** | **0.5000** | **+0.0061** | **+0.0347** |

Coverage more than doubled and the per-question effect roughly halved. The new
questions do benefit (+0.0347 where the gate fires), so **precision held in sign
but not in size**: the original gate was, in hindsight, selecting unusually easy
questions rather than generalising.

The extrapolation that matters: applying +0.0347 to every multi-hop question
would give ≈0.529 — **still short of the 0.55 gate**. Full coverage at the
current effect size does not clear it, so the next step must raise the effect per
question, not only the number of questions. That is the learned selector's real
job, and it is why spending inference on it is now justified rather than assumed.

### Phase 2e — witness position, measured 2026-11-13 — **the metric is order-blind; promote at your peril**

Every arm so far appended witnesses *after* the retrieved head. `--position top`
puts them first instead. Both use the same witnesses, so @10 is identical by
construction — `score.evidence_hit` joins the context into one blob. The
informative part is the prefix.

| position | @1 all | @3 all | @10 all | @1 fired | @3 fired |
|---|---|---|---|---|---|
| baseline, no witnesses | 0.1750 | 0.3181 | 0.4939 | 0.1488 | 0.2747 |
| **witness bottom (2d)** | **0.1750** | **0.3181** | **0.5000** | **0.1488** | **0.2747** |
| witness top | 0.1517 | 0.2881 | 0.5000 | 0.0153 | 0.1027 |

Two findings:

1. **Pushing witnesses to the front is destructive.** On the questions that fire
   it costs 0.13 at @1 and 0.17 at @3: the witnessing turn is an intermediate
   hop's sentence ("Caroline works at the gym"), not the sentence that answers
   ("Tom joined the gym in May"). A production wiring must therefore append,
   never promote.
2. **@10 cannot see any of this.** Because `score.evidence_hit` is set-based, the
   same ten memories score identically in any order — which also means the
   +0.0061 in Phase 2d is an *inclusion* effect, not a ranking effect. Any future
   claim that bridges improve ranking must be made at @1/@3.

### Phase 2f — the loss was displacement, not bridge quality, measured 2026-11-18 — **best result so far, gate still open**

Hypothesis stated before the run: *broad candidate generation plus overlap-ranking of
witnesses lifts multi-hop @10 by ≥0.01.* The first half was wrong-edge; the run found
something better.

| arm | coverage | overall @10 | delta | fired-subset delta |
|---|---|---|---|---|
| broad gate, displaces (2d) | 17% | 0.5000 | +0.0061 | +0.0347 |
| subject gate, displaces (2c) | 89% | 0.4685 | −0.0254 | −0.0287 |
| subject + **overlap** ranking, displaces | 89% | 0.4553 | −0.0386 | −0.0436 |
| subject + **novelty** ranking, displaces | 89% | 0.4452 | −0.0487 | −0.0550 |
| subject, **additive**, cap 2 | 89% | 0.5037 | +0.0098 | +0.0110 |
| subject, additive, cap 3 | 89% | 0.5078 | +0.0139 | +0.0157 |
| subject, additive, cap 5 | 89% | 0.5136 | +0.0197 | +0.0222 |
| **subject, additive, cap 8** | **89%** | **0.5167** | **+0.0228** | **+0.0257** |
| broad, additive, cap 3 | 17% | 0.5017 | +0.0078 | +0.0449 |

**Two ranking signals were tested and rejected.** Overlap with the question is
*anti*-correlated with usefulness: a multi-hop question names the bridge entity, while
the answering sentence names an entity the question does not contain. Inverting the
signal (prefer novelty) was worse still. Read alone, that looks like "lexical signals
cannot rank witnesses".

**The follow-up isolates the real cause.** Every arm above spends one recall slot per
witness, so a 10-slot context loses a retrieved turn for every sentence it gains.
`--extra-budget` appends witnesses *after* the full top-10 instead. The same selector
that lost 2.5 points at 89% coverage then **gains** them: +0.0139 at cap 3, rising
monotonically to **+0.0228** at cap 8. So the previous negative results measured
displacement, not bridge quality, and the ranking signals are not yet fairly tested —
they were all measured with a confound that dominates them.

**Gate status: not cleared.** 0.5167 against 0.55 leaves 0.033, and the curve is
sublinear (+4 points per three extra witnesses between cap 5 and 8), so the next three
witnesses will not close it. The cost is also real: eight extra sentences on top of ten
retrieved turns is roughly +80% context, traded for +2.3 points.

Two things this changes:

1. **The mechanism to build is additive**, not filtering: bridges should extend the
   evidence budget, because their value is a sentence the retriever missed, and their
   damage is a sentence it had already found.
2. **Ranking has to be re-measured under the additive budget** before anything is
   concluded about lexical signals — with displacement removed, a good ranker has room
   to show up, and a bad one can only cost the extra slots.

Untested and stated as such: single-hop impact (the harness scores multi-hop only),
token cost measured with a real tokenizer, and whether the curve saturates before it
reaches the gate.

**Shipped as opt-in, not as a default.** `infinitydb.bridge.recall_with_bridges(db,
question, user_id=..., n=10, max_bridges=8)` returns the normal result with the
witnesses appended and labelled `source: "bridge"`, plus the chosen edges. It is a
library call, not a change to `recall()`: the gate is not cleared, so the default
path is untouched. The API is covered by `tests/test_bridge_additive.py`
(witnesses never replace a retrieved turn, the selector is respected, witnesses
deduplicate); the harness numbers above were produced by the probe, not by this
entry point, and the two should be reconciled before any release note quotes them.

### Phase 2i — assembly depth: the first change that beats plain retrieval at equal slots, measured 2026-11-18 — **positive**

Gate 2b had survived four mechanisms. The fifth attempt started as MMR diversity and
ended somewhere unexpected: **the diversity was irrelevant, and the ranking was
truncating its own chunk assembly.**

`_recall_turn_windows` merges adjacent turns into chunks, and only then returns `n`
chunks. Chunk assembly traversed `3 * n` ranked items — 30 when a caller asks for 10.
A chunk that owns one hop of a multi-hop question absorbs its neighbour, and that
neighbour usually sits *below* position 30, so the shallow traversal never reached it.
Separating the two numbers (`assembly_depth`) fixed a second problem with the same
cause: when assembly yields fewer than `n` chunks the engine runs `_recall_atoms` again
to top up the slots, so the shallow path was doing a *second full scoring pass*.

| LoCoMo, `top_k=10`, 1,986 questions | previous default | depth 240 | delta |
|---|---|---|---|
| evidence_recall@10 | 0.780 | **0.8069** | **+0.0269** |
| answer_in_context | 0.545 | **0.5584** | +0.0134 |
| multi-hop | 0.4939 | **0.5359** | **+0.0420** |
| single-hop | 0.9128 | 0.9326 | +0.0198 |
| temporal | 0.7984 | 0.8281 | +0.0297 |
| open-domain | 0.3557 | 0.4010 | +0.0452 |
| adversarial | 0.9148 | 0.9327 | +0.0179 |
| median context tokens | 711 | 963 | +35% |
| p50 query latency | 10.7 ms | 12.6 ms | +1.9 ms |

Every category improves; nothing regresses. Depth sweep on multi-hop: 60 → 0.5195,
120 → 0.5287, 240 → 0.5359, 1000 → 0.5458 — monotone but flattening, and the cost of
1000 is a longer assembly loop for no gain worth 0.01.

**The honest cost is context, not CPU.** Median returned context grows 711 → 963
tokens (+35%) because chunks carry their neighbours. At `k=10` that is the mechanism
working as designed — the second hop arrives inside an existing slot — but it is a
real increase in what the caller pays for, and it belongs next to the gain.

Two lessons this phase produced, both about measurement rather than retrieval:

1. **A parameter that changes nothing is a bug report.** λ=1.0 (pure relevance) beat
   the diversity weighting, which is impossible unless `recall(n=10)` and
   `recall(n=40)[:10]` disagree. Chasing that contradiction found the truncation.
2. **The measured headline and the mechanism must be re-derived together.** Changing
   a default is what makes a benchmark row stale, so the arena was re-run in full
   rather than patching the number by hand; `arena.py verify` reproduces 0.8069 from
   the committed rows.

Tables in `DILIGENCE.md`, `LEADERBOARD.md`, `COMPETITIVE_TEARDOWN.md` and
`BENCHMARK_INTEGRITY_AUDIT.md` quote the pre-2i default (0.7795–0.780). They are kept
as historical measurements of that configuration and are not silently rewritten.

### Phase 2h — query expansion at equal slots, measured 2026-11-18 — **negative, and the default PRF is load-bearing**

Gate 2b asks for a mechanism that beats plain retrieval *at the same slot count*.
Bridges failed it. The next candidate was cheaper and better motivated:
pseudo-relevance feedback. A multi-hop question names one hop and not the other, so
the top hits often contain the missing name; re-querying with it ranks the answer
instead of spending extra slots.

| arm (281 multi-hop questions, k=10) | @1 | @3 | @10 | vs baseline |
|---|---|---|---|---|
| baseline (engine PRF 0.15) | 0.1750 | 0.3181 | **0.4939** | — |
| + terms from top-3 hits | **0.1803** | 0.3007 | 0.4513 | −0.0426 |
| + terms from top-5 hits | **0.1803** | 0.3033 | 0.4539 | −0.0400 |

Expansion **raises precision at 1** (+0.0053, the best @1 in the programme) and
**lowers recall at 10**: the added terms re-rank the bridge sentence to the top
while pushing other needed evidence out. At equal slots that is a net loss, so
Gate 2b stays unmet — now by four mechanisms (bridges, ranking, additive budget,
expansion).

Three things this measured that are worth keeping:

1. **`recall(entities=…)` is an AND filter, not a boost.** The first arm passed the
   discovered names as `entities=` and produced an *exactly* identical 0.4939 on
   every metric. That is a no-op signature, not a negative result — the parameter
   drops any memory lacking one of the entities. Reading measured behaviour caught
   what reading the signature had not.
2. **The probe had a guard that silently disabled a new mode.** `if not bridges:
   continue` ran before the expansion branch, and the default selector produces no
   bridges, so every question short-circuited to the baseline. Two configs reported
   a pristine 0.0000 delta before that was found. A mode that cannot change a number
   must be made to prove it runs.
3. **The engine's built-in PRF is load-bearing.** Ablating `recall(expansion=…)` on
   the same questions:

   | expansion | @10 |
   |---|---|
   | 0.00 | 0.4755 (−0.0184) |
   | **0.15 (default)** | **0.4939** |
   | 0.30 | 0.4795 (−0.0144) |

   So the shipped default is the best of the three measured points, and the ~1.8
   points it contributes are real rather than decorative. Adding *more* expansion on
   top (the arms above) pushes it past that optimum.

The pattern across phases 2c–2h is now unambiguous: on this dataset, every
mechanism that changes *which* ten memories are returned loses to plain hybrid
retrieval at ten. The remaining honest options are a better ranking signal (costs a
model), a larger budget (costs context), or accepting the ceiling.

### Phase 2g — the equal-budget control, measured 2026-11-18 — **bridges lose to "retrieve more"**

Phase 2f gained +0.0228 by appending witnesses on top of `top_k`. That arm used 18
context slots, and nothing in the comparison held slots constant. This control does:
same questions, same 18 items, but plain retrieval instead of bridges.

| arm (249 questions where the selector fires) | slots | recall |
|---|---|---|
| bridges appended (10 retrieved + 8 witnesses) | 18 | 0.5164 |
| **plain retrieval, no bridges** | **18** | **0.5714** |
| delta | — | **−0.0550** |

**The entire Phase 2f gain was context, not bridges.** Given the same budget, the
retriever picks better material than a quoted witness every time. Nine arms of this
phase — gating, scoping, ranking, additive budgets — were all variations on a
mechanism that loses to the simplest possible alternative.

Two consequences, both uncomfortable and both worth writing down:

1. **The bridge programme is closed at equal cost.** Nothing here is shippable as a
   retrieval improvement. `recall_with_bridges` remains in the library as an opt-in
   inspection tool, with a note that its measured value is negative against an
   equal-slot baseline; it should not be recommended to anyone for recall quality.
2. **The gate itself needs a second form.** Plain retrieval at 18 slots scores
   0.5714, above the 0.55 gate — but the gate is defined at `top_k=10`, so that does
   not clear it, and quoting it would be comparing budgets. The gate is therefore
   restated as two conditions, both at matched budget:

   > **Gate 2a:** multi-hop ≥ 0.55 at `top_k=10`, single-hop down ≤ 0.005.
   > **Gate 2b:** any multi-hop mechanism must beat plain retrieval *at the same
   > slot count*, with the slot count reported. Currently: **not met by −0.0550**.

What this says about the remaining work: multi-hop is a *retrieval-budget* problem
before it is a graph problem. The cheapest honest path to better multi-hop numbers is
a larger `k` with its token cost stated — not a bridge. Any mechanism that wants to
justify itself now has a concrete bar: beat plain retrieval with the same slots.

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
