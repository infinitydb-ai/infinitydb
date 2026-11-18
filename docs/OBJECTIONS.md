# Complaints this project deserves

If you are evaluating a memory layer, your job is to disbelieve it. This page
lists the objections worth raising, what this project can answer, and — for five
of them — the fact that the answer is **it doesn't**.

Everything below is checkable with one command:

```bash
pip install infinitydb-ai
infinitydb-evidence            # runs this machine, prints PASS/FAIL and the known limits
infinitydb-evidence --with-arena  # also recomputes the published benchmark table
```

The command exits non-zero if any check fails, so it can be run in CI or during a
procurement review rather than being taken on trust.

---

## Objections with an answer

### "Your benchmark is self-reported."

Run `./scripts/verify.sh` and `python benchmarks/arena/arena.py verify`. Both
recompute the table from committed per-question rows and **fail** if a number does
not reproduce. Tampering is detectable by design: the arena caught a committed
`0.9` that recomputed to `0.7167` during development.

### "You'll lock me in."

Apache-2.0, zero runtime dependencies, SQLite on your disk. `infinitydb-migrate`
imports mem0 exports, JSONL and plain text; nothing is uploaded and no key is
required. Audit bundles are plain JSON with no InfinityDB service in the loop.

### "I can't prove what it remembered."

`infinitydb-audit export/verify` produces a bundle a third party verifies offline,
with an anchor digest for edit and deletion evidence. What it does *not* prove is
written in [`AUDIT.md`](AUDIT.md): history rewritten before the first export is
undetectable.

### "It's slow or it phones home."

`infinitydb-evidence` measures ingest and recall on your machine. The default path
makes no network call and calls no model — that is a test, not a claim
(`tests/test_demo.py` runs the demo with proxies pointed at a dead port).

### "Extraction quality will be poor without a model."

Correct, and it is a design choice: deterministic patterns only. The engine takes
an `extractor=` callable if you want model-written facts; that path is not measured
here and is not what the benchmark rows describe.

---

## Objections with no answer — the honest five

### 1. "Your multi-hop numbers lose to just retrieving more."

True. Bridge witnesses appended on top of `top_k` scored **0.5164**, while plain
retrieval with the **same 18 slots** scored **0.5714** on the same questions
(Phase 2g). The bridge mechanism does not earn its context. Nine experiments in
`SELF_IMPROVEMENT.md` reduce to that.

Consequence: multi-hop is a retrieval-budget problem before it is a graph problem,
and any mechanism now has to beat plain retrieval at equal slots. Currently nothing
here does.

### 2. "You lose to mem0 on open-domain questions."

Yes — that is in the published per-category table. On commonsense questions the
deterministic extractor has nothing to work with.

### 3. "Nobody has reproduced your results."

Nobody has. Every number was produced by this project, on one machine. The arena is
reproducible in principle and the rows are committed, which is not the same as an
independent replication.

### 4. "One dataset."

LoCoMo in full, plus a stratified LongMemEval sample. No third dataset, and no
held-out split used for tuning retrieval — the gate was written before the runs, but
the dataset was not held out.

### 5. "No encryption, no real multi-tenancy, no scale evidence."

All true. SQLite file permissions are the access control; `user_id` is a string, not
a security boundary; the largest measured store is 5,000 facts. This is a library
for a single application, not a hosted service, and treating it as one would be a
mistake.

---

## What would change our mind

Each of these is a measurable event, not an opinion:

| Claim | Test that would settle it |
|---|---|
| "multi-hop works" | beat 0.5714 at 18 slots, or 0.4939 at 10 slots, with the slot count reported |
| "it scales" | 1M atoms, concurrent writers, p95 recall under a stated bound |
| "it generalises" | a third dataset the repo has never been tuned against |
| "it's safe to host" | encryption at rest, per-tenant keys, auth — none of which exist |

Until then, the accurate description is: **strong deterministic single-hop recall, a
genuinely unusual audit path, first-class MCP support, and a multi-hop story that is
still negative.** If that is not what you need, the honest recommendation is not to
use this.
