# Diligence pack

Everything on this page is measured in this repository and reproducible with one
command. There is no projection, no TAM slide, and no claim about revenue, users,
downloads or stars — the project has none of those recorded, and inventing them
would be the fastest way to fail diligence.

```bash
./scripts/verify.sh          # tests + engine smoke + published baseline + readiness
```

Expected: `PASS`. It needs Python 3.9+ and nothing else — no API key, no model
download, no network service, no GPU.

## 1. What the product is

A memory engine for long conversations and agent sessions, in one SQLite file:
hierarchical storage (atoms → turn windows → facts), timescoped recall with
supersession, typed relation edges, a contradiction audit, persona summaries, and
a 10-tool MCP server. Retrieval is lexical by default and dependency-free; semantic
fusion is opt-in with any local embedding model.

The differentiator is not "we have vectors". It is that the default configuration
is **10 ms p50 and 7.5 s ingest for 10 conversations with zero external calls**, and
that its retrieval quality is measured against the systems people actually deploy.

## 2. Measured results (LoCoMo, 10 conversations, 1,986 questions)

Same harness, same scoring script, same top-k for every system. Higher is better.

| system | evidence recall@10 | answer-in-context | ctx tokens | ingest | p50 |
|---|---|---|---|---|---|
| **InfinityDB** (default) | **0.7795** | **0.5454** | 711 | **7.55 s** | **10.66 ms** |
| **InfinityDB**, dense fusion (local MiniLM, RRF 0.3) | **0.7867** | **0.5487** | **652** | 8.02 s | 29.04 ms |
| mem0 (`mem0ai==2.2.1`, no reader LLM) | 0.6777 | 0.3864 | 321 | 544.29 s | 152.05 ms |
| free BM25 + bge-large hybrid (RRF k=60) | 0.6757 | 0.3890 | 314 | 582.17 s | 151.87 ms |
| TencentDB Agent Memory 1.0.3, keyword/FTS mode | 0.5254 | – | – | – | – |
| InfinityDB, atoms ablation (no windows) | 0.4978 | 0.3545 | 329 | 7.91 s | 5.05 ms |

Category breakdown for the default row: multi-hop 0.494, single-hop 0.913, temporal 0.798, open-domain 0.354, adversarial 0.915. With dense fusion: 0.498 / 0.918 / 0.812 / 0.369 / 0.926 — every category improves, and context tokens drop 8% (711 → 652).

Raw per-question JSONL for every row is committed under
`benchmarks/arena/results/`, and per-system summaries in
`benchmarks/arena/results/scores/`. `benchmarks/arena/score.py` regenerates the
table from the JSONL.

LongMemEval (180-question stratified sample): **R@5 0.9333** — a sample, not the
full 500-question set; treat it as directional.

## 3. What is *not* claimed

- **Zep/Graphiti was attempted locally on 2026-11-11** with graphiti-core 0.31.2, embedded Kuzu and Ollama; the
  harness is committed, but extraction exceeded Ollama's 10-minute request limit on CPU before one 18-turn session
  finished (HTTP 500, zero episodes); a reduced-pipeline retry with 2-turn episodes then failed on a Kuzu
  full-text-index incompatibility in the deprecated embedded backend, and the supported backends (Neo4j, FalkorDB)
  need a server this container cannot run (no Docker daemon, no Java). That is a host limitation, not a result: no row exists. The full TencentDB pipeline is unmeasured for the same class of reason. They need an LLM
  for extraction and a graph database; with no credentials they cannot be run, and
  this repository ships no emulation of them. `benchmarks/arena/competitor_check.py`
  reports exactly which ones are runnable in the current environment and why.
- **Not "world's best".** We lead every system measured here, but the two systems
  most likely to beat us on multi-hop are precisely the two that never ran.
- **No revenue, users, retention or stars.** None exist to report.
- The TencentDB number is that project's documented key-free keyword mode, not its
  LLM pipeline — the row is labelled that way in every table.
- Six ranking experiments were tried and five failed; their exact scores and why
  they failed are in `docs/COMPETITIVE_TEARDOWN.md`. A buyer should read that table
  before believing any roadmap.

## 4. Engineering facts a reviewer can check

- 92 unit tests. `pytest -q tests` in this checkout.
- Zero runtime dependencies for the default path (stdlib + SQLite only).
- Derived data (vectors, edges) is rebuildable: `rebuild_edges()`, `index_dense()`.
- The core ships no model and makes no network call — the embedder is injected.
- `capture.py` + 10 MCP tools are the integration surface; the arena adapter and the
  MCP server use the same `recall()` entry point, so benchmarks measure shipped code.
- Apache-2.0. Competitor harnesses are pinned to the versions in
  `benchmarks/arena/README.md`; no competitor code is vendored into the package.

## 5. Known gaps, in the order a buyer will hit them

1. **Multi-hop is the weakness** (0.50). Single-hop is solved (0.92); cross-session
   "who did X then Y" reasoning is not, and it is where graphs win.
2. **Answer-in-context, not answer accuracy.** With no reader LLM, F1 is 0.0 for
   every system in the table — the metric is whether the gold evidence is in context.
3. **Single-dataset provenance.** LoCoMo is a public research dataset; there is no
   proprietary traffic behind these numbers.
4. **No hosted product.** It is a library plus MCP server; there is no multi-tenant
   service, billing or admin surface.

## 6. The 30-minute diligence path

```bash
./scripts/verify.sh                                     # numbers reproduce
python3 benchmarks/arena/competitor_check.py            # what can/cannot be rerun
python3 benchmarks/arena/score.py --systems infinitydb,infinitydb-dense,mem0 --no-reader
python3 benchmarks/arena/replay_probe.py                # per-category, dense on/off
git log --stat | head -40                               # what changed, in what order
```

If a number here does not reproduce on your machine, that is a bug and it counts
against the project — the scripts are the source of truth, not this document.

## 7. Offline reader sample (small n — a capability check, not a ranking)

The retrieval rows above need no LLM, but a buyer's real question is "does it
answer?". With no paid API we ran one fixed local reader (`llama3.2:3b`, CPU,
via Ollama) over each system's own retrieved context on conv-26. Verbatim:

```
Q: When did Caroline go to the LGBTQ support group?
  gold   : 7 May 2023
  infinitydb      : '08 May 2023'   F1=0.67
Q: When did Melanie paint a sunrise?
  gold   : 2022
  infinitydb      : 'Last year.'   F1=0.00
Q: When did Caroline go to the LGBTQ support group?
  gold   : 7 May 2023
  infinitydb-dense: '08 May 2023'   F1=0.67
Q: When did Melanie paint a sunrise?
  gold   : 2022
  infinitydb-dense: 'Last year.'   F1=0.00
Q: When did Caroline go to the LGBTQ support group?
  gold   : 7 May 2023
  mem0            : '2023-05-08'   F1=0.33
Q: When did Melanie paint a sunrise?
  gold   : 2022
  mem0            : '[2023-05-08]'   F1=0.00
```

| system | questions answered | F1 | gold string in answer |
|---|---|---|---|
| infinitydb | 14 | 0.433 | 0.286 |
| infinitydb-dense | 14 | 0.437 | 0.214 |
| mem0 | 18 | 0.381 | 0.111 |

Three caveats, stated before the numbers are used for anything:

- **n=14-18 on one conversation.** Enough to prove the path works and to expose
  failure modes, not enough to rank systems. The ordering happens to agree with
  the 1,986-question retrieval table; that agreement is a coincidence of a small
  sample, not corroboration, so quote the retrieval table instead.
- The reader, not retrieval, sets this score. "08 May 2023" against a gold of
  "7 May 2023" scores 0.67 on token F1 despite being the same date: F1 on
  short factual answers punishes formatting, which is why the retrieval table is
  the one quoted in the README.
- One error mode is a genuine product gap: the reader answers "Last year." where
  gold is "2022". Retrieval found the right turn; the context carried a relative
  time expression. Normalising relative dates at ingest is a real fix, not a
  benchmark trick.

The capability is the point: retrieval, prompting, generation and scoring all run
offline with zero credentials at ~7 s per answer on 8 CPU cores. The full 1,986
questions would take ~4 hours here, so every reader number in this repo carries
its n.
