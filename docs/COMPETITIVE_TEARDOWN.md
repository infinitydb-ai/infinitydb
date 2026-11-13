# Competitive teardown: how other memory engines are built

This is a code-level read of the memory engines InfinityDB competes with. It is
written to answer one question honestly: *what do they actually do, and which of
it can we match without an API key?*

**Method.** The published wheels were downloaded with `pip download --no-deps`
and unzipped; everything below cites the shipped module and line, not a blog
post. Nothing was run: Zep/Graphiti, Letta, Memobase and Cognee all need an LLM
provider, an embedding endpoint, or a graph/vector server, and this environment
has no keys. **No competitor is emulated and no head-to-head score is claimed
for them.** Only mem0 (no-LLM mode) and a free BM25+dense baseline were actually
measured; those numbers live in
[`benchmarks/arena`](../benchmarks/arena/README.md).

| Package | Version read | Licence | What it is |
| --- | --- | --- | --- |
| `mem0ai` | 2.2.1 | Apache-2.0 | Memory layer over a vector store; LLM writes facts |
| `graphiti-core` | 0.31.2 | Apache-2.0 | Zep's open-source temporal knowledge graph |
| `letta` | 0.34.12 | Apache-2.0 | Agent runtime with self-editing memory blocks |
| `memobase` | 0.0.27 | Apache-2.0 | User-profile memory service client |
| `cognee` | 1.6.7 | Apache-2.0 | ECL pipeline: graph + vector + summaries |
| `mempalace` | shipped repo + docs | MIT | Local-first verbatim store, palace index, MCP server |
| `TencentDB-Agent-Memory` | shipped repo + docs | MIT | L0–L3 layered memory, team memory, context compression |

Reading shipped source for architecture is fair use of a permissive licence; no
code was copied into this repository.

---

## 1. mem0 2.2.1 — LLM writes, vector reads

**Write path** (`mem0/memory/main.py`, `_add_to_vector_store`, l.881). With
`infer=True` (the default) a turn costs *two* model calls:

1. extraction — `ADDITIVE_EXTRACTION_PROMPT` (`mem0/configs/prompts.py`, l.470)
   turns the messages into self-contained facts. The prompt is explicit about
   the two things that matter: it resolves every relative date against the
   **observation date** ("User went to Paris the week of May 15, 2023"), and it
   deduplicates against "recently extracted memories" (up to 20) plus existing
   memories.
2. linking — new facts come back with `linked_memory_ids` pointing at related
   existing memories. This replaced the older destructive
   `ADD/UPDATE/DELETE/NONE` flow (`get_update_memory_messages`, l.406), which
   rewrote memory in place; v2.2.1 is **additive** with links.

**Read path** (`search`, l.1393): one embedding of the query, a filtered vector
search (`top_k=20` by default, `threshold=0.1`), optional `rerank`, and
`show_expired`/`reference_date` for time-scoped reads.

**What is genuinely good here:** extraction quality. An LLM writing "User
started a nursing job at the clinic on 8 May 2023" beats any rule engine, and
date grounding is what makes temporal questions answerable later.

**What it costs:** two model calls per turn. On LoCoMo that is 544 s of ingest
for ten conversations with `infer=False`; with inference it is far more, and it
is the reason mem0's published numbers are not comparable to ours.

## 2. Graphiti 0.31.2 (Zep) — bi-temporal graph

**Write path** (`graphiti.py`, `add_episode` l.1334): an LLM extracts entities
and edges from an episode, then
`utils/maintenance/edge_operations.py::resolve_extracted_edges` (l.424) decides,
per candidate, whether it duplicates, extends or **contradicts** an existing
edge. A contradiction does not delete: the old edge gets `invalid_at` set to the
new edge's `valid_at` (l.727–741), so the graph keeps both facts with a validity
interval. That is real bi-temporality — event time and ingestion time — and it
is the feature most often quoted in Zep's marketing.

**Read path** (`search/search_config.py`): three search methods per entity type —
`cosine_similarity`, `bm25`, `bfs` (graph expansion, `MAX_SEARCH_DEPTH = 3`) —
combined with one of five rerankers: `rrf`, `node_distance`, `episode_mentions`,
`mmr`, `cross_encoder`. Defaults: `DEFAULT_MIN_SCORE = 0.6`,
`DEFAULT_MMR_LAMBDA = 0.5`, cross-encoder `BAAI/bge-reranker-v2-m3`
(`cross_encoder/bge_reranker_client.py`, l.36).

Two details worth stealing conceptually:

- Their MMR (`search/search_utils.py`, l.1912) is **not** the greedy version in
  the original paper. It scores each candidate once as
  `λ·sim(q,d) + (λ−1)·max_i sim(d, d_i)` using a full similarity matrix. Cheap,
  but it only reorders; it cannot change *which* candidates were fetched.
- `episode_mentions` reranks by how many source episodes mention a node — graph
  evidence as a popularity prior.

**Why we cannot claim anything against it:** every one of these stages needs an
LLM at write time and an embedding model at read time. Reading the code tells us
what it does, not how well it does it.

## 3. Letta 0.34.12 — memory as an editable context

Letta (MemGPT's successor) splits memory by *who edits it*: in-context "core
memory blocks" the agent rewrites with tools, an archival store searched by
embedding, and recall over raw message history. The shipped wheel is a compiled
distribution, so this section is its published architecture rather than a line
audit. The idea is the important part: a small, always-present block that the
model curates itself, plus a large searchable store behind it. It optimises for
agent control of its own context, not for retrieval recall on a fixed corpus.

## 4. Memobase 0.0.27 — the profile, not the transcript

`memobase/core/user.py` shows the data model: a user has `UserProfile` rows with
`topic`, `sub_topic` and `content`, built from `ProfileDelta` objects produced
during a "flush". Retrieval returns that compact profile instead of passages.
This is the most token-frugal design in the group — a paragraph of distilled
facts — and it is the right shape for "who is this user" questions. It is the
wrong shape for "what exactly did she say about the flat in April", which is
what an evidence-recall benchmark measures.

## 5. Cognee 1.6.7 — ECL and graph-context

Cognee's own naming is the teardown: Extract, Cognify, Load. Chunks are
summarised, entities/edges extracted, and retrieval
(`modules/retrieval/graph_completion_retriever.py`, `get_context_from_objects`
l.314) renders retrieved triplets into a textual context block, with separate
retrievers for summaries, Cypher and scoped chunk search. The lesson is
representation: the context handed to a model is *rendered from a graph*, not
concatenated passages. It is also the most LLM-dependent pipeline of the five.

## 6. MemPalace — verbatim storage, structured scoping

MemPalace is the closest competitor to this repository's premise: local-first,
no API key, no LLM deciding what to keep. Its published claim is **96.6% R@5 on
LongMemEval with zero API calls**.

The design is the opposite of mem0's. Nothing is summarised or extracted:
conversations are stored verbatim in ChromaDB and retrieved by semantic search.
The differentiator is the *index*: conversations are filed into wings (person or
project), rooms (topic or session), halls (memory type) and finally drawers
(verbatim text), plus a temporal entity graph in SQLite with validity windows,
and a 36–47 tool MCP server with capture hooks and per-agent "diaries".

Their own reported ablation is the interesting part: search scoped by wing+room
scores 94.8% against 70.8% unscoped, i.e. **+34% from filters, not from a better
ranker**. That is the same effect we measured independently on LoCoMo: pooling
ten conversations into one store costs 4.7 points of recall, and `session_id`
recovers all of it. Two systems with different code arriving at the same
conclusion is the strongest evidence either of them has on this axis.

Where they are ahead: distribution (Claude Code plugin, hooks, 59k stars), the
breadth of the MCP surface, and the fact that their published benchmark is the
one people quote.

Where we are ahead: recall latency (10.7 ms p50 in-process vs a ChromaDB +
embedding round trip), zero dependencies (they need ChromaDB, and embeddings for
semantic search), deterministic provenance and point-in-time reads, and
equal-context comparison against a real product rather than a single number.

Our LongMemEval run (R@5 0.933 on a stratified 180-question sample) is deliberately
*not* claimed as a win: different sample size, same metric.

## 7. TencentDB Agent Memory — layers, compression, teams

Open-sourced under MIT (`TencentCloud/TencentDB-Agent-Memory`), and the most
direct architectural competitor to InfinityDB: an L0–L3 pyramid of
conversations → atomic memories → scene blocks → core persona, with recall that
pulls a bounded number of each layer.

Three things it does that we did not, at the time of writing:

1. **Context compression** for long sessions, with published results on agent
   benchmarks rather than retrieval-only datasets: WideSearch pass rate 33% →
   50% at −61% tokens, SWE-bench 58.4% → 64.2% at −33%, and PersonaMem accuracy
   48% → 76%. Those are end-to-end agent numbers with a reader model, which is a
   stronger (and more expensive) claim than any retrieval proxy here.
2. **Team memory**: `team_id` scoping with permissioned sharing, so one member's
   or agent's experience becomes reusable by the team. We have `user_id` and L4
   policies, not sharing-with-ACL.
3. **Distribution**: an OpenClaw plugin, a Hermes gateway and SDKs, so adoption
   is one command inside an existing agent platform.

It requires Node ≥ 22, an OpenAI-compatible model endpoint and (for teams) a
server; this environment has none of those, so it is read, not run. Its
extraction quality comes from an LLM, which is exactly the axis we cannot
measure without a key — hence the `extractor=` hook in this repository: the same
harness can run "InfinityDB + a model" whenever someone supplies one.

---

## What InfinityDB already matches

| Capability | Graphiti | mem0 | InfinityDB |
| --- | --- | --- | --- |
| No LLM required at write or read | no | only `infer=False` | **yes** |
| Temporal validity (`as_of` point-in-time reads) | edges, bi-temporal | `reference_date`/expiry | atoms + `as_of`, event `ts` |
| Automatic invalidation of outdated facts | LLM resolution | LLM update | rules for single-valued facts, `supersede_fact()` otherwise |
| Lexical retrieval | bm25 over edges | no | porter-stemmed FTS5 over turn windows |
| Dense retrieval | yes | yes | no (by design: no model in the core) |
| Graph traversal in retrieval | bfs depth 3 | no | causal edges only, not traversal |
| Reranking | 5 options | LLM/optional | optional cross-encoder plug-in (measured, off) |
| Query expansion | no | LLM extraction | pseudo-relevance feedback, `expansion=0.15` |
| Scoped retrieval (session / entity filters) | bfs by entity type | filters | `session_id`, `entities`, in-scope IDF |
| Token budget for the prompt | no | no | `budget_tokens` |
| Provenance / audit trail | no | history table | SHA-256 per atom, `get_proof()` |
| Typed relation graph | traversal, LLM-built | bfs, LLM-built | edges written at ingest, `relations()` |
| Contradiction handling | LLM detects | LLM detects | rule-based supersession + audit trail (`contradictions()`) |
| Agent transcript capture | no | no | `python -m infinitydb.capture` |
| MCP tool surface | no | no | 10 tools (recall, add, persona, correct, forget, policy, proof, graph, contradictions, stats) |
| Third-party dependencies | many | many | **zero** |

## Gap list, in priority order

**Closed since this teardown was written:** scoped retrieval (`session_id`,
`entities`, with corpus statistics computed inside the scope — it recovers the
entire contamination loss and is measured by `arena/scoped_probe.py`), the
`extractor=` hook for model-written facts, a typed relation graph with a
contradiction audit, a ten-tool MCP server, and
incremental transcript capture. What is left:

1. **LLM-grade extraction (blocked).** The single biggest quality difference.
   Without keys we cannot run it, and faking it would repeat the emulator
   mistake this repo already retracted. The mitigation is honest: our
   deterministic extraction is a floor, and the retrieval layer is strong enough
   that it still leads on the one competitor we can run.
2. **Graph traversal at read time (built, gate failed).** Typed edges are now
   written at ingest and `relations(hops=2)` walks them, including inbound hops
   through a shared object. Feeding those reached terms into recall was measured
   and rejected: multi-hop evidence recall fell 0.4939 → 0.4484 and single-hop
   0.9128 → 0.8845, even at half query weight, because OR-ing more terms into a
   BM25 query imports distractors faster than it finds the bridge. The graph
   therefore ships as an explicit query surface (`relations`, `entity_graph`),
   not as a ranking hint. A *compositional* second pass was then built and also
   failed: asking the bridge value as its own query and giving it the tail slots
   scored 0.4863 multi-hop at one slot and 0.4754 at two (single-hop 0.9110 /
   0.9075). Inspecting the bridges showed why — the deterministic patterns
   extract phrases like "all the yellow leaves" for a question about where
   someone works, so the second pass spends real evidence slots on noise. The
   decision rule that separates a bridge from a coincidence is exactly the part
   a model does. That is gate 2, and it is why multi-hop stays open.
3. **Free-text contradiction detection (partly closed).** Same subject, same
   typed predicate and a different object now supersedes the older fact, as does
   opposite polarity of the same object; `contradictions()` returns the audit
   trail and `as_of` keeps the history. First-person prose now works both ways ("I moved
   to Lisbon" vs "I live in Madrid" both classify as `location`, so the newer
   supersedes the older and `contradictions()` records it), and so does
   third-person phrasing about the store's subject ("User relocated to Berlin"
   vs "User is based in Munich"). What is still missing is pronoun coreference
   across turns ("she moved to Lisbon" in a two-person transcript) and a
   subject-less statement, both of which need either a parser or a model.
4. **Profile compaction (shipped, unclaimed).** `persona_summary()` distils the
   resolved persona into one paragraph. It is a summary of already-resolved
   state, so it cannot invent a fact, but no retrieval-quality number is claimed
   for it: the earlier selection-bias problem (only facts that matched a pattern
   are in the persona) still applies.

## Experiment log (what we tried because of this teardown)

All numbers: 10 LoCoMo conversations, 1,986 questions, `top_k=10`, no reader.

| Change | Idea from | evidence_recall@10 | answer_in_context | Verdict |
| --- | --- | --- | --- | --- |
| baseline (turn windows, stemming, merge) | — | 0.7632 | 0.5253 | — |
| query expansion 0.15 (PRF) | mem0's paraphrase-robust extraction | 0.7781 | 0.5454 | **kept**, default |
| base-verb / past-tense token fix ("move" vs "moved") | found while testing capture | **0.7795** | **0.5454** | **kept** (a real bug: the candidate scored zero and was dropped) |
| dense fusion: local MiniLM embeddings over turn windows, RRF with lexical at weight 0.3 | Graphiti hybrid retrieval (vector + full-text + graph in one call) | **0.7867** | **0.5487** | **kept**, opt-in (`dense=`): +0.9% relative recall, 8% fewer context tokens, 2.7× p50 latency. Weight is the knob — 1.0 regressed to 0.7768 |
| session/entity scoping + in-scope IDF | MemPalace wing/room filters | 0.7799 (pooled) vs 0.7331 unscoped | 0.5450 | **kept**, measured by `scoped_probe.py` |
| candidate hook for `extractor=` | mem0/TencentDB LLM extraction | n/a (needs a model) | n/a | **shipped as an opt-in callable**, not measured |
| MMR diversity reranking | Graphiti `mmr` reranker | unchanged | unchanged | reverted (chunks do not overlap; Jaccard ≈ 0) |
| cross-encoder `ms-marco-MiniLM-L-6` | Graphiti `cross_encoder` | 0.7852 | 0.5500 | shipped as `rerank=` plug-in, **off** (13× latency for <1%) |
| date-window boost for "last week"-style queries | mem0 date grounding | 0.7781 | 0.5448 | reverted (no measurable effect on this dataset) |
| two-hop entity bridging (`hops=2`: rare terms of the top passages re-searched, indirect hits in reserved tail slots) | multi-hop and open-domain questions | 0.4939 → 0.4850 | — | reverted: the bridge pool's own evidence recall was 0.22, so it displaced working context instead of adding to it |
| typed-edge query expansion (`relations()` terms OR-ed into the BM25 query at 0.5 weight) | Graphiti entity edges | 0.4484 (multi-hop) / 0.8845 (single-hop) | — | reverted: reaching the bridge entity also imports its distractors; graph kept as an explicit query surface instead |
| compositional second pass (bridge value re-queried, tail slots replaced) | Graphiti/Letta multi-hop traversal | 0.4863 (1 slot) / 0.4754 (2 slots) | — | reverted: deterministic bridges are noisy ("all the yellow leaves" for "where does Caroline work"), so the second pass costs more evidence than it finds |
| query-aspect coverage weighting (multiply BM25 by matched-aspect ratio, γ=0.5–1.0) | multi-hop questions that name several aspects | 0.4831 | — | reverted: +0.002 overall came from single-hop and adversarial only; multi-hop and open-domain both fell, so it is a trade, not a gain |

The PRF sweep was 0.1 → 0.7763/0.5402, **0.15 → 0.7781/0.5454**, 0.2 →
0.7745/0.5409, 0.3 → 0.7623/0.5299.

## What this document does not say

It does not say InfinityDB is the best memory engine. It says: on the one public
retrieval benchmark we can run offline, with every raw row committed, it beats
the two no-LLM systems we could actually execute; and the systems we did not
execute are better at the things that need a model, which is why we did not
pretend to beat them.


## Verified competitor access status (November 2026)

Checked against the vendors' own pages and repositories, because older guides
are still widely circulated and are wrong now:

- **Zep Community Edition was retired in April 2025.** Zep's open-source effort
  moved to Graphiti (Apache-2.0). Graphiti is a framework: it needs Neo4j,
  FalkorDB or Neptune, plus an LLM for extraction and an embedder. Self-hosting
  "Zep" today means assembling that yourself; the managed product is Zep Cloud.
  Source: <https://blog.getzep.com/announcing-a-new-direction-for-zeps-open-source-strategy/>
- **TencentDB Agent Memory is not TencentDB for MySQL.** It is a Node memory
  plugin (Node >= 22) on a local SQLite + sqlite-vec backend with BM25+vector
  RRF recall, requiring an OpenAI-compatible LLM key. Its published numbers are
  agent-task deltas (WideSearch, SWE-bench, PersonaMem), not LoCoMo recall.
  Source: <https://github.com/TencentCloud/TencentDB-Agent-Memory>
- Grapiti's own retrieval is vector + full-text + graph traversal in one call,
  which is why our multi-hop gap (0.494) is the honest place to expect it to
  win — that gap is stated above rather than hidden.

Setup steps, cost estimates and the required metric set live in
[`COMPETITOR_BENCHMARK_PLAYBOOK.md`](COMPETITOR_BENCHMARK_PLAYBOOK.md).
`benchmarks/arena/competitor_check.py` reports which of them is runnable here.

### TencentDB Agent Memory, measured (November 2026)

Their retriever does run without a key, so it was measured rather than described.
Version 1.0.3, their own `executeMemorySearch` in the documented
`recall.strategy=keyword` (FTS5/BM25) mode, one L1 record per verbatim turn
written through their store API, same 10 LoCoMo conversations, same 1,977
questions, same scorer:

| | evidence recall@10 | multi-hop | single-hop | temporal |
| :--- | :--- | :--- | :--- | :--- |
| TencentDB Agent Memory 1.0.3, keyword mode | **0.5254** | 0.189 | 0.593 | 0.598 |
| InfinityDB (default) | **0.7795** | 0.494 | 0.913 | 0.798 |
| InfinityDB, atom-only ablation (no turn windows) | 0.4978 | — | — | — |

Two honest readings. First, their turn-level keyword retriever beats our
atom-only ablation (0.5254 vs 0.4978) — the difference is that their records are
whole turns while our ablation scores fragments, which is exactly why the turn
window exists. Second, against the shipped engine they are 33% behind on
evidence recall, and the gap is widest on multi-hop (0.189 vs 0.494) and
single-hop (0.593 vs 0.913).

What is **not** measured: their LLM extraction and their embedding/vector branch,
which is where their headline PersonaMem/WideSearch claims come from. This row
must never be presented as "TencentDB Agent Memory vs InfinityDB" without the
mode qualifier — it is their keyword retriever against our default engine.

Reproduce: `benchmarks/arena/systems/tencentdb_keyword/` (npm install, then the
harness per conversation), summary in
`benchmarks/arena/results/scores/tencentdb_keyword.json`.

Dense fusion weight sweep (same 1,977-question replay, MiniLM vectors, fusion
weight is the only variable — no re-ingest needed):

| weight | evidence recall@10 |
|---|---|
| 0.2 | 0.8158 |
| **0.3** | **0.8182** (shipped) |
| 0.5 | 0.8097 |
| 0.7 | 0.7839 |

The optimum is broad on the low side and falls off fast above 0.5, which is what
reciprocal rank fusion predicts: once dense outranks lexical, exact-term matches
(a name, a date, a code) get buried. 0.3 is shipped; anyone re-tuning has the
curve.
