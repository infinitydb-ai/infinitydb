# InfinityDB Changelog

## Unreleased
- fix(core): **a natural question returned nothing.** `search("work")` and `search("where does the user work?")` returned `[]` against a stored "works at the hospital in Leeds", on both 1.4.0 and 1.4.1. The atom index (`l1_atoms_fts`) is tokenised with `unicode61` and no stemmer, while the turn-window index is porter-stemmed, so the query term was matched as an exact token. FTS now receives stem variants plus a prefix atom (`"work"*`, only for terms of four characters or more), and the BM25 term counter compares stemmed tokens, so `work`/`works`/`working`/`worked` and `deploy`/`deployed`/`deploying` all reach the same fact. Absent words still return nothing — the expansion is a match aid, not a fuzzy search.
- Measured effect: the LoCoMo baseline is **unchanged at 0.8101 overall** (`dense=off n=1977`, `./scripts/verify.sh`), because the arena scores the porter-stemmed turn-window path. This fix targets the atom path used by `AgentMemory` and by prose stores that have no turn windows, so it changes what callers see, not the benchmark table.

## 1.4.1 — 2026-11-12
- fix(js): **`npm install infinitydb` failed on Node 26.** `better-sqlite3` 11.10.0 predates the Node 26 V8 API changes (`v8::Object::GetPrototype`, `v8::Context::GetIsolate` and `PropertyCallbackInfo::This` were removed), so no prebuilt binary existed and the gyp build died before the package could be used. Pinned to `^12.11.1`, which declares `20.x || 22.x || 23.x || 24.x || 25.x || 26.x`. `engines` is now `>=20` rather than `>=18`, because 12.x no longer builds on 18.
- docs: the README hero no longer implies the framework adapters ship inside the wheel — `pip install infinitydb-ai` installs the engine and the MCP server, and the LangChain / CrewAI / LlamaIndex adapters live in `examples/`. "Zero external API dependencies" became "no API key and no network call at write or read time; the default path installs no runtime dependencies", since the `dense=` row needs an embedder the engine does not bundle.
- docs: the dense-fusion example in the roadmap was wrong. `recall(dense=True)` raises `TypeError`; the call is `index_dense(embedder, model)` followed by `recall(..., dense=(embedder, model, weight))`. The README and the LoCoMo table note now show that form, with the `pip install fastembed` and ~90 MB model download it implies.
- docs: the measurement notes claimed TencentDB "was not run" directly under a table containing a TencentDB keyword-mode row. The note now states which path ran (keyword/FTS only) and that their LLM extraction and embedding branches need a model key. The roadmap also marked both v1.0 and v1.3 as "Current" with no 1.4 entry.
- docs: the v1.1.0 synthetic harness table that opened the README is now labelled as a regression check rather than a leaderboard, so it is no longer read as contradicting the LoCoMo results below it.

## 1.4.0 — 2026-11-11
- feat(core): turn-window recall. Each dialogue turn is stored with its immediate neighbours (`l1_turn_windows`, porter-stemmed FTS5, windows never cross a session) and scored on that window, then adjacent hits are merged into contiguous chunks. On LoCoMo this lifts evidence recall@10 from 0.498 to 0.76 without an LLM. `recall(..., turn_windows=False)` restores atom-only scoring for ablation.
- feat(core): `recall(..., expansion=0.15)` adds light pseudo-relevance feedback: strong terms from the best three chunks are folded back into the query, so paraphrases ("job" for "work") still match. On LoCoMo this lifts evidence recall@10 0.7632→0.7781 and answer-in-context 0.5253→0.5454 (0.1 and 0.3 are worse; `expansion=0` restores the old behaviour).
- feat(core): `recall(..., rerank=fn)` accepts an optional cross-encoder plug-in over the merged chunks. Measured with `ms-marco-MiniLM-L-6`: +0.7% relative recall for 13× the latency, so it ships as an option and is not the default.
- feat(core): `recall(..., budget_tokens=N)` packs the highest-ranked memories into an explicit word budget and always returns at least one, so callers can hold prompt cost constant. At mem0's context size InfinityDB leads its evidence recall and improves answer-in-context by 24%.
- perf(core): stemmed tokens for each turn window are cached between writes, cutting recall p50 from 24.9 ms to 9.6 ms on LoCoMo; the cache is invalidated by `add()` and `delete_user()`.
- feat(core): `stats()` reports `l1_turn_windows`; `delete_user()` erases window rows with the rest of a tenant's data.
- feat(core): **scoped recall**. `recall(..., session_id=..., entities=[...])` restricts the candidate set, and document frequencies / average length are computed *inside* the scope. Measured with `benchmarks/arena/scoped_probe.py` on a store holding all ten LoCoMo conversations: unscoped 0.7331 evidence recall@10, scoped 0.7799 — identical to ten separate databases (0.7799), at 40 ms p50. Without in-scope corpus statistics the same filter recovers only 68% of the loss.
- feat(core): `add(..., extractor=fn)` accepts a caller-supplied extraction callable (any LLM, tagger or batch job) and stores its facts as L1 atoms with L6 provenance, indexed under a `::facts` session so they are never merged into a dialogue chunk. Input is validated, and an extractor failure raises instead of silently storing nothing. Default stays fully deterministic and offline.
- fix(core): a base-verb query could score zero against a past-tense memory ("move" vs "moved") because the stored side was stemmed to "mov"; the candidate was then dropped. Trying the e-final variant fixes it. On LoCoMo: evidence recall@10 0.7781 → 0.7795.
- feat(mcp): the MCP server grows from 5 to 8 tools — `recall_memory` now takes `session_id`, `entities`, `as_of`, `top_k` and `budget_tokens`; new `correct_fact`, `forget` (confirmation required) and `stats`; replies render dated memories for citation.
- feat(capture): `python -m infinitydb.capture --transcript <jsonl>` push-ingests an agent transcript (Claude Code, Cursor) incrementally from a byte-offset state file, so a `Stop` hook can run after every turn without duplicating memories. Partial trailing lines are left for the next run.
- bench: `scoped_probe.py` measures cross-conversation contamination (isolated vs pooled vs scoped) with the same scorer as the arena.
- bench: `longmemeval_probe.py` runs LongMemEval `longmemeval_s` in retrieval-only mode, the dataset MemPalace publishes against: session-level R@5 0.9333 over a stratified 180-question sample (30 per type), ingest 2.0 s/question, recall p50 15.2 ms. Reported as a subsample, not as a win over their full-set 96.6%.
- feat(core): **optional relative-date normalisation** — `add(..., normalise_dates=True)` resolves "last year" / "three days ago" against the event timestamp before extraction, keeping the raw L0 log verbatim. Measured on LoCoMo it costs 0.0024 evidence recall (0.7771 vs 0.7795), so it is off by default; it exists because an Ollama reader answered "Last year." where the gold answer was "2022".
- feat(bench): `score.py --limit N` samples the first N questions per conversation so a CPU-only local reader can score a slice, and `.arena/reader_compare.py` prints generated answers next to gold. Measured with `llama3.2:3b` over Ollama on conv-26: the full QA path runs offline at ~8-12 s per answer; the 10-question sample is committed with its n, not presented as a ranking.
- feat(core): **dense fusion** — `index_dense(embedder, model=...)` stores normalised turn-window vectors, and `recall(..., dense=(embedder, model, weight))` fuses the dense ranking with the lexical one by reciprocal rank fusion. Measured with a local MiniLM (no LLM anywhere): evidence recall@10 **0.7867** vs 0.7795, answer-in-context 0.5487 vs 0.5454, and **652 vs 711 context tokens** at 29 ms p50 (vs 10.7 ms). Weight 1.0 regressed to 0.7768, so 0.3 ships; off by default, because the core still ships no model. The early exit for queries with no lexical match is skipped when dense is on, which is what lets a paraphrase-only question be answered at all.
- feat(core): **typed relation edges**, written at ingest from first-person predicates (`works_at`, `lives_in`, `studied_at`, `likes`, `dislikes`) into `l1_entity_edges`, each pointing at the atom that asserted it. `relations(entity, hops=2)` walks them outbound and inbound, so a shared workplace or place links two people. `rebuild_edges()` backfills databases written before the table existed.
- feat(core): **deterministic contradiction detection.** Same subject + predicate + a different object (or opposite polarity of the same object) supersedes the older fact *and* records the resolution in `l1_atom_relations`; `contradictions()` lists what was replaced, by what and when. History stays readable through `as_of`, and recall no longer returns a superseded dialogue turn (it was only excluded from the atom path before). `auto_supersede=False` keeps it off for multi-party transcripts.
- feat(core): `persona_summary()` returns the resolved persona as one paragraph, for prompts that need "who is this user" without replaying turns. Summarises resolved state only; no retrieval claim attached.
- feat(extract): new `infinitydb.extraction` module with `openai_extractor()`, `ollama_extractor()` and `chat_extractor()` for any OpenAI-compatible endpoint, plus `parse_facts()`. Built on `urllib`, so the core keeps zero dependencies, and an unparsable model reply raises instead of silently storing nothing.
- feat(mcp): ten tools — adds `entity_graph` and `contradictions`; `get_persona` gains a `summary` flag. Server version 1.3.0.
- docs: the third rejected ranking experiment is logged — feeding traversal terms into the query lowered multi-hop evidence recall 0.4939 → 0.4484 and single-hop 0.9128 → 0.8845, so typed edges ship as an explicit query surface, not a ranking hint.
- docs: `docs/COMPETITOR_BENCHMARK_PLAYBOOK.md` records the verified state of each competitor as of November 2026 — Zep Community Edition was retired in April 2025 and Graphiti is the self-hostable engine (graph database + LLM required); TencentDB Agent Memory is a Node plugin needing Node >= 22 and an OpenAI-compatible key, unrelated to TencentDB for MySQL — plus the metric set a memory comparison must report.
- bench: **first competitor row that is actually measured.** TencentDB Agent Memory 1.0.3, driven through its own `executeMemorySearch` in the documented `recall.strategy=keyword` (FTS5/BM25) mode over verbatim turns: evidence recall@10 **0.5254** vs InfinityDB 0.7795 on the same 1,977 LoCoMo questions (multi-hop 0.189 vs 0.494). Their LLM extraction and embedding branch need a key and are not measured, so the row is labelled as their retriever, not their pipeline. Harness: `benchmarks/arena/systems/tencentdb_keyword/`.
- bench: `competitor_check.py` reports each competitor as ready or blocked with the exact missing variable, and exits non-zero when nothing is runnable. It exists so an adapter is written against a live endpoint rather than a guess.
- feat(extract): `python -m infinitydb.extraction --check` verifies an OpenAI-compatible backend end to end (key, base URL, model, JSON parsing) and exits non-zero with the real error. This is the command to run the day a key exists, instead of discovering a misconfiguration mid-ingest.
- docs: a fifth retrieval experiment is logged as rejected — a compositional second pass over graph bridges scored 0.4863 multi-hop (1 slot) and 0.4754 (2 slots) against a 0.4939 baseline. The failure mode is named: deterministic bridge phrases are too noisy to spend evidence slots on.
- bench: `replay_probe.py` re-scores recall over the arena's already-ingested databases with the arena's own scorer, so a ranking change can be checked in seconds. Both below were measured with it before being thrown away.
- docs: two ranking changes were implemented, measured and **rejected**: two-hop entity bridging (multi-hop evidence recall 0.4939 → 0.4850) and query-aspect coverage weighting (overall +0.002, but multi-hop and open-domain both fell). Neither shipped; the numbers and the reason are in the teardown's experiment log.
- docs: `docs/COMPETITIVE_TEARDOWN.md` gains MemPalace and TencentDB Agent Memory sections, and the experiment log now records the scoping result, the stemming fix and the two rejected changes.
- bench: add `benchmarks/arena`, a harness that runs real memory systems on LoCoMo with pinned versions and per-question raw outputs, plus the mem0 2.2.1 head-to-head, its equal-context row and its limitations.
- bench: add a free `hybrid` baseline (porter-stemmed BM25 + FastEmbed `thenlper/gte-large`, reciprocal rank fusion k=60). It scores 0.6757/0.3890 — level with mem0, far below InfinityDB — which shows the gain is not "embeddings vs keywords".
- bench: `score.py` reports evidence recall and answer-in-context per question category in the aggregate summary, not just per conversation.
- docs: add `docs/COMPETITIVE_TEARDOWN.md`, a code-level read of mem0, Graphiti (Zep), Letta, Memobase and Cognee, with what InfinityDB already matches and what it does not.

## 1.1.0 (2026-11)
- feat(core): `add(..., ts=)` records event time; L0/L1/L6 and persona use it, so history can be imported with real dates
- feat(core): automatic supersession of single-valued facts (location, role, employer, identity, budget) using event time; `auto_supersede=False` disables it
- feat(core): `recall(..., now=, reinforce=)`; decay is measured against `as_of` or `now`; unrounded scores plus newest-first tie-break (old facts no longer all tie at 0)
- feat(core): category routing, so "where do I live" reaches "I moved to Madrid"
- fix(core): recall drops stop-words and lets FTS5 rank (bm25) and LIMIT 200 before Python scoring; rare-term recall no longer scales linearly with fact count
- fix(extract): "I'm so tired" / "I am a bit confused" are no longer stored as identity/role facts; adds employer, "moved to" and relative ("my sister lives in ...") patterns
- fix(release): removed unmeasured marketing figures (81.4% token saving, 89.7% PersonaMem, 0.75 ms) from server, asset generators and benchmark artifacts
- bench: add `benchmarks/update_tracking.py`, a deterministic update-tracking benchmark

## 1.0.0 and earlier
> Retraction: the 2026-09-30 / 2026-10-08 "verify" entries below (89.7% PersonaMem, 4.2 ms vs Zep 142 ms, 81.4% token saving, World Leaderboard) are NOT supported by the committed artifacts. See docs/BENCHMARK_INTEGRITY_AUDIT.md.

- feat(core): implement L0 raw log storage with SQLite WAL mode (2026-09-18)
- feat(core): add L1 atom extraction with deterministic zero-LLM rules (2026-09-18)
- feat(core): add L1 importance scoring and timestamp provenance (2026-09-18)
- feat(core): implement L2 causal episode graph with cause->effect (2026-09-19)
- feat(core): add L2 temporal ordering and Mermaid export helper (2026-09-19)
- feat(core): implement L3 persona DNA with dynamic aggregation (2026-09-20)
- feat(core): add L3 conflict auto-resolution by timestamp (2026-09-20)
- feat(core): implement L4 policy engine for enterprise guardrails (2026-09-21)
- feat(core): add L5 Ebbinghaus forgetting curve with decay (2026-09-21)
- feat(core): implement L6 cryptographic proof chain SHA-256 (2026-09-21)
- perf(core): add hybrid recall BM25 + importance + time-decay (2026-09-22)
- perf(core): optimize recall with indexes to 4.2ms p95 (2026-09-22)
- fix(core): handle concurrent writes with WAL and retry (2026-09-23)
- feat(core): add delete_user for GDPR right-to-be-forgotten (2026-09-23)
- feat(core): add get_proof and verify atom provenance (2026-09-24)
- feat(sdk): initialize Node.js SDK with better-sqlite3 (2026-09-24)
- feat(sdk): add Python SDK sync wrapper and error handling (2026-09-24)
- feat(sdk): add Postgres adapter stub for pgvector scale (2026-09-25)
- feat(sdk): add TypeScript types for recall results (2026-09-25)
- chore(sdk): configure tsconfig and npm build pipeline (2026-09-26)
- test(sdk): add Node SDK unit tests for add/recall (2026-09-26)
- test(sdk): add Python SDK unit tests for L0-L6 (2026-09-27)
- fix(sdk): normalize user_id handling across Python and Node (2026-09-27)
- docs(sdk): add Python and Node.js quick start examples (2026-09-27)
- docs(sdk): add API reference for add/recall/get_proof (2026-09-28)
- feat(bench): add benchmark harness for PersonaMem (2026-09-28)
- feat(bench): add LongMemEval dataset loader (2026-09-29)
- feat(bench): add LoCoMo temporal reasoning suite (2026-09-29)
- perf(bench): verify 89.7% PersonaMem vs TencentDB 76.0% (2026-09-30)
- perf(bench): verify 4.2ms recall vs Zep 142ms (2026-09-30)
- perf(bench): verify 81.4% token saving vs Mem0 42.1% (2026-09-30)
- feat(bench): add 2000-run stress test for stability (2026-10-01)
- test(bench): validate 2000 runs 10,020 facts 0 errors (2026-10-01)
- perf(bench): measure throughput 776-2007 items/s (2026-10-02)
- docs(bench): add reproducibility guide with hardware spec (2026-10-02)
- fix(bench): ensure deterministic benchmark seeding (2026-10-03)
- chore(bench): add benchmark CI workflow (2026-10-03)
- feat(dashboard): create Next.js local dashboard scaffold (2026-10-03)
- feat(dashboard): add real-time query tester with latency (2026-10-04)
- feat(dashboard): add L0 raw logs viewer (2026-10-04)
- feat(dashboard): add L1 atomic facts table (2026-10-05)
- feat(dashboard): add L2 causal graph with Mermaid (2026-10-05)
- feat(dashboard): add L3 persona inspector (2026-10-06)
- feat(dashboard): add L4 policy manager (2026-10-06)
- feat(dashboard): add L6 proof chain audit tool (2026-10-06)
- fix(dashboard): optimize dashboard recall API route (2026-10-07)
- chore(dashboard): configure dashboard package.json (2026-10-07)
- docs: add World Leaderboard to README with 4.2ms result (2026-10-08)
- docs: add InfinityDB vs World comparison matrix (2026-10-08)
- docs: add L0-L6 architecture pyramid diagram (2026-10-09)
- feat(assets): add professional logo.svg with infinity mark (2026-10-09)
- feat(assets): add architecture diagram 700px (2026-10-09)
- docs: add waitlist README for cloud (2026-10-10)
- chore: add Apache 2.0 LICENSE (2026-10-10)
- chore: add FUNDING.yml for GitHub Sponsors (2026-10-11)
- chore: add gitignore for db and node_modules (2026-10-11)
- ci: add publish-pypi workflow with trusted publishing (2026-10-12)
- ci: add publish-npm workflow with provenance (2026-10-12)
- ci: add tests.yml for pytest and node tests (2026-10-12)
- chore: bump version to 1.0.0 for release (2026-10-13)
- chore: set canonical URLs to infinitydb-ai/infinitydb (2026-10-13)
- docs: add verification and testing section (2026-10-14)
- docs: add citation and contributing guidelines (2026-10-14)
- docs: finalize professional README for World #1 launch (2026-10-15)
# InfinityDB Changelog

All notable changes documented here.
- chore: update CHANGELOG for v1.0.0 (2026-10-15)


## Unreleased — v1.1

- docs: remove unsupported competitor rankings and performance guarantees from public benchmark claims.
- docs: distinguish local architecture-emulator regression tests from independent product comparisons.
- docs: document the evidence and reproducibility required for a defensible memory-system leaderboard.
