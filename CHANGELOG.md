# InfinityDB Changelog

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
