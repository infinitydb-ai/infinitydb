<p align="center">
  <img src="assets/logo.svg" width="480" alt="InfinityDB Logo"/>
</p>

<h3 align="center">Remember Everything. Forget Nothing That Matters.</h3>

<p align="center">
  InfinityDB is a high-performance, open-source agent memory engine.<br/>
  Symbolic short-term memory + layered long-term memory. No API key and no network call at write or read time; the default path installs no runtime dependencies. Runs locally on SQLite or Postgres.
</p>

<p align="center">
  <a href="https://github.com/infinitydb-ai/infinitydb"><img src="https://img.shields.io/github/stars/infinitydb-ai/infinitydb?style=social" alt="GitHub Stars"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/></a>
  <a href="https://pypi.org/project/infinitydb-ai/"><img src="https://img.shields.io/pypi/v/infinitydb-ai" alt="PyPI"/></a>
  <a href="https://www.npmjs.com/package/infinitydb"><img src="https://img.shields.io/npm/v/infinitydb" alt="NPM"/></a>
  <a href="https://smithery.ai/skills/diwakarsrd28/infinitydb"><img src="https://img.shields.io/badge/Smithery-Skill-orange" alt="Smithery Skill"/></a>
  <a href="https://huggingface.co/spaces/Diwakarsrd/infinitydb"><img src="https://img.shields.io/badge/HuggingFace-Live_Demo-yellow" alt="Hugging Face Demo"/></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/node-%3E%3D20.0-green" alt="Node"/>
</p>

<p align="center">
  <b>Agents remember. Humans innovate.</b><br/>
  MCP server included · LangChain, CrewAI and LlamaIndex adapters in <code>examples/</code><br/>
  <a href="https://github.com/infinitydb-ai/infinitydb">github.com/infinitydb-ai/infinitydb</a>
</p>

---

## Why InfinityDB

Most memory systems compress history into an opaque vector pile. When recall fails, you are left guessing. InfinityDB builds a structured, verifiable semantic pyramid.

*   **Token-Aware Memory:** Stores raw multi-turn conversations and tool transcripts outside the model context. Actual token savings depend on the caller tokenizer and prompt construction.
*   **Deterministic Fact Extraction:** L1 Atomic Facts are extracted with local rules and no LLM call. Extraction quality is limited to the implemented patterns; latency depends on workload and hardware.
*   **Temporal Fact History:** `add(..., ts=)` records when something happened. A newer single-valued fact (location, role, employer, identity, budget) automatically supersedes the older one, and `recall(..., as_of=ts)` returns what was current at that time. Multi-valued facts (preferences, relations) are never superseded automatically, and the engine does not infer contradictions between free-text facts; use `supersede_fact()` for those.
*   **Cryptographic Proof Chain:** Every fact contains an immutable SHA-256 hash linking to its exact source slice in the raw log.
*   **Tenant-Scoped Guardrails (L4 Policies):** Combines global baseline policies with tenant-specific SOPs; a tenant's private policies are excluded from other tenants' recall.
*   **Indexed Hybrid Retrieval:** SQLite FTS5 (porter-stemmed) narrows keyword candidates; strong terms from the best few hits are fed back into the query (light pseudo-relevance feedback), which recovers paraphrases such as "job" for "work"; importance weighting and Ebbinghaus time-decay rank the retrieved facts. A cross-encoder reranker can be plugged in through `recall(..., rerank=fn)` if you want the last few percent and can pay for it. Latency depends on dataset, hardware, and workload.

---

## Regression checks (harness output, not a leaderboard)

Run offline with `python benchmarks/update_tracking.py` and `python benchmarks/reproducible.py`. Synthetic, single machine, no external system involved: these two scripts exist to catch regressions in update tracking and retrieval latency. They are **not** cross-system comparisons. Head-to-head results against other memory engines are the LoCoMo table in the next section.

The columns below are the harness baseline recorded at v1.1.0; current output is checked by `./scripts/verify.sh`.

| Check | v1.0.0 | v1.1.0 |
|---|---|---|
| Update tracking: current answer ranked first (300 users, 1-4 moves each, 3,000 distractor facts) | 27% | 100% |
| Update tracking: outdated fact in top 3 | 75% | 0% |
| `benchmarks/reproducible.py` recall p50 / p95 | 9.1 / 12.4 ms | 1.2 / 1.9 ms |
| Recall p50, 30,000 facts, query with a rare term | 1,667 ms | 0.4 ms |

Limits: the update-tracking test only uses facts the built-in rules can parse ("I live in X", "I moved to X"); free-text facts outside those patterns are stored but not auto-superseded. Queries made only of very common words still scan many rows (25 ms at 30,000 facts).

## Benchmark status

A reproducible head-to-head harness now runs **actual memory systems** on the public LoCoMo dataset, with pinned versions and per-question raw outputs: see [benchmarks/arena](benchmarks/arena/README.md). It replaces the earlier architecture emulators, which must not be read as product comparisons.

Measured 2026-11-10 on one 8-vCPU CPU-only host, all 10 conversations, 1,986 questions, top-10 retrieval, no LLM at write or read time:

| System | evidence recall@10 | answer in context | context tokens | ingest | search p50 |
| --- | --- | --- | --- | --- | --- |
| **InfinityDB** | **0.780** | **0.545** | 711 | **7.5 s** | **10.7 ms** |
| **InfinityDB**, `dense=` (local MiniLM fusion) | 0.787 | 0.549 | **652** | 8.0 s | 29.0 ms |
| **InfinityDB**, `budget_tokens=320` | 0.696 | 0.479 | 325 | 8.4 s | 10.8 ms |
| mem0 2.2.1 OSS (`infer=False`) | 0.678 | 0.386 | **321** | 544.3 s | 152.1 ms |
| TencentDB Agent Memory 1.0.3, keyword/FTS mode (no LLM) | 0.525 | — | — | — | — |
| Free BM25 + dense hybrid, RRF-fused | 0.676 | 0.389 | **314** | 582.2 s | 151.9 ms |
| InfinityDB, atom-only ablation | 0.498 | 0.355 | 329 | 7.9 s | 5.1 ms |

The TencentDB Agent Memory row is retrieval-only on purpose. Their L1 extraction
and their embedding branch both need a model key, so only their documented
`recall.strategy=keyword` (FTS5/BM25) path could run; it was driven through their
own `executeMemorySearch` over the same verbatim turns, and no reader was used,
so answer-in-context, tokens and latency are left blank rather than estimated.
Reproduce it with `benchmarks/arena/systems/tencentdb_keyword/`.

The hybrid row is a deliberately strong "free stack" baseline: porter-stemmed BM25
over the same turns plus `thenlper/gte-large` embeddings, fused with reciprocal
rank fusion (k=60), all local. It does not beat mem0 here — fusing the two
signals does not fix what a bag-of-words index and one embedding lose — and it
costs mem0-class ingest time.

The `dense=` row is not the default configuration: it needs an embedding model you supply yourself. The arena used `fastembed` with `sentence-transformers/all-MiniLM-L6-v2` — one ~90 MB download, no API key. The engine bundles no model and declares no runtime dependencies, so enabling it means installing an embedder yourself (`pip install fastembed`), running `db.index_dense(embedder, "minilm")`, then calling `recall(..., dense=(embedder, "minilm", 1.0))`.

What these numbers do and do not say:

- InfinityDB retrieves more annotated evidence (+15.0% relative) and puts the gold answer in context considerably more often (+41% relative) than mem0's no-LLM mode, at ~72× faster ingestion and ~14× faster search — but with ~2.2× more context by default.
- Prompt cost can be held constant: `recall(..., budget_tokens=N)` packs the best memories into a word budget. At mem0's ~320-word context, InfinityDB still leads on evidence recall (0.696 vs 0.678) and puts the gold answer in context 24% more often; it reaches mem0's answer-in-context rate at ~200 words.
- These are retrieval proxies scored without a reader model, on one dataset, on one machine, with heuristics tuned on that same dataset. They are not a general leaderboard.
- **Zep/Graphiti** was not run at all: it requires a graph database and model credentials this environment does not have, and three local attempts failed in ways documented in `benchmarks/arena/README.md`. **TencentDB** appears only in its documented keyword/FTS mode (row above); their LLM extraction and embedding paths need a model key and were not run. No comparison against either system's full pipeline is claimed.
- No independent third party has reproduced these results.

Two further measurements, same host, same scoring code:

- **Scoped recall under contamination.** Every row above gives each conversation
  its own database, which is not how an agent runs. Pooling all ten conversations
  into one store with unscoped recall drops evidence recall@10 from 0.780 to
  **0.733**; the same store recalled with `session_id=<conversation>` returns
  **0.780** — identical to ten separate databases, at 40 ms p50 across a 10×
  larger corpus. Scoping computes corpus statistics inside the scope; without
  that, it recovers only 68% of the loss.
- **LongMemEval (`longmemeval_s`), the dataset MemPalace publishes against.**
  Session-level **R@5 = 0.933** over a stratified 180-question sample
  (30 per question type, answer hit if any returned memory comes from an
  annotated answer session, ~54 sessions / ~550 turns per question, no reader
  LLM). MemPalace reports 96.6% R@5 for the full 500; this is a 180-question
  subsample, so treat it as close, not as a win. The weak type is
  `single-session-preference` (0.667): preference questions want the L3 profile,
  which recall does not yet merge in.

The repository's own reproducible harness remains the regression tool for latency and correctness. See [the benchmark integrity audit](docs/BENCHMARK_INTEGRITY_AUDIT.md) and [benchmark reproduction notes](benchmarks/README.md).

How the other memory engines are built — mem0, Graphiti (Zep), Letta, Memobase, Cognee — is documented from their shipped source in [the competitive teardown](docs/COMPETITIVE_TEARDOWN.md), together with what we adopted from them and the changes that were measured and rejected.

## Architecture: The L0-L6 Memory Pyramid

Instead of an opaque vector pile, InfinityDB constructs a neurosymbolic hierarchy:

```
L0 RAW LOGS       -> Verbatim conversation & tool logs stored on disk/SQLite (offloaded from context)
L1 ATOM FACTS     -> Extracted atomic facts with SQLite FTS5 keyword index
L2 EPISODE GRAPH  -> Causal & temporal relationships ("X happened because Y") with Mermaid export
L3 PERSONA DNA    -> Dynamic user identity profile with automated conflict resolution
L4 POLICY ENGINE  -> Enterprise compliance rules, security policies, and tool SOPs
L5 FORGETTING     -> Ebbinghaus forgetting curve decay with reinforcement
L6 PROOF CHAIN    -> Cryptographic SHA-256 provenance linking every atom back to exact L0 raw text
```

The stored raw logs and extracted records are inspectable; extraction is rule-based and does not guarantee lossless semantic coverage.

---

## Verify it yourself

Every number in this README is reproducible offline, with no API key:

```bash
./scripts/verify.sh
```

That runs the test suite, an engine smoke test, the published LoCoMo baseline, and
the competitor readiness check. The full claim inventory — what is measured, what
is not, and the known gaps ranked — is in [docs/DILIGENCE.md](docs/DILIGENCE.md).

## Installation

### Python
```bash
pip install infinitydb-ai
```

Local development:
```bash
git clone https://github.com/infinitydb-ai/infinitydb.git
cd infinitydb
pip install -e ".[dev]"
```

### Node.js / TypeScript
```bash
npm install infinitydb
```

---

## Quick Start

### Python (Drop-in Agent Memory)
```python
from infinitydb import AgentMemory

mem = AgentMemory("agent_memory.db")

# Store conversation: auto-extracts atomic facts with zero LLM API cost ($0)
mem.add("User hates long answers and loves python. Project code is ABC123", user_id="sarah_c")

# Recall latency depends on workload and hardware
facts = mem.search("project code?", user_id="sarah_c", limit=3)
print(facts)
# -> ["Project code is ABC123", "User prefers concise answers", "User loves python"]
```

### Python (Full L0-L6 Engine)
```python
from infinitydb import InfinityDB

db = InfinityDB("agent_memory.db")

# 1. Ingest conversation (builds L0-L6 automatically)
log_id = db.add(
    "I am Sarah Connor. I work as an Infrastructure Lead. "
    "I prefer Rust because it provides zero-cost memory safety. "
    "My budget is $40k.",
    user_id="sarah_c"
)

# 2. Indexed keyword retrieval with importance and time-decay scoring
context = db.recall("what language does Sarah prefer?", user_id="sarah_c")
print("Latency:", context["latency_ms"], "ms")
print("Top Fact:", context["results"][0]["fact"])
print("Causal Link:", context["causal_graph"])

# 3. Temporal facts: preserve history and explicitly supersede
old_fact = context["results"][0]["id"]
new_log = db.add("I switched to Go for high-concurrency microservices.", user_id="sarah_c")
new_fact = db.conn.execute(
    "SELECT id FROM l1_atoms WHERE source_id = ?", (new_log,)
).fetchone()["id"]
db.supersede_fact(old_fact, new_fact, user_id="sarah_c")
# Current recall returns the new fact; as_of=<unix timestamp> can inspect history.

# 4. L6 Cryptographic Proof Verification
atom_id = context["results"][0]["id"]
proof = db.get_proof(atom_id)
print("Proof Hash:", proof["hash"])
print("Raw Snippet:", proof["raw_snippet"])

# 5. Enterprise Policies (L4)
db.add_policy("Do not expose internal VPC credentials in agent responses", domain="security", priority=2.0)

# 6. GDPR Compliance
db.delete_user("sarah_c")
```

### Node.js / TypeScript
```javascript
const { InfinityDB } = require("infinitydb");

const db = new InfinityDB();

db.add(
  "I am Alex. I work as a Cloud Architect in London. I prefer PostgreSQL because of reliability.",
  "user_alex"
);

const result = db.recall("what database does Alex use?", { userId: "user_alex" });
console.log(`Recall latency: ${result.latencyMs}ms`);
console.log("Results:", result.results);

const proof = db.getProof(result.results[0].id);
console.log("Verified provenance:", proof.verified, proof.hash);
```

---

## Interactive Local Dashboard

```bash
python dashboard/server.py
```
Open `http://localhost:8080` to inspect:
* Real-time query tester with latency readout
* L0 Raw Logs viewer
* L1 Atomic Facts table
* L2 Causal Graph visualizer (Mermaid-compatible)
* L3 Dynamic Persona inspector
* L4 Enterprise Policies SOP manager
* L6 Cryptographic Proof Chain audit tool

---

## SQLite durability

For file-backed databases, both SDKs use WAL journaling with
`synchronous=FULL`. In WAL mode, SQLite synchronizes the write-ahead log at commit,
providing stronger commit durability at a potential write-throughput cost. Actual
durability still depends on the filesystem, storage device, operating system, and their
honoring of synchronization requests. In-memory databases do not configure file-backed
WAL settings.

---

## Resource lifecycle (Python)

Close persistent and in-memory database handles when the application is finished with
them. Both the core engine and the high-level `AgentMemory` wrapper support context
managers:

```python
from infinitydb import AgentMemory

with AgentMemory("agent_memory.db") as memory:
    memory.add("I prefer explicit cleanup.", user_id="agent-1")
    facts = memory.search("cleanup", user_id="agent-1")
```

For long-lived objects managed manually, call `memory.close()` (or
`db.close()` on the core `InfinityDB` instance). This closes the SQLite connection;
it does not delete the database file.

---

## Verification & Testing

```bash
pytest
python benchmarks/reproducible.py --iterations 1000 --output benchmark-results.json
node infinitydb-js/test.js
```

The legacy `benchmarks/benchmark.py` and `benchmarks/run_benchmark.py` commands now route to the same single-engine harness. They no longer produce synthetic competitor rankings. See [benchmark methodology](benchmarks/REPRODUCIBLE_BENCHMARK.md).

---

## Integrations

Example adapters for leading agent frameworks live in [`examples/`](examples/) in this repository. They are **not** packaged in the PyPI wheel: `pip install infinitydb-ai` gives you the engine and the MCP server, so clone the repo if you want these adapters. No cloud required.

### LangChain
Compatible with LangChain chains and agents.
```python
from examples.langchain_integration import InfinityDBLangChainMemory
from infinitydb import InfinityDB

db = InfinityDB("agent_memory.db")
memory = InfinityDBLangChainMemory(db=db, user_id="sarah_c")
```

See `examples/langchain_integration.py` for full chain example.

### CrewAI
Shared memory with role-based partitioning and L4 governance guardrails for multi-agent teams.
```python
from examples.crewai_integration import InfinityDBCrewMemory
from infinitydb import InfinityDB

db = InfinityDB("agent_memory.db")
memory = InfinityDBCrewMemory(db=db, agent_role="researcher")
```

See `examples/crewai_integration.py`.

### LlamaIndex
Drop-in conversational memory for LlamaIndex query engines, chat engines, and agent pipelines.
```python
from examples.llamaindex_integration import InfinityDBLlamaMemory

memory = InfinityDBLlamaMemory(db_path="agent_memory.db", user_id="robert_m")
context = memory.get(input_text="What are user requirements?")
```

See `examples/llamaindex_integration.py`.

### Vercel AI SDK
Middleware for Node.js / Next.js AI applications that enriches prompt context with local memory retrieval.
```javascript
const { InfinityDBVercelMemory } = require("./examples/vercel_ai_integration");

const memory = new InfinityDBVercelMemory({ userId: "alex_dev" });
const enrichedPrompt = await memory.enrichPrompt("Generate component architecture");
```

See `examples/vercel_ai_integration.js`.

### Claude Code, Cursor & any MCP host
Model Context Protocol server over stdio, zero dependencies, ten tools:
`recall_memory` (scoped by session, entity, point-in-time and token budget),
`entity_graph` (typed relation traversal) and `contradictions` (supersession audit),
`add_memory`, `get_persona`, `correct_fact`, `forget`, `add_policy`,
`verify_proof`, `stats`.
```bash
python -m infinitydb.mcp_server
```

Capture is the other half: MCP is pull (the agent has to decide to look), so
`python -m infinitydb.capture --transcript <session.jsonl>` push-ingests an
agent transcript incrementally — safe to run from a `Stop` hook after every
turn, because a byte-offset state file means only new lines are read and
nothing is stored twice.
```json
{"hooks": {"Stop": [{"hooks": [{"type": "command",
  "command": "python -m infinitydb.capture --transcript \"$CLAUDE_TRANSCRIPT_PATH\""}]}]}}
```

See `docs/MCP_SERVER.md` for host configuration and hook setup.

**Registry:** [NPM](https://www.npmjs.com/package/infinitydb) `infinitydb` | [PyPI](https://pypi.org/project/infinitydb-ai/) `infinitydb-ai` | [Smithery](https://smithery.ai/skills/diwakarsrd28/infinitydb) `diwakarsrd28/infinitydb` | [Hugging Face](https://huggingface.co/spaces/Diwakarsrd/infinitydb)

---

## Roadmap

*   **v1.4 (Current):** Dense fusion over turn windows (`index_dense(...)` then `recall(..., dense=(embedder, model, weight))`, opt-in), typed relation edges, scoped recall that holds up in a pooled multi-tenant store, typed supersession with `as_of` recall, opt-in relative-date normalisation, and the LoCoMo arena with per-question raw output. Install with `pip install -U infinitydb-ai` or `npm i infinitydb`.
*   **v1.3:** Typed relation edges written at ingest (`relations()`, `entity_graph`), deterministic contradiction detection with a supersession audit (`contradictions()`), model-backed extraction module (`openai_extractor` / `ollama_extractor`), persona compaction, ten-tool MCP server, fast replay probe.
*   **v1.2:** Scoped recall (`session_id` / `entities`), caller-supplied extraction hook, eight-tool MCP server, incremental transcript capture, LongMemEval + scoped-recall probes.
*   **v1.0:** Open-source L0-L6 Brain Engine, SQLite WAL mode, Python and Node.js SDKs, local dashboard, benchmark suite.
*   **With a key:** `python -m infinitydb.extraction --check` validates the extraction backend, then `add(..., extractor=openai_extractor())` (or `ollama_extractor()` for a free local model) turns turns into model-written facts. The hook is shipped and tested; the measured quality number is pending a key.
*   **Next:** Direct Postgres + `pgvector` adapter for very large stores, Redis caching tier, read-time graph expansion for multi-hop questions.
*   **Cloud (Waitlist):** Hosted multi-tenant managed clusters with SSO, RBAC, and SOC2. See [waitlist/README.md](waitlist/README.md).

---

## Open Agent Memory Specification (OAMS-1.0)

InfinityDB implements and references the [Open Agent Memory Specification (STANDARD.md)](STANDARD.md), defining the standard L0 to L6 neurosymbolic interface for AI agents under the Apache 2.0 license.

---

## 1-Click Cloud Deployment

Deploy a dedicated, persistent InfinityDB memory microservice to Fly.io or Docker:

```bash
# Local containerized deployment via Docker Compose
docker-compose up -d

# Deploy to Fly.io
fly launch
```

REST endpoints exposed:
* `POST /add` - Ingest dialogue turns and extract atomic facts
* `POST /search` - Hybrid memory recall
* `POST /recall` - Full L0-L6 recall with causal graph and policies
* `GET /health` - Liveness check on SQLite WAL
* `GET /stats` - Memory layer statistics

---

## Contributing

Contributions are welcome. Please open an issue to discuss design proposals before submitting pull requests.

## Citation

If you use InfinityDB in research, please cite: `https://github.com/infinitydb-ai/infinitydb`

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

<!-- leaderboard update -->

<!-- comparison matrix update -->

<!-- architecture pyramid update -->

<!-- testing guide -->

<!-- citation info -->

<!-- launch ready -->
