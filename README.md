<p align="center">
  <img src="assets/logo.svg" width="480" alt="InfinityDB Logo"/>
</p>

<h3 align="center">Remember Everything. Forget Nothing That Matters.</h3>

<p align="center">
  InfinityDB is a high-performance, open-source agent memory engine.<br/>
  Symbolic short-term memory + layered long-term memory. Zero external API dependencies. Runs locally on SQLite to Postgres.
</p>

<p align="center">
  <a href="https://github.com/infinitydb-ai/infinitydb"><img src="https://img.shields.io/github/stars/infinitydb-ai/infinitydb?style=social" alt="GitHub Stars"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/></a>
  <a href="https://pypi.org/project/infinitydb-ai/"><img src="https://img.shields.io/pypi/v/infinitydb-ai" alt="PyPI"/></a>
  <a href="https://www.npmjs.com/package/infinitydb"><img src="https://img.shields.io/npm/v/infinitydb" alt="NPM"/></a>
  <a href="https://smithery.ai/skills/diwakarsrd28/infinitydb"><img src="https://img.shields.io/badge/Smithery-Skill-orange" alt="Smithery Skill"/></a>
  <a href="https://huggingface.co/spaces/Diwakarsrd/infinitydb"><img src="https://img.shields.io/badge/HuggingFace-Live_Demo-yellow" alt="Hugging Face Demo"/></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/node-%3E%3D18.0-green" alt="Node"/>
</p>

<p align="center">
  <b>Agents remember. Humans innovate.</b><br/>
  LangChain | CrewAI | MCP Compatible<br/>
  <a href="https://github.com/infinitydb-ai/infinitydb">github.com/infinitydb-ai/infinitydb</a>
</p>

---

## Why InfinityDB

Most memory systems compress history into an opaque vector pile. When recall fails, you are left guessing. InfinityDB builds a structured, verifiable semantic pyramid.

*   **Token-Aware Memory:** Stores raw multi-turn conversations and tool transcripts outside the model context. Actual token savings depend on the caller tokenizer and prompt construction.
*   **Deterministic Fact Extraction:** L1 Atomic Facts are extracted with local rules and no LLM call. Extraction quality is limited to the implemented patterns; latency depends on workload and hardware.
*   **Temporal Fact History:** Facts remain in their original provenance chain. Applications can explicitly mark a newer fact as superseding an older one and query memory as of a Unix timestamp; the engine does not guess semantic contradictions.
*   **Cryptographic Proof Chain:** Every fact contains an immutable SHA-256 hash linking to its exact source slice in the raw log.
*   **Tenant-Scoped Guardrails (L4 Policies):** Combines global baseline policies with tenant-specific SOPs; a tenant's private policies are excluded from other tenants' recall.
*   **Indexed Hybrid Retrieval:** SQLite FTS5 narrows keyword candidates; importance weighting and Ebbinghaus time-decay rank the retrieved facts. Latency depends on dataset, hardware, and workload.

---

## Benchmark status

**No independent leaderboard ranking is claimed yet.** The benchmark artifacts currently committed to this repository do not support the historical competitor-comparison numbers previously shown here. In particular, the checked-in run reports 0.0 PersonaMem accuracy for InfinityDB, a measured token saving of 63.2% (not 81.4%), and the 2,000-iteration stress artifact reports 8.238 ms p95 recall (not 4.2 ms).

The existing benchmark harness includes local architecture emulators; it does not execute the actual TencentDB, Mem0, Zep, or Letta products. Those emulator outputs must not be presented as direct product comparisons.

Until a benchmark is independently reproducible against pinned implementations and datasets, treat performance figures as workload-specific observations—not guarantees or rankings. See [the benchmark integrity audit](docs/BENCHMARK_INTEGRITY_AUDIT.md) and [benchmark reproduction notes](benchmarks/README.md).

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

InfinityDB provides drop-in adapters for leading agent frameworks. No cloud required.

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

### Claude Desktop & Cursor (MCP)
Model Context Protocol server over stdio. Works with any MCP host.
```bash
python -m infinitydb.mcp_server
```

See `docs/MCP_SERVER.md` for Claude Desktop and Cursor configuration.

**Registry:** [NPM](https://www.npmjs.com/package/infinitydb) `infinitydb` | [PyPI](https://pypi.org/project/infinitydb-ai/) `infinitydb-ai` | [Smithery](https://smithery.ai/skills/diwakarsrd28/infinitydb) `diwakarsrd28/infinitydb` | [Hugging Face](https://huggingface.co/spaces/Diwakarsrd/infinitydb)

---

## Roadmap

*   **v1.0 (Current):** Open-source L0-L6 Brain Engine, SQLite WAL mode, Python and Node.js SDKs, local dashboard, comprehensive benchmark suite.
*   **v1.1 (Upcoming):** Direct Postgres + `pgvector` adapter for 100M+ memories, Redis caching tier, MCP server for Claude / Cursor.
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
