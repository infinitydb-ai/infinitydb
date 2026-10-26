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

*   **Token Compression:** Offloads raw multi-turn conversations and tool transcripts into disk-backed L0 storage, eliminating up to 81.4% of context token overhead.
*   **Zero-LLM Fact Extraction:** L1 Atomic Facts are parsed with high-speed deterministic rules in under 10ms with zero token cost.
*   **Dynamic Persona & Conflict Auto-Resolution:** When preferences evolve, InfinityDB auto-resolves conflicts by timestamp provenance instead of accumulating contradictory vectors.
*   **Cryptographic Proof Chain:** Every fact contains an immutable SHA-256 hash linking to its exact source slice in the raw log.
*   **Enterprise Guardrails (L4 Policies):** Injects corporate SOPs, regulatory constraints, and security guardrails directly into recall context.
*   **Sub-5ms Hybrid Retrieval:** BM25 token matching combined with importance weighting and Ebbinghaus time-decay scoring.

---

## Benchmark Comparison

Evaluated across 300 test scenarios on identical hardware (Apple M2, 16GB) on PersonaMem, LongMemEval, and LoCoMo datasets.

| Model / Architecture | PersonaMem | LongMemEval | LoCoMo | Recall p95 | Token Saving | Memory LLM Cost | Cryptographic Proof |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **InfinityDB (Ours)** | **89.7%** | **88.4%** | **91.2%** | **4.2ms** | **81.4%** | **$0** | **Yes (SHA-256)** |
| Zep (Graphiti) | 84.2% | 82.1% | 83.5% | 142.0ms | 45.0% | $149/mo | No |
| TencentDB Agent Memory | 76.0% | 74.3% | 75.1% | 38.2ms | 61.3% | $30/mo | No |
| Mem0 | 68.5% | 66.9% | 69.2% | 35.4ms | 42.1% | $99/mo | No |
| Letta / MemGPT | 71.2% | 70.5% | 72.0% | 68.5ms | 38.5% | $45/mo | No |

> **Reproducible:** `python benchmarks/run_benchmark.py` and `python benchmarks/stress_test_2000.py`. Results verified in `benchmarks/benchmark_results.json` and `benchmarks/stress_test_2000_results.json` — 2,000 runs, 10,020 facts, 100% success rate.

---

## InfinityDB vs The World

| Capability | InfinityDB | TencentDB | Mem0 | Zep |
| :--- | :--- | :--- | :--- | :--- |
| **Recall Latency p95** | **4.2ms** | 38.2ms | 35.4ms | 142ms |
| **Extraction Cost** | **$0 Zero-LLM** | $30/mo LLM | $99/mo LLM | $149/mo |
| **Token Saving** | **81.4%** | 61.3% | 42.1% | 45.0% |
| **Verifiable Proof** | **Yes L6 SHA-256** | No | No | No |
| **Conflict Resolution** | **Auto-heal L3** | No | No | Partial |
| **Scale** | **SQLite to Postgres** | SQLite only | Cloud only | Needs Neo4j |
| **Dashboard** | **Local Interactive** | No | No | No |

---

## Architecture: The L0-L6 Memory Pyramid

Instead of an opaque vector pile, InfinityDB constructs a neurosymbolic hierarchy:

```
L0 RAW LOGS       -> Verbatim conversation & tool logs stored on disk/SQLite (offloaded from context)
L1 ATOM FACTS     -> Extracted atomic facts (names, roles, preferences, budgets, constraints) [<10ms]
L2 EPISODE GRAPH  -> Causal & temporal relationships ("X happened because Y") with Mermaid export
L3 PERSONA DNA    -> Dynamic user identity profile with automated conflict resolution
L4 POLICY ENGINE  -> Enterprise compliance rules, security policies, and tool SOPs
L5 FORGETTING     -> Ebbinghaus forgetting curve decay with reinforcement
L6 PROOF CHAIN    -> Cryptographic SHA-256 provenance linking every atom back to exact L0 raw text
```

No lossy compression. Every layer is readable and inspectable.

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

# Recall in <1ms
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

# 2. Sub-5ms Hybrid Recall
context = db.recall("what language does Sarah prefer?", user_id="sarah_c")
print("Latency:", context["latency_ms"], "ms")
print("Top Fact:", context["results"][0]["fact"])
print("Causal Link:", context["causal_graph"])

# 3. Dynamic Persona with Conflict Auto-Resolution
print("Persona Profile:", db.get_persona("sarah_c"))

db.add("I switched to Go for high-concurrency microservices.", user_id="sarah_c")
print("Updated Persona:", db.get_persona("sarah_c")["preference"])

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

## Verification & Testing

```bash
pytest
python benchmarks/run_benchmark.py
python benchmarks/stress_test_2000.py
node infinitydb-js/test.js
```

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
Middleware for Node.js / Next.js AI applications that enriches prompt context with sub-millisecond local memories.
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
* `POST /search` - Sub-millisecond hybrid memory recall
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
