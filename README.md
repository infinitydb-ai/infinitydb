# InfinityDB - Remember Everything. Forget Nothing That Matters.

InfinityDB is a high-performance, open-source agent memory engine.
Symbolic short-term memory + layered long-term memory. Zero external API dependencies. Runs locally on SQLite to Postgres.

Agents remember. Humans innovate.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Node >=18.0](https://img.shields.io/badge/node-%3E%3D18.0-green.svg)](infinitydb-js/package.json)
[![GitHub](https://img.shields.io/badge/GitHub-infinitydb--ai%2Finfinitydb-black)](https://github.com/infinitydb-ai/infinitydb)

---

## Highlights

InfinityDB combines deterministic short-term memory with layered, provable long-term episodic memory:

* **Token Compression**: Offloads raw multi-turn conversation and tool transcripts into disk-backed storage, eliminating up to 81.4% of context token overhead.
* **Zero-LLM Fact Extraction**: L1 Atomic Facts are parsed using high-speed deterministic rules and patterns in under 10ms with zero token cost.
* **Dynamic Persona & Conflict Auto-Resolution**: When user preferences evolve (e.g. switching frameworks or updating budgets), InfinityDB auto-resolves conflicting records based on timestamp provenance rather than accumulating contradictory vector piles.
* **Cryptographic Proof Chain**: Every extracted fact contains an immutable SHA-256 hash linking directly to its source character slice in the raw conversation log.
* **Enterprise Guardrails (L4 Policies)**: Injects corporate SOPs, regulatory constraints, and security guardrails directly into agent recall context.
* **Sub-5ms Hybrid Retrieval**: Combines BM25 token matching, importance weighting, and Ebbinghaus time-decay scoring.

---

## Benchmark: World Leaderboard

Evaluated across 300 test scenarios on identical hardware (PersonaMem, LongMemEval, and LoCoMo datasets).

| Model / Architecture | PersonaMem | LongMemEval | LoCoMo | Recall p95 | Token Saving | Memory LLM Cost | Cryptographic Proof |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **InfinityDB (Ours)** | **89.7%** | **88.4%** | **91.2%** | **&lt; 5.0ms** | **81.4%** | **$0** (Local/Deterministic) | **Yes (SHA-256)** |
| Zep (Graphiti) | 84.2% | 82.1% | 83.5% | 142.0ms | 45.0% | $149/mo | No |
| TencentDB Agent Memory | 76.0% | 74.3% | 75.1% | 38.2ms | 61.3% | $30/mo | No |
| Mem0 | 68.5% | 66.9% | 69.2% | 35.4ms | 42.1% | $99/mo | No |
| Letta / MemGPT | 71.2% | 70.5% | 72.0% | 68.5ms | 38.5% | $45/mo | No |

Reproduce locally:
```bash
python benchmarks/run_benchmark.py
```

---

## Architecture: The L0-L6 Memory Pyramid

Instead of an opaque vector pile, InfinityDB constructs a structured neurosymbolic memory hierarchy:

```
L0 RAW LOGS       -> Verbatim conversation & tool logs stored on disk/SQLite (offloaded from context)
L1 ATOM FACTS     -> Extracted atomic facts (names, roles, preferences, budgets, constraints) [<10ms]
L2 EPISODE GRAPH  -> Causal & temporal relationships ("X happened because Y") with Mermaid export
L3 PERSONA DNA    -> Dynamic user identity profile with automated conflict resolution
L4 POLICY ENGINE  -> Enterprise compliance rules, security policies, and tool SOPs
L5 FORGETTING     -> Ebbinghaus forgetting curve decay with spaced-repetition access reinforcement
L6 PROOF CHAIN    -> Cryptographic SHA-256 provenance linking every atom back to exact L0 raw text
```

---

## Installation

### Python
```bash
pip install infinitydb
```
Or for local development:
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

### Python

```python
from infinitydb import InfinityDB

# Initialize persistent memory database (uses SQLite WAL mode)
db = InfinityDB("agent_memory.db")

# 1. Ingest conversation (builds L0-L6 layers automatically)
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

# Later statement: user changes preference
db.add("I switched to Go for high-concurrency microservices.", user_id="sarah_c")
# Persona automatically resolves to the latest preference
print("Updated Persona:", db.get_persona("sarah_c")["preference"])

# 4. L6 Cryptographic Proof Verification
atom_id = context["results"][0]["id"]
proof = db.get_proof(atom_id)
print("Proof Hash:", proof["hash"])
print("Raw Snippet:", proof["raw_snippet"])

# 5. Enterprise Policies (L4)
db.add_policy("Do not expose internal VPC credentials in agent responses", domain="security", priority=2.0)

# 6. GDPR Compliance / Right-to-be-Forgotten
db.delete_user("sarah_c")
```

### Node.js / TypeScript

```javascript
const { InfinityDB } = require("infinitydb");

const db = new InfinityDB();

// Ingest memory
db.add(
  "I am Alex. I work as a Cloud Architect in London. I prefer PostgreSQL because of reliability.",
  "user_alex"
);

// Hybrid recall
const result = db.recall("what database does Alex use?", { userId: "user_alex" });
console.log(`Recall latency: ${result.latencyMs}ms`);
console.log("Results:", result.results);

// Inspect proof
const proof = db.getProof(result.results[0].id);
console.log("Verified provenance:", proof.verified, proof.hash);
```

---

## Interactive Local Dashboard

InfinityDB includes an interactive visual explorer:

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

## Roadmap

* **v1.0 (Current)**: Open-source L0-L6 Brain Engine, SQLite WAL mode, Python and Node.js SDKs, local interactive dashboard, comprehensive benchmark suite.
* **v1.1 (Upcoming)**: Direct Postgres + `pgvector` adapter for 100M+ memory scaling, Redis caching tier, MCP (Model Context Protocol) server for Claude / Cursor.
* **Cloud (Waitlist)**: Hosted multi-tenant managed clusters with enterprise SSO, RBAC, and SOC2 compliance. See [waitlist/README.md](waitlist/README.md).

---

## Verification & Testing

Run full test suite:
```bash
pytest
```

Run comparative benchmarks:
```bash
python benchmarks/run_benchmark.py
```

Run JavaScript test suite:
```bash
node infinitydb-js/test.js
```

---

## Contributing

Contributions are welcome. Please open an issue to discuss design proposals before submitting pull requests.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
