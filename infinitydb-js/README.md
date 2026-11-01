# InfinityDB

> High-Performance Neurosymbolic Agent Memory Engine for JavaScript and TypeScript.

[![NPM Version](https://img.shields.io/npm/v/infinitydb?color=blue)](https://www.npmjs.com/package/infinitydb)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)

InfinityDB gives AI agents long-term neurosymbolic memory with deterministic recall, sub-millisecond latency, dynamic persona conflict resolution, and cryptographic SHA-256 audit proofs.

---

## Installation

```bash
npm install infinitydb
```

```bash
yarn add infinitydb
# or
pnpm add infinitydb
```

---

## Quick Start

### 1. Ingestion and Causal Recall

```javascript
const { InfinityDB } = require("infinitydb");
// Or in TypeScript:
// import { InfinityDB } from "infinitydb";

const db = new InfinityDB();

// Ingest agent dialogue / user interactions (L0 Log -> L1 Atomic Facts -> L2 Episode Graph)
const logId = db.add(
  "Sarah prefers TypeScript over JavaScript because of strict static typing. Her project budget is $15,000.",
  "usr_sarah"
);

// Recall relevant memory with causal facts, policy checks, and proof chains
const memory = db.recall("what language does Sarah prefer?", { userId: "usr_sarah" });

console.log("Latency:", memory.latencyMs, "ms");
console.log("Recalled Facts:", memory.results);
console.log("Causal Graph:", memory.causalGraph);
```

### 2. Dynamic Persona & Automatic Conflict Resolution

```javascript
// Initial preference
db.add("I prefer TypeScript for backend APIs", "usr_sarah");
console.log(db.getPersona("usr_sarah")); 
// => { preference: ["TypeScript for backend APIs"] }

// User changes their mind later:
db.add("I now prefer Rust for mission-critical core engines", "usr_sarah");

// Conflicting older preference is automatically reconciled
console.log(db.getPersona("usr_sarah"));
// => { preference: ["Rust for mission-critical core engines"] }
```

### 3. Explicit temporal supersession and historical recall

When a user explicitly replaces a fact, preserve the original fact and its provenance while marking the newer fact as its successor:

```javascript
const oldLog = db.add("I prefer Python for data work.", "usr_sarah");
const oldFact = Array.from(db.atoms.values()).find(a => a.sourceId === oldLog);
const cutoff = oldFact.ts + 0.001;

const newLog = db.add("I prefer Rust for systems work.", "usr_sarah");
const newFact = Array.from(db.atoms.values()).find(a => a.sourceId === newLog);

db.supersedeFact(oldFact.id, newFact.id, "usr_sarah");

// Current recall returns the active fact; historical recall can still see the old one.
db.recall("Python", { userId: "usr_sarah" }).results; // []
db.recall("Python", { userId: "usr_sarah", asOf: cutoff }).results;
```

Supersession is explicit; the SDK does not infer semantic contradictions. Both facts and their proof records remain stored. Relations are tenant-scoped and cycle-checked. SQLite mode persists relations across reopen.

### 4. Cryptographic Audit Proofs (SHA-256 Chain)

Every atomic fact is hashed and anchored into an immutable tamper-evident proof chain:

```javascript
const atomId = memory.results[0].id;
const proof = db.getProof(atomId);

console.log(proof);
// {
//   atomId: "ATOM_...",
//   hash: "a4f8c2...",
//   verified: true,
//   timestamp: 1729100000000
// }
```

### 5. Enterprise Policies & Guardrails

```javascript
// Register a global policy or a tenant-scoped policy
db.addPolicy("Never log plaintext passwords or API keys", "security", 2.0);
db.addPolicy("Tenant-specific retention rule", "retention", 1.5, "usr_sarah");

const recall = db.recall("database credentials", { userId: "usr_sarah" });
console.log(recall.policies); // Global + usr_sarah policies only
```

### 6. Tenant data deletion

```javascript
// Remove this tenant’s stored logs, facts, episodes, and persona state
const result = db.deleteUser("usr_sarah");
console.log(result); // Per-layer deletion counts
```

This API reports database deletion results only; it does not certify legal or regulatory compliance.

### 7. Backwards-Compatible Key-Value Store

```javascript
db.put("session:123", "active");
const val = db.get("session:123");
db.delete("session:123");
```


### 8. Durable SQLite storage (Node.js)

Install the package dependencies, then provide a database path to enable durable storage:

```javascript
const { InfinityDB } = require("infinitydb");
const db = new InfinityDB({ dbPath: "./data/infinitydb.sqlite" });

db.add("I prefer SQLite for durable memory.", "user_123");
console.log(db.recall("SQLite", { userId: "user_123" }).results);

// Close cleanly before process shutdown. Reopen with the same dbPath to recover state.
db.close();
```

When `dbPath` is omitted, the SDK remains in-memory for backwards compatibility.
The SQLite-backed mode uses WAL, FULL synchronous commits, a busy timeout, and normalized
L0/L1/L2/L3/L4/L6 tables with transactional row upserts. Existing v1 JSON-snapshot databases
are migrated transactionally to schema v2 on first open. SDK instances refresh their view
from SQLite before public reads and mutations, and sequence allocation is serialized by a
SQLite write transaction. This is not yet a guarantee of serializable concurrent read-modify-write
operations across processes; use a single writer until that coordination layer is added.
Back up the database using SQLite's backup facilities.

---

## Architecture (L0 – L6 Brain)

| Layer | Component | Function |
| :--- | :--- | :--- |
| **L0** | Raw Interaction Store | Append-only raw logs with sub-millisecond writes |
| **L1** | Atomic Fact Extractor | Deterministic, zero-hallucination semantic extraction |
| **L2** | Episodic Graph | Causal edge tracking between related memory episodes |
| **L3** | Dynamic Persona | Identity state with automatic chronological conflict resolution |
| **L4** | Enterprise Policy Engine | Real-time policy and compliance injection |
| **L5** | Ebbinghaus Decay | Recency and frequency weighted memory decay |
| **L6** | SHA-256 Proof Chain | Cryptographically verifiable audit trail |

---

## TypeScript Support

InfinityDB includes full type definitions out of the box (`index.d.ts`). No `@types/infinitydb` required.

---

## Links

- **Documentation & Benchmarks**: [https://github.com/infinitydb-ai/infinitydb](https://github.com/infinitydb-ai/infinitydb)
- **Python Package (PyPI)**: [`pip install infinitydb-ai`](https://pypi.org/project/infinitydb-ai/)
- **License**: Apache 2.0
