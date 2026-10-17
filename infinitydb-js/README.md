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

### 3. Cryptographic Audit Proofs (SHA-256 Chain)

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

### 4. Enterprise Policies & Guardrails

```javascript
// Register policies that automatically govern recalls
db.addPolicy("Never log plaintext passwords or API keys", "security", 2.0);

const recall = db.recall("database credentials", { userId: "usr_sarah" });
console.log(recall.policies); // Injected active governance policies
```

### 5. GDPR Right-to-be-Forgotten Compliance

```javascript
// Instantly and irreversibly wipe all user logs, facts, and persona state
const result = db.deleteUser("usr_sarah");
console.log(result.gdprCompliant); // true
```

### 6. Backwards-Compatible Key-Value Store

```javascript
db.put("session:123", "active");
const val = db.get("session:123");
db.delete("session:123");
```

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
