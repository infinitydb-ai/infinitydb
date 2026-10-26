# Open Agent Memory Specification (OAMS-1.0)

**Standard Identifier:** `urn:agent-memory:spec:1.0`  
**License:** Apache License 2.0  
**Status:** Living Standard  
**Maintainer:** InfinityDB Open Source Initiative (`infinitydb-ai/infinitydb`)

---

## 1. Abstract

The Open Agent Memory Specification (OAMS-1.0) defines a unified, deterministic, and verifiable architecture for autonomous agent memory systems. Existing agent memory systems suffer from three fundamental deficiencies:
1. **Network & Extraction Latency:** Remote vector database lookups and LLM-based fact extraction introduce 150ms to 500ms of latency per conversational turn.
2. **Economic Inefficiency:** Invoking frontier LLMs to extract atomic facts burns context tokens, generating substantial operating costs ($0.02 to $0.05 per user interaction).
3. **Absence of Provenance:** Pure high-dimensional vector embeddings lack deterministic audit trails, leading to unresolvable persona contradictions and compliance violations.

OAMS-1.0 resolves these limitations by establishing a six-tier neurosymbolic memory hierarchy (L0 through L6) operating on local storage engines in Write-Ahead Logging (WAL) mode.

---

## 2. The L0–L6 Architecture

An implementation conforming to OAMS-1.0 MUST provide the following hierarchical tiers:

```
+-------------------------------------------------------------------+
| L6: Cryptographic Proof Chain (SHA-256 Provenance & Tamper-Proof) |
+-------------------------------------------------------------------+
| L5: Temporal Retention Engine (Ebbinghaus Forgetting Curve)       |
+-------------------------------------------------------------------+
| L4: Enterprise Policy Injection (SOPs & Safety Guardrails)        |
+-------------------------------------------------------------------+
| L3: Dynamic Persona DNA (Conflict Auto-Resolution)                |
+-------------------------------------------------------------------+
| L2: Causal Episode Graph (Event Dependencies & Edge Transitions)  |
+-------------------------------------------------------------------+
| L1: Atomic Fact Extraction (Deterministic Symbolic Slot-Filling)  |
+-------------------------------------------------------------------+
| L0: Raw Conversation & Tool Transcripts (Serialized Ingestion)    |
+-------------------------------------------------------------------+
```

### 2.1. Layer 0: Raw Ingestion Log (L0)
* **Definition:** An append-only log storing verbatim conversation turns, tool calls, and environment observations.
* **Requirement:** Each record must contain a unique UUID, an ISO-8601 UTC timestamp, the actor identifier, and the uncompressed raw text.

### 2.2. Layer 1: Deterministic Atomic Fact Extraction (L1)
* **Definition:** Symbolic extraction of discrete predicate-object tuples from dialogue transcripts without invoking external LLM inference endpoints.
* **Performance Guarantee:** Fact parsing MUST complete in less than 10 milliseconds per dialogue turn.
* **Cost Guarantee:** Zero external token expenditure ($0.00).

### 2.3. Layer 2: Causal Episode Graph (L2)
* **Definition:** Directed acyclic graph (DAG) capturing causal, temporal, and conditional relationships between agent decisions and real-world outcomes.
* **Schema:** `(source_event_id, target_event_id, relation_type, confidence)` where relation types include `CAUSES`, `PRECEDES`, `MODIFIES`, and `INVALIDATES`.

### 2.4. Layer 3: Dynamic Persona DNA (L3)
* **Definition:** Structured state machine tracking user preferences, behavioral traits, and operating environment constraints.
* **Conflict Auto-Resolution:** When conflicting assertions occur (e.g., preference shifting from language A to language B), the system MUST evaluate timestamp recency and confidence to deterministically supersede stale data.

### 2.5. Layer 4: Enterprise Policy Injection (L4)
* **Definition:** Rule-based compliance gate that enforces standard operating procedures (SOPs), privacy guardrails (GDPR/HIPAA), and organizational restrictions prior to context compilation.
* **Precedence:** L4 policies take absolute priority over user-generated L1 facts.

### 2.6. Layer 5: Temporal Retention Engine (L5)
* **Definition:** Implementation of the Ebbinghaus Forgetting Curve to model retention decay and spaced repetition reinforcement.
* **Mathematical Formulation:**

$$R(t) = \exp\left(-\frac{\Delta t}{S}\right)$$

Where:
* $R(t)$ is the retention probability score $[0.0, 1.0]$.
* $\Delta t$ is the elapsed time in days since the memory was last accessed or updated.
* $S$ is memory stability, defined as:

$$S = 1.0 + (3.0 \times \text{importance}) + \ln(1 + \text{access\_count})$$

Every successful memory recall MUST increment the `access_count`, actively increasing memory stability $S$ over time.

### 2.7. Layer 6: Cryptographic Proof Chain (L6)
* **Definition:** Cryptographic verification linking extracted L1 facts directly to exact character offsets within L0 raw transcripts.
* **Hash Standard:** SHA-256.
* **Provenance Tuple:** `(atom_id, sha256_hash, source_log_id, character_range, timestamp)`.

---

## 3. Core Interface Specification

Conforming implementations must expose the following three core functions:

### 3.1. Store Operation (`add`)
```python
def add(
    content: str,
    user_id: str = "default",
    session_id: str | None = None,
    importance: float = 0.9
) -> str
```
* **Inputs:** Raw dialogue text or tool output, user identifier, optional session key, importance float $[0.0, 1.0]$.
* **Output:** Deterministic log identifier.
* **Side Effects:** Atomizes facts into L1, updates L2 causal links, updates L3 persona, and computes L6 cryptographic proof.

### 3.2. Recall Operation (`search` / `recall`)
```python
def recall(
    query: str,
    n: int = 5,
    user_id: str | None = None,
    include_policies: bool = True
) -> dict[str, Any]
```
* **Latency Guarantee:** Sub-5.0ms p95 latency on local SSD storage.
* **Output Structure:** Must return matched facts, active causal links, governing enterprise policies, latency in milliseconds, and token savings ratio.

### 3.3. Right-to-be-Forgotten (`delete_user`)
```python
def delete_user(user_id: str) -> dict[str, int]
```
* **Requirement:** Cryptographically purge all L0 logs, L1 facts, L2 graph edges, L3 personas, and L6 proofs associated with the given user identifier in compliance with GDPR Article 17.

---

## 4. Reference Implementation

The canonical reference implementation of OAMS-1.0 is maintained by the InfinityDB project:
* **Python Distribution:** `pip install infinitydb-ai` (Package: `infinitydb-ai`)
* **Node.js Distribution:** `npm install infinitydb` (Package: `infinitydb`)
* **Source Repository:** `https://github.com/infinitydb-ai/infinitydb`

---

## 5. Security & Privacy Considerations

1. **Local-First Isolation:** Conforming storage engines MUST support fully embedded operation without transmitting unencrypted conversational state across external networks.
2. **Auditability:** Every recalled fact must be verifiable against raw source input via L6 SHA-256 hashes.
3. **Data Erasure:** Deletion operations must be atomic and non-recoverable across all indexed tables.
