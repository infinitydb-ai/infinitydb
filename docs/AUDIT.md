# Verifiable memory: audit bundles and erasure receipts

Most memory layers answer "what does the agent remember?" Some answer "when did it
learn that?" None of them hand you something you can give to a **third party who
does not trust you** — a regulator, a client, an opposing lawyer — and have them
check it themselves, offline, without your database.

That is what this is:

```python
from infinitydb import InfinityDB, audit

db = InfinityDB("memory.db")
bundle = audit.export_bundle(db, user_id="alice")   # JSON, self-contained
audit.verify_bundle(bundle)                          # -> {"ok": True, "checked": 42, ...}
```

Over MCP the same thing is two tools: `audit_export` and `audit_verify`, plus
`erase_with_receipt` for deletion evidence.

## From the shell

`pip install infinitydb-ai` installs `infinitydb-audit`; `python -m infinitydb.audit`
is the same thing. No Python needed to run or check an export.

```bash
# 1. export — prints the digest you should anchor
infinitydb-audit export --db memory.db --user alice --out alice-audit.json
#    digest:     dff5faf5f17f77fc…
#    chain_head: e01d5f6f14e163f1…

# 2. hand the file over. They verify it on their machine, offline:
infinitydb-audit verify alice-audit.json

# 3. months later, prove nothing changed since you anchored it:
infinitydb-audit verify alice-audit.json --anchor dff5faf5f17f77fc…

# deletion evidence
infinitydb-audit receipt --db memory.db --user alice --yes --out receipt.json
infinitydb-audit verify-receipt receipt.json
```

Exit codes are the contract, so a shell script can branch on them without parsing
text: **0** verified, **1** verification failed, **2** unusable input. `--json`
prints the full failure report; `--no-text` exports digests without the facts;
`receipt` refuses to run without `--yes`.

## What is in a bundle

```json
{
  "schema": "infinitydb-audit/1",
  "user_id": "alice",
  "created_at": 1763000000.0,
  "count": 42,
  "records": [
    {
      "atom_id": "A1f…", "fact": "I moved to Lisbon in March 2024.",
      "fact_sha256": "9c1…", "ts": 1709251200.0,
      "hash": "b7e…", "source_log_id": "L0c…",
      "char_start": 0, "char_end": 36,
      "source_sha256": "3ad…", "proof_ok": true,
      "chain": "e41…"
    }
  ],
  "chain_head": "e41…",
  "digest": "5f2…",
  "text_included": true
}
```

Three bindings, each doing a different job:

| Field | Binds | Catches |
|---|---|---|
| `hash` | the fact to the log it was extracted from (`sha256(fact + source_log_id)`) | a fact swapped for a different one |
| `source_sha256` | the exported record to the original source text | the source passage being rewritten |
| `chain` / `chain_head` | each record to all previous ones | reordering, removal, insertion |

`include_text=False` exports the digests without the facts, for handing over an
integrity record without disclosing content.

## What it proves — and what it does not

This distinction decides whether the feature is useful to you, so it is stated
plainly rather than buried.

| Situation | Guarantee |
|---|---|
| You keep the bundle `digest` somewhere the operator cannot rewrite (a git commit, an append-only log, an email to the client) and compare later | **A later bundle that does not match proves the record was edited or deleted.** This is the property that survives an audit. |
| You verify a bundle with no earlier anchor | Internal consistency only: every fact matches its source snippet, nothing was reordered, nothing was removed *since the bundle was built*. |
| The operator rewrote history *before* the first export | **Undetectable.** No export format can fix this; only an external anchor recorded at the time can. |

So the honest headline is: **an anchored bundle is deletion- and edit-evident; an
unanchored one is self-consistent.** Claiming more than that would be the sort of
overclaim this repository exists to avoid.

Also out of scope, deliberately:

- The bundle is **not signed**. There is no key material, no PKI, no "verified by
  InfinityDB" claim. Signing adds a trust root that then has to be managed, and
  the anchor comparison already gives you the property that matters.
- The store is **not encrypted at rest** by this feature. File permissions remain
  the only access control; see the limits in [`MCP_SERVER.md`](MCP_SERVER.md).
- `proof_ok: false` means the *source text no longer matches* the stored proof —
  for example because a row was edited directly in SQLite. It is reported per
  record and makes `verify_bundle` fail, but it is not silently corrected.

## Erasure receipts

```python
receipt = audit.erase_with_receipt(db, user_id="alice", confirm=True)
audit.verify_receipt(receipt)   # -> True
```

The receipt records the pre-erasure bundle digest, the post-erasure digest, the
atom count removed, and per-layer deletion counts from `delete_user`. Combined with
an anchored bundle from earlier, someone can check three things without access to
your database: the facts that were there are gone, nothing else changed, and the
receipt itself was not edited after the fact.

`erase_with_receipt` refuses without `confirm=True` — the same guard the MCP
`forget` tool uses.

## Measured cost

`python scripts/measure_audit.py` — raw output in [`audit_cost.json`](audit_cost.json).
Linux 6.1 x86_64, Python 3.12.3, one machine:

| Facts | Export | Verify | Bundle | Bytes/fact |
|---|---|---|---|---|
| 100 | 3.2 ms | 2.2 ms | 52 KB | 537 |
| 1,000 | 24.3 ms | 17.3 ms | 522 KB | 535 |
| 5,000 | 125.6 ms | 91.0 ms | 2.6 MB | 536 |

Both steps are linear in fact count, which is the property that matters: a tenant
with 5,000 facts exports and verifies in about a fifth of a second. Verification
re-derives every digest from the bundle alone — it never touches the database, so
a third party runs it on their own machine with `pip install infinitydb-ai` and
nothing else.

## When to use it

- **Regulated deployments** where someone will ask "prove what the assistant knew
  on the 3rd, and prove you deleted it when asked." Anchor the digest on every
  export; keep the anchors outside the operator's control.
- **Disputes**: export the bundle at the moment of the dispute and email the
  digest to the counterparty. From then on, any change is detectable.
- **Data-deletion requests**: anchor a bundle, erase, hand over the receipt.
- **Not** for proving the agent behaved correctly. This records what was stored,
  not what was said to the user.
