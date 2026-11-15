"""Portable, tamper-evident audit of what an agent remembered.

Every other memory layer gives you rows. This gives you a **bundle** you can hand
to someone who does not trust you, and a verifier they can run offline, against
nothing but the bundle:

    bundle = export_bundle(db, user_id="alice")
    verify_bundle(bundle)              # -> {"ok": True, ...}

What a bundle proves depends entirely on one habit, so it is stated first and
again in `docs/AUDIT.md`:

* **With an anchor** — the bundle digest recorded somewhere the operator cannot
  rewrite (a git commit, an append-only log, an email to a client) — a later
  bundle that no longer matches the anchor proves the record was edited or
  deleted. That is the property that matters for audits.
* **Without an anchor** — a bundle only proves internal consistency: every fact
  matches the source snippet it was extracted from, nothing was reordered, and
  nothing was dropped since the chain was built. An operator who rewrites history
  *before* the first export produces a perfectly valid bundle. No design fixes
  that without an external anchor, and claiming otherwise would be dishonest.

Each record's `hash` is the engine's own L6 digest, `sha256(fact + source_log_id)`,
which binds a fact to the log it came from. The chain is
`sha256(previous_chain + record_digest)`, so reordering, removal or an edited
field all change `chain_head`.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from infinitydb.core.proof import verify_proof

SCHEMA = "infinitydb-audit/1"


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _record_digest(record: dict[str, Any]) -> str:
    """Digest of the fields a verifier re-derives. Chain fields are excluded."""
    return _sha256({key: record[key] for key in
                    ("atom_id", "fact", "ts", "hash", "source_log_id",
                     "char_start", "char_end", "source_sha256")})


def export_bundle(db, user_id: str = "default", include_text: bool = True) -> dict[str, Any]:
    """Export one tenant's memory as a bundle a third party can verify.

    `include_text=False` keeps the digests but drops the facts themselves, for
    handing over an integrity record without disclosing content.
    """
    records: list[dict[str, Any]] = []
    chain = "0" * 64
    for row in db.audit_records(user_id):
        raw = row.get("raw") or ""
        proof_ok = bool(row.get("hash")) and verify_proof(
            row["hash"], row["fact"], row["source_log_id"], raw,
            row["char_start"], row["char_end"],
        )
        record = {
            "atom_id": row["atom_id"],
            "fact": row["fact"] if include_text else None,
            "fact_sha256": _sha256(row["fact"]),
            "ts": row["ts"],
            "hash": row.get("hash") or "",
            "source_log_id": row.get("source_log_id") or "",
            "char_start": row.get("char_start"),
            "char_end": row.get("char_end"),
            "source_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "proof_ok": proof_ok,
        }
        chain = hashlib.sha256((chain + _record_digest(record)).encode("utf-8")).hexdigest()
        record["chain"] = chain
        records.append(record)

    body = {"schema": SCHEMA, "user_id": user_id, "records": records}
    return {
        **body,
        "created_at": round(time.time(), 3),
        "count": len(records),
        "chain_head": chain,
        "digest": _sha256(body),
        "text_included": include_text,
    }


def verify_bundle(bundle: dict[str, Any], anchor: str | None = None) -> dict[str, Any]:
    """Re-derive every digest and the chain. Nothing here trusts the bundle."""
    failures: list[dict[str, Any]] = []
    if bundle.get("schema") != SCHEMA:
        return {"ok": False, "checked": 0, "failures": [
            {"reason": "schema", "detail": f"expected {SCHEMA}, got {bundle.get('schema')!r}"}]}

    records = bundle.get("records") or []
    if bundle.get("count") != len(records):
        failures.append({"reason": "count", "detail":
                         f"header says {bundle.get('count')}, bundle has {len(records)} records"})

    chain = "0" * 64
    for index, record in enumerate(records):
        fact = record.get("fact")
        if fact is None:
            if record.get("fact_sha256") is None:
                failures.append({"reason": "missing_fact", "index": index})
            # Text was withheld at export: the digest still binds the chain.
        elif _sha256(fact) != record.get("fact_sha256"):
            failures.append({"reason": "fact_edited", "index": index,
                             "atom_id": record.get("atom_id")})

        expected_hash = hashlib.sha256(
            f"{fact if fact is not None else ''}{record.get('source_log_id', '')}".encode("utf-8")
        ).hexdigest()
        if fact is not None and record.get("hash") and expected_hash != record["hash"]:
            failures.append({"reason": "proof_hash_mismatch", "index": index,
                             "atom_id": record.get("atom_id")})

        try:
            digest = _record_digest(record)
        except KeyError as missing:
            failures.append({"reason": "field_missing", "index": index,
                             "detail": str(missing)})
            continue
        chain = hashlib.sha256((chain + digest).encode("utf-8")).hexdigest()
        if record.get("chain") != chain:
            failures.append({"reason": "chain_broken", "index": index,
                             "atom_id": record.get("atom_id")})
            chain = record.get("chain") or chain  # keep going to report more than one

    body = {"schema": bundle.get("schema"), "user_id": bundle.get("user_id"),
            "records": records}
    recomputed = _sha256(body)
    if recomputed != bundle.get("digest"):
        failures.append({"reason": "bundle_digest_mismatch",
                         "detail": f"recomputed {recomputed[:16]}…"})
    if chain != bundle.get("chain_head"):
        failures.append({"reason": "chain_head_mismatch"})

    anchored = None
    if anchor is not None:
        anchored = (anchor == recomputed)
        if not anchored:
            failures.append({"reason": "anchor_mismatch",
                             "detail": "this bundle is not the one that was anchored"})

    return {
        "ok": not failures,
        "checked": len(records),
        "failures": failures,
        "digest": recomputed,
        "anchor_checked": anchored,
    }


def erase_with_receipt(db, user_id: str = "default", confirm: bool = False) -> dict[str, Any]:
    """Erase a tenant and return a receipt naming exactly what was removed.

    The receipt carries the digest of the pre-erasure bundle. Holding that digest
    separately is what lets someone check later that the erased facts are really
    gone and that nothing else changed.
    """
    if not confirm:
        raise PermissionError("erase_with_receipt requires confirm=True")
    before = export_bundle(db, user_id=user_id)
    erased = db.delete_user(user_id)
    after = export_bundle(db, user_id=user_id)
    receipt = {
        "schema": "infinitydb-erasure/1",
        "user_id": user_id,
        "erased_at": round(time.time(), 3),
        "bundle_digest_before": before["digest"],
        "bundle_digest_after": after["digest"],
        "chain_head_before": before["chain_head"],
        "atoms_erased": before["count"],
        "atoms_remaining": after["count"],
        "layers": erased,
    }
    receipt["digest"] = _sha256(receipt)
    return receipt


def verify_receipt(receipt: dict[str, Any]) -> bool:
    """Check a receipt's own digest; it is the operator's signed-off claim."""
    payload = {key: value for key, value in receipt.items() if key != "digest"}
    return _sha256(payload) == receipt.get("digest")
