"""Cryptographic provenance verification for InfinityDB atoms.

A proof is valid only when the stored atom hash matches its source log and
the source character range still contains the exact extracted fact.
"""
from __future__ import annotations

import hashlib


def verify_proof(
    hash_str: str,
    fact: str,
    source_log_id: str,
    raw_text: str,
    char_start: int,
    char_end: int,
) -> bool:
    """Verify both the atom digest and its exact source-text range."""
    if not isinstance(char_start, int) or not isinstance(char_end, int):
        return False
    if char_start < 0 or char_end < char_start or char_end > len(raw_text):
        return False
    source_fact = raw_text[char_start:char_end]
    if source_fact != fact:
        return False
    expected = hashlib.sha256((fact + source_log_id).encode("utf-8")).hexdigest()
    return hashlib.compare_digest(expected, hash_str)
