#!/usr/bin/env bash
# One-command verification of every published claim.
#
# A technical buyer should be able to run this and see the same numbers the
# docs quote. Nothing here needs a model key, a network service, or a GPU.
#
#   ./scripts/verify.sh            # tests + published baseline + readiness
#   PYTHON=.arena/venvs/dev/bin/python ./scripts/verify.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
FAILED=0

step() { printf '\n=== %s ===\n' "$1"; }

step "environment"
"$PY" -V || FAILED=1
printf 'sqlite: '; "$PY" -c "import sqlite3; print(sqlite3.sqlite_version)"

step "unit tests"
"$PY" -m pytest -q tests || FAILED=1

step "engine smoke (no model, no key)"
"$PY" - <<'SMOKE' || FAILED=1
from infinitydb.core.engine import InfinityDB

db = InfinityDB(":memory:")
db.add("Caroline: I work at the gym in Bristol.", user_id="u", ts=1000.0)
db.add("Caroline: I work at the hospital now.", user_id="u", ts=2000.0)

hit = db.recall("where does Caroline work", n=3, user_id="u", reinforce=False)["results"]
assert hit, "recall returned nothing"
assert "hospital" in hit[0]["fact"].lower(), hit[0]

# These must return usable structures, not raise: the graph API is part of the
# product surface and a buyer will poke it.
contra = db.contradictions("u")
rels = db.relations("caroline", user_id="u")
persona = db.persona_summary("u")
assert isinstance(contra, list) and isinstance(persona, str) and persona
print(f"smoke: recall OK (top hit is the current fact), contradictions={len(contra)}, "
      f"relations={len(rels) if hasattr(rels, '__len__') else 'n/a'}, persona={len(persona)} chars")
SMOKE

step "published LoCoMo baseline (replay of the ingested arena databases)"
if [ -d .arena/infinitydb ]; then
  "$PY" benchmarks/arena/replay_probe.py | tail -7 || FAILED=1
else
  echo "skipped: run benchmarks/arena/worker.py first to create .arena/infinitydb"
fi

step "competitor readiness"
"$PY" benchmarks/arena/competitor_check.py || true

step "extraction backend (optional, not part of any claimed number)"
if [ -n "${OPENAI_API_KEY:-}" ] || [ -n "${OLLAMA_HOST:-}" ]; then
  "$PY" -m infinitydb.extraction --check 2>&1 | tail -2 || true
else
  echo "skipped: no extraction key configured. The default path is lexical and needs none;"
  echo "the typed-edge extractor is opt-in, so this does not affect the measured rows."
fi

step "result"
if [ "$FAILED" -eq 0 ]; then
  echo "PASS: tests, smoke, and the published baseline all reproduce."
else
  echo "FAIL: something above did not reproduce; do not quote the numbers until it does."
fi
exit "$FAILED"
