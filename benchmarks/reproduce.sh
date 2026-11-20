#!/usr/bin/env bash
# Reproduce InfinityDB's published benchmark numbers from a clean checkout.
#
#   ./benchmarks/reproduce.sh            # full suites (~35 min on 8 cores)
#   ./benchmarks/reproduce.sh --smoke    # 1 LoCoMo conversation + 6 questions
#
# The script verifies what it is running (commit, Python, CPU count, zero
# runtime dependencies), downloads the two datasets if missing and refuses to
# continue when a SHA-256 does not match the pin, then runs the same harnesses
# that produced the committed artifacts and prints a scorecard comparing the
# new run against those baselines.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SUITE="full"
BACKEND="sqlite"
OUT="$REPO_ROOT/benchmarks/.cache/repro"
DATA_DIR="$REPO_ROOT/benchmarks/data"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke|--full)
            SUITE="${1#--}"
            shift
            ;;
        --backend)
            BACKEND="${2:-}"
            shift 2
            ;;
        --out)
            OUT="${2:-}"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="${2:-}"
            shift 2
            ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "$BACKEND" != "sqlite" && "$BACKEND" != "postgres" ]]; then
    echo "--backend must be sqlite or postgres" >&2
    exit 2
fi
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "python3 not found; set PYTHON=/path/to/python" >&2
    exit 2
fi

mkdir -p "$OUT" "$DATA_DIR"

COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
PY_VERSION="$("$PYTHON" -c 'import platform; print(platform.python_version())')"
CPU_COUNT="$("$PYTHON" -c 'import os; print(os.cpu_count() or 1)')"

echo "=== InfinityDB benchmark reproduction ==="
echo "commit:  $COMMIT"
echo "python:  $PY_VERSION"
echo "cpus:    $CPU_COUNT"
echo "suite:   $SUITE"
echo "backend: $BACKEND"
echo

# --- Preconditions ---------------------------------------------------------

if ! grep -qE '^dependencies = \[\]' pyproject.toml; then
    echo "ERROR: pyproject.toml no longer declares 'dependencies = []'." >&2
    echo "The dependency-free default path is part of what is being reproduced." >&2
    exit 1
fi
echo "OK: dependencies = []"

"$PYTHON" - "$COMMIT" "$PY_VERSION" "$CPU_COUNT" "$SUITE" "$BACKEND" "$OUT" <<'PY'
import json, sys
commit, version, cpus, suite, backend, out = sys.argv[1:7]
json.dump(
    {
        "commit": commit,
        "python": version,
        "cpus": int(cpus),
        "suite": suite,
        "backend": backend,
    },
    open(f"{out}/environment.json", "w"),
    indent=2,
)
PY

# --- Datasets --------------------------------------------------------------

LME_URL="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
LME_SHA256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
LOCOMO_URL="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_SHA256="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"

fetch() {
    local url="$1" dest="$2" want="$3" name="$4"
    if [[ -f "$dest" ]]; then
        local got
        got="$(sha256sum "$dest" | cut -d' ' -f1)"
        if [[ "$got" == "$want" ]]; then
            echo "OK: $name (cached, sha256 verified)"
            return 0
        fi
        echo "checksum mismatch for cached $name: $got != $want" >&2
        exit 1
    fi
    echo "downloading $name ..."
    if ! curl -fsSL --retry 3 --retry-delay 2 -o "$dest.part" "$url"; then
        echo "download failed: $url" >&2
        rm -f "$dest.part"
        exit 1
    fi
    local got
    got="$(sha256sum "$dest.part" | cut -d' ' -f1)"
    if [[ "$got" != "$want" ]]; then
        echo "checksum mismatch for $name: $got != $want" >&2
        rm -f "$dest.part"
        exit 1
    fi
    mv "$dest.part" "$dest"
    echo "OK: $name (downloaded, sha256 verified)"
}

fetch "$LME_URL" "$DATA_DIR/longmemeval_s_cleaned.json" "$LME_SHA256" "longmemeval_s_cleaned.json"
mkdir -p "$REPO_ROOT/benchmarks/.cache"
fetch "$LOCOMO_URL" "$REPO_ROOT/benchmarks/.cache/locomo10.json" "$LOCOMO_SHA256" "locomo10.json"
echo

# --- Suites ----------------------------------------------------------------

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
DSN_ARGS=()
if [[ "$BACKEND" == "postgres" ]]; then
    if [[ -z "${INFINITYDB_DSN:-}" ]]; then
        echo "ERROR: --backend postgres needs INFINITYDB_DSN" >&2
        exit 2
    fi
    DSN_ARGS=(--dsn "$INFINITYDB_DSN")
fi

STARTED="$(date +%s)"

if [[ "$SUITE" == "smoke" ]]; then
    echo "--- LoCoMo (smoke: 1 conversation, 20 questions, unbudgeted)"
    "$PYTHON" benchmarks/locomo_local_compare.py \
        --engines infinitydb --max-convos 1 --max-qa 20 --top-k 10 \
        --output "$OUT/locomo-smoke.json"
    echo
    if [[ "$BACKEND" == "postgres" ]]; then
        echo "--- LongMemEval-S (smoke: 1 question per ability, postgres)"
    else
        echo "--- LongMemEval-S (smoke: 1 question per ability)"
    fi
    "$PYTHON" benchmarks/longmemeval_bench.py \
        --data-path "$DATA_DIR/longmemeval_s_cleaned.json" \
        --sample 1 --budget-tokens 600 "${DSN_ARGS[@]}" \
        --output "$OUT/longmemeval-smoke.json"
    LOCOMO_ARGS_FOR_SCORECARD=(--locomo "$OUT/locomo-smoke.json" --locomo-engine infinitydb)
    LME_JSON="$OUT/longmemeval-smoke.json"
else
    if [[ "$BACKEND" == "postgres" ]]; then
        echo "--- LoCoMo skipped: benchmarks/locomo_local_compare.py has no PostgreSQL mode"
        LOCOMO_ARGS_FOR_SCORECARD=(--locomo-skip)
    else
        echo "--- LoCoMo (4 conversations, 584 questions, k=10, budget 420)"
        "$PYTHON" benchmarks/locomo_local_compare.py \
            --engines infinitydb --max-convos 4 --max-qa 500 --top-k 10 \
            --budget-tokens 420 --budget-only \
            --output "$OUT/locomo-budget-420.json"
        LOCOMO_ARGS_FOR_SCORECARD=(--locomo "$OUT/locomo-budget-420.json" --locomo-engine infinitydb_budget_420)
    fi
    echo
    echo "--- LongMemEval-S (500 questions, k=10, budget 600)"
    "$PYTHON" benchmarks/longmemeval_bench.py \
        --data-path "$DATA_DIR/longmemeval_s_cleaned.json" \
        --budget-tokens 600 "${DSN_ARGS[@]}" \
        --output "$OUT/longmemeval-500.json"
    LME_JSON="$OUT/longmemeval-500.json"
fi

ELAPSED="$(( $(date +%s) - STARTED ))"
echo
echo "--- Scorecard"
"$PYTHON" benchmarks/scorecard.py \
    --lme "$LME_JSON" \
    --baseline-lme benchmarks/longmemeval-s-calibrated-results.json \
    "${LOCOMO_ARGS_FOR_SCORECARD[@]}" \
    --suite "$SUITE" \
    --elapsed "$ELAPSED" \
    --json "$OUT/scorecard.json"
echo
echo "scorecard: $OUT/scorecard.json"
