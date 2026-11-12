#!/usr/bin/env bash
# Reproduce the LoCoMo memory arena on one machine.
#
#   bash benchmarks/arena/run.sh                    # all systems, all conversations
#   bash benchmarks/arena/run.sh infinitydb         # one system only
#   bash benchmarks/arena/run.sh "infinitydb mem0"  # a subset
#
# InfinityDB runs on the current interpreter (it has no third-party
# dependencies). mem0 and the free hybrid baseline are installed into
# .arena/venvs/mem0 from the pinned requirements files, because a competitor
# must run with its own pinned stack.
set -euo pipefail

cd "$(dirname "$0")/../.."

CONVS="conv-26 conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50"
SYSTEMS="${1:-infinitydb mem0 hybrid}"
MEM0_VENV=".arena/venvs/mem0"

if [[ " $SYSTEMS " == *" mem0 "* && ! -x "$MEM0_VENV/bin/python" ]]; then
  echo "creating mem0 virtualenv"
  python3 -m venv "$MEM0_VENV"
  "$MEM0_VENV/bin/pip" install -r benchmarks/arena/requirements-mem0.txt
fi

# The hybrid baseline shares that virtualenv: it is FastEmbed + numpy, which
# mem0 already pins, and it must not leak into InfinityDB's interpreter.
if [[ " $SYSTEMS " == *" hybrid "* && ! -x "$MEM0_VENV/bin/python" ]]; then
  echo "creating hybrid virtualenv"
  python3 -m venv "$MEM0_VENV"
  "$MEM0_VENV/bin/pip" install -r benchmarks/arena/requirements-hybrid.txt
fi

for system in $SYSTEMS; do
  if [[ "$system" == "mem0" || "$system" == "hybrid" ]]; then
    PY="$MEM0_VENV/bin/python"
  else
    PY="python3"
  fi
  for conv in $CONVS; do
    "$PY" benchmarks/arena/worker.py \
      --system "$system" --conv "$conv" \
      --out "benchmarks/arena/results/$system/$conv.jsonl" \
      --workdir ".arena/$system/$conv" --force
  done
done

# Retrieval metrics need no LLM. Add --reader only where a local Ollama is
# serving the reader model (see score.py) to also measure end-to-end QA.
python3 benchmarks/arena/score.py --systems "${SYSTEMS// /,}" --no-reader
