"""Fast Windows-compatible sanity checks for the benchmark arena.

This intentionally avoids downloading embedding models or contacting APIs.
It validates the local dataset, the real InfinityDB adapter, and the scorer.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from arena import locomo  # noqa: E402
from arena.score import answer_in_context, evidence_hit, f1  # noqa: E402
from arena.systems.infinitydb_system import InfinityDBSystem  # noqa: E402

def main() -> int:
    dataset = locomo.load()
    assert len(dataset) == 10, f"expected 10 LoCoMo conversations, got {len(dataset)}"
    conversation = locomo.by_id(dataset, "conv-26")
    assert conversation.sessions and conversation.qa, "conv-26 must contain sessions and QA"
    assert f1("Paris", "Paris") == 1.0
    assert answer_in_context(["The trip was to Paris."], "Paris")
    assert evidence_hit(
        ["Speaker A: I visited Paris last summer."],
        ["Speaker A: I visited Paris last summer."],
    ) == 1.0

    with tempfile.TemporaryDirectory(prefix="infinitydb-arena-smoke-") as temp_dir:
        system = InfinityDBSystem(temp_dir, conversation.sample_id)
        try:
            system.reset()
            system.ingest(conversation.sessions[0])
            result = system.search(conversation.qa[0].question, 10)
            assert isinstance(result, list), "adapter search must return a list"
            assert all(isinstance(item, str) for item in result), "memories must be strings"
        finally:
            system.close()

    print("PASS: LoCoMo dataset parses (10 conversations)")
    print("PASS: retrieval scorer sanity checks")
    print("PASS: InfinityDB adapter reset/ingest/search/close")
    print("NOTE: mem0/hybrid model downloads and full runs are intentionally not part of smoke mode")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
