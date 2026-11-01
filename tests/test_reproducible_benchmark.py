"""Regression tests for the honest, single-engine benchmark harness."""
import json
from pathlib import Path

from benchmarks.reproducible import percentile, run


def test_nearest_rank_percentile() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 95) == 4.0


def test_reproducible_harness_reports_correctness(capsys) -> None:
    result = run(5)
    capsys.readouterr()
    assert result["iterations"] == 5
    assert result["correctness"] == {
        "tenant_a_recall": True,
        "tenant_b_isolated": True,
        "source_proof_verified": True,
    }
    assert result["recall_latency_ms"]["samples"] == 100
    assert result["storage"]["database_and_wal_bytes"] > 0
    assert "No third-party product comparison" in result["limitations"][-1]


def test_result_matches_shared_schema_contract() -> None:
    schema = json.loads((Path(__file__).parent.parent / "benchmarks" / "result-schema-v1.json").read_text())
    result = run(5)
    required = set(schema["required"])
    assert required <= result.keys()
    assert result["benchmark"] == schema["properties"]["benchmark"]["const"]
    assert result["schema_version"] == schema["properties"]["schema_version"]["const"]
    assert result["engine"] == "python"
    assert result["environment"]["runtime"]["name"] == "python"
    assert set(result["ingestion"]) == {"elapsed_seconds", "operations_per_second"}
    assert set(result["recall_latency_ms"]) == {"p50", "p95", "p99", "samples"}
