"""Regression tests for the honest, single-engine benchmark harness."""
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
