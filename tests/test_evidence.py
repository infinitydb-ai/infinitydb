"""The self-audit command is the contract with a skeptic; test what it reports."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from infinitydb import evidence

ROOT = Path(__file__).resolve().parents[1]


def test_limits_are_stated_and_include_the_multi_hop_failure():
    areas = {area for area, _ in evidence.LIMITS}
    assert "multi-hop retrieval" in areas, "the weakest result must be in the list"
    multi = dict(evidence.LIMITS)["multi-hop retrieval"]
    assert "0.5714" in multi and "loses" in multi, "the limit must name the number that beat us"
    assert "independent reproduction" in areas


def test_cli_reports_every_check_and_exits_zero():
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(ROOT)}
    proc = subprocess.run([sys.executable, "-m", "infinitydb.evidence", "--json"],
                          capture_output=True, text=True, timeout=300, env=env, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["failed"] == 0 and report["passed"] >= 5
    names = {row["check"] for row in report["results"]}
    assert {"demo", "audit bundle", "migration", "mcp transport", "recall latency"} <= names
    assert all(row["detail"] for row in report["results"])


def test_a_crashing_check_is_reported_not_raised(monkeypatch, capsys):
    def boom():
        raise RuntimeError("provider down")

    monkeypatch.setattr(evidence, "CHECKS", [("fake", boom)])
    assert evidence.main([]) == 1, "a failing check must fail the command"
    out = capsys.readouterr().out
    assert "FAIL" in out and "provider down" in out
