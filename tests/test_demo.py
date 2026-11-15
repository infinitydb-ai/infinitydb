"""The demo is the first thing a new user runs, so it is tested like an API."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from infinitydb import demo

ROOT = Path(__file__).resolve().parents[1]


def test_demo_recalls_the_facts_it_wrote(capsys):
    assert demo.main([]) == 0
    out = capsys.readouterr().out
    assert "Lisbon" in out, "the location must be recalled"
    assert "data engineer" in out, "the extracted job fact must be recalled"
    assert "forty thousand euros" in out or "40,000" in out
    assert "stored  facts=" in out


def test_demo_accepts_a_custom_query(capsys):
    assert demo.main(["--query", "where does the user live", "-n", "1"]) == 0
    out = capsys.readouterr().out
    assert "where does the user live" in out
    assert "Lisbon" in out


def test_demo_runs_as_a_module_without_network():
    """`python -m infinitydb.demo` is the documented command; keep it working.

    `-S` drops site-packages so an accidental third-party import fails loudly,
    and an unreachable proxy proves nothing dials out.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(ROOT),
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
        "NO_PROXY": "",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "infinitydb.demo"],
        capture_output=True, text=True, timeout=60, env=env, cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "Lisbon" in proc.stdout


@pytest.mark.parametrize("name", ["infinitydb-demo", "infinitydb-mcp"])
def test_console_scripts_are_declared(name):
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f"{name} = " in pyproject
