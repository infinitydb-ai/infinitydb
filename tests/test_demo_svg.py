"""The README animation must keep telling the truth.

`assets/demo.svg` is generated from a real `python -m infinitydb.demo` run, so it
drifts the moment the demo changes. These tests fail when that happens instead of
letting the image quietly advertise output the code no longer produces.
"""
from __future__ import annotations

import html
import re
import xml.dom.minidom
from pathlib import Path

import pytest

from infinitydb import demo

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets" / "demo.svg"
TIMING = re.compile(r"\d+(?:\.\d+)?\s*ms")


def _demo_lines() -> list[str]:
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert demo.main([]) == 0
    return buffer.getvalue().rstrip("\n").split("\n")


@pytest.fixture(scope="module")
def svg() -> str:
    assert ASSET.exists(), "assets/demo.svg is missing; run scripts/make_demo_svg.py"
    return ASSET.read_text(encoding="utf-8")


def test_asset_is_well_formed_svg(svg):
    document = xml.dom.minidom.parseString(svg)  # raises on malformed XML
    root = document.documentElement
    assert root.tagName == "svg"
    assert int(root.getAttribute("width")) > 500
    assert int(root.getAttribute("height")) > 300
    assert root.getAttribute("role") == "img"
    assert root.getAttribute("aria-labelledby")


def test_every_non_timing_line_of_the_demo_is_in_the_asset(svg):
    """Timings vary per machine, so they are excluded; everything else must match."""
    recorded = svg
    missing = []
    for line in _demo_lines():
        if not line.strip() or TIMING.search(line):
            continue
        if html.escape(line) not in recorded:
            missing.append(line)
    assert not missing, f"asset no longer matches the demo output: {missing[:3]}"


def test_asset_contains_the_facts_a_reader_should_see(svg):
    for expected in ("Lisbon", "data engineer", "forty thousand euros", "SQLite-WAL",
                     "uvx --from infinitydb-ai infinitydb-mcp"):
        assert expected in svg


def test_asset_is_static_and_portable(svg):
    """A README image must not execute anything or fetch anything."""
    assert "<script" not in svg.lower()
    # The SVG namespace URI is mandatory; every other remote reference is not.
    body = svg.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "http://" not in body and "https://" not in body
    assert "xlink:href" not in svg
    assert ASSET.stat().st_size < 120_000, "keep the README light"


def test_animation_is_opt_out_able(svg):
    """prefers-reduced-motion must show the whole transcript, not a blank frame."""
    assert "prefers-reduced-motion" in svg
    reduced = svg.split("prefers-reduced-motion", 1)[1]
    assert "animation: none" in reduced and "opacity: 1" in reduced
