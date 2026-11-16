#!/usr/bin/env python3
"""Render the real demo output as an animated terminal SVG for the README.

The asset has to be *true*: it is generated from `python -m infinitydb.demo`, not
hand-drawn, and a test fails if the committed SVG stops matching what the demo
prints. Timings change per machine, so those are marked as an example run rather
than pinned; every fact, query and count is checked.

    python scripts/make_demo_svg.py                 # writes assets/demo.svg
    python scripts/make_demo_svg.py --out /tmp/x.svg

Why SVG and not a GIF: no host, no binary, scales on retina, and the full text
lives in <title>/<desc> so it is readable by screen readers and indexable.
"""
from __future__ import annotations

import argparse
import contextlib
import html
import io
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "demo.svg"

BG = "#0b0f14"
BAR = "#161d26"
FG = "#d7e2ee"
DIM = "#7d8fa3"
ACCENT = "#5eead4"
WARN = "#fbbf24"

# 14px monospace advance is ~8.4px in Menlo/SF Mono but ~9.3px in Linux
# DejaVu Sans Mono, and the asset is viewed on both; budget for the wider one.
CHAR_W = 9.4
LINE_H = 23
PAD_X = 26
TOP = 74
STEP = 0.55       # seconds between lines appearing
LOOP = 22         # seconds for one full cycle


def demo_lines() -> list[str]:
    """Run the demo in-process and capture exactly what it prints."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        from infinitydb import demo

        demo.main([])
    return buffer.getvalue().rstrip("\n").split("\n")


def _colourize(line: str) -> str:
    """Same colouring rules the README would use for a terminal block."""
    if line.startswith("query:"):
        return ACCENT
    if line.startswith("Next:") or line.startswith("  claude mcp"):
        return DIM
    if "ms" in line and line.startswith("  "):
        return FG
    if line.startswith("InfinityDB demo"):
        return WARN
    return FG


def render(lines: list[str], when: str) -> str:
    cols = max(len(line) for line in lines)
    width = int(PAD_X * 2 + cols * CHAR_W)
    height = TOP + len(lines) * LINE_H + 28

    body: list[str] = []
    for index, line in enumerate(lines):
        y = TOP + index * LINE_H
        colour = _colourize(line)
        delay = round(0.3 + index * STEP, 2)
        body.append(
            f'  <text class="ln" x="{PAD_X}" y="{y}" fill="{colour}" '
            f'style="animation-delay:{delay}s">{html.escape(line) or " "}</text>'
        )

    cursor_y = TOP + len(lines) * LINE_H
    cursor_delay = round(0.3 + len(lines) * STEP, 2)
    body.append(
        f'  <rect class="cursor" x="{PAD_X}" y="{cursor_y - 12}" width="9" height="16" '
        f'fill="{ACCENT}" style="animation-delay:{cursor_delay}s"/>'
    )

    flat = "\n".join(lines)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">InfinityDB demo: ingest three turns and recall facts, in memory, with no model calls</title>
  <desc id="d">{html.escape(flat)}</desc>
  <style>
    text {{ font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; font-size: 14px; white-space: pre; }}
    .ln {{ opacity: 1; animation: reveal {LOOP}s linear infinite both; }}
    .cursor {{ opacity: 0; animation: blink 1.05s steps(1) infinite; }}
    @keyframes reveal {{
      0% {{ opacity: 0; }}
      2% {{ opacity: 1; }}
      93% {{ opacity: 1; }}
      97%, 100% {{ opacity: 0; }}
    }}
    @keyframes blink {{ 0%, 50% {{ opacity: 1; }} 50.01%, 100% {{ opacity: 0; }} }}
    @media (prefers-reduced-motion: reduce) {{
      .ln {{ animation: none; opacity: 1; }}
      .cursor {{ animation: none; opacity: 1; }}
    }}
  </style>
  <rect width="{width}" height="{height}" rx="10" fill="{BG}"/>
  <rect width="{width}" height="44" rx="10" fill="{BAR}"/>
  <rect y="34" width="{width}" height="10" fill="{BAR}"/>
  <circle cx="22" cy="22" r="5.5" fill="#ff5f57"/>
  <circle cx="42" cy="22" r="5.5" fill="#febc2e"/>
  <circle cx="62" cy="22" r="5.5" fill="#28c840"/>
  <text x="86" y="27" fill="{DIM}" font-size="12.5">python -m infinitydb.demo · {html.escape(when)}</text>
{chr(10).join(body)}
</svg>
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    lines = demo_lines()
    when = f"{platform.system()} {platform.machine()}, python {platform.python_version()}"
    svg = render(lines, when)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out} ({len(svg.encode('utf-8'))} bytes, {len(lines)} lines, "
          f"generated {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
