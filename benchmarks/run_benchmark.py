"""Deprecated compatibility entry point for the reproducible benchmark.

This command now measures InfinityDB only. It does not emulate or compare
third-party products. See benchmarks/REPRODUCIBLE_BENCHMARK.md.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.reproducible import main


if __name__ == "__main__":
    main()
