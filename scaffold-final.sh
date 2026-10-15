#!/usr/bin/env bash
set -euo pipefail

mkdir -p infinitydb/core infinitydb-js .github/workflows dashboard benchmarks waitlist tests

touch infinitydb/__init__.py
touch infinitydb/core/__init__.py

cat > README.md <<'EOF'
# InfinityDB

InfinityDB is a high-performance, open-source agent memory engine.

## Project status

Initial development scaffold. Performance claims and benchmark results must be independently reproduced before publication.

## Components

- `infinitydb/`: Python memory engine
- `infinitydb-js/`: JavaScript package
- `benchmarks/`: Reproducible benchmark suite
- `dashboard/`: Project dashboard
- `waitlist/`: Waitlist application

## Python installation

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
EOF

cat > pyproject.toml <<'EOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "infinitydb"
version = "1.0.0"
description = "An open-source agent memory engine"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
authors = [{name = "InfinityDB contributors"}]
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6"]

[tool.hatch.build.targets.wheel]
packages = ["infinitydb"]

[tool.pytest.ini_options]
testpaths = ["tests"]
EOF

cat > infinitydb/core/engine.py <<'EOF'
"""Minimal local memory engine for the initial InfinityDB scaffold."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Memory:
    key: str
    content: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class InfinityDB:
    """In-memory key-based store; persistence is not implemented yet."""

    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}

    def put(self, key: str, content: str) -> Memory:
        if not key.strip():
            raise ValueError("key cannot be empty")
        if not content.strip():
            raise ValueError("content cannot be empty")
        memory = Memory(key=key, content=content)
        self._memories[key] = memory
        return memory

    def get(self, key: str) -> Memory | None:
        return self._memories.get(key)

    def delete(self, key: str) -> bool:
        return self._memories.pop(key, None) is not None

    def __len__(self) -> int:
        return len(self._memories)
EOF

cat > infinitydb/__init__.py <<'EOF'
"""InfinityDB agent memory engine."""

from infinitydb.core.engine import InfinityDB, Memory

__all__ = ["InfinityDB", "Memory"]
EOF

cat > infinitydb/core/__init__.py <<'EOF'
"""Core InfinityDB components."""
EOF

cat > tests/test_engine.py <<'EOF'
from infinitydb import InfinityDB


def test_put_and_get():
    db = InfinityDB()
    db.put("user:1", "likes Python")
    assert db.get("user:1").content == "likes Python"


def test_delete():
    db = InfinityDB()
    db.put("x", "value")
    assert db.delete("x")
    assert db.get("x") is None


def test_empty_key_rejected():
    db = InfinityDB()
    try:
        db.put(" ", "value")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
EOF

cat > infinitydb-js/package.json <<'EOF'
{
  "name": "@infinitydb-ai/infinitydb",
  "version": "1.0.0",
  "description": "JavaScript client for InfinityDB",
  "main": "index.js",
  "license": "MIT"
}
EOF

cat > infinitydb-js/index.js <<'EOF'
"use strict";

// JavaScript client implementation is not yet available.
module.exports = {};
EOF

cat > .github/workflows/tests.yml <<'EOF'
name: Tests

on:
  push:
  pull_request:

jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install -e ".[dev]"
      - run: pytest
EOF

cat > benchmarks/README.md <<'EOF'
# Benchmarks

Add reproducible workloads, baselines, environment details, and raw results here.
Do not publish unverified performance claims.
EOF

cat > dashboard/README.md <<'EOF'
# InfinityDB Dashboard

Dashboard implementation pending.
EOF

cat > waitlist/README.md <<'EOF'
# InfinityDB Waitlist

Waitlist implementation pending.
EOF

cat > .gitignore <<'EOF'
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
.env
*.db
node_modules/
dist/
build/
*.egg-info/
EOF

cat > LICENSE <<'EOF'
MIT License

Copyright (c) 2026 InfinityDB contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
EOF

echo "InfinityDB scaffold created."
