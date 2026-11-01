# InfinityDB Next — v2 alpha workspace

This is an isolated alpha workspace integrated alongside the existing InfinityDB 1.x implementation. It does not replace the stable `infinitydb` Python package or `infinitydb-js` Node SDK.

## Included capabilities

- SQLite WAL-backed tenant-scoped memory records, validity windows and relationship edges.
- FTS5 lexical retrieval with AND-first matching and a substring fallback when FTS5 is unavailable.
- Atomic batch ingestion, evidence capture and SHA-256 evidence-chain verification.
- Context-pack compilation with a conservative character-based token estimate.
- Python reference package, optional REST adapter (see `src/infinitydb_next/api.py`), TypeScript HTTP client and dependency-free Rust score-fusion primitives.

## Install and test

```bash
cd alpha-v2
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

## Explicit alpha limitations

This workspace does **not** implement vector/ANN retrieval, embedding providers, predictive prefetch, autonomous consolidation, multi-agent federation, production authentication, encryption-at-rest or MCP transport. The REST adapter is a development-only interface and must remain bound to localhost until authentication and transport security are implemented. The token estimate is not tokenizer-accurate. Evidence hash chains detect alteration but do not prove authorship.

Benchmark artifacts are workload-specific synthetic measurements; they are not competitor comparisons or production guarantees. See `docs/BENCHMARKS.md` before publishing performance claims.
