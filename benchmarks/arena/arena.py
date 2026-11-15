#!/usr/bin/env python3
"""Standalone runner for the LoCoMo memory arena.

One entry point for the whole benchmark, so a number can be produced, checked
and cited without reading the harness source:

    python benchmarks/arena/arena.py run    --systems infinitydb --convs conv-26
    python benchmarks/arena/arena.py cite   --results benchmarks/arena/results
    python benchmarks/arena/arena.py verify --results benchmarks/arena/results

`run` drives `worker.py` (one process per system) and then `score.py`, writing a
`manifest.json` beside the results that pins the dataset hash, the system
versions, the interpreter, the host and the exact command. `cite` renders the
markdown table and BibTeX block from that manifest, so a quoted number carries
its provenance with it. `verify` recomputes every retrieval metric from the raw
per-question rows and fails if it disagrees with the committed summary.

Retrieval metrics need no model. `qa_f1` / `gold_in_answer` stay 0 unless a
local reader is serving; `run --reader` forwards that flag to `score.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parent
ROOT = ARENA_DIR.parents[1]
if str(ARENA_DIR.parent) not in sys.path:
    sys.path.insert(0, str(ARENA_DIR.parent))

from arena import locomo  # noqa: E402

VERSION = "1.0.0"
SYSTEMS = ["infinitydb", "infinitydb-dense", "mem0", "hybrid", "tencentdb-keyword"]
DEFAULT_CONVS = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43",
                 "conv-44", "conv-47", "conv-48", "conv-49", "conv-50"]
# Metrics that are deterministic given the same rows, so `verify` can compare
# them exactly. Latency and ingest time are measured, not reproducible.
DETERMINISTIC_KEYS = ("questions", "answer_in_context", "evidence_recall@k",
                      "context_tokens_median", "adversarial_refusal_rate")
TOLERANCE = 1e-9

# Systems that need a competitor's own dependency tree, isolated in a venv.
FOREIGN_VENV_SYSTEMS = {"mem0", "hybrid"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _dataset_manifest() -> dict:
    path = Path(locomo.DATA_PATH)
    rows = list(locomo.load())
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "conversations": len(rows),
        "questions": sum(len(c.qa) for c in rows),
        "turns": sum(c.turn_count for c in rows),
    }


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "sqlite": _sqlite_version(),
    }


def _sqlite_version() -> str:
    try:
        import sqlite3

        return sqlite3.sqlite_version
    except Exception:  # pragma: no cover - sqlite is always present
        return "unknown"


def _venv_python(path: Path, requirements: str) -> Path:
    """Create a pinned competitor venv on first use, then reuse it."""
    python = path / "bin" / "python"
    if python.exists():
        return python
    print(f"[arena] creating venv {path} from {requirements}")
    subprocess.run([sys.executable, "-m", "venv", str(path)], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-q", "-r",
                    str(ARENA_DIR / requirements)], check=True)
    return python


def _python_for(system: str, mem0_venv: Path) -> Path:
    if system in FOREIGN_VENV_SYSTEMS:
        req = ("requirements-mem0.txt" if system == "mem0" else "requirements-hybrid.txt")
        return _venv_python(mem0_venv, req)
    return Path(sys.executable)


def _run_workers(args: argparse.Namespace) -> list[str]:
    results = Path(args.results)
    workdir = Path(args.workdir)
    mem0_venv = Path(args.mem0_venv)
    completed: list[str] = []
    for system in args.systems:
        python = _python_for(system, mem0_venv)
        for conv in args.convs:
            target = results / system / f"{conv}.jsonl"
            if target.exists() and not args.force:
                print(f"[arena] {system} {conv}: reusing {target}")
                completed.append(f"{system}/{conv}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            cmd = [str(python), str(ARENA_DIR / "worker.py"),
                   "--system", system, "--conv", conv,
                   "--out", str(target),
                   "--workdir", str(workdir / system / conv),
                   "--top-k", str(args.top_k)]
            if args.max_qa:
                cmd += ["--max-qa", str(args.max_qa)]
            if args.force:
                cmd += ["--force"]
            print(f"[arena] {system} {conv}: running")
            subprocess.run(cmd, check=True, cwd=ROOT)
            completed.append(f"{system}/{conv}")
    return completed


def _run_scorer(args: argparse.Namespace) -> Path:
    out = Path(args.results) / "scores"
    cmd = [sys.executable, str(ARENA_DIR / "score.py"),
           "--results", str(args.results), "--out", str(out),
           "--systems", ",".join(args.systems)]
    if args.convs:
        cmd += ["--convs", ",".join(args.convs)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if not args.reader:
        cmd += ["--no-reader"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return out


def _system_versions(systems: list[str]) -> dict:
    """Record the pinned competitor versions the README claims are in use."""
    versions: dict[str, str] = {}
    for system in systems:
        if system.startswith("infinitydb"):
            versions[system] = f"this repository @ {_git('rev-parse', '--short', 'HEAD') or 'unknown'}"
            continue
        requirements = {
            "mem0": "requirements-mem0.txt",
            "hybrid": "requirements-hybrid.txt",
        }.get(system)
        if requirements:
            lines = (ARENA_DIR / requirements).read_text(encoding="utf-8").splitlines()
            versions[system] = "; ".join(line.strip() for line in lines
                                         if line.strip() and not line.startswith("#"))
        else:
            versions[system] = "see systems/ adapter"
    return versions


def cmd_run(args: argparse.Namespace) -> int:
    args.systems = [s for s in args.systems if s]
    started = time.time()
    Path(args.results).mkdir(parents=True, exist_ok=True)
    completed = _run_workers(args)
    scores_dir = _run_scorer(args)
    manifest = {
        "arena_version": VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.time() - started, 1),
        "command": " ".join(["python", "benchmarks/arena/arena.py", *sys.argv[1:]]),
        "dataset": _dataset_manifest(),
        "systems": args.systems,
        "system_versions": _system_versions(args.systems),
        "system_cores": SYSTEMS,
        "conversations": args.convs,
        "top_k": args.top_k,
        "max_qa": args.max_qa or "all",
        "reader": bool(args.reader),
        "environment": _environment(),
        "git": {"commit": _git("rev-parse", "HEAD"),
                "dirty": bool(_git("status", "--porcelain"))},
        "runs": completed,
        "scores": str(scores_dir.relative_to(ROOT)) if scores_dir.is_relative_to(ROOT) else str(scores_dir),
    }
    manifest_path = Path(args.results) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\n[arena] wrote {manifest_path}")
    print(f"[arena] next: python benchmarks/arena/arena.py cite --results {args.results}")
    return 0


def _load_manifest(results: Path) -> dict:
    path = results / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_verify(args: argparse.Namespace) -> int:
    results = Path(args.results)
    committed = results / "scores" / "summary.json"
    if not committed.exists():
        print(f"FAIL: {committed} missing — run `arena.py run` first")
        return 1
    reference = json.loads(committed.read_text(encoding="utf-8"))
    manifest = _load_manifest(results)

    expected = manifest.get("dataset", {}).get("sha256")
    actual = _sha256(Path(locomo.DATA_PATH))
    if expected and expected != actual:
        print(f"FAIL: dataset changed since this run\n  manifest {expected}\n  on disk  {actual}")
        return 1
    print(f"dataset sha256 {actual[:16]}… matches the manifest" if expected
          else f"dataset sha256 {actual[:16]}… (no manifest to compare against)")

    with tempfile.TemporaryDirectory() as tmp:
        systems = ",".join(reference.keys())
        cmd = [sys.executable, str(ARENA_DIR / "score.py"),
               "--results", str(results), "--out", tmp,
               "--systems", systems, "--no-reader"]
        if manifest.get("conversations"):
            cmd += ["--convs", ",".join(manifest["conversations"])]
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        recomputed = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))

    failures = 0
    for system, data in reference.items():
        new = recomputed.get(system, {}).get("aggregate")
        if not new:
            print(f"FAIL: {system} could not be rescored")
            failures += 1
            continue
        old = data["aggregate"]
        drift = {k: (old.get(k), new.get(k)) for k in DETERMINISTIC_KEYS
                 if old.get(k) is not None and abs((old.get(k) or 0) - (new.get(k) or 0)) > TOLERANCE}
        if drift:
            failures += 1
            print(f"FAIL: {system} does not reproduce")
            for key, (before, after) in drift.items():
                print(f"  {key}: committed {before} != recomputed {after}")
        else:
            print(f"ok   {system}: questions={new['questions']} "
                  f"evRec={new['evidence_recall@k']} inCtx={new['answer_in_context']}")
    print("\nPASS: every retrieval metric reproduces from the raw rows" if not failures
          else f"\nFAIL: {failures} system(s) did not reproduce")
    return 0 if not failures else 1


def _markdown_table(summary: dict) -> str:
    header = ("| system | questions | evidence_recall@10 | answer_in_context "
              "| ctx tokens (median) | p50 ms |\n|---|---|---|---|---|---|")
    rows = [header]
    for system in sorted(summary):
        a = summary[system]["aggregate"]
        rows.append(f"| `{system}` | {a['questions']} | {a['evidence_recall@k']:.4f} "
                    f"| {a['answer_in_context']:.4f} | {a['context_tokens_median']:.0f} "
                    f"| {a['latency_ms']['p50']:.2f} |")
    return "\n".join(rows)


def cmd_cite(args: argparse.Namespace) -> int:
    results = Path(args.results)
    summary_path = results / "scores" / "summary.json"
    if not summary_path.exists():
        print(f"FAIL: {summary_path} missing — run `arena.py run` first")
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = _load_manifest(results)
    dataset = manifest.get("dataset", {})
    commit = manifest.get("git", {}).get("commit", "")[:7] or _git("rev-parse", "--short", "HEAD")

    lines = [
        "# LoCoMo memory arena results",
        "",
        f"Produced by `benchmarks/arena/arena.py` v{manifest.get('arena_version', VERSION)}.",
        "",
        _markdown_table(summary),
        "",
        "## Provenance",
        "",
        f"- Dataset: `{dataset.get('path', 'locomo10.json')}` "
        f"({dataset.get('conversations', '?')} conversations, "
        f"{dataset.get('questions', '?')} questions)",
        f"- Dataset SHA-256: `{dataset.get('sha256', 'not recorded — no manifest')}`",
        f"- Repository commit: `{commit or 'unknown'}`",
        f"- Systems: {', '.join(f'`{s}`' for s in manifest.get('systems', sorted(summary)))}",
        f"- top_k: {manifest.get('top_k', 10)} | reader LLM: "
        f"{'yes' if manifest.get('reader') else 'none (retrieval metrics only)'}",
        f"- Generated: {manifest.get('generated_at', 'unknown')}",
        f"- Host: {manifest.get('environment', {}).get('platform', 'unknown')}, "
        f"{manifest.get('environment', {}).get('cpu_count', '?')} CPUs, "
        f"Python {manifest.get('environment', {}).get('python', '?')}, "
        f"SQLite {manifest.get('environment', {}).get('sqlite', '?')}",
        f"- Command: `{manifest.get('command', 'see README')}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python benchmarks/arena/arena.py run --systems {','.join(manifest.get('systems', ['infinitydb']))} \\",
        "  --convs " + ",".join(manifest.get("conversations", DEFAULT_CONVS)),
        "python benchmarks/arena/arena.py verify --results " + str(results),
        "```",
        "",
        "## Cite",
        "",
        "```bibtex",
        "@misc{infinitydb_arena_" + (commit or "head") + ",",
        "  title  = {LoCoMo Memory Arena: reproducible retrieval benchmark for agent memory systems},",
        "  author = {InfinityDB contributors},",
        "  year   = {" + str(time.gmtime().tm_year) + "},",
        "  note   = {Dataset SHA-256 " + str(dataset.get("sha256", "unknown"))[:16] + "},",
        "  url    = {https://github.com/infinitydb-ai/infinitydb/tree/main/benchmarks/arena}",
        "}",
        "```",
        "",
        "Retrieval metrics only: `evidence_recall@10` and `answer_in_context` are",
        "computed without any language model, so they reproduce on a CPU-only machine.",
        "They are not answer accuracy. See the arena README for limitations.",
        "",
    ]
    text = "\n".join(lines)
    out = Path(args.out) if args.out else results / "CITATION_RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if not args.quiet:
        print(text)
    print(f"[arena] wrote {out}")
    return 0


def _split(values: str | list[str]) -> list[str]:
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for value in values:
        out.extend(part for part in value.replace(",", " ").split() if part)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="arena.py",
        description="Run, check and cite the LoCoMo memory arena.",
        epilog="Retrieval metrics need no model; no API key is used anywhere.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run systems and score them")
    run.add_argument("--systems", default=["infinitydb"], type=_split,
                     help=f"comma/space separated subset of {SYSTEMS} (default: infinitydb)")
    run.add_argument("--convs", default=DEFAULT_CONVS, type=_split,
                     help="conversation ids (default: the ten LoCoMo conversations)")
    run.add_argument("--results", default=str(ARENA_DIR / "results"))
    run.add_argument("--workdir", default=str(ROOT / ".arena"))
    run.add_argument("--mem0-venv", default=str(ROOT / ".arena" / "venvs" / "mem0"))
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--max-qa", type=int, default=0, help="0 = every question")
    run.add_argument("--limit", type=int, default=0, help="score only the first N questions per conversation")
    run.add_argument("--reader", action="store_true", help="also run the local Ollama reader")
    run.add_argument("--force", action="store_true", help="re-run workers that already produced rows")
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="recompute metrics from raw rows and compare")
    verify.add_argument("--results", default=str(ARENA_DIR / "results"))
    verify.set_defaults(func=cmd_verify)

    cite = sub.add_parser("cite", help="render a citable results table with provenance")
    cite.add_argument("--results", default=str(ARENA_DIR / "results"))
    cite.add_argument("--out", default="")
    cite.add_argument("--quiet", action="store_true")
    cite.set_defaults(func=cmd_cite)

    args = ap.parse_args(argv)
    if not hasattr(args, "systems"):
        args.systems = []
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
