"""Deterministic update-tracking benchmark for InfinityDB.

Question answered: after a user's single-valued fact changes several times, does recall()
rank the CURRENT value first, and does it keep returning outdated values?

Workload (seeded): N users each state a city 1-4 times ("I live in X.", later "I moved to Y."),
interleaved across users, mixed with distractor preference statements. Each user is then asked
"Where do I live?". Limits: only phrasing the built-in rules parse; synthetic; single machine.

    python benchmarks/update_tracking.py --users 300 --noise 3000 --output update-tracking.json
"""
from __future__ import annotations

import argparse, json, platform, random, re, sqlite3, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infinitydb import InfinityDB  # noqa: E402

CITIES = "Paris Berlin Tokyo Lima Cairo Oslo Rome Madrid Lagos Seoul Hanoi Dubai Delhi Quito Accra Perth Basel Turin Porto Split".split()


def run(users: int, noise: int, seed: int, mixed_phrasing: bool) -> dict:
    rng = random.Random(seed)
    names = [f"user{i}" for i in range(users)]
    moves = {u: [] for u in names}
    for u in names:
        last = None
        for _ in range(rng.randint(1, 4)):
            city = rng.choice([c for c in CITIES if c != last]); moves[u].append(city); last = city
    truth = {u: moves[u][-1] for u in names}
    order = [u for u in names for _ in moves[u]]
    rng.shuffle(order)
    cursor = {u: 0 for u in names}
    stream = []
    for u in order:
        i = cursor[u]; cursor[u] += 1
        verb = "I live in" if (i == 0 or not mixed_phrasing) else "I moved to"
        stream.append((u, f"{verb} {moves[u][i]}."))
    for i in range(noise):
        stream.insert(rng.randrange(len(stream) + 1), (rng.choice(names), f"I prefer tool{rng.randint(0, 299)} for project {i}."))

    db = InfinityDB(":memory:")
    t = 1_700_000_000.0
    for u, text in stream:
        t += 60
        db.add(text, user_id=u, ts=t)
    now = t + 60

    correct = stale = 0
    lat = []
    for u in names:
        t0 = time.perf_counter()
        res = db.recall("Where do I live?", n=3, user_id=u, now=now, reinforce=False)
        lat.append((time.perf_counter() - t0) * 1000)
        cities = [m.group(1) for r in res["results"] if (m := re.search(r"(?:live in|moved to) (\w+)", r["fact"]))]
        correct += bool(cities) and cities[0] == truth[u]
        stale += any(c != truth[u] for c in cities)
    lat.sort()
    return {
        "benchmark": "infinitydb-update-tracking", "users": users, "facts_written": len(stream),
        "seed": seed, "mixed_phrasing": mixed_phrasing,
        "current_answer_first": round(correct / users, 4),
        "outdated_fact_in_top3": round(stale / users, 4),
        "recall_latency_ms": {"p50": round(statistics.median(lat), 3), "p95": round(lat[int(users * 0.95)], 3)},
        "environment": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version, "platform": platform.platform()},
        "limitations": ["Synthetic workload using phrasing the built-in rules parse.",
                        "Measures update tracking only, not general retrieval quality.",
                        "Single machine; latency varies with hardware."],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300); ap.add_argument("--noise", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=5); ap.add_argument("--output", default=None)
    ap.add_argument("--same-phrasing", action="store_true", help='always "I live in X." (default mixes in "I moved to X.")')
    a = ap.parse_args()
    out = run(a.users, a.noise, a.seed, not a.same_phrasing)
    text = json.dumps(out, indent=2)
    print(text)
    if a.output:
        Path(a.output).write_text(text + "\n", encoding="utf-8")
