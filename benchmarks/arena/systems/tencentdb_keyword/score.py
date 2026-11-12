"""Score the TencentDB keyword-mode rows with the arena's own evidence metric."""
import json, math, sys
sys.path.insert(0, "benchmarks"); sys.path.insert(0, ".")
from arena import locomo
from arena.score import evidence_hit

convs = sys.argv[1:]
dataset = {c.sample_id: c for c in locomo.load()}
all_scores, buckets = [], {}
for conv in convs:
    conversation = dataset[conv]
    texts = conversation.turn_text_by_dia()
    import os
    path = f".arena/tencentdb/out/{conv}.json"
    if not os.path.exists(path):
        print(f"skip {conv}: no rows")
        continue
    data = json.load(open(path))
    for row, qa in zip(data["rows"], conversation.qa):
        ev = [texts[e] for e in qa.evidence if e in texts]
        score = evidence_hit([r["content"] for r in row["results"]], ev)
        if math.isnan(score):
            continue
        all_scores.append(score)
        buckets.setdefault(qa.category_name, []).append(score)
print(f"tencentdb-agent-memory (FTS keyword mode) n={len(all_scores)} "
      f"recall@10={sum(all_scores)/max(1,len(all_scores)):.4f}")
for name, values in sorted(buckets.items()):
    print(f"  {name:28s} {sum(values)/len(values):.4f}  (n={len(values)})")
