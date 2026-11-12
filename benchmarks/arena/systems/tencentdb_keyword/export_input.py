"""Export LoCoMo conversations to the JSON the TencentDB harness reads."""
import json, sys
sys.path.insert(0, "benchmarks"); sys.path.insert(0, ".")
from arena import locomo
convs = sys.argv[1:] or [c.sample_id for c in locomo.load()]
for c in locomo.load():
    if c.sample_id not in convs:
        continue
    turns = [{"id": f"{c.sample_id}:{t.dia_id}", "text": t.text if hasattr(t, "text") else str(t),
              "ts": t.ts} for t in c.all_turns()]
    qa = [{"q": q.question, "cat": q.category_name, "ev": list(q.evidence)} for q in c.qa]
    json.dump({"conv": c.sample_id, "turns": turns, "qa": qa},
              open(f".arena/tencentdb/input/{c.sample_id}.json", "w"))
    print(c.sample_id, len(turns), "turns", len(qa), "qa")
