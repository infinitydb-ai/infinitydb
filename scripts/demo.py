"""20-second terminal demo. No key, no network, no model download.

Run with: python3 scripts/demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infinitydb.core.engine import InfinityDB

db = InfinityDB(":memory:")
t = 1700000000.0
turns = [
    ("Maya: I work at the gym in Leeds.", 0),
    ("Maya: my sister lives in Berlin and works at Siemens.", 10 * 86400),
    ("Maya: I work at the hospital in Leeds.", 30 * 86400),
    ("Maya: I moved to Manchester.", 60 * 86400),
]
for text, offset in turns:
    db.add(text, user_id="maya", ts=t + offset)
    print(f"  + stored  {text}")

print("\nQ: where does Maya work today?")
top = db.recall("where does Maya work", n=3, user_id="maya", reinforce=False)["results"][0]
print(f"  top hit   {top['fact'].strip()[:78]}")

print("\nQ: what changed over time?")
for conflict in db.contradictions("maya")[:2]:
    print(f"  {conflict['category']}: {conflict['old_fact']!r} -> {conflict['new_fact']!r}")

print("\nQ: who is connected to Maya?")
for edge in db.relations("maya", user_id="maya")[:3]:
    print(f"  {edge['subject']} -[{edge['predicate']}]-> {edge['object']}")

print("\npersona:")
print("  " + db.persona_summary("maya"))
