from infinitydb_next import MemoryStore

with MemoryStore("demo.sqlite3") as memory:
    memory_id = memory.add_memory(
        "The agent should cite evidence when recalling durable facts.",
        tenant_id="demo",
        raw_evidence="User instruction captured during onboarding.",
    )
    print("stored:", memory_id)
    for hit in memory.search("cite evidence", tenant_id="demo"):
        print(hit.memory.content, hit.score, hit.matched_by)
    print("evidence chain valid:", memory.verify_evidence_chain(tenant_id="demo"))
