/**
 * Drive TencentDB Agent Memory's real retrieval path locally, with no LLM key.
 *
 * Their L1 extraction needs a model, so this feeds their store turn-level
 * records directly and calls their own `executeMemorySearch` — i.e. the
 * `recall.strategy=keyword` configuration, FTS5/BM25 only. Labelled as such.
 */
import fs from "node:fs";
import { VectorStore } from "./node_modules/@tencentdb-agent-memory/memory-tencentdb/src/core/store/sqlite.ts";
import { executeMemorySearch } from "./node_modules/@tencentdb-agent-memory/memory-tencentdb/src/core/tools/memory-search.ts";

const conv = process.argv[2];
if (!conv) {
  console.error("usage: tsx harness.mts <conv-id>");
  process.exit(2);
}
const data = JSON.parse(fs.readFileSync(`input/${conv}.json`, "utf8"));
const dbPath = `run/${conv}.db`;
for (const suffix of ["", "-wal", "-shm"]) fs.rmSync(dbPath + suffix, { force: true });

const store = new VectorStore(dbPath, 0);
await store.init();
console.log(`fts available: ${store.isFtsAvailable()}`);

let n = 0;
for (const turn of data.turns) {
  const iso = new Date(turn.ts * 1000).toISOString();
  const ok = store.upsertL1(
    {
      id: `m${n++}`,
      content: turn.text,
      type: "episodic",
      priority: 50,
      scene_name: "chat",
      source_message_ids: [turn.id],
      metadata: {},
      timestamps: [iso],
      createdAt: iso,
      updatedAt: iso,
      sessionKey: conv,
      sessionId: conv,
    },
    undefined,
  );
  if (!ok) {
    console.error("upsert failed at", turn.id);
    process.exit(3);
  }
}

const out: unknown[] = [];
for (const qa of data.qa) {
  const result = await executeMemorySearch({
    query: qa.q,
    limit: 10,
    vectorStore: store,
  });
  out.push({
    q: qa.q,
    strategy: result.strategy,
    results: (result.results ?? []).map((r: any) => ({ content: r.content, score: r.score })),
  });
}
if (out.length) console.log("sample:", JSON.stringify(out[0]).slice(0, 400));
fs.writeFileSync(`out/${conv}.json`, JSON.stringify({ conv, rows: out }));
console.log(`done: ${n} records, ${out.length} queries -> out/${conv}.json`);
store.close();
