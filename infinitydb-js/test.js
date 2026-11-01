const assert = require("assert");
const { InfinityDB } = require("./index.js");

console.log("[*] Running InfinityDB JavaScript test suite...");

const db = new InfinityDB();

// 1. Legacy put & get
const mem = db.put("usr:1", "likes TypeScript");
assert.strictEqual(mem.content, "likes TypeScript");
assert.strictEqual(db.get("usr:1").content, "likes TypeScript");
assert.strictEqual(db.delete("usr:1"), true);
assert.strictEqual(db.get("usr:1"), null);

// 2. Add and Recall
const logId = db.add("I am Sarah, I prefer TypeScript because it has strict static types. My budget is $15k", "usr_sarah");
assert.ok(logId.startsWith("L0_"));

const recallRes = db.recall("what does Sarah prefer?", { userId: "usr_sarah" });
assert.ok(recallRes.results.length > 0);
assert.ok(recallRes.latencyMs >= 0);
assert.strictEqual(recallRes.proofAvailable, true);
assert.ok(recallRes.causalGraph.length > 0);

// 3. Proof Chain
const atomId = recallRes.results[0].id;
const proof = db.getProof(atomId);
assert.ok(proof !== null);
assert.strictEqual(proof.atomId, atomId);
assert.strictEqual(proof.verified, true);
assert.strictEqual(proof.hash.length, 64);

// 4. Persona & Conflict Resolution
const persona1 = db.getPersona("usr_sarah");
assert.ok(persona1.preference.includes("TypeScript"));

// Update preference
db.add("I prefer Rust for mission-critical core engines", "usr_sarah");
const persona2 = db.getPersona("usr_sarah");
assert.ok(persona2.preference.includes("Rust"));

// 5. Policies
db.addPolicy("Never log plaintext passwords", "security", 2.0);
const res2 = db.recall("login issue", { userId: "usr_sarah" });
assert.ok(res2.policies.length >= 1);

// 6. GDPR Delete
const wipe = db.deleteUser("usr_sarah");
assert.strictEqual(wipe.gdprCompliant, true);
assert.strictEqual(db.recall("Rust", { userId: "usr_sarah" }).results.length, 0);


// 7. Input validation and query isolation
assert.throws(() => db.add("   "), /non-empty string/);
assert.throws(() => db.add("valid", " "), /userId/);
assert.throws(() => db.add("valid", "usr_sarah", null, 1.5), /between 0 and 1/);
assert.throws(() => db.recall("Rust", { n: 0 }), /positive integer/);
assert.deepStrictEqual(db.recall("   ", { userId: "usr_sarah" }).results, []);
assert.strictEqual(db.recall("   ", { userId: "usr_sarah" }).tokensSaved, null);

// 8. Importance is applied to stored facts
const lowId = db.add("I prefer SQLite.", "low_importance", null, 0.1);
const highId = db.add("I prefer SQLite.", "high_importance", null, 1.0);
const lowAtom = Array.from(db.atoms.values()).find(a => a.sourceId === lowId);
const highAtom = Array.from(db.atoms.values()).find(a => a.sourceId === highId);
assert.ok(lowAtom.importance < highAtom.importance);

// 9. Proof verification detects source tampering
const proofDb = new InfinityDB();
const proofLog = proofDb.add("I prefer Rust for safe systems.", "proof_user");
const proofAtom = Array.from(proofDb.atoms.values()).find(a => a.sourceId === proofLog);
assert.strictEqual(proofDb.getProof(proofAtom.id).verified, true);
proofDb.logs.get(proofLog).raw = "tampered source";
assert.strictEqual(proofDb.getProof(proofAtom.id).verified, false);


 // 10. SQLite durability across process-style reopen
const fs = require("fs");
const os = require("os");
const path = require("path");
const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "infinitydb-sqlite-"));
const dbPath = path.join(tempDir, "memory.sqlite");
try {
  const durable = new InfinityDB({ dbPath });
  const durableLog = durable.add("I prefer SQLite for durable agent memory.", "durable_user");
  durable.addPolicy("Never expose secrets", "security", 3);
  const durableAtom = Array.from(durable.atoms.values()).find(a => a.sourceId === durableLog);
  const durableAtomId = durableAtom.id;
  assert.strictEqual(durable.getProof(durableAtomId).verified, true);
  durable.close();

  const reopened = new InfinityDB({ dbPath });
  assert.ok(reopened.logs.has(durableLog));
  assert.ok(reopened.atoms.has(durableAtomId));
  assert.ok(reopened.getPersona("durable_user").preference.includes("SQLite"));
  assert.strictEqual(reopened.getProof(durableAtomId).verified, true);
  assert.strictEqual(reopened.recall("SQLite", { userId: "durable_user" }).results.length, 1);
  assert.strictEqual(reopened.stats().l4Policies, 1);
  reopened.deleteUser("durable_user");
  reopened.close();

  const afterDelete = new InfinityDB({ dbPath });
  assert.strictEqual(afterDelete.recall("SQLite", { userId: "durable_user" }).results.length, 0);
  assert.strictEqual(afterDelete.stats().l0RawLogs, 0);
  afterDelete.close();
} finally {
  // Shared temporary directory is removed after the migration tests below.
}

// sdk tests


// 11. Normalized schema, legacy snapshot migration, and cross-instance visibility.
const schemaDb = require("better-sqlite3");
const normalizedPath = path.join(tempDir, "normalized.sqlite");
try {
  const first = new InfinityDB({ dbPath: normalizedPath });
  const second = new InfinityDB({ dbPath: normalizedPath });
  const firstLog = first.add("I prefer PostgreSQL for production storage.", "shared_user");
  assert.ok(second.recall("PostgreSQL", { userId: "shared_user" }).results.length > 0);
  const secondLog = second.add("I prefer Redis for cache workloads.", "shared_user");
  assert.ok(first.recall("Redis", { userId: "shared_user" }).results.length > 0);
  first.close();
  second.close();

  const raw = new schemaDb(normalizedPath);
  const tables = raw.prepare(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
  ).all().map(row => row.name);
  assert.ok(tables.includes("l0_logs"));
  assert.ok(tables.includes("l1_atoms"));
  assert.ok(tables.includes("l6_proofs"));
  assert.ok(!tables.includes("infinitydb_state"));
  assert.strictEqual(raw.prepare("SELECT count(*) AS n FROM l0_logs").get().n, 2);
  raw.close();

  // Simulate a v1 snapshot database and ensure it migrates without data loss.
  const legacyPath = path.join(tempDir, "legacy-v1.sqlite");
  const legacy = new schemaDb(legacyPath);
  legacy.exec("CREATE TABLE infinitydb_state (id INTEGER PRIMARY KEY, schema_version INTEGER NOT NULL, payload TEXT NOT NULL, updated_at TEXT)");
  const legacyState = {
    seq: 7,
    logs: [[ "log_old", { id: "log_old", userId: "old_user", sessionId: null, raw: "I prefer SQLite.", ts: 1 } ]],
    atoms: [[ "atom_old", { id: "atom_old", userId: "old_user", fact: "I prefer SQLite.", category: "preference", sourceId: "log_old", importance: 0.9, accessCount: 0, ts: 1 } ]],
    episodes: [[ "episode_old", { id: "episode_old", userId: "old_user", cause: "SQLite", effect: "I prefer it", relation: "causes", sourceId: "log_old", ts: 1 } ]],
    personas: [[ "old_user:preference", { userId: "old_user", key: "preference", value: "I prefer SQLite.", confidence: 0.92, updatedAt: 1, sourceAtomId: "atom_old" } ]],
    policies: [[ "policy_old", { id: "policy_old", rule: "Keep tenant data isolated", domain: "security", priority: 2, createdAt: 1 } ]],
    proofs: [[ "atom_old", { atomId: "atom_old", hash: "hash", sourceLogId: "log_old", rawSnippet: "I prefer SQLite.", charStart: 0, charEnd: 16, ts: 1 } ]],
    legacyMemories: [[ "old_key", { key: "old_key", content: "legacy value", createdAt: "2026-01-01T00:00:00.000Z" } ]]
  };
  legacy.prepare("INSERT INTO infinitydb_state VALUES (1, 1, ?, CURRENT_TIMESTAMP)").run(JSON.stringify(legacyState));
  legacy.close();
  const migrated = new InfinityDB({ dbPath: legacyPath });
  assert.ok(migrated.logs.has("log_old"));
  assert.ok(migrated.atoms.has("atom_old"));
  assert.ok(migrated.episodes.has("episode_old"));
  assert.strictEqual(migrated.getPersona("old_user").preference, "I prefer SQLite.");
  assert.strictEqual(migrated.recall("SQLite", { userId: "old_user" }).policies[0].rule, "Keep tenant data isolated");
  assert.strictEqual(migrated.get("old_key").content, "legacy value");
  assert.strictEqual(migrated._seq, 7);
  migrated.add("I prefer Rust.", "old_user");
  migrated.close();
  const migratedRaw = new schemaDb(legacyPath);
  assert.strictEqual(migratedRaw.prepare("SELECT value FROM infinitydb_meta WHERE key='schema_version'").get().value, "2");
  assert.strictEqual(migratedRaw.prepare("SELECT count(*) AS n FROM l0_logs").get().n, 2);
  migratedRaw.close();
} finally {
  fs.rmSync(tempDir, { recursive: true, force: true });
}


// 12. Temporal supersession and historical recall (in-memory).
const temporal = new InfinityDB();
const oldLog = temporal.add("I prefer Python for data engineering.", "temporal_user");
const oldFact = Array.from(temporal.atoms.values()).find(a => a.sourceId === oldLog && a.category === "preference");
const beforeSupersession = oldFact.ts;
const newLog = temporal.add("I prefer Rust for systems programming.", "temporal_user");
const newFact = Array.from(temporal.atoms.values()).find(a => a.sourceId === newLog && a.category === "preference");
temporal.supersedeFact(oldFact.id, newFact.id, "temporal_user");
assert.strictEqual(temporal.recall("Python", { userId: "temporal_user" }).results.length, 0);
assert.strictEqual(temporal.recall("Python", { userId: "temporal_user", asOf: beforeSupersession }).results.length, 1);
assert.strictEqual(temporal.recall("Rust", { userId: "temporal_user", asOf: beforeSupersession }).results.length, 0);
assert.throws(() => temporal.supersedeFact(newFact.id, oldFact.id, "temporal_user"), /cycle/);
assert.throws(() => temporal.supersedeFact(oldFact.id, newFact.id, "other_user"), /belong/);
assert.throws(() => temporal.recall("Python", { asOf: Infinity }), /finite Unix timestamp/);
assert.strictEqual(temporal.getProof(oldFact.id).verified, true);

// 13. Temporal relation survives SQLite reopen and is removed on tenant deletion.
const temporalDir = fs.mkdtempSync(path.join(os.tmpdir(), "infinitydb-temporal-"));
const temporalPath = path.join(temporalDir, "temporal.sqlite");
const temporalSql = new InfinityDB({ dbPath: temporalPath });
const temporalOldLog = temporalSql.add("I prefer Java for backend services.", "temporal_sql_user");
const temporalOld = Array.from(temporalSql.atoms.values()).find(a => a.sourceId === temporalOldLog);
const temporalCutoff = temporalOld.ts + 0.001;
const temporalNewLog = temporalSql.add("I prefer Go for backend services.", "temporal_sql_user");
const temporalNew = Array.from(temporalSql.atoms.values()).find(a => a.sourceId === temporalNewLog);
temporalSql.supersedeFact(temporalOld.id, temporalNew.id, "temporal_sql_user");
temporalSql.close();
const temporalReopened = new InfinityDB({ dbPath: temporalPath });
assert.strictEqual(temporalReopened.recall("Java", { userId: "temporal_sql_user" }).results.length, 0);
assert.strictEqual(temporalReopened.recall("Java", { userId: "temporal_sql_user", asOf: temporalCutoff }).results.length, 1);
assert.strictEqual(temporalReopened.relations.length, 1);
temporalReopened.deleteUser("temporal_sql_user");
assert.strictEqual(temporalReopened.relations.length, 0);
temporalReopened.close();
fs.rmSync(temporalDir, { recursive: true, force: true });

console.log("[PASS] All JavaScript tests passed successfully!");
