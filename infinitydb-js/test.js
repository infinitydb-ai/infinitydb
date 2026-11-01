const assert = require("assert");
const { InfinityDB } = require("./index.js");

console.log("[*] Running InfinityDB JavaScript test suite...");
const fs = require("fs");
const path = require("path");
const policyFixture = JSON.parse(fs.readFileSync(
  path.join(__dirname, "..", "tests", "fixtures", "sdk_policy_conformance.json"),
  "utf8"
));


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

// Tenant-scoped policies are visible only to their tenant; global policies remain shared.
const policyDb = new InfinityDB();
policyDb.addPolicy("Global policy", "security", 1);
policyDb.addPolicy("Tenant A only", "security", 2, "tenant_a");
assert.deepStrictEqual(policyDb.recall("policy", { userId: "tenant_a" }).policies.map(p => p.rule), ["Tenant A only", "Global policy"]);
assert.deepStrictEqual(policyDb.recall("policy", { userId: "tenant_b" }).policies.map(p => p.rule), ["Global policy"]);
assert.deepStrictEqual(policyDb.recall("policy").policies.map(p => p.rule), ["Global policy"]);
policyDb.deleteUser("tenant_a");
assert.deepStrictEqual(policyDb.recall("policy", { userId: "tenant_a" }).policies.map(p => p.rule), ["Global policy"]);
assert.throws(() => policyDb.addPolicy("bad", "security", 1, ""), /tenantId/);
policyDb.close();

// 6. GDPR Delete
const wipe = db.deleteUser("usr_sarah");
assert.ok(wipe.l0LogsDeleted >= 1);
assert.ok(wipe.l1AtomsDeleted >= 1);
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
const os = require("os");
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
  assert.ok(first.recall("PostgreSQL", { userId: "shared_user" }).results.length > 0);
  assert.strictEqual(first.recall("nonexistentterm", { userId: "shared_user" }).results.length, 0);
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

  // Existing schema-v2 databases rebuild a missing FTS index on open.
  const ftsRepair = new schemaDb(normalizedPath);
  ftsRepair.prepare("DELETE FROM infinitydb_meta WHERE key='l1_atoms_fts_version'").run();
  ftsRepair.prepare("INSERT INTO l1_atoms_fts(l1_atoms_fts) VALUES ('delete-all')").run();
  ftsRepair.close();
  const repaired = new InfinityDB({ dbPath: normalizedPath });
  assert.ok(repaired.recall("PostgreSQL", { userId: "shared_user" }).results.length > 0);
  repaired.deleteUser("shared_user");
  assert.strictEqual(repaired.recall("PostgreSQL", { userId: "shared_user" }).results.length, 0);
  repaired.close();

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


// 12. SQLite writes are incremental: adding one memory must not rewrite old rows.
const incrementalDir = fs.mkdtempSync(path.join(os.tmpdir(), "infinitydb-incremental-"));
const incrementalPath = path.join(incrementalDir, "incremental.sqlite");
try {
  const incremental = new InfinityDB({ dbPath: incrementalPath });
  incremental.add("I prefer Python for analytics.", "incremental_user");
  const audit = new schemaDb(incrementalPath);
  audit.exec("CREATE TABLE write_audit (table_name TEXT NOT NULL)");
  for (const table of ["l0_logs", "l1_atoms", "l2_episodes", "l3_personas", "l4_policies", "l6_proofs"]) {
    audit.exec(`CREATE TRIGGER audit_${table}_update AFTER UPDATE ON ${table} BEGIN INSERT INTO write_audit VALUES ('${table}'); END`);
  }
  audit.close();

  incremental.add("I prefer Rust for systems programming.", "incremental_user");
  const verify = new schemaDb(incrementalPath);
  // Updating a preference legitimately updates its persona row; no historical
  // log, atom, episode, policy, or proof row should be rewritten.
  const updatedTables = verify.prepare(
    "SELECT table_name, count(*) AS n FROM write_audit GROUP BY table_name"
  ).all();
  assert.deepStrictEqual(updatedTables, [{ table_name: "l3_personas", n: 1 }]);
  assert.strictEqual(verify.prepare("SELECT count(*) AS n FROM l0_logs").get().n, 2);
  verify.close();

  incremental.deleteUser("incremental_user");
  incremental.close();
  const afterIncrementalDelete = new InfinityDB({ dbPath: incrementalPath });
  assert.strictEqual(afterIncrementalDelete.stats().l0RawLogs, 0);
  assert.strictEqual(afterIncrementalDelete.stats().l1AtomFacts, 0);
  afterIncrementalDelete.close();
} finally {
  fs.rmSync(incrementalDir, { recursive: true, force: true });
}

// 13. Temporal supersession and historical recall (in-memory).
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

// 14. Temporal relation survives SQLite reopen and is removed on tenant deletion.
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

// 15. Fallback extraction preserves exact L0 source ranges for L6 proofs.
const fallback = new InfinityDB();
const fallbackConversation = "   The deployment uses isolated build workers. A separate staging environment validates every release!";
const fallbackLog = fallback.add(fallbackConversation, "fallback_user");
const fallbackAtoms = Array.from(fallback.atoms.values()).filter(atom => atom.sourceId === fallbackLog);
assert.strictEqual(fallbackAtoms.length, 2);
for (const atom of fallbackAtoms) {
  const fallbackProof = fallback.getProof(atom.id);
  assert.strictEqual(fallbackProof.verified, true);
  assert.strictEqual(
    fallback.logs.get(fallbackLog).raw.slice(...fallbackProof.charRange),
    atom.fact
  );
}
assert.strictEqual(fallbackAtoms[0].fact, "The deployment uses isolated build workers");
assert.strictEqual(fallbackAtoms[1].fact, "A separate staging environment validates every release");
fallback.close();

console.log("[PASS] All JavaScript tests passed successfully!");

// Shared cross-SDK policy conformance fixture.
const conformanceDb = new InfinityDB();
const globalPolicy = policyFixture.global_policy;
const tenantPolicy = policyFixture.tenant_policy;
conformanceDb.addPolicy(globalPolicy.rule, globalPolicy.domain, globalPolicy.priority);
conformanceDb.addPolicy(tenantPolicy.rule, tenantPolicy.domain, tenantPolicy.priority, tenantPolicy.tenant_id);
const policyRules = userId => conformanceDb.recall("policy", userId ? { userId } : {}).policies.map(p => p.rule);
assert.deepStrictEqual(policyRules(policyFixture.tenant_a), policyFixture.expected.tenant_a);
assert.deepStrictEqual(policyRules(policyFixture.tenant_b), policyFixture.expected.tenant_b);
assert.deepStrictEqual(policyRules(), policyFixture.expected.unscoped);
conformanceDb.deleteUser(policyFixture.tenant_a);
assert.deepStrictEqual(policyRules(policyFixture.tenant_a), policyFixture.expected.tenant_a_after_delete);
conformanceDb.close();


// Shared temporal behavior fixture: current recall, historical recall, and proof preservation.
const sharedTemporal = policyFixture.temporal;
const sharedTemporalDb = new InfinityDB();
const sharedOldLog = sharedTemporalDb.add(sharedTemporal.old_fact, sharedTemporal.user_id);
const sharedOldAtom = Array.from(sharedTemporalDb.atoms.values()).find(a => a.sourceId === sharedOldLog);
const sharedHistoricalCutoff = sharedOldAtom.ts + 0.000001;
const sharedNewLog = sharedTemporalDb.add(sharedTemporal.new_fact, sharedTemporal.user_id);
const sharedNewAtom = Array.from(sharedTemporalDb.atoms.values()).find(a => a.sourceId === sharedNewLog);
sharedTemporalDb.supersedeFact(sharedOldAtom.id, sharedNewAtom.id, sharedTemporal.user_id);
const sharedCurrent = sharedTemporalDb.recall(sharedTemporal.query, { userId: sharedTemporal.user_id });
const sharedHistorical = sharedTemporalDb.recall(sharedTemporal.query, { userId: sharedTemporal.user_id, asOf: sharedHistoricalCutoff });
assert.strictEqual(sharedCurrent.results.length, 1);
assert.ok(sharedCurrent.results[0].fact.includes(sharedTemporal.expected_current_fact));
assert.strictEqual(sharedHistorical.results.length, 1);
assert.ok(sharedHistorical.results[0].fact.includes(sharedTemporal.expected_historical_fact));
assert.strictEqual(sharedTemporalDb.getProof(sharedOldAtom.id).verified, true);
sharedTemporalDb.close();


// Shared proof verification fixture: tampering with the source log must invalidate it.
const sharedProof = policyFixture.proof;
const sharedProofDb = new InfinityDB();
const sharedProofLog = sharedProofDb.add(sharedProof.source_text, sharedProof.user_id);
const sharedProofAtom = Array.from(sharedProofDb.atoms.values()).find(a => a.sourceId === sharedProofLog);
assert.strictEqual(sharedProofDb.getProof(sharedProofAtom.id).verified, true);
sharedProofDb.logs.get(sharedProofLog).raw = sharedProof.tampered_source;
assert.strictEqual(sharedProofDb.getProof(sharedProofAtom.id).verified, false);
sharedProofDb.close();


// Shared tenant data fixture: facts stay isolated and deleting one tenant preserves the other.
const sharedTenantData = policyFixture.tenant_data;
const sharedTenantDb = new InfinityDB();
sharedTenantDb.add(sharedTenantData.fact_a, sharedTenantData.tenant_a);
sharedTenantDb.add(sharedTenantData.fact_b, sharedTenantData.tenant_b);
const sharedTenantA = sharedTenantDb.recall(sharedTenantData.query, { userId: sharedTenantData.tenant_a }).results;
const sharedTenantB = sharedTenantDb.recall(sharedTenantData.query, { userId: sharedTenantData.tenant_b }).results;
assert.strictEqual(sharedTenantA.length, 1);
assert.ok(sharedTenantA[0].fact.includes(sharedTenantData.expected_a));
assert.strictEqual(sharedTenantB.length, 1);
assert.ok(sharedTenantB[0].fact.includes(sharedTenantData.expected_b));
sharedTenantDb.deleteUser(sharedTenantData.tenant_a);
assert.deepStrictEqual(sharedTenantDb.recall(sharedTenantData.query, { userId: sharedTenantData.tenant_a }).results, []);
const sharedTenantRemaining = sharedTenantDb.recall(sharedTenantData.query, { userId: sharedTenantData.tenant_b }).results;
assert.strictEqual(sharedTenantRemaining.length, 1);
assert.ok(sharedTenantRemaining[0].fact.includes(sharedTenantData.expected_b));
sharedTenantDb.close();


// Shared input validation fixture: reject invalid writes and keep empty-query recall non-broad.
const sharedValidation = policyFixture.validation;
const sharedValidationDb = new InfinityDB();
assert.throws(() => sharedValidationDb.add(sharedValidation.whitespace_conversation, sharedValidation.valid_user));
assert.throws(() => sharedValidationDb.add(
  sharedValidation.valid_conversation,
  sharedValidation.valid_user,
  null,
  sharedValidation.invalid_importance
));
const sharedEmptyRecall = sharedValidationDb.recall(sharedValidation.empty_query, { userId: sharedValidation.valid_user });
assert.deepStrictEqual(sharedEmptyRecall.results, []);
assert.strictEqual(sharedEmptyRecall.tokensSaved, null);
sharedValidationDb.close();


// Tenant deletion must not mutate the in-memory view when its SQLite transaction rolls back.
const deleteRollbackDir = fs.mkdtempSync(path.join(os.tmpdir(), "infinitydb-delete-rollback-"));
const deleteRollbackPath = path.join(deleteRollbackDir, "delete-rollback.sqlite");
const deleteRollbackDb = new InfinityDB({ dbPath: deleteRollbackPath });
const deleteRollbackLog = deleteRollbackDb.add(
  "I prefer Rust for transactional deletion tests.", "delete_rollback_tenant"
);
deleteRollbackDb.addPolicy(
  "Tenant-only deletion rollback policy", "security", 2, "delete_rollback_tenant"
);
const deleteRollbackAtom = Array.from(deleteRollbackDb.atoms.values())
  .find(atom => atom.sourceId === deleteRollbackLog);
deleteRollbackDb._storage.exec(`
  CREATE TRIGGER reject_tenant_atom_delete
  BEFORE DELETE ON l1_atoms
  BEGIN
    SELECT RAISE(ABORT, 'simulated tenant deletion failure');
  END;
`);
assert.throws(
  () => deleteRollbackDb.deleteUser("delete_rollback_tenant"),
  /simulated tenant deletion failure/
);
assert.ok(deleteRollbackDb.logs.has(deleteRollbackLog));
assert.ok(deleteRollbackDb.atoms.has(deleteRollbackAtom.id));
assert.ok(deleteRollbackDb.proofs.has(deleteRollbackAtom.id));
assert.strictEqual(deleteRollbackDb.relations.length, 0);
assert.strictEqual(deleteRollbackDb.policies.size, 1);
assert.strictEqual(
  deleteRollbackDb._storage.prepare("SELECT count(*) AS count FROM l0_logs WHERE user_id=?")
    .get("delete_rollback_tenant").count,
  1
);
deleteRollbackDb._storage.exec("DROP TRIGGER reject_tenant_atom_delete");
const deleteRollbackResult = deleteRollbackDb.deleteUser("delete_rollback_tenant");
assert.strictEqual(deleteRollbackResult.l0LogsDeleted, 1);
assert.strictEqual(deleteRollbackResult.l1AtomsDeleted, 1);
assert.strictEqual(deleteRollbackResult.l4TenantPoliciesDeleted, 1);
assert.strictEqual(deleteRollbackDb.logs.size, 0);
assert.strictEqual(deleteRollbackDb.atoms.size, 0);
assert.strictEqual(deleteRollbackDb.proofs.size, 0);
assert.strictEqual(deleteRollbackDb.policies.size, 0);
deleteRollbackDb.close();
fs.rmSync(deleteRollbackDir, { recursive: true, force: true });
