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
  fs.rmSync(tempDir, { recursive: true, force: true });
}

console.log("[PASS] All JavaScript tests passed successfully!");

// sdk tests
