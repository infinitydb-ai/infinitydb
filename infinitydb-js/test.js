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

console.log("[PASS] All JavaScript tests passed successfully!");

// sdk tests
