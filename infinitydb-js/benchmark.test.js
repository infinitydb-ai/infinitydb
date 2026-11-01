"use strict";

const assert = require("assert");
const { percentile, run } = require("./benchmark.js");

assert.strictEqual(percentile([9, 1, 5, 3, 7], 50), 5);
assert.strictEqual(percentile([9, 1, 5, 3, 7], 95), 9);
assert.throws(() => percentile([], 50), /no samples/);
assert.throws(() => run(0), /positive safe integer/);

const result = run(5, null, true);
assert.strictEqual(result.iterations, 5);
assert.strictEqual(result.correctness.tenant_a_recall, true);
assert.strictEqual(result.correctness.tenant_b_isolated, true);
assert.strictEqual(result.correctness.source_proof_verified, true);
assert.strictEqual(result.recall_latency_ms.samples, 100);
assert.ok(result.storage.database_and_wal_bytes > 0);

process.stdout.write("[PASS] Node.js benchmark harness tests passed.\n");
