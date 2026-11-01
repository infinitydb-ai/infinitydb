"use strict";

// Reproducible single-engine benchmark for the checked-out Node.js SQLite backend.
// Run from the repository root:
//   node infinitydb-js/benchmark.js --iterations 1000 --output benchmark-results-node.json

const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { performance } = require("perf_hooks");
const { InfinityDB } = require("./index.js");

const FACTS = [
  "I prefer Python for data processing.",
  "I prefer Rust for systems programming.",
  "I prefer TypeScript for web development.",
  "I prefer Go for network services.",
  "I prefer Kotlin for Android development."
];

function percentile(values, pct) {
  if (!values.length) throw new Error("cannot calculate a percentile from no samples");
  const ordered = [...values].sort((a, b) => a - b);
  const rank = Math.max(1, Math.ceil((pct / 100) * ordered.length));
  return ordered[rank - 1];
}

function parseArgs(argv) {
  const options = { iterations: 1000, output: null };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--iterations") {
      options.iterations = Number(argv[++i]);
    } else if (argv[i] === "--output") {
      options.output = argv[++i];
    } else {
      throw new Error(`unknown argument: ${argv[i]}`);
    }
  }
  if (!Number.isSafeInteger(options.iterations) || options.iterations < 1) {
    throw new Error("--iterations must be a positive safe integer");
  }
  return options;
}

function run(iterations, output = null, quiet = false) {
  if (!Number.isSafeInteger(iterations) || iterations < 1) {
    throw new Error("iterations must be a positive safe integer");
  }

  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "infinitydb-node-bench-"));
  const dbPath = path.join(tempDir, "benchmark.sqlite");
  let db;
  let dbBytes = 0;
  let walBytes = 0;
  let pageCount = 0;
  let pageSize = 0;
  let sqliteVersion = "unknown";
  let stats = null;
  let result;

  try {
    db = new InfinityDB({ dbPath });
    const ingestStart = performance.now();
    for (let i = 0; i < iterations; i += 1) {
      const userId = `tenant-${i % 10}`;
      db.add(FACTS[i % FACTS.length], userId, null, 0.85);
    }
    const ingestSeconds = (performance.now() - ingestStart) / 1000;

    const isolationDb = new InfinityDB();
    try {
      isolationDb.add("I prefer Python for isolated tenant testing.", "tenant-a");
      const tenantA = isolationDb.recall("Python", { userId: "tenant-a" });
      const tenantB = isolationDb.recall("Python", { userId: "tenant-b" });
      const proofId = tenantA.results.length ? tenantA.results[0].id : null;
      const proof = proofId ? isolationDb.getProof(proofId) : null;
      const correctness = {
        tenant_a_recall: tenantA.results.length > 0,
        tenant_b_isolated: tenantB.results.length === 0,
        source_proof_verified: Boolean(proof && proof.verified)
      };
      if (!Object.values(correctness).every(Boolean)) {
        throw new Error(`benchmark correctness checks failed: ${JSON.stringify(correctness)}`);
      }
      result = { correctness };
    } finally {
      isolationDb.close();
    }

    const recallLatenciesMs = [];
    const recallCount = Math.max(100, Math.min(iterations, 5000));
    for (let i = 0; i < recallCount; i += 1) {
      const userId = `tenant-${i % 10}`;
      const started = performance.now();
      db.recall("prefer Python Rust TypeScript Go Kotlin", { n: 5, userId });
      recallLatenciesMs.push(performance.now() - started);
    }

    pageCount = db._storage.pragma("page_count", { simple: true });
    pageSize = db._storage.pragma("page_size", { simple: true });
    sqliteVersion = db._storage.prepare("SELECT sqlite_version() AS version").get().version;
    stats = db.stats();
    result = {
      benchmark: "infinitydb-node-sqlite-local",
      benchmark_version: 1,
      iterations,
      recall_samples: recallCount,
      correctness: result.correctness,
      ingestion: {
        elapsed_seconds: Number(ingestSeconds.toFixed(6)),
        operations_per_second: Number((iterations / Math.max(ingestSeconds, 1e-12)).toFixed(3))
      },
      recall_latency_ms: {
        p50: Number(percentile(recallLatenciesMs, 50).toFixed(4)),
        p95: Number(percentile(recallLatenciesMs, 95).toFixed(4)),
        p99: Number(percentile(recallLatenciesMs, 99).toFixed(4)),
        samples: recallLatenciesMs.length
      },
      storage: {},
      environment: {
        node: process.version,
        platform: `${os.platform()} ${os.release()}`,
        machine: os.arch(),
        sqlite: sqliteVersion,
        cpu_count: os.cpus().length,
        git_commit: process.env.GITHUB_SHA || process.env.GIT_COMMIT || null
      },
      dataset: {
        sha256: crypto.createHash("sha256").update(FACTS.join("\n"), "utf8").digest("hex"),
        tenant_count: 10,
        fact_templates: FACTS.length,
        storage_backend: "temporary file-backed SQLite"
      },
      engine_stats: stats,
      limitations: [
        "Measures only the checked-out InfinityDB implementation.",
        "Synthetic deterministic workload; not a representative customer trace.",
        "Results depend on hardware, OS, SQLite build, filesystem, and workload.",
        "No third-party product comparison or token-savings estimate is performed."
      ]
    };
  } finally {
    if (db) {
      db.close();
      if (fs.existsSync(dbPath)) dbBytes = fs.statSync(dbPath).size;
      const walPath = `${dbPath}-wal`;
      if (fs.existsSync(walPath)) walBytes = fs.statSync(walPath).size;
    }
    if (result) {
      result.storage = {
        sqlite_file_bytes_after_close: dbBytes,
        wal_file_bytes_after_close: walBytes,
        database_and_wal_bytes: dbBytes + walBytes,
        page_count: pageCount,
        page_size_bytes: pageSize
      };
    }
    fs.rmSync(tempDir, { recursive: true, force: true });
  }

  const rendered = JSON.stringify(result, null, 2);
  if (output) fs.writeFileSync(output, rendered + "\n", "utf8");
  process.stdout.write(rendered + "\n");
  return result;
}

if (require.main === module) {
  try {
    const options = parseArgs(process.argv.slice(2));
    run(options.iterations, options.output);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}

module.exports = { percentile, run };
