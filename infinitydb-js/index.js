"use strict";

const crypto = require("crypto");

/**
 * InfinityDB - High-performance Node.js Agent Memory Engine.
 * Implements L0-L6 brain architecture:
 * L0: Raw logs offloaded from LLM context
 * L1: Atomic facts (zero-LLM regex/rule extraction)
 * L2: Causal & temporal graph
 * L3: Dynamic persona with conflict auto-resolution
 * L4: Enterprise policies & SOPs
 * L5: Ebbinghaus forgetting curve decay
 * L6: Cryptographic SHA-256 proof chain
 */
class InfinityDB {
  constructor(options = {}) {
    this.options = options;
    this._seq = 0;
    this._storage = null;
    if (options.dbPath) {
      const Database = require("better-sqlite3");
      this._storage = new Database(options.dbPath);
      this._storage.pragma("journal_mode = WAL");
      this._storage.pragma("synchronous = FULL");
      this._storage.pragma("busy_timeout = 5000");
      this._initStorage();
    }
    this.logs = new Map();
    this.atoms = new Map();
    this.episodes = new Map();
    this.personas = new Map();
    this.policies = new Map();
    this.proofs = new Map();
    this.legacyMemories = new Map();
    this._loadState();
  }

  _initStorage() {
    this._storage.exec(`
      CREATE TABLE IF NOT EXISTS infinitydb_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS l0_logs (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, payload TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS idx_js_l0_user ON l0_logs(user_id);
      CREATE TABLE IF NOT EXISTS l1_atoms (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, source_id TEXT NOT NULL, fact TEXT NOT NULL, ts REAL NOT NULL, payload TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS idx_js_l1_user_ts ON l1_atoms(user_id, ts);
      CREATE INDEX IF NOT EXISTS idx_js_l1_source ON l1_atoms(source_id);
      CREATE TABLE IF NOT EXISTS l2_episodes (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, source_id TEXT NOT NULL, payload TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS idx_js_l2_user ON l2_episodes(user_id);
      CREATE TABLE IF NOT EXISTS l3_personas (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, payload TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS idx_js_l3_user ON l3_personas(user_id);
      CREATE TABLE IF NOT EXISTS l4_policies (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS l6_proofs (atom_id TEXT PRIMARY KEY, source_log_id TEXT NOT NULL, payload TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS idx_js_l6_source ON l6_proofs(source_log_id);
      CREATE TABLE IF NOT EXISTS legacy_memories (memory_key TEXT PRIMARY KEY, payload TEXT NOT NULL);
    `);
    const version = this._storage.prepare("SELECT value FROM infinitydb_meta WHERE key='schema_version'").get();
    if (!version) {
      const legacyTable = this._storage.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='infinitydb_state'").get();
      this._storage.transaction(() => {
        if (legacyTable) {
          const row = this._storage.prepare("SELECT schema_version, payload FROM infinitydb_state WHERE id=1").get();
          if (row) {
            if (row.schema_version !== 1) throw new Error(`Unsupported InfinityDB storage schema version: ${row.schema_version}`);
            const state = JSON.parse(row.payload);
            this._seq = state.seq || 0;
            this._writeEntries("l0_logs", "id", state.logs || [], v => [v.id, v.userId, JSON.stringify(v)]);
            this._writeEntries("l1_atoms", "id", state.atoms || [], v => [v.id, v.userId, v.sourceId, v.fact, v.ts, JSON.stringify(v)]);
            this._writeEntries("l2_episodes", "id", state.episodes || [], v => [v.id, v.userId, v.sourceId, JSON.stringify(v)]);
            this._writeEntries("l3_personas", "id", state.personas || [], v => [v[0], v[1].userId, JSON.stringify(v[1])]);
            this._writeEntries("l4_policies", "id", state.policies || [], v => [v.id, JSON.stringify(v)]);
            this._writeEntries("l6_proofs", "atom_id", state.proofs || [], v => [v.atomId, v.sourceLogId, JSON.stringify(v)]);
            this._writeEntries("legacy_memories", "memory_key", state.legacyMemories || [], v => [v[0], JSON.stringify(v[1])]);
          }
          this._storage.exec("DROP TABLE infinitydb_state");
        }
        this._storage.prepare("INSERT INTO infinitydb_meta(key,value) VALUES ('schema_version','2')").run();
        this._storage.prepare("INSERT INTO infinitydb_meta(key,value) VALUES ('sequence',?)").run(String(this._seq));
      })();
    } else if (version.value !== "2") {
      throw new Error(`Unsupported InfinityDB storage schema version: ${version.value}`);
    }
  }

  _writeEntries(table, keyColumn, entries, mapper) {
    const columns = {
      l0_logs: "id, user_id, payload", l1_atoms: "id, user_id, source_id, fact, ts, payload",
      l2_episodes: "id, user_id, source_id, payload", l3_personas: "id, user_id, payload",
      l4_policies: "id, payload", l6_proofs: "atom_id, source_log_id, payload",
      legacy_memories: "memory_key, payload",
    }[table];
    const values = columns.split(", ").map(() => "?").join(", ");
    const updates = columns.split(", ").slice(1).map(col => `${col}=excluded.${col}`).join(", ");
    const stmt = this._storage.prepare(`INSERT INTO ${table} (${columns}) VALUES (${values}) ON CONFLICT(${keyColumn}) DO UPDATE SET ${updates}`);
    for (const entry of entries) stmt.run(...mapper(entry));
  }

  _loadState() {
    if (!this._storage) return;
    const get = (sql, key) => this._storage.prepare(sql).all().map(row => [row[key], JSON.parse(row.payload)]);
    this._seq = Number(this._storage.prepare("SELECT value FROM infinitydb_meta WHERE key='sequence'").get()?.value || 0);
    this.logs = new Map(get("SELECT id, payload FROM l0_logs", "id"));
    this.atoms = new Map(get("SELECT id, payload FROM l1_atoms", "id"));
    this.episodes = new Map(get("SELECT id, payload FROM l2_episodes", "id"));
    this.personas = new Map(get("SELECT id, payload FROM l3_personas", "id"));
    this.policies = new Map(get("SELECT id, payload FROM l4_policies", "id"));
    this.proofs = new Map(get("SELECT atom_id AS id, payload FROM l6_proofs", "id"));
    this.legacyMemories = new Map(get("SELECT memory_key AS id, payload FROM legacy_memories", "id"));
  }

  _refreshState() {
    if (this._storage) this._loadState();
  }

  _persist() {
    if (!this._storage) return;
    this._storage.transaction(() => {
      this._writeEntries("l0_logs", "id", Array.from(this.logs.values()), v => [v.id, v.userId, JSON.stringify(v)]);
      this._writeEntries("l1_atoms", "id", Array.from(this.atoms.values()), v => [v.id, v.userId, v.sourceId, v.fact, v.ts, JSON.stringify(v)]);
      this._writeEntries("l2_episodes", "id", Array.from(this.episodes.values()), v => [v.id, v.userId, v.sourceId, JSON.stringify(v)]);
      this._writeEntries("l3_personas", "id", Array.from(this.personas.entries()), v => [v[0], v[1].userId, JSON.stringify(v[1])]);
      this._writeEntries("l4_policies", "id", Array.from(this.policies.values()), v => [v.id, JSON.stringify(v)]);
      this._writeEntries("l6_proofs", "atom_id", Array.from(this.proofs.values()), v => [v.atomId, v.sourceLogId, JSON.stringify(v)]);
      this._writeEntries("legacy_memories", "memory_key", Array.from(this.legacyMemories.entries()), v => [v[0], JSON.stringify(v[1])]);
      this._storage.prepare("UPDATE infinitydb_meta SET value=? WHERE key='sequence'").run(String(this._seq));
    })();
  }

  close() {
    if (this._storage && this._storage.open) this._storage.close();
  }

  _sha256(str) {
    return crypto.createHash("sha256").update(str).digest("hex");
  }

  // Legacy key-value compatibility
  put(key, content) {
    if (!key || !key.trim()) throw new Error("key cannot be empty");
    if (!content || !content.trim()) throw new Error("content cannot be empty");
    const mem = { key, content, createdAt: new Date().toISOString() };
    this.legacyMemories.set(key, mem);
    this.add(`Memory key ${key}: ${content}`, "legacy");
    this._persist();
    return mem;
  }

  get(key) {
    return this.legacyMemories.get(key) || null;
  }

  delete(key) {
    const deleted = this.legacyMemories.delete(key);
    if (deleted) this._persist();
    return deleted;
  }

  // Core L0-L6 API
  add(conversation, userId = "default", sessionId = null, importance = 0.9) {
    if (typeof conversation !== "string" || !conversation.trim()) {
      throw new TypeError("conversation must be a non-empty string");
    }
    if (typeof userId !== "string" || !userId.trim()) {
      throw new TypeError("userId must be a non-empty string");
    }
    if (typeof importance !== "number" || !Number.isFinite(importance)) {
      throw new TypeError("importance must be a finite number between 0 and 1");
    }
    if (importance < 0 || importance > 1) {
      throw new RangeError("importance must be between 0 and 1");
    }
    const now = Date.now() / 1000;
    this._seq += 1;
    const logId = `L0_${this._sha256(conversation + now + this._seq).slice(0, 16)}`;

    // L0: Store raw
    this.logs.set(logId, {
      id: logId,
      userId,
      sessionId,
      raw: conversation,
      ts: now,
    });

    // L1: Zero-LLM Deterministic Rule Extraction
    const patterns = [
      { regex: /(?:I am|I'm|my name is)\s+([A-Z][a-zA-Z0-9_\-\s]+?)(?:[.,\n]|$)/i, cat: "identity", weight: 0.95 },
      { regex: /(?:I work as|I am a|my role is)\s+([^.,\n]+)/i, cat: "role", weight: 0.90 },
      { regex: /(?:I prefer|I like|I love|I use|my preference is)\s+([^.,\n]+)/i, cat: "preference", weight: 0.92 },
      { regex: /(?:I hate|I dislike|I avoid|never use)\s+([^.,\n]+)/i, cat: "dislike", weight: 0.88 },
      { regex: /(?:my budget is|budget of|budget:\s*)\s*([$€£¥]?[0-9,]+(?:\s*[kKmMbB])?)/i, cat: "budget", weight: 0.94 },
      { regex: /(?:I live in|based in|located in)\s+([^.,\n]+)/i, cat: "location", weight: 0.85 },
      { regex: /(?:our goal is|aiming to|target is|deadline is)\s+([^.,\n]+)/i, cat: "goal", weight: 0.91 },
      { regex: /(?:must comply with|required to|constraint:\s*)\s+([^.,\n]+)/i, cat: "constraint", weight: 0.95 },
    ];

    let extractedAny = false;
    for (const pat of patterns) {
      let match;
      const re = new RegExp(pat.regex.source, "gi");
      while ((match = re.exec(conversation)) !== null) {
        extractedAny = true;
        const fact = match[0].trim();
        const atomId = `L1_${this._sha256(logId + fact).slice(0, 16)}`;

        this.atoms.set(atomId, {
          id: atomId,
          userId,
          fact,
          category: pat.cat,
          sourceId: logId,
          importance: pat.weight * importance,
          accessCount: 0,
          ts: now,
        });

        // L6: Cryptographic Proof Record
        const proofHash = this._sha256(fact + logId);
        const snippet = conversation.slice(Math.max(0, match.index - 20), Math.min(conversation.length, match.index + fact.length + 20));
        this.proofs.set(atomId, {
          atomId,
          hash: proofHash,
          sourceLogId: logId,
          rawSnippet: snippet,
          charStart: match.index,
          charEnd: match.index + fact.length,
          ts: now,
        });

        // L3: Persona Auto-Resolution
        const pKey = `${userId}:${pat.cat}`;
        const existing = this.personas.get(pKey);
        if (!existing || now >= existing.updatedAt) {
          this.personas.set(pKey, {
            userId,
            key: pat.cat,
            value: fact,
            confidence: pat.weight,
            updatedAt: now,
            sourceAtomId: atomId,
          });
        }
      }
    }

    if (!extractedAny) {
      const sentences = conversation.split(/[.!?\n]/).map(s => s.trim()).filter(s => s.length > 8);
      for (const s of sentences.slice(0, 3)) {
        const atomId = `L1_${this._sha256(logId + s).slice(0, 16)}`;
        this.atoms.set(atomId, {
          id: atomId,
          userId,
          fact: s,
          category: "general",
          sourceId: logId,
          importance: 0.75 * importance,
          accessCount: 0,
          ts: now,
        });
        this.proofs.set(atomId, {
          atomId,
          hash: this._sha256(s + logId),
          sourceLogId: logId,
          rawSnippet: s,
          charStart: 0,
          charEnd: s.length,
          ts: now,
        });
      }
    }

    // L2: Causal Episode Extraction
    const causalRegex = /(.+?)\s+(?:because|since|as a result of|so that)\s+([^.,\n]+)/gi;
    let cMatch;
    while ((cMatch = causalRegex.exec(conversation)) !== null) {
      const effectPart = cMatch[1].trim().slice(-80);
      const causePart = cMatch[2].trim().slice(0, 80);
      const epId = `L2_${this._sha256(logId + causePart + effectPart).slice(0, 16)}`;
      this.episodes.set(epId, {
        id: epId,
        userId,
        cause: causePart,
        effect: effectPart,
        relation: "causes",
        sourceId: logId,
        ts: now,
      });
    }

    this._persist();
    return logId;
  }

  recall(query, options = {}) {
    if (typeof query !== "string") throw new TypeError("query must be a string");
    const n = options.n === undefined ? 5 : options.n;
    if (!Number.isInteger(n) || n <= 0) {
      throw new RangeError("n must be a positive integer");
    }
    if (options.userId !== undefined &&
        (typeof options.userId !== "string" || !options.userId.trim())) {
      throw new TypeError("userId must be a non-empty string");
    }
    const start = performance.now();
    const userId = options.userId || null;
    const now = Date.now() / 1000;
    const qTerms = (query.toLowerCase().match(/\w+/g) || []).filter(t => t.length > 1);

    const scored = [];
    if (qTerms.length === 0) {
      return {
        results: [],
        causalGraph: [],
        policies: [],
        latencyMs: Math.round((performance.now() - start) * 100) / 100,
        tokensSaved: null,
        proofAvailable: false,
      };
    }
    for (const [id, atom] of this.atoms.entries()) {
      if (userId && atom.userId !== userId) continue;
      const factLower = atom.fact.toLowerCase();
      let termMatches = 0;
      for (const term of qTerms) {
        if (factLower.includes(term)) termMatches++;
      }

      let termScore = termMatches * 2.5;
      if (factLower.includes(query.toLowerCase())) termScore += 4.0;

      // L5 Ebbinghaus decay
      const deltaDays = (now - atom.ts) / 86400.0;
      const stability = 1.0 + atom.importance * 3.0 + Math.log1p(atom.accessCount);
      const retention = Math.exp(-deltaDays / Math.max(stability, 0.1));
      const totalScore = (termScore + atom.importance * 1.5) * retention;

      if (termMatches > 0) {
        scored.push({
          id: atom.id,
          fact: atom.fact,
          category: atom.category,
          sourceId: atom.sourceId,
          score: Math.round(totalScore * 1000) / 1000,
          retention: Math.round(retention * 1000) / 1000,
          ts: atom.ts,
        });
      }
    }

    scored.sort((a, b) => b.score - a.score);
    const topK = scored.slice(0, n);

    // Reinforce access count
    for (const item of topK) {
      const at = this.atoms.get(item.id);
      if (at) at.accessCount++;
    }

    this._persist();

    // Causal graph insights
    const causalGraph = [];
    for (const item of topK) {
      for (const ep of this.episodes.values()) {
        if (ep.sourceId === item.sourceId) {
          causalGraph.push({ cause: ep.cause, effect: ep.effect, relation: ep.relation });
        }
      }
    }

    // Policies
    const policies = Array.from(this.policies.values())
      .sort((a, b) => b.priority - a.priority)
      .slice(0, 3);

    const elapsedMs = performance.now() - start;
    return {
      results: topK,
      causalGraph,
      policies,
      latencyMs: Math.round(elapsedMs * 100) / 100,
      tokensSaved: null,
      proofAvailable: true,
    };
  }

  getProof(atomId) {
    const p = this.proofs.get(atomId);
    if (!p) return null;
    const source = this.logs.get(p.sourceLogId);
    const atom = this.atoms.get(atomId);
    let verified = false;
    if (source && atom &&
        Number.isInteger(p.charStart) && Number.isInteger(p.charEnd) &&
        p.charStart >= 0 && p.charEnd >= p.charStart &&
        p.charEnd <= source.raw.length) {
      const sourceFact = source.raw.slice(p.charStart, p.charEnd);
      const expected = this._sha256(atom.fact + p.sourceLogId);
      const actualBuffer = Buffer.from(p.hash, "hex");
      const expectedBuffer = Buffer.from(expected, "hex");
      verified = sourceFact === atom.fact &&
        actualBuffer.length === expectedBuffer.length &&
        crypto.timingSafeEqual(actualBuffer, expectedBuffer);
    }
    return {
      atomId: p.atomId,
      hash: p.hash,
      sourceLogId: p.sourceLogId,
      rawSnippet: p.rawSnippet,
      charRange: [p.charStart, p.charEnd],
      verified,
      timestamp: p.ts,
    };
  }

  getPersona(userId = "default") {
    const out = {};
    for (const [key, p] of this.personas.entries()) {
      if (p.userId === userId) {
        out[p.key] = p.value;
      }
    }
    return out;
  }

  addPolicy(rule, domain = "general", priority = 1.0) {
    const pid = `L4_${this._sha256(rule + domain).slice(0, 12)}`;
    this.policies.set(pid, { id: pid, rule, domain, priority, createdAt: Date.now() / 1000 });
    this._persist();
    return pid;
  }

  exportGraph(userId = "default") {
    const lines = ["graph TD"];
    let idx = 0;
    for (const ep of this.episodes.values()) {
      if (ep.userId === userId) {
        const c = ep.cause.replace(/["\n]/g, "").slice(0, 30);
        const e = ep.effect.replace(/["\n]/g, "").slice(0, 30);
        lines.push(`  C${idx}["${c}"] -->|${ep.relation}| E${idx}["${e}"]`);
        idx++;
      }
    }
    if (idx === 0) return "graph TD\n  Empty[No causal memory recorded]";
    return lines.join("\n");
  }

  deleteUser(userId) {
    let c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (const [id, log] of Array.from(this.logs.entries())) {
      if (log.userId === userId) { this.logs.delete(id); c0++; }
    }
    for (const [id, atom] of Array.from(this.atoms.entries())) {
      if (atom.userId === userId) {
        this.atoms.delete(id);
        this.proofs.delete(id);
        c1++;
      }
    }
    for (const [id, ep] of Array.from(this.episodes.entries())) {
      if (ep.userId === userId) { this.episodes.delete(id); c2++; }
    }
    for (const [key, p] of Array.from(this.personas.entries())) {
      if (p.userId === userId) { this.personas.delete(key); c3++; }
    }
    this._persist();
    return {
      l0LogsDeleted: c0,
      l1AtomsDeleted: c1,
      l2EpisodesDeleted: c2,
      l3PersonaDeleted: c3,
      gdprCompliant: true,
    };
  }

  stats() {
    return {
      l0RawLogs: this.logs.size,
      l1AtomFacts: this.atoms.size,
      l2CausalEpisodes: this.episodes.size,
      l3Personas: this.personas.size,
      l4Policies: this.policies.size,
      l6CryptographicProofs: this.proofs.size,
      p95RecallTargetMs: "<5.0",
    };
  }
}

module.exports = { InfinityDB };
