export interface MemoryResult {
  id: string;
  fact: string;
  category: string;
  sourceId: string;
  score: number;
  retention: number;
  ts: number;
}

export interface RecallResponse {
  results: MemoryResult[];
  causalGraph: Array<{ cause: string; effect: string; relation: string }>;
  policies: Array<{ id: string; rule: string; domain: string; priority: number }>;
  latencyMs: number;
  tokensSaved: string;
  proofAvailable: boolean;
}

export interface ProofResponse {
  atomId: string;
  hash: string;
  sourceLogId: string;
  rawSnippet: string;
  charRange: [number, number];
  verified: boolean;
  timestamp: number;
}

export class InfinityDB {
  constructor(options?: Record<string, any>);
  add(conversation: string, userId?: string, sessionId?: string | null, importance?: number): string;
  recall(query: string, options?: { n?: number; userId?: string | null }): RecallResponse;
  getProof(atomId: string): ProofResponse | null;
  getPersona(userId?: string): Record<string, string>;
  addPolicy(rule: string, domain?: string, priority?: number): string;
  exportGraph(userId?: string): string;
  deleteUser(userId: string): Record<string, any>;
  stats(): Record<string, any>;
  put(key: string, content: string): { key: string; content: string; createdAt: string };
  get(key: string): { key: string; content: string; createdAt: string } | null;
  delete(key: string): boolean;
}
