export type MemoryInput = {
  content: string;
  tenant_id?: string;
  kind?: string;
  metadata?: Record<string, unknown>;
  raw_evidence?: string;
  confidence?: number;
};

export type SearchHit = {
  id: string;
  content: string;
  kind: string;
  score: number;
  matched_by: string[];
};

export class InfinityDBClient {
  constructor(
    private readonly baseUrl = "http://127.0.0.1:8000",
    private readonly fetcher: typeof fetch = fetch,
  ) {}

  private async request<T>(path: string, body?: unknown): Promise<T> {
    const response = await this.fetcher(`${this.baseUrl}${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(`InfinityDB request failed: ${response.status} ${await response.text()}`);
    }
    return response.json() as Promise<T>;
  }

  add(input: MemoryInput): Promise<{ id: string }> {
    return this.request("/v1/memories", input);
  }

  search(query: string, tenant_id = "default", limit = 10): Promise<SearchHit[]> {
    return this.request("/v1/search", { query, tenant_id, limit });
  }

  setEmbedding(memory_id: string, embedding: number[], tenant_id = "default"): Promise<{ stored: boolean }> {
    return this.request(`/v1/memories/${encodeURIComponent(memory_id)}/embedding`, {
      embedding,
      tenant_id,
    });
  }

  searchVector(
    embedding: number[],
    tenant_id = "default",
    limit = 10,
    kinds: string[] = [],
  ): Promise<SearchHit[]> {
    return this.request("/v1/search/vector", {
      embedding,
      tenant_id,
      limit,
      kinds,
    });
  }

  searchHybrid(
    query: string,
    embedding: number[],
    tenant_id = "default",
    limit = 10,
    kinds: string[] = [],
    rrf_k = 60,
  ): Promise<SearchHit[]> {
    return this.request("/v1/search/hybrid", {
      query,
      embedding,
      tenant_id,
      limit,
      kinds,
      rrf_k,
    });
  }

  context(query: string, tenant_id = "default", limit = 10): Promise<{
    query: string;
    memories: Array<{ id: string; content: string; kind: string }>;
    evidence_ids: string[];
    estimated_tokens: number;
  }> {
    return this.request("/v1/context", { query, tenant_id, limit });
  }
}
