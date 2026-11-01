import test from "node:test";
import assert from "node:assert/strict";
import { InfinityDBClient } from "../dist/index.js";

test("add sends a JSON POST and returns the created identifier", async () => {
  let captured;
  const client = new InfinityDBClient("http://localhost:8000", async (url, options) => {
    captured = { url, options };
    return Response.json({ id: "memory-1" }, { status: 201 });
  });
  assert.deepEqual(await client.add({ content: "hello", tenant_id: "tenant-a" }), { id: "memory-1" });
  assert.equal(captured.url, "http://localhost:8000/v1/memories");
  assert.equal(captured.options.method, "POST");
  assert.deepEqual(JSON.parse(captured.options.body), { content: "hello", tenant_id: "tenant-a" });
});

test("search preserves tenant and limit parameters", async () => {
  let body;
  const client = new InfinityDBClient("http://localhost:8000", async (_url, options) => {
    body = JSON.parse(options.body);
    return Response.json([{ id: "m1", content: "hello", kind: "fact", score: 1, matched_by: ["fts5_and"] }]);
  });
  const hits = await client.search("hello", "tenant-b", 4);
  assert.deepEqual(body, { query: "hello", tenant_id: "tenant-b", limit: 4 });
  assert.equal(hits[0].id, "m1");
});

test("context returns evidence references and token estimate", async () => {
  const client = new InfinityDBClient("http://localhost:8000", async () =>
    Response.json({ query: "hello", memories: [], evidence_ids: ["e1"], estimated_tokens: 2 }));
  const result = await client.context("hello");
  assert.deepEqual(result.evidence_ids, ["e1"]);
  assert.equal(result.estimated_tokens, 2);
});

test("non-success responses include status and response text", async () => {
  const client = new InfinityDBClient("http://localhost:8000", async () =>
    new Response("tenant rejected", { status: 403 }));
  await assert.rejects(client.search("secret"), /403 tenant rejected/);
});
