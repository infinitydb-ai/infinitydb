# API Reference

Python (`from infinitydb import InfinityDB`):

- `add(conversation, user_id="default", session_id=None, importance=0.9, ts=None, auto_supersede=True, extractor=None)` — store a conversation or log. `ts` is the event time (Unix seconds) used for decay, `as_of` recall and ordering.
  - `extractor` — optional callable `extractor(text) -> [fact, ...]` for model-written facts (an LLM, a tagger, a batch job). Each item is a dict with `fact` and optional `category` (lowercase snake_case), `importance` (0–1). Extracted facts keep an L6 provenance row; a paraphrase that does not occur verbatim is recorded with `char_start = -1`. Errors raised by the extractor propagate — a provider outage is not silently an empty conversation. Default `None`: fully deterministic, offline ingestion.
- `recall(query, n=5, user_id=None, include_policies=True, as_of=None, now=None, reinforce=True, turn_windows=True, budget_tokens=None, expansion=0.15, rerank=None, session_id=None, entities=None)` — retrieve memories.
  - `n` — maximum number of memories.
  - `user_id` — tenant scope; `None` searches all users.
  - `as_of` / `now` — evaluate retrieval and decay at a point in time.
  - `reinforce` — update access counts of returned memories.
  - `turn_windows` — score each dialogue turn together with its neighbours and merge adjacent hits (default). `False` scores stored atoms on their own text.
  - `budget_tokens` — pack the highest-ranked memories until this many words are returned; at least one memory is always returned. Use it to hold prompt cost constant.
  - `expansion` — weight (0–1) for light pseudo-relevance feedback: strong terms from the best three chunks are added to the query so paraphrases ("job" for "work") still match. Default `0.15`; `0` scores the query words alone.
  - `rerank` — optional callable `rerank(query, passages) -> list[float]` applied to the merged turn chunks, e.g. a cross-encoder. The core stays dependency-free: on LoCoMo a `ms-marco-MiniLM-L-6` reranker added ~1% relative recall for ~14× the latency, so it is off by default.
  - `session_id` — restrict retrieval to one session/project/conversation. Corpus statistics (document frequencies, average length) are computed *inside* the scope, so a scoped query ranks exactly as it would against a database holding only that session.
  - `entities` — list of names/topics; a memory is considered only if it mentions every one of them. Use it to scope by person, product, or repo.
  - Scoping is measured: pooling ten conversations into one store costs 4.7 points of evidence recall@10 unscoped (0.780 → 0.733); the same store recalled with `session_id` returns 0.780, at 40 ms p50. See [benchmarks/arena/README.md](../benchmarks/arena/README.md).
- `get_proof(atom_id)` — provenance plus cryptographic verification against the raw log.
- `stats()` / `delete_user(user_id)` — layer counts and tenant erasure.


## Typed relation graph

Every dialogue turn is also mined for first-person relations — `works_at`,
`lives_in`, `studied_at`, `likes`, `dislikes` — and stored as an edge pointing
at the atom that asserted it. Edges are derived data: `db.rebuild_edges()`
re-derives them from the immutable turn index for databases written before the
edge table existed.

```python
db.add("Caroline: I work at the gym in Bristol.", user_id="u", session_id="s1", ts=t1)
db.add("Tom: I work at the gym.", user_id="u", session_id="s1", ts=t2)

db.relations("Caroline", user_id="u", hops=1)
# [{'hop': 1, 'direction': 'out', 'subject': 'caroline',
#   'predicate': 'works_at', 'object': 'the gym in Bristol', ...}]

db.relations("Caroline", user_id="u", hops=2)   # also reaches Tom, who shares the gym
```

Edges are **not** used to expand the query string. That was implemented and
measured: on LoCoMo it lowered multi-hop evidence recall from 0.4939 to 0.4484
and single-hop from 0.9128 to 0.8845, so it was reverted. The graph is an
explicit query surface, and the numbers are in `docs/COMPETITIVE_TEARDOWN.md`.

## Contradiction detection (deterministic)

Two facts that cannot both hold are resolved by event time, and the resolution
is recorded rather than performed destructively:

```python
db.add("Caroline: I work at the gym.", user_id="u", ts=1000.0)
db.add("Caroline: I work at the hospital.", user_id="u", ts=2000.0)

db.contradictions("u")
# [{'old_fact': 'I work at the gym.', 'new_fact': 'I work at the hospital.',
#   'category': 'utterance', 'resolved_at': 2000.0}, ...]

db.recall("where does Caroline work", user_id="u")            # hospital only
db.recall("where does Caroline work", user_id="u", as_of=1500.0)  # gym still there
```

The rule is structural, not semantic: same subject + same predicate + different
object, or opposite polarity (`likes` / `dislikes`) of the same object. It needs
no model, and `auto_supersede=False` keeps it off for multi-party transcripts
where two speakers legitimately hold different employers.

## Profile compaction

```python
db.persona_summary("u")
# 'Profile — identity: I am Maria; location: I live in Porto'
```

A one-paragraph distillation of the already-resolved persona, for prompts that
need "who is this user" without replaying turns. It is a summary of resolved
state; no retrieval-quality claim is attached to it.

## Model-backed extraction (optional)

```python
from infinitydb.extraction import openai_extractor, ollama_extractor, parse_facts

db.add(turn, user_id="u", extractor=openai_extractor())   # needs OPENAI_API_KEY
db.add(turn, user_id="u", extractor=ollama_extractor())   # local, free, no key
```

Verify a backend before trusting it, with the same call the engine makes:

```bash
python -m infinitydb.extraction --check                     # hosted, needs OPENAI_API_KEY
python -m infinitydb.extraction --check --backend ollama    # local, no key
```

It prints the parsed facts and exits non-zero with the actual error (missing key,
unreachable endpoint, malformed reply) rather than failing later mid-ingest.

Both are thin wrappers over any OpenAI-compatible `/chat/completions` endpoint
(`chat_extractor(base_url=..., model=...)`), built on `urllib` so the core keeps
zero dependencies. The extractor must return facts the engine validates
(`fact`, snake_case `category`, `importance` 0–1); an unparsable model reply
raises instead of storing nothing. Nothing leaves the machine unless you pass an
extractor.

## Semantic fusion (optional, still no LLM)

Lexical scoring cannot reach a paraphrase at any weight, so the engine can fuse a
dense ranking when you supply an embedder:

```python
from fastembed import TextEmbedding

model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embedder(texts):
    return [list(map(float, v)) for v in model.embed(list(texts))]

db.index_dense(embedder, model="minilm")          # once, incremental after that
result = db.recall("vehicle", n=10, dense=(embedder, "minilm", 0.3))
```

`index_dense` writes normalised vectors into `l2_vectors` (derived data, rebuildable
at any time) and skips windows that already have a vector for that model. Fusion is
reciprocal rank fusion, so no score calibration is needed between the two rankers.
Weight 0.3 measured best: 1.0 lets dense dominate and loses accuracy.

### Relative dates (opt-in)

`db.add(..., normalise_dates=True, ts=event_time)` rewrites relative expressions
into absolute ones before extraction, so a later question about a year can match:

```python
db.add("Ana: I painted a sunrise last year.", user_id="u", ts=1683522000.0,
       normalise_dates=True)
# stored turn: "I painted a sunrise 2022."
```

Off by default, and measured: on LoCoMo it changes BM25 tokens enough to cost
0.0024 evidence recall (0.7771 vs 0.7795), while making date answers reachable for
a reader. Duration phrases ("the last year", "past year") and ambiguous ones
("last summer") are deliberately left untouched, and the raw L0 log always keeps
the original wording.
