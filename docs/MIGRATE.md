# Moving from another memory layer

The reason people stay on a memory layer they already have is not loyalty — it's
that re-ingesting years of memory is a project. This makes it a command:

```bash
pip install infinitydb-ai
infinitydb-migrate mem0-export.json --db memory.db --user alice --dry-run
infinitydb-migrate mem0-export.json --db memory.db --user alice
```

No competitor SDK is installed, no network call is made, and nothing is emulated:
you hand over a file and the facts are re-extracted locally, deterministically.

## What it reads

| Source | Shape | How to get it |
|---|---|---|
| **mem0 (OSS or cloud)** | `{"results": [{"memory": "...", "user_id": "...", "created_at": "..."}]}` | `client.get_all(user_id=...)`, or the export from their console |
| **Generic JSON** | array of objects with `content` / `text` / `fact` / `memory` | any SDK's list method, dumped to a file |
| **JSONL** | one object per line, same fields, optional `role` / `speaker` | chat and transcript logs |
| **Plain text / markdown** | one memory per line | notes, exports you pasted into a file |

`--format auto` guesses from the extension; `--format json|jsonl|text` overrides.
A malformed line in JSONL is skipped rather than aborting the import.

## Field mapping

| Export field | Becomes | If absent |
|---|---|---|
| `memory` / `content` / `text` / `fact` | the raw turn written to L0, then facts extracted from it | the row is skipped |
| `user_id` / `userId` | tenant id | `--user` |
| `created_at` / `timestamp` / `ts` / `time` | event time (`ts`), so decay and `as_of` are correct | import time, **counted and reported** |
| `session_id` / `thread_id` / `conversation_id` | scoped recall boundary | unscoped |
| `role` / `speaker` | kept in the text unless it is `user` | bare text |

Timestamps accept ISO 8601 (`2023-03-01T09:00:00Z`), epoch seconds and epoch
milliseconds. Millisecond values are normalised — that mistake is common enough to
handle rather than reject.

A missing timestamp is not silently invented: the report says how many records
were stored at import time, because a wrong event time quietly corrupts temporal
recall and every answer built on it.

## Safety properties

- **Re-runs are idempotent.** Before writing, each record is checked against
  `l0_logs` for the same user and text; importing a file twice adds nothing.
  Verified by test: second run reports `imported 0, skipped 3` and the store's row
  count does not grow.
- **`--dry-run` writes nothing** but reports exactly what would be imported,
  including duplicates that would be skipped.
- **`auto_supersede=False`** during import. Your export is a historical record;
  importing it must not silently delete a newer fact that happens to be
  single-valued.
- **Verification after import**: the command prints the atom count and runs a
  recall for the first imported memory, so a silent no-op is visible immediately.

## What does not come across

Being explicit here matters more than the happy path:

- **Embeddings.** Vectors from another system are not portable — the models and
  dimensions differ. Facts are re-extracted and re-indexed locally, which is why
  the import is fast and needs no key. If you use `dense=`, re-run
  `db.index_dense(...)` afterwards.
- **Graph edges from another product.** Relations are re-derived from the text by
  this engine's own extractor; a competitor's typed edges are not imported as-is.
- **Deletion history.** `as_of` recall will see the imported facts as of their
  event times, but supersession chains that happened in the old system are not
  reconstructed.
- **Their scores, ranks and feedback.** Ranks are not comparable across systems;
  importing them would make recall worse while looking like continuity.
- **Conversation structure beyond `session_id`.** Turn boundaries within a session
  are rebuilt from the text, not from the export's threading metadata.

## The honest part

A migration command removes *one* blocker: the effort of switching. It does not
create a reason to switch. The reasons in this repo are the offline deterministic
path, the MCP integration, and the audit bundles — and multi-hop retrieval, which
is still below its gate, is not one of them. Importing into a system that retrieves
worse would be a bad trade, and nothing here pretends otherwise.

If you are evaluating both, run `infinitydb-migrate --dry-run` against your export
(no writes, no lock-in), then `python benchmarks/arena/arena.py verify` on this
repo, and compare on your own data rather than either project's numbers.
