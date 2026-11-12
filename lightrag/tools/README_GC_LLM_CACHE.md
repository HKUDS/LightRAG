# Orphaned LLM Cache GC

Operator-invoked cleanup for cache rows a document deletion could not reach.

`delete_llm_cache=True` removes a document's LLM cache rows by following the
`llm_cache_list` each chunk carries. A row whose key was never attached to its
chunk is invisible to that path: the chunks are deleted, the deletion reports
`success`, and the row stays. The row holds the extraction prompt — the chunk
text verbatim — plus the extracted entities and relations, so a stranded row
is an incomplete-deletion residue rather than housekeeping.

The write ordering that lands with the recovery-anchor work closes the window
on **new** data: a cache row is written only after its reference is recorded,
and the write is skipped when the reference cannot be recorded. What remains
are rows written before that ordering, and rows a hard kill lost mid-flight.

Recovery cannot see either set — the only other reader of a cache row's
`chunk_id` is a filtered scan, which is unaffordable on every document
deletion and unavailable on Redis. So the sweep is a deliberate, offline,
operator-invoked operation:

```bash
python -m lightrag.tools.gc_llm_cache            # report
python -m lightrag.tools.gc_llm_cache --apply    # delete the orphans
```

## Scope rules

- **Only rows carrying a non-empty `chunk_id` are candidates.** Query and
  keywords rows carry `chunk_id=None` by construction; they are the job of
  `lightrag-clean-llmqc` and are never touched here. The `summary`,
  `smartheading` and multimodal `analysis` artifacts carry no chunk reference
  either, so no liveness question can be asked about them: they are counted as
  `unscoped_rows` and left alone. A chunk-liveness sweep cannot name them at
  all — naming them would need the opposite rule (delete what no live chunk
  references), which this tool deliberately does not implement. Of the three,
  only `summary` also has no other path: `smartheading` rows ride
  `smartheading_llm_cache_ids` into a document deletion's pool and multimodal
  `analysis` rows ride the multimodal chunk's merged `llm_cache_list`, so both
  are reclaimed by that deletion and only stranded by the ordering gaps this
  tool exists for. Rows from before the field existed are the same case.
- **A candidate is deleted only once its chunk's absence is confirmed by
  `text_chunks.get_by_id_strict`.** The bulk `filter_keys` answer is only a
  candidate filter, because it fails open on a backend whose index is not
  ready — it reports every key absent — which would classify every
  chunk-scoped row as an orphan. A backend without
  `supports_strict_point_reads` is refused, and a strict read that cannot
  confirm absence stops the sweep before the first delete. A row whose chunk
  still exists is left alone even when no chunk references it: its owning
  document is alive, and deleting it would only re-bill the next reprocess.
- **Rows a document deletion deliberately retained count as dead**, because
  the chunk they name is gone. `delete_llm_cache=False` keeps extraction cache
  rows so a re-ingest is cheap; a sweep deletes them. That trade is the
  operator's call, which is why the default is a report and `--apply` must be
  asked for.

## Usage

Library (works with every configured backend combination):

```python
from lightrag.tools.gc_llm_cache import gc_orphan_llm_cache

rag = LightRAG(...)
await rag.initialize_storages()
report = await gc_orphan_llm_cache(rag)              # report only
report = await gc_orphan_llm_cache(rag, apply=True)  # delete the orphans
```

CLI — the console script `lightrag-gc-llm-cache`, or `python -m
lightrag.tools.gc_llm_cache` — with env-driven construction (`WORKING_DIR`,
`WORKSPACE`, `LIGHTRAG_KV_STORAGE` / `LIGHTRAG_GRAPH_STORAGE` /
`LIGHTRAG_DOC_STATUS_STORAGE`, each backend's own connection variables, and
`EMBEDDING_DIM`; the vector store is a no-op, and neither the LLM nor the
embedder is ever called):

```bash
python -m lightrag.tools.gc_llm_cache                    # report
python -m lightrag.tools.gc_llm_cache --apply            # delete
python -m lightrag.tools.gc_llm_cache --verbose          # sample keys
python -m lightrag.tools.gc_llm_cache --batch-size 1000  # storage round trips
```

A sweep is a full namespace read (keys are materialized by the shared
per-backend scan; rows are fetched and deleted in batches), one `filter_keys`
call per batch of distinct chunk ids, and one confirming `get_by_id_strict`
read per candidate chunk. Like the other offline
tools, run it while the server is stopped: the shared pipeline status that
would let a live process notice the sweep is per-process for the file-backed
KV stores, so a CLI run cannot see an ingestion happening in another process.

## Report

```
Cache storage: PGKVStorage (workspace='acme')
Cache rows: 412 (318 chunk-scoped, 94 without a chunk_id)
Chunk-scoped: 311 live, 7 orphaned
Report only — re-run with --apply to delete the orphans.
```

The first line is what the CLI opened. A `LIGHTRAG_KV_STORAGE` the run did not
notice would otherwise look exactly like an empty workspace, so the target is
printed rather than inferred.

The report carries a bounded key sample (`orphan_keys_sample`,
`deleted_keys_sample` — 20 keys) so a large workspace cannot turn a dry run
into an unbounded print. Counts are exact either way.

A batch whose rows are still present after `delete` and
`index_done_callback` raises instead of reporting a successful deletion:
"deleted but still there" is the mirror image of the silent incomplete
deletion this tool exists to fix. One soft spot remains, on a backend whose
read path is buffer-aware: OpenSearch stages deletes and reports a staged one
as absent, so a batch whose tombstone a retryable per-item failure left in the
flush buffer still passes — there the check proves the delete was **accepted**,
not published. When the flush succeeds the tombstone is gone and publication
really is confirmed. That leftover tombstone is not loud at exit: the cache
storage's own `finalize()` is what raises "these writes have been lost", and
`LightRAG.finalize_storages()` catches a per-storage failure, logs it and
finalizes the rest rather than re-raising, so the CLI can exit 0 with
`Deleted:` already printed. On a staging backend read that count as accepted,
not published, and check the log for `Failed to finalize llm_response_cache`.

A chunks namespace destroyed between runs is not detectable either: a missing
OpenSearch index or Mongo collection is recreated empty at `initialize()`, and
reads then answer "confirmed absent" for every id, so the sweep cannot tell a
destroyed chunks store from a legitimately empty one. The dry run is the guard
— a healthy workspace that reports every chunk-scoped row as orphaned is in
that state, not offering a cleanup.

Deletion is per batch and not rolled back — a failure leaves the batches
already deleted. Every liveness check runs before the first delete, so a read
failure deletes nothing, and a sweep interrupted mid-way is completed by
running it again: the rows that remain are still orphans.

## Backend support

Key enumeration is not part of `BaseKVStorage` yet. The sweep uses the same
per-backend scans as the other offline tools — `JsonKVStorage`,
`RedisKVStorage`, `PGKVStorage`, `MongoKVStorage`, `OpenSearchKVStorage` — and
refuses with a clear error on a backend it cannot list. When bounded KV
iteration reaches `BaseKVStorage`, `_iter_key_batches` in the tool is the one
function that has to change.
