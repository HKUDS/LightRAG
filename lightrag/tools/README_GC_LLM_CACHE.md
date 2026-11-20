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
  confirm absence stops the sweep before the first delete.
- **A row that a live chunk references is kept, whatever its `chunk_id`
  says.** The cache key hashes the prompt, which carries the chunk text but no
  document identity, so two documents with one identical paragraph are two
  chunks sharing one row, and the row's `chunk_id` names only its first
  writer. When that writer is deleted, the other chunk still lists the key in
  its `llm_cache_list`, and the deletion rebuild reads it as that chunk's
  evidence (see *Deletion rebuild evidence* in
  `docs/design/PurgeRecoveryContract.md`). Removing a shared initial row makes
  the next rebuild re-extract the chunk (an LLM call, and possibly different
  names); removing a shared gleaning row silently drops what only that
  gleaning named. So before the first delete the sweep reads every live
  chunk's `llm_cache_list` once (`text_chunks.iter_rows`, which raises rather
  than ending early) and keeps each referenced key; the report counts them as
  `referenced_rows`.
- **A row whose chunk still exists is left alone even when no chunk
  references it.** Its owning document is alive, and the row is still
  reachable by key: a reprocess, or a deletion rebuild that re-extracts that
  chunk, hits it and re-attaches it instead of calling the LLM. Chunks written
  before `llm_cache_list` existed reference nothing at all, so a "delete what
  no chunk references" rule would empty their cache. Rows a re-extraction
  stopped referencing (superseded gleaning, a key from an older configuration)
  are therefore kept while their chunk lives, and reclaimed by a sweep once
  its document is deleted.
- **Rows a document deletion deliberately retained count as dead**, because
  the chunk they name is gone. `delete_llm_cache=False` keeps extraction cache
  rows so a re-ingest is cheap; a sweep deletes them. That trade is the
  operator's call, which is why the default is a report and `--apply` must be
  asked for. In particular, **never sweep in the middle of the name-drift
  recovery** (*Recovery from name drift* in
  `docs/design/PurgeRecoveryContract.md`): it deletes a document with
  `delete_llm_cache=false` and then ingests the same content again, relying on
  exactly those rows to answer from the cache. Swept in between, the re-ingest
  calls the LLM for every chunk and may drift again.
- **A document whose resume purge was interrupted** (its chunk rows deleted,
  the re-chunking not yet done — a hard kill or a failure in between) leaves
  its rows naming chunks that no longer exist, so the sweep counts them as
  orphans. Its retry would otherwise hit them; swept, it calls the LLM for
  every chunk. Finish or retry such documents before an `--apply` run.

## Usage

Library (see supported backends below):

```python
from lightrag import LightRAG
from lightrag.tools.gc_llm_cache import gc_orphan_llm_cache

rag = LightRAG(...)  # Use your deployment's storage configuration.
await rag.initialize_storages()
try:
    report = await gc_orphan_llm_cache(rag)  # Report only.
    # Pass apply=True to delete the orphans after reviewing the report.
finally:
    await rag.finalize_storages()
```

CLI — the console script `lightrag-gc-llm-cache`, or `python -m
lightrag.tools.gc_llm_cache` — with env-driven construction (`WORKING_DIR`,
`WORKSPACE`, `LIGHTRAG_KV_STORAGE` / `LIGHTRAG_GRAPH_STORAGE` /
`LIGHTRAG_DOC_STATUS_STORAGE`, each backend's own connection variables, and
`EMBEDDING_DIM`; the vector store is a no-op, and neither the LLM nor the
embedder is ever called). The CLI loads `.env` from the current working
directory. Environment variables already set in the process take precedence
over values from that file.

```bash
python -m lightrag.tools.gc_llm_cache                    # report
python -m lightrag.tools.gc_llm_cache --apply            # delete
python -m lightrag.tools.gc_llm_cache --verbose          # sample keys
python -m lightrag.tools.gc_llm_cache --batch-size 1000  # storage round trips
```

A sweep reads the cache namespace through `iter_rows` in pages, retains keys
for deduplication and orphan classification, makes one `filter_keys`
call per batch of distinct chunk ids, and one confirming `get_by_id_strict`
read per candidate chunk. Like the other offline
tools, run it while the server is stopped: the shared pipeline status that
would let a live process notice the sweep is per-process for the file-backed
KV stores, so a CLI run cannot see an ingestion happening in another process.

## Report

```
Cache storage: PGKVStorage (workspace='acme')
Cache rows: 412 (318 chunk-scoped, 94 without a chunk_id)
Chunk-scoped: 309 live, 2 referenced by another live chunk, 7 orphaned
Report only — re-run with --apply to delete the orphans.
```

The first line is what the CLI opened. A `LIGHTRAG_KV_STORAGE` the run did not
notice would otherwise look exactly like an empty workspace, so the target is
printed rather than inferred.

The report carries a bounded key sample (`orphan_keys_sample`,
`deleted_keys_sample` — 20 keys) so a large workspace cannot turn a dry run
into an unbounded print. Counts are exact either way.

After each batch, the sweep verifies that no cache operations remain buffered
and that the deleted keys are absent. If either check fails, the sweep raises
and the CLI exits with an error. This includes a retryable OpenSearch bulk
failure that leaves a buffered deletion hiding a row still on disk.

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

The cache backend must implement `BaseKVStorage.iter_rows`. The supported
backends are `JsonKVStorage`, `RedisKVStorage`, `PGKVStorage`,
`MongoKVStorage` and `OpenSearchKVStorage`.
