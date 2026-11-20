# Hybrid Retrieval Contract (PostgreSQL)

Read this before touching `lightrag/kg/postgres_impl.py` (`hybrid_query`, the
lexical schema and backfill), `lightrag/kg/lexical_tokenizer.py`, the dispatch
in `lightrag/operate.py` (`_get_node_data`, `_get_vector_context`), or
`QueryParam.enable_hybrid` / `BaseVectorStorage.supports_hybrid_query`.

`PGVectorStorage` can combine its pgvector dense search with PostgreSQL native
full-text search (`tsvector` + GIN), fused by Reciprocal Rank Fusion. The
feature is optional and off by default. This document states the rules, the
accepted residues and the recovery paths; the module docstrings state the
per-call contract and point here.

## Context

The request comes from the hybrid-retrieval discussion in issue #3198. An
earlier attempt put a BM25 index inside the application process (#3377); this
implementation takes the engine-side direction instead: PostgreSQL's own
full-text index, no new dependency, no in-process index, no new storage role.

## Scope

- PostgreSQL only. Every other vector backend keeps plain dense retrieval even
  when `enable_hybrid` is True.
- Entity seeding and chunk recall. The lexical leg reads `entity_name` for the
  entity table and `content` for the chunk table. Relationships get the schema
  for uniformity but are never queried lexically.
- The dense leg, the graph traversal, reranking and token budgets are
  unchanged. Only the retrieval step that fills the candidate list changes.

## Capability flag and the default-off guarantee

`BaseVectorStorage.supports_hybrid_query` is a `ClassVar[bool] = False`.
`PGVectorStorage` sets it to True. A consumer must check both the flag and
`QueryParam.enable_hybrid`; either one being false keeps the plain `query`
path. The flag is a `ClassVar` on purpose: the storage bases are dataclasses,
so a bare annotated attribute would become an `__init__` field and change every
backend's constructor signature.

`QueryParam.enable_hybrid` is declared last. `QueryParam` is a plain dataclass
and SDK callers may construct it positionally, so new fields belong at the end.
The default is `HYBRID_BY_DEFAULT`, and only the literal string `true`
(case-insensitive) enables it, mirroring `RERANK_BY_DEFAULT`. The API field is
`Optional[bool] = None`: omitted means "server default", an explicit value
overrides it.

**The guarantee:** with `enable_hybrid` off, retrieval output is byte-identical
to the pre-hybrid code. This is pinned by
`tests/pipeline/test_default_off_identity.py`: 55 documents, 220 CJK+EN chunks,
66 entities, 65 relations, 20 queries across all five modes, on the file
backends and on PostgreSQL, compared hash for hash against pinned manifests
(`tests/pipeline/default_off_manifest_file.json` and
`tests/pipeline/default_off_manifest_pg.json`). The query cache key also stays on the legacy
form when the flag is off, so answers cached before the feature existed keep
hitting.

Rules for that test: changing the fixture or the query list invalidates the
proof. Regenerate both manifests against a fresh `origin/main` checkout and
re-compare before updating the JSON. Never normalize a difference away; a hash
mismatch is a retrieval behavior change and must be explained. The file and PG
manifests already differ from each other only in the order of relations tied on
`(degree, weight)`, because each backend has its own deterministic edge
enumeration order.

## Lexical tokenizer

`lightrag/kg/lexical_tokenizer.py` is a pure string module: NFKC normalization,
lowercase, then one token sequence. The write side
(`to_tsvector('simple', tokenize_for_index(text))`) and the query side
(`to_tsquery('simple', tokenize_for_query(text))`) MUST both consume it. There
is no second tokenizer, and PostgreSQL's default parser never splits CJK on its
own.

- CJK runs become overlapping bigrams; a one-character run becomes a unigram.
- Every other alphanumeric run is one word token, truncated to 128 characters.
- Every emitted token is alphanumeric, so the reserved tsquery characters
  (`& | ! ( ) : * <-> ' \`) cannot reach `to_tsquery` from user input. The
  query string joins tokens with ` & ` (AND); phrase proximity is not part of
  this contract.
- Output is capped at 4096 tokens per call. An empty or separator-only query
  yields `""`; the caller must skip the lexical leg rather than feed `""` to
  `to_tsquery`.
- Changing any rule changes stored lexemes. The NULL-row backfill does **not**
  repair this: rows that already have a tsvector are not selected. A rule
  change requires resetting `content_tsv` to NULL for the affected rows (or a
  full re-tokenization) and then letting the backfill run.

## Schema and index lifecycle

Every PG vector table carries `content_tsv TSVECTOR`. New tables get the column
in their DDL. Existing tables get
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR`: nullable, no
default, metadata-only, so the statement does not rewrite the table.
Pre-existing rows stay NULL until the backfill covers them.

The GIN index is `CREATE INDEX CONCURRENTLY IF NOT EXISTS ... USING gin
(content_tsv)`. `CONCURRENTLY` refuses transaction blocks, so it runs on a bare
pooled connection in autocommit. A session advisory lock keyed on the index
name serializes concurrent instances; without it one instance could drop an
index another is still building. Only an INVALID index is dropped and rebuilt,
up to three attempts per startup. A valid index is never touched.

Failure to build is a residue, not a crash: the error is logged and the next
startup retries. Until then, lexical retrieval degrades as described below.

The backfill runs as a background task, never as a blocking startup step. Per
batch it selects rows of this workspace whose tsvector is NULL, tokenizes them
with the same application-side function, and applies one transaction of
`UPDATE ... WHERE workspace=$2 AND id=$3 AND content_tsv IS NULL`. Batch size
comes from `POSTGRES_LEXICAL_BACKFILL_BATCH_SIZE` (default 1000, must be a
positive integer; invalid values fail at construction). A kill or cancellation
rolls back the batch in flight, so the next startup resumes from the NULL rows.
`finalize` cancels and awaits the task before releasing the shared client.

Write-path maintenance is part of the upsert statements: chunks and relations
tokenize `content`, entities tokenize `entity_name`, so new or updated rows are
never NULL. The schema exists independent of the retrieval flag; writes
maintain it even when `enable_hybrid` is off.

## Query path

`hybrid_query` runs the dense leg by calling `query` unchanged: same cosine
threshold, same `top_k`. It then tokenizes the query and, if tokens exist, runs
the lexical leg:

```sql
WHERE workspace = $1
  AND content_tsv @@ to_tsquery('simple', $2)
ORDER BY ts_rank_cd(content_tsv, to_tsquery('simple', $2)) DESC, <key>
LIMIT $3
```

The lexical leg takes `top_k` and applies no cosine threshold. PostgreSQL
would happily seqscan without the index, so index absence cannot be inferred
from a query error. Each hybrid query probes `pg_index.indisvalid`; if the
index is missing or invalid, or the lexical query raises, that one query
degrades to the dense result.

Fusion is application-side `rrf_fuse` with k = 60: a record scores the sum over
its legs of `1 / (60 + rank)`, ranks 1-based. Records union by key
(`entity_name` for entities, `id` for chunks); on overlap the vector-leg record
object wins, so its original fields reach the caller. Ties break by vector-leg
rank, then lexical rank, then key. The result is at most `top_k` records in
fusion order.

**Fusion-order invariant:** downstream stages consume the returned order and
must not re-sort by `distance`. `_get_node_data` and `_get_vector_context`
return the backend order as-is, and the later truncation in
`lightrag/utils.py::process_chunks_unified` preserves it. This is pinned by
`tests/test_operate_hybrid_order.py`, where a distance-descending re-sort would
demonstrably reorder the records.

A lexical-only record gains `distance = cosine_better_than_threshold`. That is
a sentinel, not a similarity: it exists so a downstream candidate that applies
the cosine threshold admits the record. It must not be compared against real
distances or used for ranking.

Only entities and chunks are supported. Any other namespace raises
`ValueError` at the entry point, rather than silently returning dense results.

## Degradation and warning dedupe

The lexical leg is fail-open for that query: a missing or invalid index, or an
error from the lexical SQL, returns the dense result. The first failure of a
streak logs one warning; further failures stay silent until a lexical query
succeeds, which re-arms the warning. The flag is per storage instance, so a
recovered workspace does not require a restart to log a new failure.

Two cases are intentionally silent:

- Empty tokenization (a query with no alphanumeric content) skips the lexical
  leg without a warning. It is a valid query, not a failure.
- An incomplete backfill does not fail. The index is valid and the lexical leg
  simply sees fewer rows, so the result degrades toward dense without a
  warning. This is the accepted lag described below.

Errors from the dense leg still propagate. Degradation only covers the lexical
leg.

## Cache partitioning

The answer cache key includes a hybrid component only when the flag is on
(`operate.kg_query` and `operate.naive_query`). With the flag off, the key is
the legacy key, so entries written before the feature existed keep hitting.
With the flag on, the key differs, so an answer built from a different
retrieval context cannot be served. `_ANSWER_CACHE_POLICY_VERSION` is not
bumped for this reason.

Keyword extraction is retrieval-independent and keeps its existing key. The
same conditional component is used for the KG and naive paths.

## Concurrency and multiple workers

The index and the column live in PostgreSQL, shared by every worker. There is
no process-local index state and no new scheduling guard. Multiple instances
may run their own backfill task; the `WHERE content_tsv IS NULL` predicate and
the per-batch `UPDATE ... WHERE ... IS NULL` make the work idempotent, at the
cost of duplicated effort. The only cross-instance coordination is the
advisory lock around the concurrent index build.

The per-query `indisvalid` probe reads live catalog state and caches nothing.
A rebuild that finishes between two queries is visible to the next query
without a restart.

## Rollback

Turning `enable_hybrid` off is a complete rollback of behavior: the dispatch
falls back to `query`, the cache key returns to the legacy form, and the
byte-identical guarantee above applies. The column and the GIN index are
harmless residue while off: they cost storage and write maintenance, no read
path uses them.

Removing them is manual and intentionally not automated:

```sql
DROP INDEX CONCURRENTLY IF EXISTS <lexical_index>;
ALTER TABLE <table> DROP COLUMN IF EXISTS content_tsv;
```

Dropping the column is metadata-only but does not reclaim the space until the
table is rewritten; schedule it like any other destructive DDL. The measured
cost of keeping it is on the order of a few percent on upsert p95 and about
0.2x the vector index size on the CJK corpus used for evaluation (see
`lightrag/evaluation/README_HYBRID_BENCH.md`).

## Accepted residues

1. **Backfill lag.** Between a deployment and the end of the background
   backfill, the lexical leg covers only rows written since the upgrade (plus
   whatever the backfill has reached). Retrieval degrades toward dense, which
   is the harmless direction. Recovery: let the backfill finish, or restart and
   let it resume from the remaining NULL rows.
2. **Invalid index after failed concurrent builds.** A cancelled or failed
   `CREATE INDEX CONCURRENTLY` can leave an INVALID index. It is detected and
   dropped on the next attempt, three attempts per startup. If all fail, the
   error is logged and lexical retrieval stays degraded until a later startup.
   Recovery is automatic; no manual drop is required.
3. **Tokenizer rule changes need re-tokenization.** The backfill only selects
   NULL rows. Rows that already have a tsvector keep the old token sequence
   until it is reset. This is deliberate: the backfill must not rewrite the
   whole table on every startup.
4. **Relationships carry the schema but are not queried.** Their `content_tsv`
   is maintained on write for uniformity. A future relation hybrid leg would
   use it without another migration.
5. **The sentinel distance.** Lexical-only records carry the cosine threshold
   as their `distance`. This is an admission ticket for downstream filters, not
   a score. Any code that treats it as a similarity is wrong.
6. **One catalog probe per hybrid query.** The index-validity probe is an extra
   round trip. Caching it would go stale after a rebuild, drop or failed build,
   which is exactly when it matters. The cost is accepted and measured.
7. **Write amplification while off.** The tsvector is maintained on every
   upsert regardless of the retrieval flag, so deployments that never enable
   hybrid still pay the write and space cost. The alternative, conditionally
   maintaining the column, would make the first hybrid query see a table full
   of NULL rows and require a full backfill at enable time.
8. **Non-PostgreSQL backends.** They advertise no capability and ignore the
   flag. There is no in-process fallback and no second index.

## Verification map

| contract | pinned by |
| --- | --- |
| tokenizer string rules | `tests/kg/postgres_impl/test_lexical_tokenizer.py` |
| schema, backfill, concurrent index, invalid-index retry | `tests/kg/postgres_impl/test_postgres_lexical_schema.py` |
| fusion math, sentinel, degradation, live PG | `tests/kg/postgres_impl/test_postgres_hybrid_query.py`, `tests/kg/postgres_impl/test_hybrid_fusion.py` |
| capability flag and graceful fallback | `tests/kg/test_hybrid_capability.py` |
| default-off byte identity | `tests/pipeline/test_default_off_identity.py` |
| fusion order downstream | `tests/test_operate_hybrid_order.py` |
| cache key partition | `tests/pipeline/test_query_cache_hybrid_partition.py` |
| positional compatibility and env default | `tests/pipeline/test_query_param_hybrid_compat.py` |
| API field default | `tests/api/routes/test_query_hybrid_default.py` |

The evaluation harness that produced the performance and quality numbers is
documented in `lightrag/evaluation/README_HYBRID_BENCH.md`.
