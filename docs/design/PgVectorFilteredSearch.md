# PostgreSQL filtered vector search (iterative index scan)

Status: mitigation, not a guarantee. This document states what
`PostgreSQLDB.iterative_scan_statement` and `_fetch_with_iterative_scan` fix,
what they leave, and why.

## The defect

`chunks` / `entities` / `relationships` queries are

```sql
WHERE workspace = $1 AND content_vector <=> $q < $t
ORDER BY content_vector <=> $q LIMIT $k
```

while the HNSW index is global: one table holds every workspace. When the
planner walks the index, it receives only the `hnsw.ef_search` (default 40)
nearest candidates of the *whole table* and applies `workspace` afterwards. A
workspace holding a small share of the table gets about `40 x share` rows, i.e.
usually **0 rows and no error**.

## What the setting does

`SET hnsw.iterative_scan = strict_order` (pgvector >= 0.8) lets the index scan
resume and fetch further batches until `LIMIT` is satisfied, or a scan budget
is exhausted. `strict_order` keeps nearest-first order and the distance bound;
`relaxed_order` may return rows out of order and is not acceptable here.

It is applied with **`SET LOCAL` in the query's own transaction**
(`_fetch_with_iterative_scan`: `BEGIN`, `SET LOCAL`, the query, `COMMIT`), not
as a session setting at connection init or pool reset. Behind a
transaction-pooling proxy (PgBouncer / Supavisor in transaction mode, a
deployment `env.example` already anticipates) the backend that took a session
`SET` can be returned to the proxy before the `SELECT`, which then runs with
the scan off and no error. A setting made in the same transaction as the query
is the only one guaranteed to be in force when the scan runs, in session and
transaction pooling alike, and nothing has to be replayed after `RESET ALL`.

Cost: two to three extra round trips per vector query (measured locally, trivial
statement: 0.3 ms to 1.3 ms; the absolute figure grows with network latency,
against scans that already take 5-30 ms). The setting is plain `SET LOCAL`;
do not `LOAD 'vector'` first. A not-yet-loaded extension GUC is kept as a
placeholder and adopted when the library loads, whereas `LOAD` is refused for
non-superusers (managed databases).

If the `SET LOCAL` fails, only that transaction is aborted: the failure is
logged once and the query runs without the setting, as before this change.
A failure of the query itself is re-raised and does not fall back to a plain
query. The normal transient-error retry around `PostgreSQLDB.query`
(`_run_with_retry`) still applies to it, as to every other query.

### Old pgvector (< 0.8)

The parameter does not exist, and the failure is **silent on a new
connection**: `SET hnsw.<unknown>` is accepted as a placeholder and only
dropped when the library first loads, with a server-side `WARNING`; the client
sees no exception. Once the library is loaded on the connection the same
`SET` raises `InvalidNameError`. The outcome therefore cannot be read from the
exception, so the extension version is probed once at startup
(`_probe_iterative_scan_support`): below 0.8 the `SET` is skipped and one
warning is logged. If the version cannot be read, the `SET LOCAL` is still
attempted and its failure falls back as described above.

## Measured behaviour

PostgreSQL 18.4, pgvector 0.8.6, 212,240 rows of 16-d random vectors, HNSW
`m=16`, `LIMIT 20`, 15 queries per cell. `off` is the behaviour before the
change.

Threshold loose (every row qualifies), mean rows returned / median latency:

| workspace share (rows) | off | strict_order | strict + `max_scan_tuples=200k` + `work_mem=64MB` |
|---|---|---|---|
| 4.7% (10,000) | 1.9 rows / 1 ms | 20 / 5 ms | 20 / 5 ms |
| 0.94% (2,000) | 0.3 / 1 ms | 20 / 18 ms | 20 / 21 ms |
| 0.09% (200) | 0.0 / 1 ms | 16 / 31 ms | 20 / 127 ms |
| 0.02% (40) | 0.0 / 1 ms | 3 / 30 ms | 20 / 568 ms |

Threshold tight (almost no row qualifies): 0 rows in every configuration, but
latency rises from ~1 ms to ~30 ms, because the scan runs to its budget before
giving up.

## What remains

1. **Very small workspaces.** The scan is bounded by `hnsw.max_scan_tuples`
   (default 20,000) *and* by memory (`work_mem` x `hnsw.scan_mem_multiplier`).
   Raising only one of them changes nothing; both must be raised. The cost is
   the latency above, paid by every query.
2. **Latency on "few qualify" queries.** Where fewer than `LIMIT` rows satisfy
   the distance threshold (common for entities / relationships), the scan always
   runs to its budget. A ~30 ms floor replaces ~1 ms. This was measured on a
   small table; larger rows or dimensions cost more.
3. **VCHORDRQ** is not covered; its filtering behaviour was not examined.
4. **IVFFlat is covered, with a weaker guarantee** (see below).
5. **pgvector < 0.8** keeps the old behaviour (a warning is logged).

## IVFFlat

IVFFlat has the same defect: it scans `ivfflat.probes` lists (default 1 of
`lists`) and filters afterwards. Measured (212,240 rows, `lists=100`,
`LIMIT 20`, threshold loose):

| workspace share (rows) | default (`probes=1`) | `relaxed_order` |
|---|---|---|
| 4.7% (10,000) | 20 rows | 20 rows |
| 0.94% (2,000) | 19.1 rows | 20 rows |
| 0.09% (200) | **2.1 rows** | 20 rows (14.6 ms vs 2.4 ms) |

IVFFlat accepts **only** `relaxed_order`; `strict_order` is ignored with a
server WARNING. Under `relaxed_order` the rows are **not in distance order**
(30 of 30 sampled queries) and the planner adds no sort, so the result would
break the nearest-first contract the callers rely on. The IVFFlat query is
therefore wrapped (`_with_distance_resort`):

```sql
WITH relaxed AS MATERIALIZED (<template> + distance AS _distance)
SELECT * FROM relaxed ORDER BY _distance + 0
```

This is the pgvector-documented remedy. `MATERIALIZED` forces the scan to
finish before the sort, and `+ 0` keeps the planner from deciding the sort is
redundant because the index "already" ordered the rows. It restores *order* only: which rows
make the top `LIMIT` is still decided by the approximate scan, as it always was
for an ANN index. The helper column is removed before rows reach callers.

## Rejected / deferred

* Raising `max_scan_tuples` / `work_mem` by default: the table above shows the
  price, and the right value depends on the deployment's workspace-size
  distribution.
* Partitioning by workspace or per-workspace partial indexes: the structural
  fix (the query becomes tenant-local, as Qdrant's tenant index does in
  `VectorSpaceProvenance.md`), at the cost of DDL per workspace. Deferred.
* Materialising the workspace first (`WITH scoped AS MATERIALIZED ...`): exact,
  but linear in the workspace size on every query.

## Index choice for multi-workspace deployments

When many workspaces share one vector table, prefer **`HNSW` or
`HNSW_HALFVEC` with pgvector >= 0.8**: a strict-order iterative scan, no
re-sort, no ordering caveat. `IVFFLAT` is protected too but relies on the
re-sort wrapper and on `ivfflat.probes` for recall. `VCHORDRQ` is unexamined.
A single-workspace deployment is not exposed to this defect whatever the index.

### When the shared table is the wrong shape: Qdrant

The defect exists because workspaces share one global ANN index and the
workspace filter runs after it. Iterative scan reduces the symptom; it does not
remove the shape. For a deployment with many workspaces, or many small ones next
to a large one, the backend built around tenant-local search is the structural
answer:

* `QdrantVectorDBStorage` keeps all workspaces in one collection but tags every
  point with `workspace_id`, indexes that field as a tenant
  (`KeywordIndexParams(is_tenant=True)`), and builds the collection with
  `HnswConfigDiff(m=0, payload_m=16)`: **no global graph, one graph per tenant**.
  A workspace-filtered query is tenant-local by construction, so it cannot hit
  this defect and needs no scan budget. See `VectorSpaceProvenance.md`.
* `MilvusVectorDBStorage` is not exposed either, for a different reason: it
  uses one collection per workspace (`{workspace}_{namespace}_{model_suffix}`),
  so there is no shared index. Its cost is collection count, not recall.

What this is **not** a recommendation about:

* **It is about multiple workspaces, not multiple Gunicorn workers.** Worker
  processes only need a network-reachable store; PostgreSQL, Qdrant and Milvus
  are all fine, and the file-backed stores are the ones that are not.
* **It is not free.** PGVectorStorage can share one PostgreSQL with the KV,
  graph (AGE) and doc-status stores; Qdrant adds a second system to operate,
  back up and monitor. Changing the vector backend means rebuilding the vector
  data (a full re-embedding, e.g. `lightrag-rebuild-vdb` against the new backend); the other three stores are unaffected.
* **Latency and recall were not measured here.** The Qdrant argument is
  structural (what the index can and cannot return), not a benchmark.

Pick PostgreSQL + HNSW (pgvector >= 0.8) when workspaces are few or large and a
single database is worth more than tenant-local search; pick Qdrant when the
workspace count is large, sizes are very uneven, or the defect above has
already been observed.
