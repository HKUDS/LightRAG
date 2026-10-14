# PostgreSQL filtered vector search (HNSW iterative scan)

Status: mitigation, not a guarantee. This document states what
`PostgreSQLDB.configure_hnsw_iterative_scan` fixes, what it leaves, and why.

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

It is session state, so it is issued on every new connection and again after
`RESET ALL` on pool release. It is a plain `SET`: a not-yet-loaded extension
GUC is stored as a placeholder and adopted when the library loads, whereas
`LOAD 'vector'` is refused for non-superusers (managed databases).

### Old pgvector (< 0.8)

The parameter does not exist, and the failure is **silent on a new
connection**: `SET hnsw.<unknown>` is accepted as a placeholder and only
dropped when the library first loads, with a server-side `WARNING`; the client
sees no exception. After the library is loaded (the pool-reset path) the same
`SET` raises `InvalidNameError`. The outcome therefore cannot be read from the
exception, so the extension version is probed once at startup
(`_probe_hnsw_iterative_scan_support`): below 0.8 the `SET` is skipped and one
warning is logged. If the version cannot be read, the `SET` is still attempted.

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
3. **IVFFlat and VCHORDRQ** are not covered.
   * IVFFlat has the same defect. It scans only `ivfflat.probes` lists
     (default 1 of `lists`) and filters afterwards. Measured (212,240 rows,
     `lists=100`, `LIMIT 20`, threshold loose): a 0.09% workspace (200 rows)
     returned 2.1 rows; 0.94% (2,000) returned 19.1; 4.7% returned 20. With
     `ivfflat.iterative_scan = relaxed_order` all three returned 20 (14.6 ms
     for the small one vs 2.4 ms).
   * IVFFlat accepts **only** `relaxed_order`; `strict_order` is rejected with
     a server WARNING and ignored. Rows may then arrive out of distance order,
     which the callers do not tolerate today, so it was not applied.
   * VCHORDRQ filtering behaviour was not examined.
4. **pgvector < 0.8** keeps the old behaviour (a warning is logged).

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

When many workspaces share one vector table, use **`HNSW` or `HNSW_HALFVEC`
with pgvector >= 0.8**. It is the only combination this change protects, and
the only one with a strict-order iterative scan. `IVFFLAT` shows the same
silent-empty result and has no strict-order remedy; `VCHORDRQ` is unexamined.
A single-workspace deployment is not exposed to this defect whatever the index.
