# Graph Storage Migration (`PGGraphStorage` → `PGTableGraphStorage`)

Offline tool for moving an already-extracted entity-relation graph from the
Apache AGE backend (`PGGraphStorage`) to the plain-table backend
(`PGTableGraphStorage`) without re-indexing the source documents.

Changing `LIGHTRAG_GRAPH_STORAGE` normally means the previously extracted
graph is simply not visible to the new backend, so the documents have to be
re-processed — an LLM-cost-bearing operation for a graph that already exists.
This tool copies the graph itself, through the public storage API only
(`get_all_nodes` / `get_all_edges` to read, `upsert_nodes_batch` /
`upsert_edges_batch` to write, `remove_nodes` / `remove_edges` to undo). It
issues no raw SQL, so canonicalization, endpoint handling and identity rules
stay in the already-tested storage layer.

Only the graph moves. Vector and KV storages are untouched, and because the
migrated graph keeps the same entity and relation identities, existing vector
data stays valid — nothing is re-embedded and nothing is re-extracted. The LLM
cache is a separate concern with its own tool (see
[README_MIGRATE_LLM_CACHE.md](README_MIGRATE_LLM_CACHE.md)).

Re-indexing remains the general guidance for changing storage backends. This
tool is an advanced path for one specific pair.

## Before you run it

**Stop every LightRAG writer first.** The tool reads the source graph, writes
it to the target, and then verifies the result. A concurrent writer mutating
either side inside that window invalidates the comparison, and the tool cannot
lock the backends. It refuses on the conditions it can detect, but a live
writer is not one of them — this is an operational requirement, the same one
`rebuild_vdb` carries.

**The target graph slice must be empty.** The migration is defined as
populating an empty slice; that invariant is also what makes its failure
handling exact (see below). A non-empty target is refused before anything is
written.

Both backends read the usual `POSTGRES_USER`, `POSTGRES_PASSWORD`,
`POSTGRES_DATABASE`, `POSTGRES_HOST` and `POSTGRES_PORT` variables.

## Usage

```bash
# Dry run (default): reads and checks everything, writes nothing.
python -m lightrag.tools.migrate_graph_storage

# Perform the migration.
python -m lightrag.tools.migrate_graph_storage --apply
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--apply` | off | Perform the migration. Without it the run is read-only. |
| `--workspace` | `WORKSPACE` env | Workspace to migrate. |
| `--force-empty-target` | off | **Dangerous.** `drop()` the whole non-empty target graph slice before migrating, instead of refusing. |
| `--source-backend` | `PGGraphStorage` | Source graph storage class name. |
| `--target-backend` | `PGTableGraphStorage` | Target graph storage class name. |

`--source-backend` / `--target-backend` are an extension seam, not an open
door: only allow-listed pairs run, and Phase 1 allow-lists exactly
`PGGraphStorage` → `PGTableGraphStorage`. Any other pair is rejected before a
backend is even constructed.

The dry run performs the same enumeration and the same refusal checks as the
real run, so it predicts what `--apply` will do. With `--force-empty-target`
it reports that the apply run *would* drop the pre-existing slice, but never
drops anything itself.

Exit code is 0 on success and non-zero on any failure.

## The report

Both modes print a dict to stdout, following the convention of the other
tools:

```python
{'mode': 'dry-run', 'source_backend': 'PGGraphStorage',
 'target_backend': 'PGTableGraphStorage', 'nodes': 8123, 'edges': 41022,
 'verified': False, 'compensated': False, 'target_non_empty': False,
 'would_drop_target_slice': False,
 'written_node_count': 8123, 'written_edge_count': 41022}
```

`mode` is `dry-run` or `apply`. `verified` is true only when the post-write
comparison passed. The destructive-consequence field is mode-symmetric:
`would_drop_target_slice` in a dry run, `dropped_target_slice` in an apply run
— so the preview cannot look clean while hiding that `--force-empty-target`
will destroy the existing slice.

## What makes it refuse

Every check below runs *before* the first write, and each one aborts the run
rather than migrating a graph it cannot faithfully reproduce:

- the storage pair is not allow-listed;
- the target graph slice is not empty (unless `--force-empty-target`);
- a node carries no `entity_id` — the target requires one, and a source graph
  can legitimately contain such a node (for example NetworkX creates an
  edge's unknown endpoint as an attribute-less node);
- the same node id appears more than once with differing properties — the
  write path would silently keep only the last;
- `a→b` and `b→a` exist with differing properties. The target stores
  undirected edges in canonical order, so the two would merge and one payload
  would be lost. Phase 1 fails closed here rather than picking a winner;
- more than one directed row exists for the same ordered pair — a violation of
  the invariant that makes the canonical merge lossless.

Verification after the write re-runs the source-side checks and compares node
ids, canonical edges and properties. It is driven from the source because the
target has already canonicalized and therefore cannot reveal what was lost.
Property comparison is type-strict: `1`, `1.0`, `True` and `"1"` are four
different values, because all four are distinguishable downstream.

## If a write fails

The tool removes exactly what the run wrote — the migrated node ids plus every
edge endpoint, since the target auto-creates missing endpoints without
reporting them — using `remove_edges` first and then `remove_nodes` to respect
the foreign key. It never uses `drop()` for this; whole-slice deletion is
reachable only through `--force-empty-target`. The written set is computed
before the first write, so a failure at any point still has the complete set,
and removing it is exact precisely because the slice started empty.

Compensation is not atomic. There is no transaction spanning two backends, so
a crash during compensation can leave the target partially populated. Two
things make that recoverable: re-running the tool refuses a non-empty target
rather than compounding the mess, and if compensation itself fails the report
carries the full `written_node_ids` and `written_edge_keys` so the remaining
rows can be removed by hand.

## What this tool does not do

- **It does not migrate any other pair.** The per-backend precondition and
  compensation hooks exist but are unimplemented; adding a pair means adding
  its entry and its hooks deliberately.
- **It does not merge into a populated target.** Empty-target is a
  precondition, not a convenience.
- **It does not run in a single transaction.** Two backends, no shared
  transaction; the failure handling above is the substitute.
- **It does not protect you from a live writer.** See the preconditions.
- **It does not move vectors, KV data or document status.**
