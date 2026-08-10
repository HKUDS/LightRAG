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

**Stop every LightRAG writer first — the tool cannot check this for you.**
It reads the source graph, writes it to the target, and then verifies the
result. A concurrent writer mutating either side inside that window
invalidates the comparison, and the tool can neither detect a live writer nor
lock the backends. Every other precondition below is enforced; this one is
purely operational, and it is the assumption the whole design rests on. The
same requirement applies to `rebuild_vdb`.

**The target graph slice must be empty.** The migration is defined as
populating an empty slice; that invariant is also what makes its failure
handling exact (see below). A non-empty target is refused before anything is
written.

Both backends read the usual `POSTGRES_USER`, `POSTGRES_PASSWORD`,
`POSTGRES_DATABASE`, `POSTGRES_HOST` and `POSTGRES_PORT` variables.

## Usage

```bash
# Dry run (default): reads and checks everything, migrates nothing.
python -m lightrag.tools.migrate_graph_storage

# Perform the migration.
python -m lightrag.tools.migrate_graph_storage --apply
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--apply` | off | Perform the migration. Without it no graph data is written (see the schema caveat below). |
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

**A dry run performs no migration writes, but it is not read-only at the
schema level.** Both backends have to be initialized before their contents can
be inspected, and `initialize()` is what creates the tables `PGTableGraphStorage`
needs (and, on the AGE side, the graph and its indexes) and runs any legacy-edge
normalization. Against a database that has never hosted this backend, a dry run
therefore creates schema. It writes no graph data and moves nothing.

**`--workspace` is cross-checked, not trusted.** The backends resolve their
workspace as `POSTGRES_WORKSPACE` env > the value passed in > `"default"`, so
the environment can outrank the flag. The tool compares what was requested
against what the backends actually resolved to, refuses when they differ, and
reports the resolved workspace — the slice a destructive run acted on is never
left implicit.

Exit code is 0 on success and non-zero on any failure.

## The report

Both modes print a dict to stdout, following the convention of the other
tools:

```python
{'mode': 'dry-run', 'source_backend': 'PGGraphStorage',
 'target_backend': 'PGTableGraphStorage', 'workspace': 'default',
 'nodes': 8123, 'edges': 41022,
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
- more than one directed row *reaches the tool* for the same ordered pair — a
  violation of the invariant that makes the canonical merge lossless. Note the
  limit of this backstop: AGE enumerates edges with `SELECT DISTINCT source,
  target, properties`, so byte-identical parallel relationships are already
  collapsed to one row before the tool sees them. It can therefore only catch
  parallels whose payloads differ — which is the case that would lose data;
  identical parallels collapse losslessly.

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

A failed `drop()` under `--force-empty-target` aborts the run rather than
proceeding: the backend reports that failure by returning an error status
instead of raising, so an unchecked call would let the migration write into,
and later compensate against, rows it does not own.

Compensation is not atomic. There is no transaction spanning two backends, so
a crash during compensation can leave the target partially populated. Two
things make that recoverable: re-running the tool refuses a non-empty target
rather than compounding the mess, and if compensation itself fails the report
carries the full `written_node_ids` and `written_edge_keys` so the remaining
rows can be removed by hand.

## What this tool does not do

- **It does not migrate any other pair.** `PRECONDITION_HOOKS` and
  `COMPENSATION_HOOKS` are named placeholders — declared, but not yet consulted
  anywhere; the Phase 1 behaviour is hardcoded. Adding a pair means wiring them
  as well as allow-listing it.
- **It does not merge into a populated target.** Empty-target is a
  precondition, not a convenience.
- **It does not run in a single transaction.** Two backends, no shared
  transaction; the failure handling above is the substitute.
- **It does not protect you from a live writer.** See the preconditions.
- **It does not move vectors, KV data or document status.**
