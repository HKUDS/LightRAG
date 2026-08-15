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

**A dry run migrates nothing, but it is not read-only, and its blast radius is
not limited to one workspace.** Both backends have to be initialized before
their contents can be inspected, and `initialize()` is not a passive step. Two
different scopes are involved:

- *Scoped to the workspace being migrated:* `_normalize_legacy_edges`
  **deletes and re-inserts edge rows** (reversed duplicates, legacy
  orderings) for this workspace and namespace only; the AGE side creates the
  graph and its indexes.
- *Table-wide, across every workspace and namespace:* the schema migration in
  `PGTableGraphStorage`'s DDL. On a table whose primary key predates the
  `namespace` column it **DELETEs duplicate rows** from `lightrag_graph_nodes`
  and `lightrag_graph_edges` (keeping the most recently updated row per key)
  before rebuilding the primary keys, and when the endpoint foreign keys are
  absent it then **DELETEs every orphan edge** — an edge whose endpoint node
  is missing — from the whole table. That sweep cannot be narrowed: as
  `pgtable_impl` puts it, "the sweep is table-wide because the FK is a global
  composite constraint and cannot be validated per workspace." Both paths are
  one-time (first install, or the first run after the namespace-PK upgrade)
  and are skipped once the constraints exist, but the run that performs them
  touches rows belonging to workspaces `--workspace` never named.

So a dry run against one workspace can still modify others, on a legacy
schema. What it never does is migrate data or copy anything between backends —
that is the guarantee, and it is narrower than "writes nothing". Take a backup
of both tables before the first run against a pre-`namespace` schema.

**`--workspace` is cross-checked, not trusted.** The backends resolve their
workspace as `POSTGRES_WORKSPACE` env > `config.ini` `[postgres] workspace` >
the value passed in > `"default"`, so either the environment or a checked-in
config file can outrank the flag. The tool compares what was requested
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

`nodes` / `edges` in a dry run also double as a sizing signal for the apply
run: both modes hold the whole graph in memory (see **It does not stream**
below), so a dry run against the real source is the cheapest way to gauge how
much memory the real run will need.

## What makes it refuse

Every check below runs *before* the first write, and each one aborts the run
rather than migrating a graph it cannot faithfully reproduce:

- the storage pair is not allow-listed;
- the target graph slice is not empty (unless `--force-empty-target`);
- a node has no usable identity — `id` or `entity_id` missing, not a string,
  or the two disagreeing. Both backends can legitimately produce such a row: a
  vertex created with no properties is legal in AGE and enumerates as
  `{"id": None}`, and NetworkX creates an edge's unknown endpoint as an
  attribute-less node;
- the same node id appears more than once, **whether or not the payloads
  match**. The target keeps one row per id, so two physical source vertices
  become one node and the graph's node count changes;
- `a→b` and `b→a` both exist, **whether or not the payloads match**. The
  target stores one row per canonical pair, and AGE computes degree by
  counting relationship rows — so collapsing an identical reciprocal preserves
  every property while changing degree and traversal. LightRAG's own write
  path matches edges undirected, so it does not create reciprocals: finding
  one means the source violates the invariant this tool assumes;
- a payload contains a value PostgreSQL `jsonb` cannot store faithfully:
  `NaN`, `Infinity` and `-Infinity` (which `agtype` represents and `jsonb`
  rejects outright), or `-0.0` (which `jsonb` accepts but normalises to `0.0`,
  and this tool's comparison treats signed zeros as different — so the write
  would succeed and then fail its own verification). The general rule: if the
  target cannot round-trip a value, refuse before the destructive step rather
  than discover it after the point of no return;
- more than one directed row *reaches the tool* for the same ordered pair — a
  violation of the invariant that makes the canonical merge lossless. Note the
  limit of this backstop: AGE enumerates edges with `SELECT DISTINCT source,
  target, properties`, so byte-identical parallel relationships are already
  collapsed to one row before the tool sees them. It can therefore only catch
  parallels whose payloads *differ*. **This is the one cardinality gap the tool
  cannot close**: byte-identical same-direction parallels are invisible to it,
  and collapsing them changes degree without losing any property. Everything
  else that would change cardinality — reciprocals, duplicate ids — is refused
  outright.

Verification after the write re-runs the source-side checks and compares node
ids, canonical edges and properties. It is driven from the source because the
target has already canonicalized and therefore cannot reveal what was lost.
Property comparison is type-strict: `1`, `1.0`, `True` and `"1"` are four
different values, and `-0.0` differs from `0.0` — all are distinguishable
downstream.

**What `verified` claims — and what it cannot.** The tool only ever sees what
the source's `get_all_nodes` / `get_all_edges` return, so its claim is scoped
to that enumerated view: **every node and edge the enumerator returned came
across with its identity, its payload, and its count intact.** Within that
view nothing silently collapses — reciprocals and duplicate ids are refused
before the write rather than merged and blessed.

`verified` does **not** prove that the physical AGE graph was reproduced. Two
things are already lost or hidden before this tool is handed a row:

- **Physical multiplicity.** AGE enumerates with `SELECT DISTINCT source,
  target, properties`, so two byte-identical relationships between the same
  ordered pair arrive as one row. AGE's own degree counts relationship rows, so
  such a source has degree 2 where the target will have 1 — and `verified` will
  still be true, because both sides match the one row the enumerator showed.
  Reading raw multiplicity would mean going around the storage API, which this
  tool deliberately does not do.
- **Payload keys named `id`, `source` or `target`.** `PGGraphStorage`'s
  enumerator overwrites them with identity values when flattening each row, so
  a business property under one of those names is gone upstream and no
  comparison here can notice.

If exact physical fidelity matters for your graph, check for duplicate
relationships directly in the AGE tables before migrating.

## If a write fails

The tool removes exactly what the run wrote — the migrated node ids plus every
edge endpoint, since the target auto-creates missing endpoints without
reporting them — using `remove_edges` first and then `remove_nodes` to respect
the foreign key. It never uses `drop()` for this; whole-slice deletion is
reachable only through `--force-empty-target`. The written set is computed
before the first write, so a failure at any point still has the complete set,
and removing it is exact precisely because the slice started empty.

`--force-empty-target` has an irreducible window. Every source check runs
before the drop, so a malformed source cannot cost you the slice — but nothing
proves the *write* will succeed. If the migration fails after the drop, the
compensation removes what this run wrote and the target ends up empty: the
pre-existing data is gone and the new data never landed. That is the bargain
the flag asks for; take a backup before using it.

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
- **It does not stream.** The storage API enumerates whole graphs, so a run
  holds the source graph, the derived write plan and — during verification —
  the target graph in memory at the same time. Peak memory grows with the
  graph, roughly 2x its decoded size, and no setting bounds it: the fail-closed
  checks (duplicate ids, reciprocals, parallel pairs) are global properties
  that need the whole-graph view. For a very large graph, run the tool on a
  host sized accordingly, or migrate workspace by workspace. Running out of
  memory *inside* the process (`MemoryError`) is handled like any other
  failure — compensation removes what the run wrote — but a kernel OOM kill
  terminates the process outright: nothing compensates, the report never
  prints, and under `--force-empty-target` it lands after the drop. Size the
  host before reaching for that flag.
- **It does not protect you from a live writer.** See the preconditions.
- **It does not disambiguate workspaces that share an AGE graph.** AGE derives
  its graph name by replacing every non-alphanumeric character with `_`, so
  `team-a` and `team_a` name the same AGE graph while PGTable keeps them as
  distinct workspaces. Check which graph a workspace actually resolves to
  before migrating one whose name contains punctuation.
- **It does not move vectors, KV data or document status.**
