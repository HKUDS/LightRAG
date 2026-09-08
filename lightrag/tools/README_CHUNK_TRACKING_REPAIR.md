# Offline Chunk-Tracking Repair

`entity_chunks` and `relation_chunks` are the authoritative chunk-level
provenance for graph objects. A crash can leave an orphan row whose graph object
no longer exists. Since `BaseKVStorage` does not provide complete key
enumeration, those rows cannot be found and repaired individually. This tool
removes them by replacing one or both namespaces from the set of current graph
keys.

## When to use this tool

Use this tool when an entity or relation has already been removed from the
graph but its chunk-tracking key may still remain. Known examples include a
process exit or task cancellation after a graph deletion committed but before
the corresponding tracking cleanup was persisted, and an entity deletion whose
now-unknown incident relation rows could not all be cleaned up.

The tool preserves an authoritative row for a live graph object when that row
still exists. If the row is missing, it can reconstruct it only when a matching
cached extraction result provides chunk-level evidence. It cannot invent
evidence when both sources are missing.

This is not a general repair for every LightRAG consistency problem:

| Problem | Correct action |
| --- | --- |
| Tracking key exists but its graph object is gone | Use this tool |
| Live graph object has no tracking row, but extraction cache still has its evidence | This tool can reconstruct the row |
| Live graph object has neither an existing row nor cached evidence | Restore the extraction cache or re-ingest; the tool blocks by default |
| A live object's tracking row contains a questionable chunk id | Investigate manually; this tool deliberately trusts live-object rows |
| `full_entities` / `full_relations` recovery anchors are incomplete | Use `kg_integrity_repair.py` |
| Vector storage differs from the graph or `text_chunks` | Use `rebuild_vdb.py` |

This tool is also different from the startup chunk-tracking migration. The
startup migration runs only for an empty namespace and seeds from the graph's
KEEP-truncated `source_id`. This repair is operator initiated, is not gated on
an empty namespace, and never treats `source_id` as authoritative evidence.

## Safety requirement

This is an **offline-only** maintenance operation. Stop every LightRAG API
server, pipeline worker, and SDK writer that uses the target backing stores and
workspace before running it, and keep them stopped until a successful apply.
An in-process API reservation cannot exclude writers in other processes or SDK
instances, while the repair drops whole namespaces.

The tool asks for this confirmation before it initializes any storage. It asks
again before applying the destructive replacement. `--yes` suppresses both
prompts and should be used only after the maintenance isolation is established.

The durable SQLite plan is a **forward-recovery plan, not a backup or rollback
image**. It contains the rows that the repair intends to write, not the orphan
rows that replacement removes, and it is normally deleted after a successful
apply. If rollback, forensic inspection, or recovery from an incorrectly chosen
non-empty workspace is required, back up the affected backend or tracking
namespaces before applying. Treat the plan as sensitive maintenance data: it
contains graph object keys, chunk ids, and their associations.

## Configuration

The CLI loads `.env` from the current working directory with
`override=False`. Existing process environment variables therefore take
precedence over values in `.env`. Run it from the project directory that holds
the intended `.env`, or provide the complete target configuration explicitly in
the environment. It uses the same selectors as `LightRAG`:

- `WORKING_DIR` and `WORKSPACE`;
- `LIGHTRAG_KV_STORAGE`, `LIGHTRAG_GRAPH_STORAGE`, and
  `LIGHTRAG_DOC_STATUS_STORAGE`;
- the selected backend's normal connection and workspace environment variables.

It uses `NoopVectorDBStorage` and never calls an LLM or embedding provider.
Before scanning, it prints the working directory, requested workspace, and the
concrete graph, KV, and document-status storage classes. Check these values
and any backend-specific workspace override before confirming an apply.

Even a dry run initializes the configured storages. It does not modify tracking
rows, but initialization may create missing tables or indexes or perform a
backend's normal compatibility setup. "Read-only" below refers to the repair
plan itself, not to all possible storage-initialization side effects.

All built-in graph backends implement the bounded whole-graph iteration this
tool requires. A third-party graph backend must implement `iter_labels()` and
`iter_edges()` without first collecting the whole graph; otherwise planning
fails closed before any tracking namespace is dropped.

## Recommended workflow

Keep the workspace offline for both commands. First build a dry-run plan:

```bash
lightrag-repair-chunk-tracking
# or: python -m lightrag.tools.chunk_tracking_repair
```

Review the configuration, warnings, and existing/planned row denominators. Then
run an apply:

```bash
lightrag-repair-chunk-tracking --apply
```

The second command scans the sources again and builds a **new** plan. A dry-run
plan is temporary and cannot later be passed to `--apply`; `--resume-plan` is
only for an apply that was interrupted after its exact operation had been
sealed. The apply invocation prints its own plan and asks for confirmation
before the first drop.

### SQLite plan location and lifecycle

The plan location depends on the run mode:

| Run mode | SQLite location | Lifecycle |
| --- | --- | --- |
| Dry run | A generated file in the operating system's temporary directory | Deleted when the run finishes; it is not reusable by a later apply |
| `--apply` | `<resolved WORKING_DIR>/.chunk_tracking_repair_plans/plan-<random-id>.sqlite3` | Deleted after success; retained if the destructive phase is interrupted or fails |
| `--apply --plan-file PATH` | The explicitly supplied `PATH` | Same lifecycle as the default apply plan |
| `--apply --resume-plan PATH` | The existing sealed plan at `PATH` | Reused in place and deleted after the resumed apply succeeds |

With the default `WORKING_DIR=./rag_storage` and when invoked from the project
root, an apply plan is therefore created under:

```text
./rag_storage/.chunk_tracking_repair_plans/plan-<random-id>.sqlite3
```

Relative `WORKING_DIR` and `--plan-file` values are resolved from the current
working directory. Before the first namespace is dropped, `--apply` prints the
absolute durable-plan path. Record that path and keep it on durable local
storage until the repair succeeds.

To choose the location explicitly:

```bash
lightrag-repair-chunk-tracking --apply --plan-file /secure/path/repair.sqlite3
```

The `--plan-file` target must not already exist. Its parent directories are
created automatically.

To repair only one namespace when the other cannot be reconstructed safely:

```bash
lightrag-repair-chunk-tracking --apply --namespace entity
lightrag-repair-chunk-tracking --apply --namespace relation
```

`--namespace` limits the blockers and destructive writes, but planning still
scans the complete corpus and graph and constructs both namespaces. Selecting
one namespace therefore does not proportionally reduce scan time or local plan
size.

For an already-isolated automated maintenance environment:

```bash
lightrag-repair-chunk-tracking --apply --yes
```

`--yes` means that the operator has already confirmed both conditions: all
writers are stopped, and the destructive replacement is accepted. It does not
stop writers, imply `--apply`, or bypass safety blockers. Without `--apply`, it
only suppresses the initial offline-confirmation prompt.

## Command-line options

| Option | Meaning and constraints |
| --- | --- |
| `--apply` | Drop and rebuild the selected tracking namespaces. Without it, the tool only builds and prints a temporary plan. |
| `--yes` | Skip the offline-isolation prompt and, during apply, the prompt that requires typing `REPAIR`. It does not imply `--apply`. |
| `--namespace both\|entity\|relation` | Select the namespace to write and validate for blockers. The default is `both`. |
| `--plan-file PATH` | Put a new apply's durable SQLite plan at `PATH`. The file must not already exist; missing parent directories are created. Intended for use with `--apply`. Mutually exclusive with `--resume-plan`. |
| `--resume-plan PATH` | Resume an interrupted apply from its sealed snapshot. Requires `--apply` and is mutually exclusive with `--plan-file`. |
| `--allow-empty-graph` | Permit clearing the selected namespace when the entire graph is intentionally empty. Use only after independently verifying the backend and workspace. |
| `--allow-missing-rows` | Accept a plan that leaves some current graph objects without tracking rows. It cannot override the hard blocker for an entirely empty namespace whose corresponding graph objects exist. |
| `-h`, `--help` | Print the CLI help and exit. |

Resume always uses the namespaces and safety overrides sealed by the original
apply. If `--namespace` is supplied during resume it must match exactly; omit it
to use the recorded value. New allow flags do not change a sealed resume plan.
For unattended recovery in an already isolated environment:

```bash
lightrag-repair-chunk-tracking --apply --resume-plan /path/from/failed/run.sqlite3 --yes
```

The plan is built completely before the first drop. Graph data is used only to
decide which entities and relations currently exist. Existing authoritative
rows for those current graph keys are retained, including empty rows created by
manual operations, and cached extraction results add any recoverable
chunk-level attribution. This preserves rows re-keyed by rename or merge—the
cache still carries the old names and cannot reconstruct those transformations.
Graph `source_id` and document-level recovery anchors are never used as tracking
evidence.

Planning is memory-bounded. Document status and graph objects are read through
bounded pages, and the distinct chunk set, graph membership indexes, and final
replacement rows are held in a disk-backed SQLite database. Dry-run plans are
temporary. Once a confirmed apply enters its destructive phase, its plan remains
available for recovery until success. Apply reads the database in fixed,
internally configured upsert batches; there is currently no CLI batch-size
option. Process-buffered KV backends are flushed after every batch, and any
operation left pending fails the apply while retaining the durable plan. Python
memory therefore grows with a batch and the largest single tracking row, not
with total document, graph-object, or attribution counts. The temporary database
requires writable local storage and disk space proportional to the plan. Check
free space before scanning a large corpus, or use `--plan-file` to select a
suitable filesystem. A backend's own baseline still applies—for example,
`NetworkXStorage` keeps its graph in memory by design—but the repair no longer
creates another full graph-sized copy.

If the graph contains entities or relations but retained rows plus cached
evidence would produce an empty corresponding namespace, apply is refused
before any drop. More generally, a plan leaving **any** current graph object
without a tracking row is blocked by default; `--allow-missing-rows` accepts
that reported degradation after the operator reviews the existing/planned row
denominators. An entirely empty graph is also refused by default because the
tool cannot distinguish a legitimate empty graph from a wrong backend/workspace
or an unavailable graph index. After independently verifying that the selected
graph is intentionally empty, `--allow-empty-graph` permits clearing its orphan
tracking rows. Unsafe dry runs exit with status 1.

`--allow-missing-rows` is an acceptance of lasting degraded provenance, not a
request to postpone those rows. A partially populated namespace is not filled
automatically by the empty-namespace startup migration. Prefer restoring cache
evidence, re-ingesting, or narrowing `--namespace`; use the override only when
the reported missing objects and their later deletion behavior are understood.

This conservative repair removes orphan **keys** but does not claim to validate
every chunk id in a row belonging to a live graph object. Such rows are the
authority, and a rename, merge, or manual source id can legitimately have no
matching extraction-cache record. Discarding them would manufacture the very
provenance loss this tool is intended to avoid.

Legacy `manual_creation` and `UNKNOWN` values are no-evidence sentinels rather
than real chunk ids and are not copied as attribution. The row itself, including
an authoritative empty row created by a manual operation, is retained.

## Reading the report

- `Corpus` reports all document-status rows, how many carry a `chunks_list`, and
  the number of distinct chunk ids found in those lists.
- `Cache` reports chunks with at least one usable cached extraction result,
  chunks whose parsed results contain extracted objects, and chunks with no
  usable result. Cache evidence only contributes when the extracted object still
  exists in the graph.
- `Graph` is the current object universe used to discard orphan keys.
- `Existing current-object rows` counts valid tracking rows already present for
  graph objects. A malformed row without a `chunk_ids` list is ignored and
  reported as a warning.
- `Planned rows` is the union of retained current-object rows and recoverable
  cached attribution. Compare it with the graph denominator before applying.
- `Written rows` appears after apply. For a namespace not selected by
  `--namespace`, its written count remains zero.

## Failure recovery

There is no transaction spanning the two namespaces. Each selected namespace is
dropped and fully written before the next is touched, limiting a failure to the
current namespace. The complete pre-drop replacement is committed and synced in
the durable SQLite plan before apply records its exact namespaces and overrides
and performs the first drop.

If an apply fails or is interrupted, keep all writers stopped and use the path
printed by the failed run:

```bash
lightrag-repair-chunk-tracking --apply --resume-plan /path/from/failed/run.sqlite3
```

Resume validates the plan format and its working-directory, workspace, storage
classes, and storage namespaces. It then re-drops and rewrites the originally
selected namespaces directly from the sealed snapshot; it does **not** scan the
now-partial tracking store. This preserves authoritative rename, merge, and
manual-creation rows that extraction cache cannot reproduce. Once the apply has
entered its destructive phase, the plan is kept after any failure or interruption
and removed only after the complete apply succeeds. A failure before that phase
does not modify tracking and does not produce a resumable plan. Do not start a
fresh scan against the partially rebuilt tracking store, and do not restart
LightRAG until resume completes successfully.

The resume must use the same working directory, effective workspace, storage
classes, storage namespaces, and backend connection target as the original
apply. Identity or SQLite integrity mismatches fail before a drop. Moving the
plan file itself is allowed as long as the configured storage identity remains
unchanged.

## Exit status

| Status | Meaning |
| --- | --- |
| `0` | Safe dry run, successful apply/resume, or an operator declining either confirmation before mutation |
| `1` | Unsafe plan, storage/scan/apply/resume failure, or another rejected operation |
| `2` | Invalid command-line syntax reported by `argparse` |

Automation should inspect both the exit status and the printed plan/report. In
particular, declining an interactive confirmation is a clean cancellation and
therefore exits with status 0.
