# Offline Chunk-Tracking Repair

`entity_chunks` and `relation_chunks` are the authoritative chunk-level
provenance for graph objects. A crash can leave an orphan row whose graph object
no longer exists. Since `BaseKVStorage` does not provide complete key
enumeration, those rows cannot be found and repaired individually. This tool
removes them by replacing one or both namespaces from the set of current graph
keys.

## Safety requirement

This is an **offline-only** maintenance operation. Stop every LightRAG API
server, pipeline worker, and SDK writer that uses the target backing stores and
workspace before running it, and keep them stopped until a successful apply.
An in-process API reservation cannot exclude writers in other processes or SDK
instances, while the repair drops whole namespaces.

The tool asks for this confirmation before it initializes any storage. It asks
again before applying the destructive replacement. `--yes` suppresses both
prompts and should be used only after the maintenance isolation is established.

## Configuration

The CLI loads `.env` and uses the same selectors as `LightRAG`:

- `WORKING_DIR` and `WORKSPACE`;
- `LIGHTRAG_KV_STORAGE`, `LIGHTRAG_GRAPH_STORAGE`, and
  `LIGHTRAG_DOC_STATUS_STORAGE`;
- the selected backend's normal connection and workspace environment variables.

It uses `NoopVectorDBStorage` and never calls an LLM or embedding provider.
Before scanning, it prints the working directory, requested workspace, and the
concrete graph, KV, and document-status storage classes. Check these values
before confirming an apply.

## Usage

Dry-run planning is the default:

```bash
lightrag-repair-chunk-tracking
# or: python -m lightrag.tools.chunk_tracking_repair
```

After reviewing the plan, apply it:

```bash
lightrag-repair-chunk-tracking --apply
```

Before the first namespace is dropped, `--apply` prints and durably seals a
recovery-plan path under `WORKING_DIR/.chunk_tracking_repair_plans/`. To choose
the location explicitly:

```bash
lightrag-repair-chunk-tracking --apply --plan-file /secure/path/repair.sqlite3
```

To repair only one namespace when the other cannot be reconstructed safely:

```bash
lightrag-repair-chunk-tracking --apply --namespace entity
lightrag-repair-chunk-tracking --apply --namespace relation
```

For an already-isolated automated maintenance environment:

```bash
lightrag-repair-chunk-tracking --apply --yes
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
temporary; apply plans remain available for recovery until success. Apply reads
the database in bounded upsert batches. Process-buffered KV backends are flushed
after every batch, and any operation left pending fails the apply while retaining
the durable plan. Python memory therefore grows with the configured batch size
and the largest single tracking row, not with total document, graph-object, or
attribution counts. The temporary database requires local disk space proportional
to the plan. A backend's own baseline still applies—for example,
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

This conservative repair removes orphan **keys** but does not claim to validate
every chunk id in a row belonging to a live graph object. Such rows are the
authority, and a rename, merge, or manual source id can legitimately have no
matching extraction-cache record. Discarding them would manufacture the very
provenance loss this tool is intended to avoid.

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
manual-creation rows that extraction cache cannot reproduce. The plan is kept
after every failed/interrupted apply and removed only after the complete apply
succeeds. A failed apply exits with status 1.
