# Offline Chunk-Tracking Repair

`entity_chunks` and `relation_chunks` are the authoritative chunk-level
provenance for graph objects. A crash can leave an orphan row or a row that
still names deleted chunks. Since `BaseKVStorage` does not provide complete key
enumeration, those rows cannot be found and repaired individually. This tool
replaces both namespaces from evidence in the extraction cache.

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

For an already-isolated automated maintenance environment:

```bash
lightrag-repair-chunk-tracking --apply --yes
```

The plan is built completely before the first drop. Source reads are
complete-or-raise, and graph data is used only to decide which entities and
relations currently exist. Attribution comes only from
`text_chunks.llm_cache_list` and `llm_response_cache`; graph `source_id` and
document-level recovery anchors are never used as tracking evidence.

If the graph contains entities or relations but cached evidence would produce
an empty corresponding namespace, apply is refused before any drop. Otherwise,
objects without usable cached evidence remain without a row and are reported as
warnings; inventing attribution would be worse than falling back to the graph's
bounded `source_id` view for those objects.

## Failure recovery

There is no transaction spanning the two namespaces. If an apply fails or is
interrupted after a drop, tracking may be partially rebuilt. Keep all writers
stopped, correct the reported backend or cache problem, and run `--apply` again
until it exits successfully. The replacement is idempotent for an unchanged
graph and extraction cache. A failed apply exits with status 1.
