# KG Integrity Audit / Recovery-Anchor Repair

Offline companion to the issue #3400 recoverable-mutation work.

Since #3400, ingestion prewrites per-document **recovery anchors**
(`full_entities` / `full_relations`) before mutating the graph, so purge,
retry, and scan rollback can always discover a document's contributions.
Installations with data ingested **before** that change (or written through
direct KG paths like `ainsert_custom_kg`, which sit outside the
document-level guarantee) may hold graph contributions with no anchor — such
data is invisible to per-document cleanup and becomes orphaned when its
document is deleted or reprocessed.

This tool enumerates the whole graph (deliberately offline-only; the hot
paths never do a graph-wide scan), maps every node/edge back to its owning
documents via chunk `source_id` → `text_chunks` → `full_doc_id`, and:

- reports per-document anchor gaps;
- reports irrecoverable orphans (contributions whose source chunks no longer
  exist) — these are never modified automatically;
- with `--apply`, unions the missing entries into the anchor rows.

## Usage

Library (works with every configured backend combination):

```python
from lightrag.tools.kg_integrity_repair import audit_kg_integrity

rag = LightRAG(...)
await rag.initialize_storages()
report = await audit_kg_integrity(rag)              # report only
report = await audit_kg_integrity(rag, apply=True)  # repair anchors
```

CLI (env-driven construction — `WORKING_DIR`, `WORKSPACE`, `EMBEDDING_DIM`,
`LIGHTRAG_*` storage selectors; never calls the LLM or embedder):

```bash
python -m lightrag.tools.kg_integrity_repair             # report
python -m lightrag.tools.kg_integrity_repair --apply     # repair
python -m lightrag.tools.kg_integrity_repair --verbose   # per-doc details
```

Run it while the server is stopped (or the workspace is otherwise idle):
the audit reads a moving target if ingestion runs concurrently.

## This tool is now the required remedy, not just a diagnostic

Purge and deletion **fail closed** when a document has no recovery proof.
Previously, absent anchor rows were read as "this document contributed
nothing", so graph cleanup was silently skipped while the chunks were deleted
anyway — which destroyed the provenance chain (`source_id` → `text_chunks` →
`full_doc_id`) that this tool needs, turning repairable data into permanent
orphans. So instead, the operation now refuses before deleting anything:

```
RecoveryAnchorMissingError: Refusing to purge document doc-…: recovery anchor
row(s) missing or unusable (full_entities, full_relations) …
```

Over the API this surfaces as **HTTP 409** with no data removed. Retrying
unchanged will refuse again — run `--apply` first, then retry.

The rule is narrower than "every purge needs a proof". What must never happen
is deleting something that *carries* attribution — a chunk row, or an anchor
row that names objects — while leaving those objects in the graph. A purge that
removes no such carrier cannot strand anything, so it needs no proof. Otherwise
one of these must hold:

| Proof | Meaning |
| --- | --- |
| Both anchor rows present | The normal case. Presence is the test, **not** whether the lists are non-empty: a row holding an empty list is a document that extracted no entities, and is a perfectly good proof. |
| `doc_status.metadata.kg_write_state == pre_graph` | Stamped at enqueue and advanced only once the anchors are durable, so it proves the document never reached its first graph mutation. Staged chunks are cleaned up with no graph access at all. |
| `doc_status.metadata.kg_purge` past `prepared` | A previous purge attempt got far enough to have deleted the anchors itself. Without this, purge's own last step would make every retry refuse forever. |
| No chunks and no populated anchor row | Nothing that carries attribution would be deleted. This is what lets a document that was queued before the `kg_write_state` marker existed, and never processed, be deleted directly — no scan, no audit. |

Three states therefore need this tool:

- documents ingested **before** #3416 landed the write-ahead anchors;
- documents written through direct KG paths such as `ainsert_custom_kg`,
  which are documented as sitting outside the document-level guarantee;
- documents ingested with `skip_kg` (`process_options` `'!'`) **before** the
  `kg_write_state` marker existed — see below.

### Documents that legitimately own nothing

`skip_kg` skips extraction and the merge entirely, so no anchor rows are ever
written. Documents ingested that way *after* this change carry
`kg_write_state=pre_graph` from enqueue and delete normally. Older ones have
neither anchors nor a marker, and anchor repair has nothing to rebuild from —
they never appear in the graph scan, because they own nothing in it.

The audit settles that case, and it is the only thing that can: it enumerates
the **whole** graph, which the hot paths deliberately never do, so a document
absent from that scan is not merely unproven but *proven empty*. Such
documents are listed under `anchorless_docs`, and `--apply` gives them empty
anchor rows — the present-and-empty pair that is the normal proof for a
document with no contributions.

Absence is only ever concluded from the completed scan. A document that does
own graph objects is repaired with its real names instead; blanking it would
manufacture a false proof and license the exact deletion this work prevents.

### What the certification assumes of the storage backends

"Proven empty" is only as strong as the enumeration it is concluded from, so
both reads are held to a complete-or-raise contract and the audit **fails
loudly instead of certifying on a partial view**:

- the `doc_status` enumeration uses `strict=True`, so a transport or
  deserialization failure raises rather than silently narrowing the set of
  documents being certified;
- the graph scan relies on `get_all_nodes()` / `get_all_edges()` never
  silently truncating. All shipped graph backends satisfy this (errors
  propagate; there is no result cap). Corrupt AGE properties raise a
  `PGGraphQueryException` rather than dropping the node or blanking the
  edge's `source_id` — a dropped node is an attribution the scan never sees,
  which is exactly how a contributing document could be falsely certified.

One sharp edge remains: a graph backend whose indices/collections are missing
(e.g. deleted out-of-band) reads as an **empty graph**, not as an error. The
audit cannot distinguish that from a workspace that truly owns no graph data —
one more reason to run it only while the server is stopped and the workspace
is healthy, as stated at the top.

Note that `--apply` reports what it found under `missing_entity_anchors` /
`missing_relation_anchors` (the diagnosis) and what it wrote under
`repaired_docs` (the action) — the first two are not emptied by a repair.

Anything listed under `orphan_entities` / `orphan_relations` cannot be
repaired by this tool: its source chunks are already gone, so no owning
document is determinable. Those need external provenance, a re-ingest, or
manual deletion.
