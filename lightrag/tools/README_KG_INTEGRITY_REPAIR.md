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
