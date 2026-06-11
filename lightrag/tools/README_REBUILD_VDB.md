# Offline Vector Storage (VDB) Rebuild Tool

`lightrag-rebuild-vdb` restores consistency between LightRAG's authoritative
data sources and its vector storages by dropping each vector storage and
rebuilding it from scratch:

| Vector storage | Authoritative source |
|---|---|
| `entities_vdb` | graph nodes |
| `relationships_vdb` | graph edges |
| `chunks_vdb` | `text_chunks` KV store |

## When do I need this?

LightRAG performs multi-step writes (graph + vector storage). If a vector
storage write fails at runtime — embedder outage, network timeout, context
overflow on a high-degree entity — the graph and the vector storage drift
apart. The most common symptom is the edge-count drift reported in issue
[#2917](https://github.com/HKUDS/LightRAG/issues/2917): graph edges with no
vector counterpart, so `local`/`hybrid` queries miss relations that exist in
the graph.

Since v(next), `amerge_entities` raises `VectorStorageConsistencyError` when
this happens, with a pointer to this tool. No data is lost in this situation:
the graph and the `text_chunks` KV store hold everything needed to rebuild
the vectors.

A full drop + rebuild also clears *reverse* orphans (records present in the
vector storage but absent from the graph), which incremental repair cannot
reliably do.

## Usage

```bash
# Stop the LightRAG Server first!
lightrag-rebuild-vdb
# or
python -m lightrag.tools.rebuild_vdb
```

The tool reads the same `.env` / environment configuration as the server
(`LIGHTRAG_GRAPH_STORAGE`, `LIGHTRAG_VECTOR_STORAGE`, `LIGHTRAG_KV_STORAGE`,
`WORKSPACE`, `WORKING_DIR`, `EMBEDDING_*`, backend connection settings) and
builds its embedding function through the exact factory the server uses —
run it with the same `.env` so rebuilt vectors live in the same embedding
space the server queries against.

Menu options:

1. **Consistency check (diagnose only)** — probes every graph entity/relation
   for a vector counterpart and reports what is missing. Run this first to
   decide whether a rebuild is worth the embedding cost. The check covers
   the graph → VDB direction only; reverse orphans can only be cleared by a
   full rebuild. Legacy reverse-order relation ids (from old custom-KG
   imports) are recognized and not misreported as missing. The check issues
   read queries only and never drops or rewrites vector data — but it is not
   strictly side-effect-free: connecting initializes each storage, and some
   backends (e.g. Qdrant) create an empty collection or payload index on
   first connect. It never drops, rewrites, or deletes existing records.
2. **Rebuild entities + relationships VDB** — sufficient for the #2917
   merge-failure scenario.
3. **Rebuild chunks VDB**.
4. **Rebuild ALL vector storages**.

## Important notes

- **Stop the server first.** The tool drops and rewrites vector storages;
  concurrent writers (any backend, not just file-based ones) can corrupt
  data or lose updates.
- **Embedding cost.** A rebuild re-embeds every affected record. On large
  datasets this means real API cost and time. Use the check mode first, and
  rebuild only the storages that need it.
- **Idempotent / crash-safe.** Sources (graph, `text_chunks`) are never
  modified. If the tool crashes between drop and rewrite, just re-run it.
- **`__created_at__` reset.** Backends that store creation timestamps in
  vector records (nano, faiss) will show fresh timestamps after a rebuild.
  No query logic depends on them.
- **Custom-KG placeholder entities.** `UNKNOWN` placeholder nodes created by
  `ainsert_custom_kg` are rebuilt faithfully from the graph; they may gain a
  vector record they previously lacked (improving their retrievability).
- **Chunk enumeration is backend-specific.** `BaseKVStorage` has no key
  enumeration API, so the tool scans each KV backend directly (JsonKV,
  Redis, PostgreSQL, MongoDB, OpenSearch). When a new KV backend is added,
  `enumerate_kv_keys()` in `rebuild_vdb.py` must be extended.

## Library usage

The core rebuild/check functions are plain async functions that accept your
own initialized storage instances:

```python
from lightrag.tools.rebuild_vdb import (
    check_vdb_consistency,
    rebuild_chunks_vdb,
    rebuild_entities_vdb,
    rebuild_relationships_vdb,
)

report = await check_vdb_consistency(graph, entities_vdb, relationships_vdb)
if not report["consistent"]:
    await rebuild_entities_vdb(graph, entities_vdb, global_config)
    await rebuild_relationships_vdb(graph, relationships_vdb, global_config)
```
