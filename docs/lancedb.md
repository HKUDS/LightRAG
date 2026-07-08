# LanceDB Storage Backend

## Overview

LanceDB is an embedded, serverless vector database built on the Lance columnar format. The LanceDB backend implements **all four LightRAG storage types in a single local directory** — no PostgreSQL, no Docker, no external services:

| LightRAG storage type | Class | What it stores |
|---|---|---|
| `KV_STORAGE` | `LanceDBKVStorage` | Full docs, text chunks, LLM response cache, entity/relation bookkeeping |
| `VECTOR_STORAGE` | `LanceDBVectorStorage` | Entity/relationship/chunk embeddings |
| `GRAPH_STORAGE` | `LanceDBGraphStorage` | Entity-relation knowledge graph (nodes + undirected edges) |
| `DOC_STATUS_STORAGE` | `LanceDBDocStatusStorage` | Document processing status |

Highlights:

- **Zero external dependencies** — everything is stored in one directory; `pip install lancedb` is the only requirement (auto-installed on first use).
- **Native async** — implemented on `lancedb.connect_async()`; no thread-pool bridging.
- **CJK-friendly full-text search** — a BM25 index with a bigram tokenizer is created on vector-storage content by default, so Chinese/Japanese/Korean text is searchable out of the box (exposed via `LanceDBVectorStorage.full_text_search()`).
- **Deferred embedding** — vector upserts buffer records and embed them in batches of `embedding_batch_num` at flush time, minimizing embedding API round-trips.
- **Embedding-model isolation** — vector table names carry the embedding model name and dimension (e.g. `demo_chunks_text_embedding_3_large_3072d`), so switching models starts clean tables instead of mixing incompatible vectors.
- **Rebuildable by design** — all data is derived from your source documents. If anything corrupts, delete the LanceDB directory and re-index.

## Installation

```bash
pip install lancedb          # or: pip install lightrag-hku[offline-storage]
```

Python 3.10+ is required. Wheels bundle the native Rust core for macOS arm64, Linux x86_64/aarch64, and Windows x64.

## Configuration

### Selecting the backend

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=your_llm_func,
    embedding_func=your_embedding_func,
    kv_storage="LanceDBKVStorage",
    vector_storage="LanceDBVectorStorage",
    graph_storage="LanceDBGraphStorage",
    doc_status_storage="LanceDBDocStatusStorage",
)
await rag.initialize_storages()   # REQUIRED
```

Or via environment variables (API server):

```bash
LIGHTRAG_KV_STORAGE=LanceDBKVStorage
LIGHTRAG_VECTOR_STORAGE=LanceDBVectorStorage
LIGHTRAG_GRAPH_STORAGE=LanceDBGraphStorage
LIGHTRAG_DOC_STATUS_STORAGE=LanceDBDocStatusStorage
```

### Environment variables

All variables are optional; they can also be placed in a `[lancedb]` section of `config.ini` (env vars take precedence).

| Variable | Default | Description |
|---|---|---|
| `LANCEDB_URI` | `<working_dir>/lancedb` | Database directory path |
| `LANCEDB_WORKSPACE` | — | Force a workspace name, overriding the instance `workspace` (compatibility escape hatch; normally leave unset) |
| `LANCEDB_READ_CONSISTENCY_INTERVAL` | `0` | Seconds between table-version checks; `0` = strong consistency |
| `LANCEDB_ENABLE_FTS` | `true` | Create a BM25 full-text index on vector-storage `content` |
| `LANCEDB_FTS_TOKENIZER` | `ngram` | `ngram` (CJK-friendly bigrams), `simple`, `whitespace`, or `jieba/default` / `lindera/*` (require [Lance language models](https://lancedb.github.io/lancedb/fts/) downloaded locally) |
| `LANCEDB_OPTIMIZE_THRESHOLD` | `64` | Compact a table after this many write operations; `0` disables auto-optimize |

### Workspace isolation

Each `LightRAG(workspace=...)` gets its own set of tables via name prefixes (e.g. `projecta_chunks`, `projecta_doc_status`), the same model used by the collection-based backends. Multiple workspaces share one LanceDB directory.

## Table layout

For a workspace `demo`, the directory contains one Lance table per namespace:

```
demo_full_docs, demo_text_chunks, demo_llm_response_cache, ...   # KV (id, payload JSON, timestamps)
demo_entities_<model>_<dim>d, demo_relationships_..., demo_chunks_...  # vectors
demo_chunk_entity_relation_nodes / _edges                        # graph
demo_doc_status                                                  # doc status
```

Records are stored as JSON strings in a `payload` column for full fidelity; fields used in SQL filters (`id`, `src_id`/`tgt_id`, `status`, `file_path`, `content_hash`, ...) are typed columns. Undirected graph edges are stored once under a canonical id derived from the sorted endpoint pair.

## Full-text search (CJK support)

`LanceDBVectorStorage` exposes an extra method beyond the standard LightRAG contract:

```python
hits = await rag.chunks_vdb.full_text_search("朱元璋", top_k=10)
# [{"id": "chunk-...", "content": "...", "score": 5.1, ...}, ...]
```

The default `ngram` tokenizer indexes character bigrams, which handles Chinese/Japanese/Korean text without external language models while still matching Latin-script terms. For dictionary-based Chinese segmentation, install the Lance jieba language model and set `LANCEDB_FTS_TOKENIZER=jieba/default`.

Rows written after index creation are still searchable (LanceDB scans the unindexed tail automatically); the periodic auto-optimize folds them into the index.

## Vector search behavior

- Searches are exact (brute-force) cosine kNN — no ANN index is created, which is the recommended configuration below ~100k vectors per table and avoids recall loss.
- `query()` returns results filtered by `cosine_better_than_threshold` (configured through `vector_db_storage_cls_kwargs`, default 0.2), with the LightRAG-standard result shape: meta fields plus `id`, `created_at`, and `distance` (which carries cosine **similarity**, matching every other LightRAG backend).

## Limitations

- **Single-process only.** The embedded database and its per-table write locks live in one process. `lightrag-gunicorn` refuses to start with multiple workers when a LanceDB backend is selected; use PostgreSQL/MongoDB/OpenSearch for multi-worker deployments.
- **`fork` multiprocessing is unsupported** by LanceDB's Rust runtime — use the `spawn` start method if you must create subprocesses.
- Old table versions are retained for MVCC; the periodic auto-optimize (`LANCEDB_OPTIMIZE_THRESHOLD`) compacts fragments and prunes old versions.
- The standalone maintenance tools (`lightrag.tools.rebuild_vdb`, `clean_llm_query_cache`, `migrate_llm_cache`) do not support LanceDB yet. Since all LanceDB data is derived, the equivalent of a rebuild is deleting the LanceDB directory and re-indexing.

## Testing

```bash
# Unit + integration tests (offline, real embedded LanceDB in tmp dirs)
python -m pytest tests/kg/lancedb_impl -v

# Cross-backend graph storage conformance suite
LIGHTRAG_GRAPH_STORAGE=LanceDBGraphStorage \
python -m pytest tests/kg/test_graph_storage.py --run-integration -v

# Runnable demo (no API keys or services needed)
python examples/lightrag_lancedb_demo.py
```

## References

- LanceDB documentation: https://lancedb.github.io/lancedb/
- Lance columnar format: https://github.com/lancedb/lance
- Backend-specific setup section in `docs/ProgramingWithCore.md`
