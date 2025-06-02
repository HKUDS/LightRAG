# LightRAG Tools

This directory contains useful tools that extend the functionality of LightRAG.

## Exporter

The `exporter.py` module provides functionality to export and import LightRAG data across different instances and storage backends.

### Features

- Storage-agnostic export/import system
- Complete data transfer (documents, chunks, entities, relationships, vectors)
- Works with any storage implementation
- Proper handling of embedding vectors
- Support for migrating between different storage backends
- Option to include/exclude cache data

### Use Cases

- Decoupling insert operations from inference across separate instances
- Migrating from one storage backend to another
- Creating backups of your knowledge graphs and vector data
- Setting up distributed processing workflows

### Example Usage

```python
from lightrag import LightRAG
from lightrag.tools.exporter import export_lightrag_data, import_lightrag_data

# Export data from a source LightRAG instance
source_instance = LightRAG(
    working_dir="./source_instance",
    kv_storage="JsonKVStorage",
    vector_storage="NanoVectorDBStorage", 
    graph_storage="NetworkXStorage"
)

# Add documents, entities, relationships to source_instance...

# Export all data to a directory
export_path = export_lightrag_data(
    lightrag_instance=source_instance,
    output_dir="./exports",
    include_cache=False
)

# Import into a target instance (potentially with different storage backends)
target_instance = LightRAG(
    working_dir="./target_instance",
    kv_storage="PGKVStorage",            # Different storage backend
    vector_storage="PGVectorStorage",    # Different storage backend
    graph_storage="PGGraphStorage"       # Different storage backend
)

# Import all data from the export directory
import_lightrag_data(
    lightrag_instance=target_instance,
    import_dir=export_path,
    include_cache=False
)
```

For a complete example, see [`examples/export_import_example.py`](../../examples/export_import_example.py).

### Exported Data Structure

The exported data is organized in a structured directory format:

```
lightrag_export_YYYYMMDD_HHMMSS/
├── config.json                # Configuration metadata
├── kv_stores/
│   ├── full_docs.json         # Original documents
│   ├── text_chunks.json       # Document chunks
│   └── llm_response_cache.json  # Optional LLM cache
├── vector_stores/
│   ├── entities_vdb.json      # Entity vectors and metadata
│   ├── relationships_vdb.json # Relationship vectors and metadata
│   └── chunks_vdb.json        # Chunk vectors and metadata
├── graph_store/
│   ├── nodes.json             # Graph nodes
│   ├── edges.json             # Graph edges
│   └── knowledge_graph.json   # Complete knowledge graph
└── doc_status/
    ├── status_counts.json     # Document status counts
    └── doc_statuses.json      # Document processing status
``` 