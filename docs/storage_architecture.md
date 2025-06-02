# LightRAG Storage Architecture

LightRAG provides a modular and flexible persistence layer with multiple storage implementations for different data types. This document outlines the storage architecture, data structures, and available storage implementations.

## Storage Types

LightRAG has four primary storage types:

1. **Key-Value (KV) Storage**: For document storage, text chunks, and LLM response caching
   - *Source*: [`lightrag/base.py:249-280`](../lightrag/base.py) - `BaseKVStorage` class
   - *Read/Write Functions*: `get_by_id()`, `get_by_ids()`, `upsert()`, `delete()`

2. **Vector Storage**: For semantic search of entities, relationships, and text chunks
   - *Source*: [`lightrag/base.py:143-223`](../lightrag/base.py) - `BaseVectorStorage` class
   - *Read/Write Functions*: `query()`, `upsert()`, `get_by_id()`, `get_by_ids()`, `delete()`

3. **Graph Storage**: For storing and querying relationships between entities and chunks
   - *Source*: [`lightrag/base.py:282-539`](../lightrag/base.py) - `BaseGraphStorage` class
   - *Read/Write Functions*: `upsert_node()`, `upsert_edge()`, `get_node()`, `get_edge()`, `delete_node()`, `get_knowledge_graph()`

4. **Document Status Storage**: For tracking document processing status
   - *Source*: [`lightrag/base.py:549-593`](../lightrag/base.py) - `DocStatusStorage` class
   - *Read/Write Functions*: `get_docs_by_status()`, `get_status_counts()`

## Storage Implementations

Each storage type can be implemented using different backends as defined in [`lightrag/kg/__init__.py:1-46`](../lightrag/kg/__init__.py):

### KV Storage Implementations
- `JsonKVStorage` - File-based JSON storage ([`lightrag/kg/json_kv_impl.py`](../lightrag/kg/json_kv_impl.py))
- `RedisKVStorage` - Redis-based KV storage ([`lightrag/kg/redis_impl.py`](../lightrag/kg/redis_impl.py))
- `PGKVStorage` - PostgreSQL-based KV storage ([`lightrag/kg/postgres_impl.py`](../lightrag/kg/postgres_impl.py))
- `MongoKVStorage` - MongoDB-based KV storage ([`lightrag/kg/mongo_impl.py`](../lightrag/kg/mongo_impl.py))

### Vector Storage Implementations
- `NanoVectorDBStorage` - In-memory vector database ([`lightrag/kg/nano_vector_db_impl.py`](../lightrag/kg/nano_vector_db_impl.py))
- `MilvusVectorDBStorage` - Milvus vector database ([`lightrag/kg/milvus_impl.py`](../lightrag/kg/milvus_impl.py))
- `ChromaVectorDBStorage` - Chroma vector database ([`lightrag/kg/chroma_impl.py`](../lightrag/kg/chroma_impl.py))
- `PGVectorStorage` - PostgreSQL with pgvector extension ([`lightrag/kg/postgres_impl.py`](../lightrag/kg/postgres_impl.py))
- `FaissVectorDBStorage` - Facebook AI Similarity Search (FAISS) ([`lightrag/kg/faiss_impl.py`](../lightrag/kg/faiss_impl.py))
- `QdrantVectorDBStorage` - Qdrant vector database ([`lightrag/kg/qdrant_impl.py`](../lightrag/kg/qdrant_impl.py))
- `MongoVectorDBStorage` - MongoDB with vector search ([`lightrag/kg/mongo_impl.py`](../lightrag/kg/mongo_impl.py))

### Graph Storage Implementations
- `NetworkXStorage` - In-memory graph using NetworkX ([`lightrag/kg/networkx_impl.py`](../lightrag/kg/networkx_impl.py))
- `Neo4JStorage` - Neo4j graph database ([`lightrag/kg/neo4j_impl.py`](../lightrag/kg/neo4j_impl.py))
- `PGGraphStorage` - PostgreSQL with graph capabilities ([`lightrag/kg/postgres_impl.py`](../lightrag/kg/postgres_impl.py))
- `AGEStorage` - Apache AGE (graph extension for PostgreSQL) ([`lightrag/kg/age_impl.py`](../lightrag/kg/age_impl.py))

### Document Status Storage Implementations
- `JsonDocStatusStorage` - File-based document status storage ([`lightrag/kg/json_doc_status_impl.py`](../lightrag/kg/json_doc_status_impl.py))
- `PGDocStatusStorage` - PostgreSQL-based document status storage ([`lightrag/kg/postgres_impl.py`](../lightrag/kg/postgres_impl.py))
- `MongoDocStatusStorage` - MongoDB-based document status storage ([`lightrag/kg/mongo_impl.py`](../lightrag/kg/mongo_impl.py))

## Namespaces

LightRAG uses namespaces to organize storage for different data types as defined in [`lightrag/namespace.py`](../lightrag/namespace.py):

```python
class NameSpace:
    KV_STORE_FULL_DOCS = "full_docs"
    KV_STORE_TEXT_CHUNKS = "text_chunks"
    KV_STORE_LLM_RESPONSE_CACHE = "llm_response_cache"

    VECTOR_STORE_ENTITIES = "entities"
    VECTOR_STORE_RELATIONSHIPS = "relationships"
    VECTOR_STORE_CHUNKS = "chunks"

    GRAPH_STORE_CHUNK_ENTITY_RELATION = "chunk_entity_relation"

    DOC_STATUS = "doc_status"
```

The namespaces are instantiated in the LightRAG class initialization in [`lightrag/lightrag.py:370-413`](../lightrag/lightrag.py)

## Data Structures and Fields

### Key-Value Storage Data Models

#### Full Documents (KV_STORE_FULL_DOCS)
Stores original documents for retrieval:
- `id`: Document identifier
- `content`: Document content
- `file_path`: Original document file path
- `metadata`: Additional document metadata

*Used in*: [`lightrag.ainsert_custom_chunks`](../lightrag/lightrag.py), [`lightrag.apipeline_process_enqueue_documents`](../lightrag/lightrag.py)

#### Text Chunks (KV_STORE_TEXT_CHUNKS)
Stores chunked document content:
- `tokens`: Number of tokens in the chunk
- `content`: Chunk text content
- `full_doc_id`: Reference to parent document
- `chunk_order_index`: Position in original document

*Used in*: [`lightrag.ainsert_custom_chunks`](../lightrag/lightrag.py), [`lightrag.apipeline_process_enqueue_documents`](../lightrag/lightrag.py)

*Source*: [`lightrag/base.py:18-22`](../lightrag/base.py) - `TextChunkSchema` class

#### LLM Response Cache (KV_STORE_LLM_RESPONSE_CACHE)
Caches LLM responses to avoid redundant calls:
- `args_hash`: Hash of input arguments
- `content`: Cached response
- `prompt`: Original prompt
- `quantized`: Whether response is quantized
- `min_val`/`max_val`: Quantization range values
- `mode`: Query mode used
- `cache_type`: Type of cache entry

*Used in*: [`operate.py:862-993`](../lightrag/operate.py) - `kg_query` function

### Vector Storage Data Models

#### Entities Vector DB (VECTOR_STORE_ENTITIES)
Stores entity embeddings for semantic search:
- `id`: Entity identifier
- `vector`: Embedding vector
- `entity_name`: Name of the entity
- `source_id`: Source document/chunk ID
- `content`: Entity content/description
- `file_path`: Origin file path

*Meta fields defined in*: [`lightrag/lightrag.py:392-397`](../lightrag/lightrag.py)

#### Relationships Vector DB (VECTOR_STORE_RELATIONSHIPS)
Stores relationship embeddings:
- `id`: Relationship identifier
- `vector`: Embedding vector
- `source_name`: Source entity name
- `target_name`: Target entity name
- `relation_id`: Type of relationship
- `description`: Relationship description
- `content`: Contextual content

*Meta fields defined in*: [`lightrag/lightrag.py:398-403`](../lightrag/lightrag.py)

#### Chunks Vector DB (VECTOR_STORE_CHUNKS)
Stores chunk embeddings:
- `id`: Chunk identifier
- `vector`: Embedding vector
- `content`: Chunk text content
- `full_doc_id`: Reference to parent document
- `chunk_order_index`: Position in original document
- `file_path`: Origin file path

*Meta fields defined in*: [`lightrag/lightrag.py:404-408`](../lightrag/lightrag.py)

### Graph Storage Data Models

#### Chunk-Entity-Relation Graph (GRAPH_STORE_CHUNK_ENTITY_RELATION)

##### Nodes
- `id`: Node identifier
- `entity_type`: Type of entity
- Custom properties based on entity type

*Read/Write APIs*: 
- Add node: [`BaseGraphStorage.upsert_node`](../lightrag/base.py)
- Get node: [`BaseGraphStorage.get_node`](../lightrag/base.py)
- Delete node: [`BaseGraphStorage.delete_node`](../lightrag/base.py)

##### Edges
- `source`: Source node ID
- `target`: Target node ID
- `type`: Type of relationship
- `weight`: Relationship strength
- `description`: Relationship description

*Read/Write APIs*: 
- Add edge: [`BaseGraphStorage.upsert_edge`](../lightrag/base.py)
- Get edge: [`BaseGraphStorage.get_edge`](../lightrag/base.py)
- Delete edge: [`BaseGraphStorage.remove_edges`](../lightrag/base.py)

### Document Status Storage

Tracks document processing status with these fields defined in the [`DocProcessingStatus` class](../lightrag/base.py:549-574):
- `content`: Original document content
- `content_summary`: Preview of document content
- `content_length`: Total document length
- `file_path`: Document file path
- `status`: Processing status (PENDING, PROCESSING, PROCESSED, FAILED)
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp
- `chunks_count`: Number of chunks after splitting
- `error`: Error message if processing failed
- `metadata`: Additional metadata

*Used in*: [`lightrag.apipeline_enqueue_documents`](../lightrag/lightrag.py), [`lightrag.apipeline_process_enqueue_documents`](../lightrag/lightrag.py)

## Storage Persistence

LightRAG handles persistence differently based on the storage implementation:

1. **File-based storage** (JSON implementations):
   - Data is loaded from disk on initialization
   - Changes are persisted to disk during `index_done_callback()`
   - Uses file locking to prevent corruption
   
   *Example*: [`JsonKVStorage.index_done_callback`](../lightrag/kg/json_kv_impl.py:63-85)

2. **In-memory storage** (NetworkX, NanoVectorDB, FAISS):
   - Data is loaded from disk on initialization
   - Changes are held in memory and persisted to disk during `index_done_callback()`
   - Uses shared locks to coordinate between processes
   
   *Example*: [`NanoVectorDBStorage.index_done_callback`](../lightrag/kg/nano_vector_db_impl.py:187-213)

3. **External database storage** (PostgreSQL, MongoDB, Neo4j, etc.):
   - Data is stored directly in the external database
   - Persistence is handled by the database system
   - `index_done_callback()` is generally a no-op
   
   *Examples*: 
   - [`PGGraphStorage.index_done_callback`](../lightrag/kg/postgres_impl.py:1196-1198)
   - [`AGEStorage.index_done_callback`](../lightrag/kg/age_impl.py:847-849)

The persistence and storage initialization/finalization is managed through functions in [`lightrag/lightrag.py:497-586`](../lightrag/lightrag.py):
- `initialize_storages()`: Initialize all storage instances
- `finalize_storages()`: Safely persist all data and close connections

## Cross-Process Coordination

LightRAG supports multi-process operation with coordination mechanisms in [`lightrag/kg/shared_storage.py`](../lightrag/kg/shared_storage.py):

- Shared locks prevent concurrent modifications using `get_storage_lock()`
- Update flags notify processes about changes made by other processes with `get_update_flag()` and `set_all_update_flags()`
- `index_done_callback()` ensures data consistency across processes

This allows multiple processes to work with the same data without conflicts.

## Data Export and Import

LightRAG provides a storage-agnostic data export/import system that enables:
1. Decoupling insert operations from inference across separate instances
2. Migrating between different storage backends
3. Creating backups of knowledge graphs and vector data
4. Setting up distributed processing workflows

### Export/Import Functionality

The export/import system is implemented in [`lightrag/tools/exporter.py`](../lightrag/tools/exporter.py) with two main functions:

- `export_lightrag_data()`: Exports all data from a LightRAG instance to a directory
- `import_lightrag_data()`: Imports data from an export directory into a LightRAG instance

The system handles all storage types (KV, Vector, Graph, Document Status) and preserves all relationships between entities and documents.

### Usage Example

```python
from lightrag import LightRAG
from lightrag.tools.exporter import export_lightrag_data, import_lightrag_data

# Export from source instance
source_instance = LightRAG(
    working_dir="./source_instance",
    kv_storage="JsonKVStorage",
    vector_storage="NanoVectorDBStorage",
    graph_storage="NetworkXStorage"
)

# Export all data to a directory
export_path = export_lightrag_data(
    lightrag_instance=source_instance,
    output_dir="./exports",
    include_cache=False  # Set to True to include LLM cache
)

# Import to target instance (potentially with different storage)
target_instance = LightRAG(
    working_dir="./inference_instance",
    kv_storage="PGKVStorage",            # Different storage backend
    vector_storage="PGVectorStorage",    # Different storage backend
    graph_storage="PGGraphStorage",      # Different storage backend
)

# Import data from export directory
import_lightrag_data(
    lightrag_instance=target_instance,
    import_dir=export_path,
    include_cache=False
)
```

A complete example demonstrating export/import functionality is available in [`examples/export_import_example.py`](../examples/export_import_example.py).

### Export Data Format

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
│   └── knowledge_graph.json   # Complete knowledge graph (for verification)
└── doc_status/
    ├── status_counts.json     # Document status counts
    └── doc_statuses.json      # Document processing status
```

### Key Features

1. **Storage-agnostic**: Works with any storage implementation
2. **Complete data transfer**: Preserves all documents, chunks, entities, relations, and vectors
3. **Vector handling**: Properly handles embedding vectors (conversion between numpy and lists)
4. **Cross-backend migration**: Supports transferring data between different storage backends
5. **Configurable**: Option to include/exclude cache data
6. **Validation**: Checks for compatibility issues (e.g., embedding dimensions)
7. **Export verification**: Exports complete knowledge graph for verification purposes

## LightRAG Storage Operations

LightRAG provides high-level functions to work with the storage layer:

### Document Management
- `insert()` / `ainsert()`: [`lightrag/lightrag.py:669-701`](../lightrag/lightrag.py) - Insert documents into the system
- `apipeline_enqueue_documents()`: [`lightrag/lightrag.py:771-870`](../lightrag/lightrag.py) - Queue documents for processing 
- `apipeline_process_enqueue_documents()`: [`lightrag/lightrag.py:872-1236`](../lightrag/lightrag.py) - Process queued documents
- `adelete_by_doc_id()`: [`lightrag/lightrag.py:1371-1632`](../lightrag/lightrag.py) - Delete documents and related data

### Entity and Relation Management
- `aedit_entity()` / `edit_entity()`: [`lightrag/lightrag.py:1788-1805`](../lightrag/lightrag.py) - Edit entity properties
- `aedit_relation()` / `edit_relation()`: [`lightrag/lightrag.py:1807-1830`](../lightrag/lightrag.py) - Edit relation properties
- `acreate_entity()` / `create_entity()`: [`lightrag/lightrag.py:1832-1851`](../lightrag/lightrag.py) - Create a new entity
- `acreate_relation()` / `create_relation()`: [`lightrag/lightrag.py:1853-1877`](../lightrag/lightrag.py) - Create a new relation
- `amerge_entities()` / `merge_entities()`: [`lightrag/lightrag.py:1879-1918`](../lightrag/lightrag.py) - Merge multiple entities

### Graph Operations
- `get_knowledge_graph()`: [`lightrag/lightrag.py:589-607`](../lightrag/lightrag.py) - Get knowledge graph for visualization
- `get_entity_info()`: [`lightrag/lightrag.py:1660-1667`](../lightrag/lightrag.py) - Get detailed information about an entity
- `get_relation_info()`: [`lightrag/lightrag.py:1669-1680`](../lightrag/lightrag.py) - Get detailed information about a relation

## Schema Examples

### PostgreSQL Schema (Selected Tables)

Implementation in [`lightrag/kg/postgres_impl.py`](../lightrag/kg/postgres_impl.py).

```sql
-- KV Storage Table
CREATE TABLE LIGHTRAG_KV_STORE (
    key VARCHAR(256) PRIMARY KEY,
    workspace VARCHAR(256),
    value JSONB,
    createtime TIMESTAMP,
    updatetime TIMESTAMP
);

-- Vector Storage Table
CREATE TABLE LIGHTRAG_VECTOR_STORE (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    entity_id VARCHAR(256),
    workspace VARCHAR(256),
    content_vector VECTOR,
    content TEXT,
    metadata JSONB,
    createtime TIMESTAMP,
    updatetime TIMESTAMP
);

-- Graph Nodes Table
CREATE TABLE LIGHTRAG_GRAPH_NODES (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    node_id VARCHAR(256),
    workspace VARCHAR(256),
    entity_type VARCHAR(256),
    keywords TEXT,
    description TEXT,
    content TEXT,
    content_vector VECTOR,
    createtime TIMESTAMP,
    updatetime TIMESTAMP
);

-- Graph Edges Table
CREATE TABLE LIGHTRAG_GRAPH_EDGES (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    relation_id VARCHAR(256),
    workspace VARCHAR(256),
    source_name VARCHAR(2048),
    target_name VARCHAR(2048),
    weight DECIMAL,
    keywords TEXT,
    description TEXT,
    content TEXT,
    content_vector VECTOR,
    createtime TIMESTAMP,
    updatetime TIMESTAMP
);

-- Document Status Table
CREATE TABLE LIGHTRAG_DOC_STATUS (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    file_path VARCHAR(1024),
    workspace VARCHAR(256),
    status VARCHAR(32),
    content TEXT,
    content_summary VARCHAR(512),
    content_length INTEGER,
    chunks_count INTEGER,
    error TEXT,
    metadata JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
``` 