# LightRAG Storage Backends

> Complete guide to storage backend configuration and implementation

**Version**: 1.4.9.1 | **Last Updated**: December 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Storage Types](#storage-types)
3. [Backend Comparison](#backend-comparison)
4. [PostgreSQL Backend](#postgresql-backend)
5. [MongoDB Backend](#mongodb-backend)
6. [Neo4j Backend](#neo4j-backend)
7. [Redis Backend](#redis-backend)
8. [File-Based Backends](#file-based-backends)
9. [Vector Databases](#vector-databases)
10. [Configuration Reference](#configuration-reference)

---

## Overview

LightRAG uses four types of storage, each with multiple backend options:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Storage Architecture                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    LightRAG Core                                 │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│          ┌───────────────────┼───────────────────┐                     │
│          │                   │                   │                     │
│          ▼                   ▼                   ▼                     │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐            │
│  │   KV Storage  │   │ Vector Store  │   │  Graph Store  │            │
│  │   (Documents, │   │  (Embeddings) │   │   (KG Nodes   │            │
│  │    Chunks,    │   │               │   │    & Edges)   │            │
│  │    Cache)     │   │               │   │               │            │
│  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘            │
│          │                   │                   │                     │
│          ▼                   ▼                   ▼                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Backend Implementations                        │   │
│  │                                                                  │   │
│  │  PostgreSQL │ MongoDB │ Redis │ Neo4j │ Milvus │ Qdrant │ FAISS │   │
│  │  JSON/File  │ NetworkX │ NanoVectorDB │ Memgraph │ ...           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Storage Types

### 1. Key-Value Storage (`BaseKVStorage`)

Stores documents, chunks, and LLM cache.

| Implementation | Description | Use Case |
|----------------|-------------|----------|
| `JsonKVStorage` | File-based JSON | Development, single-node |
| `PGKVStorage` | PostgreSQL tables | Production, multi-node |
| `MongoKVStorage` | MongoDB collections | Production, flexible schema |
| `RedisKVStorage` | Redis hash maps | High-performance caching |

### 2. Vector Storage (`BaseVectorStorage`)

Stores and queries embedding vectors.

| Implementation | Description | Use Case |
|----------------|-------------|----------|
| `NanoVectorDBStorage` | In-memory, file-persisted | Development, small datasets |
| `PGVectorStorage` | PostgreSQL + pgvector | Production, unified DB |
| `MilvusVectorDBStorage` | Milvus vector DB | Large-scale production |
| `QdrantVectorDBStorage` | Qdrant vector DB | Cloud-native production |
| `FaissVectorDBStorage` | FAISS index | Local high-performance |
| `MongoVectorDBStorage` | MongoDB Atlas Vector | MongoDB ecosystem |

### 3. Graph Storage (`BaseGraphStorage`)

Stores knowledge graph nodes and edges.

| Implementation | Description | Use Case |
|----------------|-------------|----------|
| `NetworkXStorage` | In-memory NetworkX | Development, small graphs |
| `PGGraphStorage` | PostgreSQL tables | Production, unified DB |
| `Neo4JStorage` | Native graph DB | Complex graph queries |
| `MemgraphStorage` | In-memory graph DB | Real-time analytics |
| `MongoGraphStorage` | MongoDB documents | Document-graph hybrid |

### 4. Document Status Storage (`DocStatusStorage`)

Tracks document processing status.

| Implementation | Description | Use Case |
|----------------|-------------|----------|
| `JsonDocStatusStorage` | File-based JSON | Development |
| `PGDocStatusStorage` | PostgreSQL | Production |
| `MongoDocStatusStorage` | MongoDB | Production |
| `RedisDocStatusStorage` | Redis | Distributed |

---

## Backend Comparison

### Feature Matrix

```
┌────────────────────┬─────────┬────────┬───────┬────────┬───────────┐
│     Feature        │ PG Full │ Mongo  │ Neo4j │ Mixed  │ File-Only │
├────────────────────┼─────────┼────────┼───────┼────────┼───────────┤
│ KV Storage         │    ✅   │   ✅   │   ❌  │   ✅   │    ✅     │
│ Vector Storage     │    ✅   │   ✅   │   ❌  │   ✅   │    ✅     │
│ Graph Storage      │    ✅   │   ✅   │   ✅  │   ✅   │    ✅     │
│ Doc Status         │    ✅   │   ✅   │   ❌  │   ✅   │    ✅     │
│ Multi-tenant       │    ✅   │   ✅   │   ✅  │   ✅   │    ⚠️    │
│ Horizontal Scale   │    ✅   │   ✅   │   ✅  │   ✅   │    ❌     │
│ ACID Transactions  │    ✅   │   ⚠️   │   ✅  │   ⚠️   │    ❌     │
│ Zero Dependencies  │    ❌   │   ❌   │   ❌  │   ❌   │    ✅     │
│ Graph Queries      │    ⚠️   │   ⚠️   │   ✅  │   ✅   │    ⚠️    │
│ Vector Search      │    ✅   │   ✅   │   ❌  │   ✅   │    ✅     │
└────────────────────┴─────────┴────────┴───────┴────────┴───────────┘

Legend: ✅ Full support  ⚠️ Limited  ❌ Not supported
```

### Performance Characteristics

| Backend | Write Speed | Read Speed | Memory Usage | Disk Usage |
|---------|-------------|------------|--------------|------------|
| PostgreSQL Full | Fast | Fast | Medium | Compact |
| MongoDB Full | Fast | Fast | Medium | Medium |
| Neo4j + Vector | Slow | Fast (graph) | High | Medium |
| File-based | Slow | Medium | Low | Compact |
| Milvus/Qdrant | Fast | Very Fast | High | Large |

---

## PostgreSQL Backend

### Complete PostgreSQL Setup

PostgreSQL can handle ALL storage types (recommended for production):

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",

    # All PostgreSQL backends
    kv_storage="PGKVStorage",
    vector_storage="PGVectorStorage",
    graph_storage="PGGraphStorage",
    doc_status_storage="PGDocStatusStorage",
)
```

### Environment Variables

```bash
# Required
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DATABASE=lightrag

# Optional
POSTGRES_MAX_CONNECTIONS=100
POSTGRES_SSL_MODE=prefer           # disable|allow|prefer|require|verify-ca|verify-full
POSTGRES_SSL_CERT=/path/to/cert
POSTGRES_SSL_KEY=/path/to/key
POSTGRES_SSL_ROOT_CERT=/path/to/ca

# Vector index configuration
POSTGRES_VECTOR_INDEX_TYPE=hnsw    # hnsw|ivfflat
POSTGRES_HNSW_M=16
POSTGRES_HNSW_EF=64
POSTGRES_IVFFLAT_LISTS=100
```

### Schema Overview

```sql
-- Documents table
CREATE TABLE LIGHTRAG_DOC_FULL (
    workspace VARCHAR(1024) NOT NULL,
    id VARCHAR(255) NOT NULL,
    doc_name VARCHAR(1024),
    content TEXT,
    meta JSONB,
    createtime TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
    updatetime TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (workspace, id)
);

-- Chunks table
CREATE TABLE LIGHTRAG_DOC_CHUNKS (
    workspace VARCHAR(1024) NOT NULL,
    id VARCHAR(255) NOT NULL,
    full_doc_id VARCHAR(255),
    chunk_order_index INT,
    tokens INT,
    content TEXT,
    content_summary TEXT,
    file_path VARCHAR(32768),
    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (workspace, id)
);

-- Entity vectors (pgvector extension required)
CREATE TABLE LIGHTRAG_VDB_ENTITY (
    workspace VARCHAR(1024) NOT NULL,
    id VARCHAR(255) NOT NULL,
    entity_name VARCHAR(1024),
    content TEXT,
    content_vector VECTOR(1024),  -- Adjust dimension to match embedding
    source_id TEXT,
    file_path TEXT,
    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (workspace, id)
);

-- Graph nodes
CREATE TABLE LIGHTRAG_GRAPH_NODES (
    workspace VARCHAR(1024) NOT NULL,
    id VARCHAR(255) NOT NULL,
    entity_type VARCHAR(255),
    description TEXT,
    source_id TEXT,
    file_path TEXT,
    created_at INT,
    PRIMARY KEY (workspace, id)
);

-- Graph edges
CREATE TABLE LIGHTRAG_GRAPH_EDGES (
    workspace VARCHAR(1024) NOT NULL,
    source_id VARCHAR(255) NOT NULL,
    target_id VARCHAR(255) NOT NULL,
    weight FLOAT,
    description TEXT,
    keywords TEXT,
    source_chunk_id TEXT,
    file_path TEXT,
    created_at INT,
    PRIMARY KEY (workspace, source_id, target_id)
);
```

### pgvector Index Types

```sql
-- HNSW index (recommended for accuracy)
CREATE INDEX ON LIGHTRAG_VDB_ENTITY
USING hnsw (content_vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- IVFFlat index (faster but less accurate)
CREATE INDEX ON LIGHTRAG_VDB_ENTITY
USING ivfflat (content_vector vector_cosine_ops)
WITH (lists = 100);
```

---

## MongoDB Backend

### Complete MongoDB Setup

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",

    # All MongoDB backends
    kv_storage="MongoKVStorage",
    vector_storage="MongoVectorDBStorage",
    graph_storage="MongoGraphStorage",
    doc_status_storage="MongoDocStatusStorage",
)
```

### Environment Variables

```bash
MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE=lightrag

# Atlas Vector Search (optional)
MONGO_ATLAS_CLUSTER=your-cluster
MONGO_ATLAS_API_KEY=your-api-key
```

### Collection Structure

```javascript
// Documents collection
db.lightrag_doc_full.insertOne({
    _id: "workspace:doc_id",
    workspace: "default",
    doc_id: "abc123",
    doc_name: "document.txt",
    content: "Full document text...",
    meta: { source: "upload" },
    created_at: ISODate(),
    updated_at: ISODate()
});

// Entities collection (with vector)
db.lightrag_entities.insertOne({
    _id: "workspace:entity_id",
    workspace: "default",
    entity_name: "Apple Inc.",
    entity_type: "organization",
    description: "Technology company...",
    content: "Apple Inc.\nTechnology company...",
    embedding: [0.1, 0.2, ...],  // Vector embedding
    source_id: "chunk_001,chunk_002",
    file_path: "document.txt"
});

// Graph edges collection
db.lightrag_graph_edges.insertOne({
    _id: "workspace:source:target",
    workspace: "default",
    source: "Apple Inc.",
    target: "iPhone",
    weight: 3.5,
    description: "Produces the iPhone",
    keywords: "technology,smartphone"
});
```

### Vector Search Index (Atlas)

```javascript
// Create vector search index
db.lightrag_entities.createSearchIndex({
    name: "vector_index",
    definition: {
        mappings: {
            dynamic: true,
            fields: {
                embedding: {
                    type: "knnVector",
                    dimensions: 1024,
                    similarity: "cosine"
                }
            }
        }
    }
});
```

---

## Neo4j Backend

### Neo4j for Graph Storage

Neo4j provides native graph storage with Cypher queries:

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",

    # Neo4j for graph, other backends for KV/Vector
    kv_storage="PGKVStorage",           # or JsonKVStorage
    vector_storage="PGVectorStorage",    # or other vector DB
    graph_storage="Neo4JStorage",        # Neo4j graph
    doc_status_storage="PGDocStatusStorage",
)
```

### Environment Variables

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Optional
NEO4J_DATABASE=neo4j
NEO4J_ENCRYPTED=false
```

### Graph Schema

```cypher
// Entity nodes
CREATE (e:Entity {
    entity_id: "Apple Inc.",
    entity_type: "organization",
    description: "Technology company...",
    source_id: "chunk_001",
    workspace: "default"
})

// Relationship
MATCH (a:Entity {entity_id: "Apple Inc."})
MATCH (b:Entity {entity_id: "iPhone"})
CREATE (a)-[r:RELATED_TO {
    weight: 3.5,
    description: "Produces",
    keywords: "technology"
}]->(b)
```

### Cypher Queries Used

```cypher
-- Get node with edges
MATCH (n:Entity {entity_id: $entity_id, workspace: $workspace})
OPTIONAL MATCH (n)-[r]-(m)
RETURN n, r, m

-- Get knowledge graph (BFS)
MATCH path = (start:Entity {entity_id: $label})-[*1..3]-(connected)
WHERE start.workspace = $workspace
RETURN path
LIMIT $max_nodes

-- Search nodes
MATCH (n:Entity)
WHERE n.workspace = $workspace
  AND toLower(n.entity_id) CONTAINS toLower($query)
RETURN n.entity_id
ORDER BY n.degree DESC
LIMIT $limit
```

---

## Redis Backend

### Redis for KV and Doc Status

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",

    kv_storage="RedisKVStorage",
    vector_storage="NanoVectorDBStorage",  # Redis doesn't have vector
    graph_storage="NetworkXStorage",       # Redis doesn't have graph
    doc_status_storage="RedisDocStatusStorage",
)
```

### Environment Variables

```bash
REDIS_URI=redis://localhost:6379
# or with auth
REDIS_URI=redis://user:password@localhost:6379/0
```

### Key Structure

```
# Document storage
lightrag:{workspace}:full_docs:{doc_id} -> JSON document

# Chunks storage
lightrag:{workspace}:text_chunks:{chunk_id} -> JSON chunk

# LLM cache
lightrag:{workspace}:llm_cache:{cache_key} -> JSON response

# Document status
lightrag:{workspace}:doc_status:{doc_id} -> JSON status
```

---

## File-Based Backends

### Zero-Dependency Setup

Best for development and small-scale usage:

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",

    # All file-based (default)
    kv_storage="JsonKVStorage",
    vector_storage="NanoVectorDBStorage",
    graph_storage="NetworkXStorage",
    doc_status_storage="JsonDocStatusStorage",
)
```

### File Structure

```
./rag_storage/
├── full_docs.json              # Complete documents
├── text_chunks.json            # Document chunks
├── llm_response_cache.json     # LLM cache
├── full_entities.json          # Entity metadata
├── full_relations.json         # Relation metadata
├── vdb_entities.json           # Entity vectors
├── vdb_relationships.json      # Relation vectors
├── vdb_chunks.json             # Chunk vectors
├── graph_chunk_entity_relation.graphml  # Knowledge graph
└── doc_status.json             # Processing status
```

### NanoVectorDB Format

```json
{
  "data": {
    "ent-abc123": {
      "__id__": "ent-abc123",
      "__vector__": [0.1, 0.2, 0.3, ...],
      "entity_name": "Apple Inc.",
      "content": "Apple Inc.\nTechnology company",
      "source_id": "chunk_001"
    }
  },
  "matrix": [[0.1, 0.2, ...], ...],
  "index_to_id": ["ent-abc123", ...]
}
```

---

## Vector Databases

### Milvus

```python
rag = LightRAG(
    vector_storage="MilvusVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "host": "localhost",
        "port": 19530,
        "collection_name": "lightrag_vectors"
    }
)
```

```bash
# Environment variables
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_TOKEN=your_token  # For Zilliz Cloud
```

### Qdrant

```python
rag = LightRAG(
    vector_storage="QdrantVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "collection_name": "lightrag"
    }
)
```

```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key  # Optional
```

### FAISS

```python
rag = LightRAG(
    vector_storage="FaissVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "index_type": "IVF_FLAT",  # or HNSW
        "nlist": 100
    }
)
```

---

## Configuration Reference

### Complete Environment Variables

```bash
# Storage Selection
KV_STORAGE=PGKVStorage
VECTOR_STORAGE=PGVectorStorage
GRAPH_STORAGE=PGGraphStorage
DOC_STATUS_STORAGE=PGDocStatusStorage

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secret
POSTGRES_DATABASE=lightrag
POSTGRES_MAX_CONNECTIONS=100
POSTGRES_SSL_MODE=prefer

# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE=lightrag

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Redis
REDIS_URI=redis://localhost:6379

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Memgraph
MEMGRAPH_URI=bolt://localhost:7687
```

### Programmatic Configuration

```python
from lightrag import LightRAG

rag = LightRAG(
    # Working directory
    working_dir="./rag_storage",
    workspace="my_project",  # Multi-tenant namespace

    # Storage backends
    kv_storage="PGKVStorage",
    vector_storage="PGVectorStorage",
    graph_storage="PGGraphStorage",
    doc_status_storage="PGDocStatusStorage",

    # Vector DB options
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.2,
        # Backend-specific options...
    },

    # Processing
    chunk_token_size=1200,
    chunk_overlap_token_size=100,
)
```

---

## Multi-Tenant Data Isolation

All storage backends support multi-tenant isolation:

```python
# Workspace creates isolated namespace
rag = LightRAG(
    working_dir="./rag_storage",
    workspace="tenant_a:kb_prod",  # Composite namespace
)

# Or with explicit tenant context
from lightrag.tenant_rag_manager import TenantRAGManager

manager = TenantRAGManager(
    base_working_dir="./rag_storage",
    tenant_service=tenant_service,
    template_rag=template_rag,
)

# Get tenant-specific instance
rag = await manager.get_rag_instance(
    tenant_id="tenant_a",
    kb_id="kb_prod"
)
```

### Isolation Pattern

```
┌─────────────────────────────────────────────────────────────┐
│              Multi-Tenant Data Isolation                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PostgreSQL: WHERE workspace = 'tenant_a:kb_prod:default'  │
│                                                             │
│  MongoDB: { workspace: "tenant_a:kb_prod:default" }        │
│                                                             │
│  Redis: lightrag:tenant_a:kb_prod:default:{key}            │
│                                                             │
│  Neo4j: MATCH (n {workspace: $workspace})                  │
│                                                             │
│  File: ./rag_storage/tenant_a:kb_prod/                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Migration Between Backends

### Export/Import Pattern

```python
# Export from source
source_rag = LightRAG(
    kv_storage="JsonKVStorage",
    vector_storage="NanoVectorDBStorage",
    graph_storage="NetworkXStorage",
)

# Initialize source
await source_rag.initialize_storages()

# Get all data
docs = await source_rag.full_docs.get_all()
chunks = await source_rag.text_chunks.get_all()
# ... export other data

# Import to target
target_rag = LightRAG(
    kv_storage="PGKVStorage",
    vector_storage="PGVectorStorage",
    graph_storage="PGGraphStorage",
)

await target_rag.initialize_storages()
await target_rag.full_docs.upsert(docs)
await target_rag.text_chunks.upsert(chunks)
# ... import other data
await target_rag.finalize_storages()
```

---

## Best Practices

### Production Recommendations

1. **Use PostgreSQL Full Stack** for simplicity and reliability
2. **Enable connection pooling** for high concurrency
3. **Create indexes** on frequently queried columns
4. **Monitor storage growth** and plan capacity
5. **Regular backups** with point-in-time recovery
6. **Use SSL/TLS** for database connections

### Performance Tuning

```bash
# PostgreSQL tuning
POSTGRES_MAX_CONNECTIONS=200
POSTGRES_VECTOR_INDEX_TYPE=hnsw
POSTGRES_HNSW_M=32
POSTGRES_HNSW_EF=128

# LightRAG tuning
MAX_PARALLEL_INSERT=4
EMBEDDING_BATCH_NUM=20
MAX_ASYNC=8
```

---

**Version**: 1.4.9.1 | **License**: MIT
