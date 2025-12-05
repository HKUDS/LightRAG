# LightRAG Documentation

**Version 1.4.9.2** | Graph-Enhanced Retrieval-Augmented Generation

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     ██╗     ██╗ ██████╗ ██╗  ██╗████████╗██████╗  █████╗  ██████╗       ║
║     ██║     ██║██╔════╝ ██║  ██║╚══██╔══╝██╔══██╗██╔══██╗██╔════╝       ║
║     ██║     ██║██║  ███╗███████║   ██║   ██████╔╝███████║██║  ███╗      ║
║     ██║     ██║██║   ██║██╔══██║   ██║   ██╔══██╗██╔══██║██║   ██║      ║
║     ███████╗██║╚██████╔╝██║  ██║   ██║   ██║  ██║██║  ██║╚██████╔╝      ║
║     ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝       ║
║                                                                           ║
║         Simple and Fast Graph-Enhanced RAG System                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

> **New Feature**: Multi-tenancy support with RBAC, tenant isolation, and knowledge base management.

---

## Documentation Overview

| Document | Description |
|----------|-------------|
| [Quick Start](0001-quick-start.md) | Get up and running in 5 minutes |
| [Architecture Overview](0002-architecture-overview.md) | System design, data flow, and core concepts |
| [API Reference](0003-api-reference.md) | Complete REST API documentation |
| [Storage Backends](0004-storage-backends.md) | Configure KV, vector, and graph storage |
| [LLM Integration](0005-llm-integration.md) | LLM providers and embedding models |
| [Deployment Guide](0006-deployment-guide.md) | Docker, Kubernetes, and production setup |
| [Configuration Reference](0007-configuration-reference.md) | All environment variables and options |
| [Multi-Tenancy](0008-multi-tenancy.md) | Tenant isolation and RBAC |

---

## Quick Links

### Getting Started

```bash
# Install
pip install lightrag-hku

# Start server
export OPENAI_API_KEY=sk-xxx
python -m lightrag.api.lightrag_server
```

### Python Usage

```python
from lightrag import LightRAG, QueryParam

rag = LightRAG(working_dir="./rag_storage")
await rag.ainsert("Your document text...")
result = await rag.aquery("Your question?", param=QueryParam(mode="hybrid"))
```

### REST API

```bash
# Insert document
curl -X POST http://localhost:9621/documents/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Document content..."}'

# Query
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question?", "mode": "hybrid"}'
```

### Docker

```bash
docker run -p 9621:9621 -e OPENAI_API_KEY=sk-xxx ghcr.io/hkuds/lightrag:latest
```

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LightRAG                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Documents ──▶ Chunking ──▶ Entity Extraction ──▶ Knowledge Graph          │
│                    │                                      │                  │
│                    ▼                                      ▼                  │
│              Embeddings ──────────────────────▶ Hybrid Retrieval            │
│                                                       │                      │
│                                                       ▼                      │
│                              Query ──▶ LLM Generation ──▶ Response          │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Storage Backends:                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │ KV Storage   │  │ Vector Store │  │ Graph Store  │                       │
│  │ JSON/Redis/  │  │ NanoVectorDB │  │ NetworkX/    │                       │
│  │ PostgreSQL/  │  │ pgvector/    │  │ Neo4j/AGE/   │                       │
│  │ MongoDB      │  │ Milvus/FAISS │  │ Memgraph     │                       │
│  └──────────────┘  └──────────────┘  └──────────────┘                       │
│                                                                              │
│  LLM Providers:                                                              │
│  OpenAI │ Anthropic │ Ollama │ Azure │ Bedrock │ HuggingFace │ ...         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Query Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `naive` | Basic vector similarity search | Simple lookups |
| `local` | Entity-focused retrieval | Specific facts |
| `global` | High-level community summaries | Broad questions |
| `hybrid` | Combined local + global | Balanced queries |
| `mix` | Full KG + vector integration | Complex reasoning |

---

## Deployment Options

| Option | Use Case | Guide |
|--------|----------|-------|
| **Local** | Development | `pip install lightrag-hku` |
| **Docker** | Staging/Production | [Docker Guide](0006-deployment-guide.md#2-docker-deployment) |
| **Kubernetes** | Production/Scale | [K8s Guide](0006-deployment-guide.md#3-kubernetes-deployment-helm) |

### Storage Topology

| Environment | KV | Vector | Graph |
|-------------|-----|--------|-------|
| Development | JSON | NanoVectorDB | NetworkX |
| Production | PostgreSQL | pgvector | Neo4j |
| High-Scale | Redis | Milvus | Neo4j |

---

## Configuration Quick Reference

### Essential Environment Variables

```bash
# LLM
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-xxx

# Embedding
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIM=1536

# Storage
LIGHTRAG_KV_STORAGE=JsonKVStorage
LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage
LIGHTRAG_GRAPH_STORAGE=NetworkXStorage

# Server
PORT=9621
```

See [Configuration Reference](0007-configuration-reference.md) for all options.

---

## Feature Highlights

### Knowledge Graph Integration

```
┌────────────────────────────────────────────────────┐
│              KNOWLEDGE GRAPH                        │
│                                                     │
│    [Person: Einstein] ─────developed──────▶        │
│           │                               │        │
│     born_in                        [Theory:        │
│           │                        Relativity]     │
│           ▼                               │        │
│    [Location: Germany]           describes_│       │
│                                           ▼        │
│                                  [Concept:         │
│                                   Spacetime]       │
└────────────────────────────────────────────────────┘
```

### Multi-Tenant Isolation

```
Tenant A          Tenant B          Tenant C
    │                 │                 │
    ▼                 ▼                 ▼
┌────────┐       ┌────────┐       ┌────────┐
│ KB-1   │       │ KB-1   │       │ KB-1   │
│ KB-2   │       │ KB-2   │       └────────┘
│ KB-3   │       └────────┘
└────────┘
```

---

## Resources

- **GitHub**: https://github.com/HKUDS/LightRAG
- **PyPI**: https://pypi.org/project/lightrag-hku/
- **Issues**: https://github.com/HKUDS/LightRAG/issues

---

## Document Index

1. **[Quick Start Guide](0001-quick-start.md)**
   - Installation options
   - Python SDK basics
   - REST API basics
   - Common patterns

2. **[Architecture Overview](0002-architecture-overview.md)**
   - System design diagrams
   - Core concepts (entities, relations, chunks)
   - Data flow pipeline
   - Query execution flow

3. **[API Reference](0003-api-reference.md)**
   - Document endpoints
   - Query endpoints
   - Graph endpoints
   - Admin endpoints

4. **[Storage Backends](0004-storage-backends.md)**
   - KV storage options
   - Vector storage options
   - Graph storage options
   - Configuration tables

5. **[LLM Integration](0005-llm-integration.md)**
   - Provider configurations
   - Embedding models
   - Reranking options
   - Custom implementations

6. **[Deployment Guide](0006-deployment-guide.md)**
   - Local development
   - Docker deployment
   - Kubernetes/Helm
   - Production best practices

7. **[Configuration Reference](0007-configuration-reference.md)**
   - Environment variables
   - CLI arguments
   - QueryParam options
   - Complete .env example

8. **[Multi-Tenancy Guide](0008-multi-tenancy.md)**
   - Tenant isolation
   - RBAC roles/permissions
   - TenantRAGManager
   - API endpoints

---

*Built with ❤️ by [HKUDS](https://github.com/HKUDS)*

*Multi-tenant feature contributed by [Raphaël MANSUY](https://www.elitizon.com/)*
