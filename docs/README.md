# LightRAG Documentation

This directory contains comprehensive documentation for LightRAG, including core algorithms, deployment guides, integration documentation, and advanced features.

> **📚 [COMPLETE DOCUMENTATION INDEX](DOCUMENTATION_INDEX.md)** - Navigate all documentation with cross-references and user journeys


## 📁 Documentation Structure

### 🎯 Production & Operations
- **[production/PRODUCTION_DEPLOYMENT_COMPLETE.md](production/PRODUCTION_DEPLOYMENT_COMPLETE.md)** - **AUTHORITATIVE** production guide
- [security/](security/) - Security configuration and hardening guides
- [monitoring/](production/PRODUCTION_DEPLOYMENT_COMPLETE.md#monitoring--observability) - Observability and alerting

### 🏗️ Architecture & Core
- [architecture/SYSTEM_ARCHITECTURE_AND_DATA_FLOW.md](architecture/SYSTEM_ARCHITECTURE_AND_DATA_FLOW.md) - Complete system overview
- [architecture/REPOSITORY_STRUCTURE.md](architecture/REPOSITORY_STRUCTURE.md) - Codebase organization
- [architecture/Algorithm.md](architecture/Algorithm.md) - LightRAG core algorithms and flowcharts
- [architecture/LightRAG_concurrent_explain.md](architecture/LightRAG_concurrent_explain.md) - Concurrency and parallel processing

### 🔧 Integration & Setup
- [integration_guides/](integration_guides/) - External service integrations (MCP, xAI, Docling)
- [integration_guides/rerank_integration.md](integration_guides/rerank_integration.md) - Reranking model integration

### ⚠️ Deprecated Documents
The following documents have been consolidated into authoritative guides:
- ~~`DockerDeployment.md`~~ → [Complete Production Guide](production/PRODUCTION_DEPLOYMENT_COMPLETE.md)
- ~~`production/PRODUCTION_DEPLOYMENT_GUIDE.md`~~ → [Complete Production Guide](production/PRODUCTION_DEPLOYMENT_COMPLETE.md)
- ~~`production/PRODUCTION_IMPLEMENTATION_GUIDE.md`~~ → [Complete Production Guide](production/PRODUCTION_DEPLOYMENT_COMPLETE.md)
- ~~`production/ProductionDeploymentGuide.md`~~ → [Complete Production Guide](production/PRODUCTION_DEPLOYMENT_COMPLETE.md)

### Integration Guides (`integration_guides/`)
Complete guides for integrating LightRAG with external services and protocols:

#### Model Context Protocol (MCP) - NEW! 🚀
- **[MCP Implementation Summary](integration_guides/MCP_IMPLEMENTATION_SUMMARY.md)** - **START HERE** - Complete overview of the MCP implementation
- **[MCP Integration Plan](integration_guides/MCP_INTEGRATION_PLAN.md)** - Strategic implementation roadmap
- **[MCP Implementation Guide](integration_guides/MCP_IMPLEMENTATION_GUIDE.md)** - Step-by-step development guide
- **[MCP Tools Specification](integration_guides/MCP_TOOLS_SPECIFICATION.md)** - Technical tool specifications

#### Other Integrations
- **xAI Grok Models** - Integration and troubleshooting guides
- **Enhanced Docling** - Advanced document processing configuration

## 🚀 Quick Navigation by Role

### 🆕 For New Users
1. **🎯 Start Here**: [Algorithm Overview](architecture/Algorithm.md) - Understand LightRAG's core approach
2. **🤖 Claude Integration**: [MCP Implementation Summary](integration_guides/MCP_IMPLEMENTATION_SUMMARY.md) - Use LightRAG with Claude CLI
3. **🚀 Quick Deploy**: [Production Guide - Quick Start](production/PRODUCTION_DEPLOYMENT_COMPLETE.md#quick-start-deployments) - Get LightRAG running in 5 minutes

### 👩‍💻 For Developers
- **🏗️ System Architecture**: [Complete System Overview](architecture/SYSTEM_ARCHITECTURE_AND_DATA_FLOW.md) - End-to-end system understanding
- **⚡ Concurrent Processing**: [Parallel Processing Details](architecture/LightRAG_concurrent_explain.md) - Performance optimization
- **🤖 MCP Development**: [MCP Implementation Guide](integration_guides/MCP_IMPLEMENTATION_GUIDE.md) - Build MCP integrations
- **🔧 Integration Guides**: [External Services](integration_guides/) - xAI, Docling, PostgreSQL

### 🚀 For DevOps & SRE
- **🎯 PRODUCTION DEPLOYMENT**: **[Complete Production Guide](production/PRODUCTION_DEPLOYMENT_COMPLETE.md)** - Single authoritative source
- **🔐 Security Setup**: [Security Hardening Guide](security/SECURITY_HARDENING.md) - Container and network security
- **📊 Monitoring**: [Observability Setup](production/PRODUCTION_DEPLOYMENT_COMPLETE.md#monitoring--observability) - Metrics, logs, alerting
- **🔄 Operations**: [Backup & Recovery](production/PRODUCTION_DEPLOYMENT_COMPLETE.md#backup--disaster-recovery) - Operational procedures

### 🔧 For System Integrators
- **🤖 MCP Production**: [MCP Deployment](integration_guides/MCP_IMPLEMENTATION_SUMMARY.md#deployment-options) - Enterprise MCP setup
- **🎛️ Model Optimization**: [Rerank Integration](integration_guides/rerank_integration.md) - Performance tuning
- **🗄️ Storage Backends**: [Storage Architecture](architecture/SYSTEM_ARCHITECTURE_AND_DATA_FLOW.md#storage-backend-coordination) - Database setup

## Featured: Model Context Protocol (MCP) Integration

**NEW in 2025**: LightRAG now supports the Model Context Protocol, enabling direct integration with Claude CLI and other MCP-compatible clients.

### What is MCP Integration?
The MCP integration provides:
- **11 MCP Tools** for RAG queries, document management, and graph exploration
- **3 MCP Resources** for system monitoring and configuration
- **Streaming Support** for real-time responses
- **Complete Claude CLI Integration** - Use natural language to interact with your knowledge base

### Quick Start with MCP
```bash
# Install and run MCP server
pip install mcp httpx pydantic aiofiles
python -m lightrag_mcp

# Setup Claude CLI
claude config mcp add lightrag-mcp python -m lightrag_mcp

# Start querying your knowledge base
claude mcp lightrag_query "What are the main themes in my documents?"
```

### Key MCP Features
- **📝 Document Operations**: Upload, process, and manage documents
- **🔍 Advanced Queries**: 6 different RAG query modes (hybrid, local, global, etc.)
- **🕸️ Knowledge Graph**: Explore entities and relationships
- **📊 System Monitoring**: Health checks, statistics, and cache management
- **⚡ High Performance**: Caching, streaming, and async operations

**➡️ [Get Started with MCP](integration_guides/MCP_IMPLEMENTATION_SUMMARY.md)**

## Core LightRAG Features

### RAG Capabilities
- **Hybrid Retrieval**: Combines knowledge graph and vector search
- **6 Query Modes**: naive, local, global, hybrid, mix, bypass
- **Multi-modal Support**: Text, PDFs, Office docs, images, tables
- **Citation Tracking**: Source attribution with file paths

### Knowledge Graph
- **Entity Extraction**: Automatic entity and relationship identification
- **Graph Storage**: Multiple backends (Neo4j, NetworkX, PostgreSQL, etc.)
- **Graph Visualization**: HTML and Neo4j visualization support
- **Schema Evolution**: Dynamic graph schema management

### Document Processing
- **Multi-format Support**: PDF, DOCX, PPTX, XLSX, TXT, MD, HTML
- **Chunking Strategies**: Configurable chunk sizes and overlap
- **Processing Pipeline**: Atomic operations with status tracking
- **Error Recovery**: LLM caching for quick reprocessing

### Storage & Scalability
- **4-Layer Storage**: KV, Vector, Graph, Document Status storage
- **12+ Storage Backends**: PostgreSQL, Redis, MongoDB, Milvus, etc.
- **Async Operations**: High-performance concurrent processing
- **Caching**: Multi-level caching for optimal performance

## Architecture Overview

```
Documents → Chunking → Entity Extraction → Knowledge Graph + Vector Store → Query Processing → Response Generation
     ↓
MCP Integration → Claude CLI → Natural Language Queries
```

### Storage System (4-Layer)
1. **KV Storage**: Document chunks, LLM cache (4 implementations)
2. **Vector Storage**: Embedding vectors (6 implementations)
3. **Graph Storage**: Entity relationships (5 implementations)
4. **Document Status Storage**: Processing status (4 implementations)

### API Architecture
- **REST API**: Complete FastAPI-based server
- **Web UI**: React/TypeScript frontend with graph visualization
- **MCP Server**: Model Context Protocol integration
- **Ollama API**: Compatible endpoint for Ollama clients

## Getting Started

### Prerequisites
- Python 3.9+
- LLM access (OpenAI, Ollama, xAI Grok, Azure, etc.)
- Embedding model access
- Optional: Database for production storage

### Installation
```bash
# Basic installation
pip install lightrag-hku

# With API server
pip install lightrag-hku[api]

# With MCP support
pip install mcp httpx pydantic aiofiles
```

### Quick Start
```bash
# 1. Copy configuration
cp env.example .env

# 2. Configure your LLM and embedding models in .env

# 3. Start LightRAG API server
lightrag-server

# 4. Start MCP server (optional)
python -m lightrag_mcp

# 5. Access via Web UI
open http://localhost:9621
```

## Configuration

### Environment Variables
LightRAG supports extensive configuration through environment variables:

- **LLM Configuration**: Model bindings, API keys, endpoints
- **Storage Configuration**: Database connections, storage backends
- **Processing Configuration**: Chunk sizes, concurrency limits
- **API Configuration**: Ports, authentication, CORS settings
- **MCP Configuration**: 25+ MCP-specific settings

See `env.example` in the project root for complete configuration options.

### Key Configuration Files
- `.env` - Local environment variables (not in git)
- `env.example` - Template with all options documented
- `CLAUDE.md` - Claude Code assistant instructions
- `pyproject.toml` - Python project metadata

## Usage Examples

### Basic RAG Operations
```python
from lightrag import LightRAG

# Initialize
rag = LightRAG(working_dir="./rag_storage")
await rag.initialize_storages()

# Insert documents
await rag.ainsert("Your document content here")

# Query
result = await rag.aquery("What is this document about?", mode="hybrid")
```

### MCP Integration
```bash
# Query through Claude CLI
claude mcp lightrag_query "Analyze the key themes in my research papers" --mode hybrid

# Upload documents
claude mcp lightrag_insert_file "/path/to/research_paper.pdf"

# Explore knowledge graph
claude mcp lightrag_get_graph --max-nodes 100 --format json
```

### API Usage
```bash
# REST API calls
curl -X POST "http://localhost:9621/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main topics?", "mode": "hybrid"}'
```

## Advanced Features

### Multi-modal Processing
- **RAG-Anything Integration**: PDF, Office docs, images, tables
- **Custom Knowledge Graph Insertion**: Direct graph manipulation
- **Citation Functionality**: Source tracking and attribution

### Performance & Scalability
- **Async Architecture**: High-performance concurrent operations
- **Multi-level Caching**: LLM, embedding, and query caching
- **Connection Pooling**: Optimized database connections
- **Horizontal Scaling**: Multi-instance deployments

### Production Features
- **Health Monitoring**: Comprehensive health checks
- **Metrics Collection**: Usage statistics and performance monitoring
- **Error Handling**: Graceful degradation and recovery
- **Security**: Authentication, authorization, input validation

## Contributing

### Documentation Guidelines
When adding documentation:
1. Place core LightRAG docs in the root `docs/` directory
2. Place integration guides in `integration_guides/`
3. Update this README when adding new sections
4. Follow existing markdown formatting conventions
5. Include code examples where applicable
6. Provide troubleshooting sections
7. Link related documentation

### Development
- Use clear, descriptive titles
- Include working code examples
- Provide troubleshooting information
- Link to related documentation
- Keep examples up to date with current APIs

## Troubleshooting

### Common Issues
1. **LLM Connection Issues**: Check API keys and endpoints
2. **Storage Errors**: Verify database connections and permissions
3. **Memory Issues**: Adjust chunk sizes and concurrency limits
4. **MCP Integration Issues**: Verify MCP dependencies and configuration

### Getting Help
- Check the specific integration guides for detailed troubleshooting
- Review the [MCP Implementation Summary](integration_guides/MCP_IMPLEMENTATION_SUMMARY.md) for MCP-specific issues
- Enable debug logging for detailed error information
- Check system health with monitoring endpoints

## What's New

### January 2025 - MCP Integration
- **Complete MCP Server**: 11 tools, 3 resources, streaming support
- **Claude CLI Integration**: Natural language interface to LightRAG
- **Production Ready**: Comprehensive error handling, caching, monitoring
- **Extensive Documentation**: Implementation guides, examples, troubleshooting

### Recent Updates
- Enhanced Docling configuration with 19 environment variables
- xAI Grok model integration with timeout handling
- Repository structure cleanup and organization
- Improved documentation navigation and organization

---

**Documentation Version**: 2.0
**Last Updated**: 2025-01-29
**LightRAG Version**: Latest
**MCP Integration Version**: 1.0.0

For the most up-to-date information, check the individual documentation files and the project's GitHub repository.
