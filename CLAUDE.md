# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LightRAG is a Python-based retrieval-augmented generation (RAG) system that combines knowledge graphs and vector retrieval for enhanced document processing and querying. The project includes both a core library and a web server with API endpoints.

## Key Components

### Core Architecture
- **LightRAG Core (`lightrag/`)**: Main library with knowledge graph processing, LLM integrations, and storage backends
- **API Server (`lightrag/api/`)**: FastAPI-based web server with REST API and Ollama-compatible interface
- **MCP Server (`lightrag_mcp/`)**: Model Context Protocol server for Claude CLI integration with 11 tools and 3 resources
- **Web UI (`lightrag_webui/`)**: React/TypeScript frontend for document management and graph visualization
- **Storage Backends (`lightrag/kg/`)**: Multiple implementations for KV, vector, graph, and document status storage

### Multi-Language Components
- **Python**: Core library, API server, examples, and tests
- **TypeScript/React**: Web UI with Vite build system and Bun package manager
- **Docker**: Containerization with docker-compose support

## xAI Usage Recommendations

### Production Configuration
```bash
# Optimal settings for xAI Grok models
LLM_BINDING=xai
LLM_MODEL=grok-3-mini
MAX_ASYNC=2  # Reduced concurrency prevents timeout issues
TIMEOUT=240  # 4 minutes for complex operations

# Use consistent embedding model
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
```

### Available Demo Scripts
- `lightrag_xai_demo_timeout_fix.py`: **Recommended** - includes retry logic and timeout handling
- `lightrag_xai_demo_robust.py`: Standard demo with dimension conflict prevention
- `lightrag_xai_demo.py`: Basic demo
- `test_xai_basic.py`: Simple connection test
- `diagnose_embedding_issue.py`: Troubleshooting tool

### Common Issues & Solutions
1. **Timeout errors**: Use `lightrag_xai_demo_timeout_fix.py` with built-in retry logic
2. **Dimension conflicts**: Clean working directory between embedding model changes
3. **Stream parameter conflicts**: Fixed in current implementation (uses `**kwargs`)
4. **High concurrency issues**: Set `MAX_ASYNC=2` for stability

## Development Commands

### Python Core & API Server
```bash
# Install in development mode
pip install -e .

# Install with API dependencies
pip install -e ".[api]"

# Start API server (development)
lightrag-server

# Start API server (production with Gunicorn)
lightrag-gunicorn --workers 4

# Run tests
python -m pytest tests/

# Run specific examples
cd LightRAG  # Must be in project root
python examples/lightrag_openai_demo.py
```

### MCP Server (Model Context Protocol)
```bash
# Install MCP dependencies
pip install mcp httpx pydantic aiofiles typing-extensions

# Start MCP server
python -m lightrag_mcp

# Run MCP functionality tests
python lightrag_mcp/examples/test_basic_functionality.py

# Run MCP usage demonstration
python lightrag_mcp/examples/usage_example.py

# Configure environment for MCP
# See env.example for complete MCP configuration options
LIGHTRAG_API_URL=http://localhost:9621
MCP_ENABLE_STREAMING=true
MCP_ENABLE_DOCUMENT_UPLOAD=true
MCP_CACHE_ENABLED=true
```

### Claude CLI Integration
```bash
# Setup Claude CLI with MCP server
claude config mcp add lightrag-mcp python -m lightrag_mcp

# Query through Claude CLI
claude mcp lightrag_query "What are the main themes in my documents?" --mode hybrid

# Document operations
claude mcp lightrag_insert_file "/path/to/document.pdf"
claude mcp lightrag_list_documents --limit 10

# Knowledge graph exploration
claude mcp lightrag_get_graph --max-nodes 50 --format json
claude mcp lightrag_search_entities "artificial intelligence"

# System monitoring
claude mcp lightrag_health_check
claude mcp resource "lightrag://system/config"
```

### Web UI (TypeScript/React)
```bash
cd lightrag_webui

# Development with Bun (recommended)
bun run dev

# Development with Node.js
npm run dev-no-bun

# Build for production
bun run build
# or
npm run build-no-bun

# Lint TypeScript/React code
bun run lint
# or
eslint .

# Preview production build
bun run preview
```

### Docker Development
```bash
# Build and run with docker-compose
docker compose up

# Build specific service
docker compose build lightrag

# Run in background
docker compose up -d
```

## Configuration & Environment

### Required Setup
- Copy `env.example` to `.env` and configure LLM/embedding models
- API server loads `.env` from current working directory at startup
- Environment variables take precedence over `.env` file settings

### Key Environment Variables
- `LLM_BINDING`: LLM provider (openai, ollama, azure_openai, xai, etc.)
- `LLM_MODEL`: Model name for text generation
- `EMBEDDING_BINDING`: Embedding provider
- `EMBEDDING_MODEL`: Embedding model name
- `WORKING_DIR`: Data storage directory (default: `./rag_storage`)
- `PORT`: API server port (default: 9621)

### xAI Integration
- `XAI_API_KEY`: xAI API key for Grok models
- `XAI_API_BASE`: xAI API base URL (default: https://api.x.ai/v1)
- Supported models: grok-3-mini, grok-2-1212, grok-2-vision-1212
- **Status**: ✅ Fully implemented and tested
- **Key Fixes Applied** (2025-01-28):
  - Fixed Unicode decode error (removed unnecessary `safe_unicode_decode` calls)
  - Fixed stream parameter conflict (moved to `**kwargs` handling)
  - Added timeout-resistant demo with retry logic
  - **Important**: Use `MAX_ASYNC=2` to prevent Ollama embedding timeouts

## Storage Architecture

LightRAG uses 4 storage types with multiple backend implementations:

1. **KV Storage**: Document chunks, LLM cache (JsonKVStorage, PGKVStorage, RedisKVStorage, MongoKVStorage)
2. **Vector Storage**: Embedding vectors (NanoVectorDBStorage, PGVectorStorage, MilvusVectorDBStorage, etc.)
3. **Graph Storage**: Entity relationships (NetworkXStorage, Neo4JStorage, PGGraphStorage, MemgraphStorage)
4. **Document Status Storage**: Processing status (JsonDocStatusStorage, PGDocStatusStorage, MongoDocStatusStorage)

Storage backends cannot be changed after documents are added to the system.

## API Structure

### REST API Endpoints
- `/query` - RAG queries with different modes (local, global, hybrid, mix, naive)
- `/documents/*` - Document upload, text insertion, batch processing
- `/api/chat` - Ollama-compatible chat interface
- `/health` - Server health check

### Query Modes
- **local**: Context-dependent information retrieval
- **global**: Global knowledge graph queries
- **hybrid**: Combines local and global methods
- **mix**: Integrates knowledge graph and vector retrieval
- **naive**: Basic vector search without graph enhancement

## Key Development Patterns

### Async Architecture
- Core LightRAG operations are async/await based
- Must call `await rag.initialize_storages()` and `await initialize_pipeline_status()` after creating LightRAG instance
- Always use `await rag.finalize_storages()` in cleanup

### Error Handling
- Document processing is atomic - files marked as failed if any step fails
- LLM caching enables quick recovery from errors during reprocessing
- Pipeline status tracking prevents partial document states

### Multi-modal Support
- Integrates with RAG-Anything for PDF, Office docs, images, tables
- Supports custom knowledge graph insertion
- Citation functionality with file path tracking

## Testing

Limited test coverage with basic functionality tests in `tests/`:
- `test_graph_storage.py` - Graph storage backend tests
- `test_lightrag_ollama_chat.py` - Ollama chat integration tests

Run examples to validate functionality:
```bash
# Requires OPENAI_API_KEY environment variable
python examples/lightrag_openai_demo.py
```

## Important Notes

### Initialization Requirements
Always follow this pattern when using LightRAG programmatically:
```python
rag = LightRAG(...)
await rag.initialize_storages()  # Required!
await initialize_pipeline_status()  # Required!
# ... use rag ...
await rag.finalize_storages()  # Cleanup
```

### Model Requirements
- LLM needs at least 32KB context length (64KB recommended)
- 32B+ parameter models recommended for entity extraction
- Embedding model must be consistent across indexing and querying

### Working Directory
- Examples must be run from project root directory (`cd LightRAG`)
- API server loads `.env` from current working directory
- Storage paths are relative to working directory unless absolute paths specified
