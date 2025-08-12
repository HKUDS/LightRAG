# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Production Deployment](#production-deployment)
- [Key Components](#key-components)
- [xAI Usage Recommendations](#xai-usage-recommendations)
- [Development Commands](#development-commands)
- [Configuration & Environment](#configuration--environment)
- [Storage Architecture](#storage-architecture)
- [API Structure](#api-structure)
- [Key Development Patterns](#key-development-patterns)
- [Testing](#testing)
- [Important Notes](#important-notes)
- [Build & Test Commands](#build--test-commands)
- [Monitoring & Observability](#monitoring--observability)
- [Backup & Recovery](#backup--recovery)
- [Code Style Guidelines](#code-style-guidelines)
- [Python Best Practices](#python-best-practices)
- [Development Patterns & Best Practices](#development-patterns--best-practices)
- [Communication Style](#communication-style)
- [Code Style Consistency](#code-style-consistency)
- [Code Documentation and Comments](#code-documentation-and-comments)
- [Knowledge Sharing and Persistence](#knowledge-sharing-and-persistence)

## Project Overview

LightRAG is a Python-based retrieval-augmented generation (RAG) system that combines knowledge graphs and vector retrieval for enhanced document processing and querying. The project includes both a core library and a web server with API endpoints.

### Production Features
- **Enterprise Security**: Authentication, JWT tokens, rate limiting, audit logging
- **Production Deployment**: Docker Compose, Kubernetes, monitoring, backup systems
- **Multi-Storage Support**: PostgreSQL, Redis, MongoDB, Neo4j, Qdrant, Milvus
- **Performance Optimization**: Gunicorn, connection pooling, async processing
- **Security Hardening**: Non-root containers, read-only filesystems, capability dropping

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
- **Docker**: Containerization with docker-compose support and production hardening
- **Kubernetes**: Production-grade K8s deployments with Helm charts
- **Shell Scripts**: Automated deployment and maintenance scripts

## Production Deployment

### Quick Production Setup
```bash
# Copy production environment template
cp production.env .env

# Edit configuration (required)
vim .env  # Configure API keys, database settings

# Start production stack
docker compose -f docker-compose.production.yml up -d

# Check status
docker compose -f docker-compose.production.yml ps
```

### Key Production Files
- `production.env`: Production environment template
- `docker-compose.production.yml`: Production Docker stack
- `Dockerfile.production`: Security-hardened container image
- `PRODUCTION_DEPLOYMENT_GUIDE.md`: Comprehensive deployment guide
- `SECURITY_HARDENING.md`: Security configuration details
- `scripts/start-production.sh`: Production startup script

### Security Features
- **Authentication**: JWT-based with configurable expiration
- **Rate Limiting**: Configurable per-endpoint rate limits
- **Audit Logging**: Complete request/response audit trail
- **Container Security**: Non-root users, minimal privileges
- **Network Security**: Internal container networks, exposed ports control

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
- `examples/insert_custom_kg.py`: Custom knowledge graph insertion
- `examples/raganything_example.py`: Multi-modal document processing
- `examples/rerank_example.py`: Result reranking demonstration

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
# Development environment
docker compose up

# Enhanced PostgreSQL environment (RECOMMENDED)
docker compose -f docker-compose.enhanced.yml up -d

# Production environment (security-hardened)
docker compose -f docker-compose.production.yml up -d

# Build specific service
docker compose build lightrag

# Build enhanced PostgreSQL image
./scripts/build-postgresql.sh

# Run in background
docker compose up -d

# View logs
docker compose logs -f lightrag

# Monitor resources
docker compose ps
docker stats
```

### Enhanced PostgreSQL Development
```bash
# Start with all optional services
docker compose -f docker-compose.enhanced.yml \
  --profile with-redis \
  --profile with-neo4j \
  --profile enhanced-processing \
  --profile monitoring up -d

# Test vector operations
docker compose -f docker-compose.enhanced.yml exec postgres-enhanced \
  psql -U lightrag -d lightrag -c "SELECT '[1,2,3]'::vector(3) <-> '[4,5,6]'::vector(3);"

# Test graph operations
docker compose -f docker-compose.enhanced.yml exec postgres-enhanced \
  psql -U lightrag -d lightrag -c "SELECT ag_catalog.create_graph('test_graph');"

# Monitor performance
docker compose -f docker-compose.enhanced.yml exec postgres-enhanced \
  psql -U lightrag -d lightrag -c "SELECT * FROM pg_lightrag_health;"
```

### Kubernetes Deployment
```bash
cd k8s-deploy

# Install dependencies (databases)
./databases/01-prepare.sh
./databases/02-install-database.sh

# Deploy LightRAG
./install_lightrag.sh

# Check deployment status
kubectl get pods -n lightrag
kubectl logs -f deployment/lightrag -n lightrag

# Uninstall
./uninstall_lightrag.sh
./databases/03-uninstall-database.sh
```

## Configuration & Environment

### Required Setup
- Copy `env.example` to `.env` and configure LLM/embedding models
- API server loads `.env` from current working directory at startup
- Environment variables take precedence over `.env` file settings

### Key Environment Variables

#### Core Application
- `LLM_BINDING`: LLM provider (openai, ollama, azure_openai, xai, etc.)
- `LLM_MODEL`: Model name for text generation
- `EMBEDDING_BINDING`: Embedding provider
- `EMBEDDING_MODEL`: Embedding model name
- `WORKING_DIR`: Data storage directory (default: `./rag_storage`)
- `PORT`: API server port (default: 9621)
- `HOST`: Server bind address (default: 0.0.0.0)
- `WORKERS`: Gunicorn worker processes (default: 4)

#### Production Settings
- `NODE_ENV`: Environment mode (development/production)
- `DEBUG`: Debug mode (true/false)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `RATE_LIMIT_ENABLED`: Enable rate limiting (true/false)
- `AUTH_ENABLED`: Enable authentication (true/false)
- `JWT_SECRET_KEY`: JWT signing secret
- `JWT_EXPIRE_HOURS`: JWT token expiration (default: 24)

#### Database Configuration
- `POSTGRES_HOST`: PostgreSQL host
- `POSTGRES_PORT`: PostgreSQL port (default: 5432)
- `POSTGRES_DB`: Database name
- `POSTGRES_USER`: Database user
- `POSTGRES_PASSWORD`: Database password
- `REDIS_URL`: Redis connection URL
- `MONGO_URL`: MongoDB connection URL

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
2. **Vector Storage**: Embedding vectors (NanoVectorDBStorage, PGVectorStorage, MilvusVectorDBStorage, **PGVectorStorageEnhanced**)
3. **Graph Storage**: Entity relationships (NetworkXStorage, Neo4JStorage, PGGraphStorage, MemgraphStorage, **PGGraphStorageEnhanced**)
4. **Document Status Storage**: Processing status (JsonDocStatusStorage, PGDocStatusStorage, MongoDocStatusStorage)

Storage backends cannot be changed after documents are added to the system.

### Enhanced PostgreSQL Storage (NEW)

**PostgreSQL 16 + pgvector + Apache AGE** provides significant performance improvements:

- **PGVectorStorageEnhanced**: Native vector similarity search with HNSW indexing (50-80% faster)
- **PGGraphStorageEnhanced**: Apache AGE graph database with Cypher query support
- **Multiple Distance Metrics**: Cosine, L2, Inner Product with optimized indexes
- **Bulk Operations**: High-performance batch inserts using PostgreSQL COPY
- **Production Ready**: Security hardening, monitoring, backup & recovery

#### Quick Setup
```bash
# Build enhanced PostgreSQL image
./scripts/build-postgresql.sh

# Start enhanced stack
docker compose -f docker-compose.enhanced.yml up -d

# Or migrate from standard setup
./scripts/migrate-to-enhanced-postgresql.sh
```

See `POSTGRESQL_ENHANCEMENT_GUIDE.md` for comprehensive documentation.

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

## Build & Test Commands

### Testing and linting
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lightrag tests/

# Run specific test
pytest tests/test_graph_storage.py::test_networkx_storage -v

# Run linter
ruff check lightrag/ tests/

# Format code
ruff format lightrag/ tests/

# Type checking (if mypy installed)
mypy lightrag/
```

### Production Testing
```bash
# Test production environment
docker compose -f docker-compose.production.yml exec lightrag pytest

# Health check endpoints
curl http://localhost:9621/health
curl http://localhost:9621/api/health

# Load testing (if installed)
locust -f tests/load_test.py --host=http://localhost:9621
```

## Code Style Guidelines

- **Formatting**: Black-compatible formatting via `ruff format`
- **Imports**: Sort imports with `ruff` (stdlib, third-party, local)
- **Type hints**: Use native Python type hints (e.g., `list[str]` not `List[str]`)
- **Documentation**: Google-style docstrings for all modules, classes, functions
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Function length**: Keep functions short (< 30 lines) and single-purpose
- **PEP 8**: Follow PEP 8 style guide (enforced via `ruff`)

## Python Best Practices

- **File handling**: Prefer `pathlib.Path` over `os.path`
- **Debugging**: Use `logging` module instead of `print`
- **Error handling**: Use specific exceptions with context messages and proper logging
- **Data structures**: Use list/dict comprehensions for concise, readable code
- **Function arguments**: Avoid mutable default arguments
- **Data containers**: Leverage `dataclasses` to reduce boilerplate
- **Configuration**: Use environment variables (via `python-dotenv`) for configuration
- **AWS CLI**: Validate all commands before execution (must start with "aws")
- **Security**: Never store/log AWS credentials, set command timeouts

## Development Patterns & Best Practices

- **Favor simplicity**: Choose the simplest solution that meets requirements
- **DRY principle**: Avoid code duplication; reuse existing functionality
- **Configuration management**: Use environment variables for different environments
- **Focused changes**: Only implement explicitly requested or fully understood changes
- **Preserve patterns**: Follow existing code patterns when fixing bugs
- **File size**: Keep files under 300 lines; refactor when exceeding this limit
- **Test coverage**: Write comprehensive unit and integration tests with `pytest`; include fixtures
- **Test structure**: Use table-driven tests with parameterization for similar test cases
- **Mocking**: Use unittest.mock for external dependencies; don't test implementation details
- **Modular design**: Create reusable, modular components
- **Logging**: Implement appropriate logging levels (debug, info, error)
- **Error handling**: Implement robust error handling for production reliability
- **Security best practices**: Follow input validation and data protection practices
- **Performance**: Optimize critical code sections when necessary
- **Dependency management**: Add libraries only when essential

## Monitoring & Observability

### Production Monitoring
```bash
# View application logs
docker compose -f docker-compose.production.yml logs -f lightrag

# Monitor system resources
docker stats

# Check database connections
docker compose -f docker-compose.production.yml exec postgres psql -U lightrag -d lightrag -c '\l'

# Redis monitoring
docker compose -f docker-compose.production.yml exec redis redis-cli info
```

### Health Checks
- `/health`: Basic application health
- `/api/health`: Detailed system status with dependencies
- Audit logs: `logs/audit.log`
- Application logs: Docker container logs

### Performance Metrics
- Request/response times logged in audit trail
- Database connection pool metrics
- Memory and CPU usage via Docker stats
- Rate limiting metrics in application logs

## Backup & Recovery

### Database Backup
```bash
# Automated backup (runs via cron)
./backup/backup-script.sh

# Manual database backup
docker compose -f docker-compose.production.yml exec postgres pg_dump -U lightrag lightrag > backup_$(date +%Y%m%d).sql

# Restore from backup
docker compose -f docker-compose.production.yml exec -i postgres psql -U lightrag -d lightrag < backup_20250801.sql
```

### Data Directory Backup
```bash
# Backup RAG storage data
tar -czf rag_storage_backup_$(date +%Y%m%d).tar.gz rag_storage/

# Restore RAG storage
tar -xzf rag_storage_backup_20250801.tar.gz
```

## Troubleshooting

### Common Issues

#### Authentication Problems
```bash
# Check JWT token validity
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:9621/health

# Reset authentication (if enabled)
docker compose -f docker-compose.production.yml restart lightrag
```

#### Database Connection Issues
```bash
# Check PostgreSQL connection
docker compose -f docker-compose.production.yml exec postgres psql -U lightrag -d lightrag -c "SELECT version();"

# Reset database connection pool
docker compose -f docker-compose.production.yml restart lightrag
```

#### Performance Issues
- **High memory usage**: Reduce `MAX_ASYNC` and `WORKERS` settings
- **Slow queries**: Check database indexes, consider upgrading storage backend
- **Timeout errors**: Increase `TIMEOUT` setting, reduce concurrency

#### Storage Backend Issues
- **Dimension mismatches**: Clean working directory when changing embedding models
- **Storage corruption**: Restore from backup, reinitialize storage backends
- **Migration failures**: Check logs, ensure proper database permissions

### Log Analysis
```bash
# Application logs
docker compose -f docker-compose.production.yml logs lightrag | grep ERROR

# Audit logs
tail -f logs/audit.log

# Database logs
docker compose -f docker-compose.production.yml logs postgres

# System resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## Communication Style

- NEVER suggest or offer staging files with git add commands
- When asking questions, always provide multiple numbered options when appropriate:

  - Format as a numbered list: `1. Option one, 2. Option two, 3. Option three`
  - Example: `1. Yes, continue with the changes, 2. Modify the approach, 3. Stop and cancel the operation`

- When analyzing code for improvement:

  - Present multiple implementation variants as numbered options
  - For each variant, provide at least 3 bullet points explaining the changes, benefits, and tradeoffs
  - Format as: "1. [short exmplanation of variant or shorly Variant]" followed by explanation points

- When implementing code changes:

  - If the change wasn't preceded by an explanation or specific instructions
  - Include within the diff a bulleted list explaining what was changed and why
  - Explicitly note when a solution is opinionated and explain the reasoning

- When completing a task, ask if I want to:
  1. Run task:commit (need to manually stage files first)
  2. Neither (stop here)

## Code Style Consistency

- ALWAYS respect how things are written in the existing project
- DO NOT invent your own approaches or innovations
- STRICTLY follow the existing style of tests, resolvers, functions, and arguments
- Before creating a new file, ALWAYS examine a similar file and follow its style exactly
- If code doesn't include comments, DO NOT add comments
- Use seeded data in tests instead of creating new objects when seeded data exists
- Follow the exact format of error handling, variable naming, and code organization used in similar files
- Never deviate from the established patterns in the codebase

## Code Documentation and Comments

When working with code that contains comments or documentation:

1. Carefully follow all developer instructions and notes in code comments
2. Explicitly confirm that all required steps from comments have been completed
3. Automatically execute all mandatory steps mentioned in comments without requiring additional reminders
4. Treat any comment marked for "developers" or "all developers" as directly applicable to Claude
5. Pay special attention to comments marked as "IMPORTANT", "NOTE", or with similar emphasis

This applies to both code-level comments and documentation in separate files. Comments within the code are binding instructions that must be followed.

## Knowledge Sharing and Persistence

- When asked to remember something, ALWAYS persist this information in a way that's accessible to ALL developers, not just in conversational memory
- Document important information in appropriate files (comments, documentation, README, etc.) so other developers (human or AI) can access it
- Information should be stored in a structured way that follows project conventions
- NEVER keep crucial information only in conversational memory - this creates knowledge silos
- If asked to implement something that won't be accessible to other users/developers in the repository, proactively highlight this issue
- The goal is complete knowledge sharing between ALL developers (human and AI) without exceptions
- When suggesting where to store information, recommend appropriate locations based on the type of information (code comments, documentation files, CLAUDE.md, etc.)
