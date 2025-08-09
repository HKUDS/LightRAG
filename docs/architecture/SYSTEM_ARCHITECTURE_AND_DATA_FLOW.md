# LightRAG System Architecture and Data Flow Documentation

**Related Documents**: [Complete Production Guide](../production/PRODUCTION_DEPLOYMENT_COMPLETE.md) | [Repository Structure](REPOSITORY_STRUCTURE.md) | [MCP Integration](../integration_guides/MCP_IMPLEMENTATION_SUMMARY.md) | [Documentation Index](../DOCUMENTATION_INDEX.md)

## Executive Summary

This document provides a comprehensive analysis of LightRAG's system interactions and data flow patterns across all components. LightRAG is a multi-component retrieval-augmented generation (RAG) system that combines knowledge graphs and vector retrieval for enhanced document processing and querying. The architecture spans Python core libraries, FastAPI web services, React/TypeScript frontends, MCP protocol servers, and containerized production deployments.

> **💡 For Production Deployment**: See the [Complete Production Deployment Guide](../production/PRODUCTION_DEPLOYMENT_COMPLETE.md) for detailed setup instructions.
> **💡 For Development Setup**: See the [Repository Structure Guide](REPOSITORY_STRUCTURE.md) for codebase organization.

## System Overview

LightRAG consists of four primary architectural layers:
1. **Core Processing Layer**: Python-based RAG engine with storage backends
2. **API Service Layer**: FastAPI server with authentication and routing
3. **Client Interface Layer**: Web UI, MCP server, and CLI integrations
4. **Infrastructure Layer**: Containerized deployment with monitoring and databases

## 1. End-to-End Data Flow

### 1.1 Document Ingestion Pipeline

The document ingestion process follows a comprehensive multi-stage pipeline:

```
Input Document → Document Processing → Knowledge Extraction → Storage Distribution → Status Tracking
```

#### Document Processing Workflow

**Stage 1: Document Input**
- **Entry Points**:
  - Web UI file upload (`/lightrag_webui/src/features/DocumentManager.tsx`)
  - API endpoint (`POST /documents/upload`)
  - MCP server tools (`lightrag_insert_file`)
  - Direct Python API (`rag.ainsert(content)`)

**Stage 2: Content Processing**
- **Enhanced Docling Processing** (`_process_with_enhanced_docling`):
  - Converts PDFs, Office docs, images to structured text
  - Applies OCR, table extraction, figure processing
  - Implements caching system with TTL (configurable hours)
  - Falls back to basic processing if enhanced processing fails

**Stage 3: Content Chunking** (`chunking_by_token_size`):
  - Tokenizes content using configurable tokenizer
  - Creates overlapping chunks (default: 1024 tokens, 128 overlap)
  - Generates unique chunk IDs with ordering metadata
  - Stores chunk relationships and document references

**Stage 4: Knowledge Extraction** (`extract_entities`):
  - **Entity Extraction**: LLM-powered identification of entities with types and descriptions
  - **Relationship Extraction**: Detects semantic relationships between entities
  - **Duplicate Handling**: Merges similar entities using embeddings and thresholds
  - **Validation**: Ensures extracted knowledge meets quality criteria

**Stage 5: Storage Distribution**:
```
Chunks → KV Storage (document content, LLM cache)
Entities/Relations → Graph Storage (knowledge graph structure)
Embeddings → Vector Storage (semantic search indices)
Processing Status → Document Status Storage (pipeline tracking)
```

#### Storage Backend Coordination

**KV Storage Backends**:
- `JsonKVStorage`: File-based storage for development
- `PGKVStorage`: PostgreSQL with JSON columns
- `RedisKVStorage`: Redis for high-performance caching
- `MongoKVStorage`: MongoDB document storage

**Vector Storage Backends**:
- `NanoVectorDBStorage`: Lightweight embedded storage
- `PGVectorStorage`: PostgreSQL with pgvector extension
- `MilvusVectorDBStorage`: Specialized vector database
- `QdrantVectorDBStorage`: Vector similarity search

**Graph Storage Backends**:
- `NetworkXStorage`: In-memory graph processing
- `Neo4JStorage`: Native graph database
- `PGGraphStorage`: PostgreSQL with graph extensions
- `MemgraphStorage`: High-performance graph analytics

### 1.2 Query Processing Flow

The query processing implements multiple retrieval strategies with sophisticated fusion:

```
User Query → Query Analysis → Multi-Mode Retrieval → Context Assembly → Response Generation
```

#### Query Processing Stages

**Stage 1: Query Input and Authentication**
- **Entry Points**: Web UI chat, API endpoints, MCP tools, Ollama-compatible interface
- **Authentication Flow**: JWT validation, rate limiting, audit logging
- **Parameter Processing**: Mode selection, token limits, retrieval parameters

**Stage 2: Query Analysis and Mode Selection**
- **Local Mode**: Context-dependent entity and chunk retrieval
- **Global Mode**: Knowledge graph traversal and relationship analysis
- **Hybrid Mode**: Combines local context with global knowledge
- **Mix Mode**: Integrates vector similarity with graph relationships
- **Naive Mode**: Basic vector search without graph enhancement

**Stage 3: Multi-Mode Retrieval Execution**

**Local Retrieval Process**:
```python
# Entity-centric retrieval
entities = await vector_search(query_embedding, top_k)
chunks = await get_entity_related_chunks(entities)
context = await build_local_context(entities, chunks, token_budget)
```

**Global Retrieval Process**:
```python
# Graph-based traversal
relationships = await graph_query(extracted_keywords)
entity_subgraph = await expand_subgraph(relationships, max_depth=2)
global_context = await summarize_global_knowledge(entity_subgraph)
```

**Stage 4: Context Assembly with Token Management**
- **Unified Token Control**: Distributes token budget across entities, relations, chunks
- **Priority Weighting**: Uses relevance scores to prioritize context elements
- **Truncation Strategy**: Intelligently trims context to fit model limits
- **Reranking**: Optional reranking of retrieved chunks for relevance

**Stage 5: Response Generation**
- **Prompt Construction**: Builds context-aware prompts with conversation history
- **LLM Integration**: Supports multiple LLM providers (OpenAI, xAI, Anthropic, Ollama)
- **Streaming Support**: Real-time response generation for interactive experiences
- **Caching**: LLM response caching to improve performance

### 1.3 Knowledge Graph Construction and Updates

Knowledge graph construction is an incremental, atomic process:

```
New Content → Entity Extraction → Relationship Detection → Graph Merging → Index Updates
```

#### Graph Construction Process

**Entity Processing**:
1. **Extraction**: LLM identifies entities with types and descriptions
2. **Embedding Generation**: Creates vector representations for similarity
3. **Deduplication**: Merges similar entities using cosine similarity thresholds
4. **Validation**: Ensures entity quality and consistency

**Relationship Processing**:
1. **Relationship Extraction**: Identifies semantic connections between entities
2. **Strength Scoring**: Assigns confidence scores to relationships
3. **Graph Integration**: Merges new relationships with existing graph
4. **Index Updates**: Updates search indices and traversal structures

**Atomic Updates**:
- Each document processing is transactional
- Failed processing marks documents as failed status
- Successful processing commits all storage changes
- Pipeline status tracking prevents partial states

## 2. System Interaction Patterns

### 2.1 API Server ↔ Core LightRAG Library Integration

The FastAPI server acts as a stateful wrapper around the async LightRAG core:

**Initialization Pattern**:
```python
# Server startup
rag = LightRAG(working_dir=working_dir, **config)
await rag.initialize_storages()
await initialize_pipeline_status()

# Request handling
async def query_endpoint(request: QueryRequest):
    return await rag.aquery(request.query, param=request.param)
```

**Storage Management**:
- **Shared Storage Access**: Thread-safe access to storage backends
- **Connection Pooling**: Efficient database connection management
- **Transaction Coordination**: Ensures data consistency across operations
- **Cleanup Patterns**: Proper resource cleanup with `await rag.finalize_storages()`

### 2.2 Web UI ↔ API Server Communication

The React frontend communicates with the API server through RESTful endpoints:

**Communication Flow**:
```typescript
// Frontend API calls
const response = await fetch(`${backendBaseUrl}/query`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify(queryRequest)
})
```

**Key Integration Points**:
- **Authentication**: JWT token management with automatic refresh
- **File Upload**: Multipart form handling for document uploads
- **Real-time Updates**: WebSocket or polling for document processing status
- **Graph Visualization**: Data transformation for interactive graph rendering
- **Error Handling**: Comprehensive error display and recovery

### 2.3 MCP Server ↔ Claude CLI Integration

The MCP (Model Context Protocol) server provides seamless Claude CLI integration:

**MCP Architecture**:
```python
# MCP tool registration
@mcp.tool()
async def lightrag_query_tool(query: str, mode: str = "hybrid") -> Dict[str, Any]:
    client = get_api_client()
    return await client.query(query, mode)
```

**Integration Features**:
- **11 Tools**: Query, document management, graph exploration, system monitoring
- **3 Resources**: System configuration, health status, API documentation
- **Streaming Support**: Real-time query responses through MCP protocol
- **Error Propagation**: Detailed error context for debugging

### 2.4 Storage Backend Coordination and Consistency

All storage backends implement consistent interfaces with atomic operations:

**Storage Interface Pattern**:
```python
class BaseKVStorage(ABC):
    @abstractmethod
    async def upsert(self, key: str, value: Dict[str, Any]) -> None:

    @abstractmethod
    async def get_by_id(self, id: str) -> Dict[str, Any] | None:
```

**Consistency Guarantees**:
- **Atomic Document Processing**: All-or-nothing document ingestion
- **Cross-Storage Transactions**: Coordinated updates across storage types
- **Lock Management**: Prevents concurrent modification conflicts
- **Rollback Capability**: Recovery from partial processing failures

## 3. Component Integration Points

### 3.1 Authentication and Authorization Flow

Production authentication provides comprehensive security:

**Authentication Workflow**:
```
Client Request → JWT Validation → Rate Limiting → Authorization → Request Processing → Audit Logging
```

**Security Components**:
- **JWT Management**: Token generation, validation, and refresh
- **Rate Limiting**: Per-endpoint and per-user request throttling
- **Audit Logging**: Complete request/response audit trail
- **Authorization**: Role-based access control for resources
- **Security Headers**: CORS, CSP, and other security headers

### 3.2 Rate Limiting and Security Middleware

FastAPI middleware provides layered security:

```python
# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if await is_rate_limited(request.client.host):
        raise HTTPException(429, "Rate limit exceeded")
    return await call_next(request)
```

**Security Layers**:
1. **Network Security**: Container isolation and internal networks
2. **Application Security**: Authentication, authorization, input validation
3. **Rate Limiting**: Configurable per-endpoint limits
4. **Audit Logging**: Comprehensive security event logging

### 3.3 Real-time Status Updates and Health Monitoring

The system provides comprehensive monitoring and status tracking:

**Health Check Endpoints**:
- `/health`: Basic application health
- `/api/health`: Detailed system status with dependencies
- Container health checks with custom scripts

**Status Tracking**:
- **Document Processing**: Real-time pipeline status updates
- **Storage Health**: Backend connectivity and performance monitoring
- **Resource Usage**: Memory, CPU, and connection pool metrics
- **Error Tracking**: Comprehensive error logging and alerting

### 3.4 Error Handling and Recovery Across Components

Multi-layer error handling ensures system resilience:

**Error Handling Strategy**:
```python
# Document processing error handling
try:
    await process_document(document)
    await update_document_status(document_id, "completed")
except ProcessingError as e:
    await update_document_status(document_id, "failed", error=str(e))
    logger.error(f"Document processing failed: {e}")
```

**Recovery Mechanisms**:
- **Retry Logic**: Configurable retry with exponential backoff
- **Circuit Breakers**: Prevent cascade failures
- **Graceful Degradation**: Fallback processing when enhanced features fail
- **State Recovery**: Resume processing from last known good state

## 4. Production Data Flow

### 4.1 Multi-tenant Workspace Isolation

Production deployment supports workspace isolation:

**Isolation Mechanisms**:
- **Working Directory Separation**: Each workspace has isolated storage
- **Database Schema Isolation**: Separate schemas or databases per tenant
- **Resource Quotas**: Per-workspace resource limits and monitoring
- **Access Controls**: Workspace-specific authentication and authorization

### 4.2 Backup and Recovery Data Paths

Comprehensive backup system protects data integrity:

**Backup Components**:
```bash
# Automated database backups
docker compose -f docker-compose.production.yml exec postgres pg_dump \
  -U lightrag lightrag > backup_$(date +%Y%m%d).sql

# RAG storage data backup
tar -czf rag_storage_backup_$(date +%Y%m%d).tar.gz rag_storage/
```

**Recovery Procedures**:
- **Database Recovery**: Point-in-time recovery from PostgreSQL backups
- **Storage Recovery**: File system restoration from compressed archives
- **Configuration Recovery**: Environment and configuration backup
- **Validation Scripts**: Post-recovery integrity checking

### 4.3 Monitoring and Audit Logging Integration

Production monitoring provides comprehensive observability:

**Monitoring Stack**:
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **Loki**: Log aggregation and analysis

**Audit Logging**:
- **Request/Response Logging**: Complete API interaction audit
- **Security Events**: Authentication, authorization, and access logs
- **Performance Metrics**: Response times, resource usage, error rates
- **Business Events**: Document processing, query execution, graph updates

### 4.4 Container Orchestration and Service Communication

Docker Compose orchestrates all services with proper networking:

**Service Architecture**:
```yaml
# Production service communication
networks:
  lightrag-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
  lightrag:
    depends_on:
      postgres: { condition: service_healthy }
      redis: { condition: service_healthy }
      docling-service: { condition: service_healthy }
```

**Communication Patterns**:
- **Internal Networks**: Isolated container communication
- **Health Dependencies**: Service startup orchestration
- **Load Balancing**: Nginx reverse proxy with upstream balancing
- **Service Discovery**: DNS-based service resolution

## Data Flow Sequence Diagrams

### Document Ingestion Sequence

```
User → Web UI → API Server → LightRAG Core → Storage Backends → Status Updates
  |      |         |           |              |                    |
  |      |         |           |              |                    └→ Real-time UI updates
  |      |         |           |              └→ Parallel writes to all storage types
  |      |         |           └→ Async processing pipeline with status tracking
  |      |         └→ Authentication, rate limiting, audit logging
  |      └→ File upload with progress tracking
  └→ User interaction with document manager
```

### Query Processing Sequence

```
User → Client Interface → API Server → LightRAG Core → Multi-Mode Retrieval → Response Generation
  |         |               |           |              |                      |
  |         |               |           |              |                      └→ Streaming response
  |         |               |           |              └→ Vector + Graph retrieval
  |         |               |           └→ Query analysis and mode selection
  |         |               └→ Security middleware and request processing
  |         └→ Query interface (Web UI, MCP, CLI, API)
  └→ Natural language query input
```

## Configuration and Environment Management

### Environment Variable Hierarchy

The system uses a layered configuration approach:

1. **OS Environment Variables** (highest priority)
2. **`.env` files** (per-instance configuration)
3. **Default values** (fallback configuration)

**Key Configuration Areas**:
- **LLM Integration**: Provider selection, API keys, model parameters
- **Storage Backends**: Database connections, cache settings, storage paths
- **Security Settings**: Authentication, rate limits, audit configuration
- **Performance Tuning**: Worker processes, connection pools, timeouts

### Multi-Environment Support

Different deployment environments use specialized configurations:

- **Development**: `docker-compose.yml` with local services
- **Production**: `docker-compose.production.yml` with security hardening
- **Testing**: Isolated test configurations with mock services

## Summary

LightRAG implements a sophisticated multi-component architecture with comprehensive data flow management. The system provides:

1. **Scalable Processing**: Async/await architecture with multiple worker processes
2. **Storage Flexibility**: Multiple backend options with consistent interfaces
3. **Security Integration**: Authentication, authorization, and audit logging
4. **Real-time Updates**: Status tracking and streaming responses
5. **Production Readiness**: Containerization, monitoring, and backup systems
6. **Multi-Interface Support**: Web UI, API, MCP, and CLI access

The architecture supports both development flexibility and production robustness through its layered design, comprehensive error handling, and extensive configuration options. All components interact through well-defined interfaces with proper error handling and recovery mechanisms.

## Documentation Quality Assessment

**Accuracy**: Verified against actual codebase implementation and container configurations
**Completeness**: Covers all major components and interaction patterns
**Usability**: Structured for both developers and system administrators with clear examples
**Maintainability**: Organized sections support incremental updates and expansion
