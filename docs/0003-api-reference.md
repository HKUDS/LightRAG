# LightRAG API Reference

> Complete REST API documentation for LightRAG Server

**Version**: 1.4.9.1 | **Base URL**: `http://localhost:9621`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Query Endpoints](#query-endpoints)
3. [Document Endpoints](#document-endpoints)
4. [Graph Endpoints](#graph-endpoints)
5. [Admin Endpoints](#admin-endpoints)
6. [Multi-Tenant Headers](#multi-tenant-headers)
7. [Error Handling](#error-handling)

---

## Authentication

LightRAG supports multiple authentication methods:

### API Key Authentication

```http
Authorization: Bearer <API_KEY>
```

### Basic Authentication

```http
Authorization: Basic <base64(username:password)>
```

### JWT Token (Multi-Tenant)

```http
Authorization: Bearer <JWT_TOKEN>
X-Tenant-ID: <tenant_uuid>
X-KB-ID: <knowledge_base_uuid>
```

---

## Query Endpoints

### POST `/query`

Execute a RAG query and get a response.

```http
POST /query
Content-Type: application/json
Authorization: Bearer <token>
```

#### Request Body

```json
{
  "query": "What is the capital of France?",
  "mode": "mix",
  "only_need_context": false,
  "only_need_prompt": false,
  "response_type": "Multiple Paragraphs",
  "top_k": 40,
  "chunk_top_k": 20,
  "max_entity_tokens": 6000,
  "max_relation_tokens": 8000,
  "max_total_tokens": 30000,
  "conversation_history": [
    {"role": "user", "content": "Tell me about Europe"},
    {"role": "assistant", "content": "Europe is..."}
  ],
  "user_prompt": "Focus on historical context",
  "enable_rerank": true,
  "include_references": true
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Query text (min 3 chars) |
| `mode` | string | ❌ | `mix` | Query mode: `local`, `global`, `hybrid`, `naive`, `mix`, `bypass` |
| `only_need_context` | bool | ❌ | `false` | Return only context without LLM response |
| `only_need_prompt` | bool | ❌ | `false` | Return only generated prompt |
| `response_type` | string | ❌ | `Multiple Paragraphs` | Response format |
| `top_k` | int | ❌ | `40` | Number of entities/relations to retrieve |
| `chunk_top_k` | int | ❌ | `20` | Number of text chunks to retrieve |
| `max_entity_tokens` | int | ❌ | `6000` | Max tokens for entity context |
| `max_relation_tokens` | int | ❌ | `8000` | Max tokens for relation context |
| `max_total_tokens` | int | ❌ | `30000` | Total context token budget |
| `conversation_history` | array | ❌ | `[]` | Previous conversation turns |
| `user_prompt` | string | ❌ | `null` | Additional instructions for LLM |
| `enable_rerank` | bool | ❌ | `true` | Enable chunk reranking |
| `include_references` | bool | ❌ | `true` | Include reference list |

#### Response

```json
{
  "response": "Paris is the capital of France...",
  "references": [
    {"id": "doc-123", "title": "France Geography"},
    {"id": "doc-456", "title": "European Capitals"}
  ]
}
```

---

### POST `/query/stream`

Stream a RAG query response (Server-Sent Events).

```http
POST /query/stream
Content-Type: application/json
Authorization: Bearer <token>
```

#### Request Body

Same as `/query`

#### Response (NDJSON Stream)

```json
{"references": [{"id": "doc-123", "title": "France"}]}
{"response": "Paris "}
{"response": "is "}
{"response": "the capital..."}
```

---

### POST `/query/data`

Get structured query data with full context information.

```http
POST /query/data
Content-Type: application/json
Authorization: Bearer <token>
```

#### Response

```json
{
  "status": "success",
  "message": "Query completed",
  "data": {
    "entities": [
      {
        "entity_name": "Paris",
        "entity_type": "location",
        "description": "Capital city of France...",
        "source_id": "chunk-001"
      }
    ],
    "relationships": [
      {
        "source": "France",
        "target": "Paris",
        "description": "capital of",
        "keywords": "capital, city"
      }
    ],
    "chunks": [
      {
        "id": "chunk-001",
        "content": "Paris is the capital...",
        "file_path": "france.txt"
      }
    ],
    "references": [...]
  },
  "metadata": {
    "mode": "mix",
    "high_level_keywords": ["capital", "france"],
    "low_level_keywords": ["paris", "city"]
  }
}
```

---

## Document Endpoints

### POST `/documents/text`

Insert a single text document.

```http
POST /documents/text
Content-Type: application/json
Authorization: Bearer <token>
```

#### Request Body

```json
{
  "text": "Document content to insert...",
  "file_source": "manual_input.txt",
  "external_id": "unique-doc-id-123"
}
```

#### Response

```json
{
  "status": "accepted",
  "message": "Text document accepted for processing",
  "doc_id": "abc123...",
  "track_id": "track_20251204_123456_xyz"
}
```

---

### POST `/documents/texts`

Insert multiple text documents in batch.

```http
POST /documents/texts
Content-Type: application/json
Authorization: Bearer <token>
```

#### Request Body

```json
{
  "texts": [
    "First document content...",
    "Second document content..."
  ],
  "file_sources": ["doc1.txt", "doc2.txt"],
  "external_ids": ["id-001", "id-002"]
}
```

---

### POST `/documents/file`

Upload a file for processing.

```http
POST /documents/file
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

#### Form Data

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | File to upload (txt, md, pdf, docx) |

#### Response

```json
{
  "status": "uploaded",
  "message": "File uploaded successfully",
  "filename": "document.pdf",
  "track_id": "track_20251204_123456_xyz"
}
```

---

### POST `/documents/files`

Upload multiple files for processing.

```http
POST /documents/files
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

---

### POST `/documents/scan`

Scan and process all files in the input directory.

```http
POST /documents/scan
Authorization: Bearer <token>
```

#### Response

```json
{
  "status": "scanning_started",
  "message": "Scanning process initiated",
  "track_id": "scan_20251204_123456_xyz"
}
```

---

### GET `/documents`

List all documents with pagination.

```http
GET /documents?page=1&page_size=20&status=processed&sort_field=created_at&sort_order=desc
Authorization: Bearer <token>
```

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | `1` | Page number |
| `page_size` | int | `20` | Items per page (max 100) |
| `status` | string | - | Filter by status: `pending`, `processing`, `processed`, `failed` |
| `sort_field` | string | `created_at` | Sort field |
| `sort_order` | string | `desc` | Sort order: `asc`, `desc` |

#### Response

```json
{
  "documents": [
    {
      "id": "abc123",
      "file_path": "document.txt",
      "status": "processed",
      "chunks_count": 15,
      "created_at": "2025-12-04T10:30:00Z",
      "updated_at": "2025-12-04T10:35:00Z"
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 20,
  "total_pages": 8
}
```

---

### GET `/documents/{doc_id}`

Get document details by ID.

```http
GET /documents/{doc_id}
Authorization: Bearer <token>
```

---

### DELETE `/documents/{doc_id}`

Delete a document and its associated data.

```http
DELETE /documents/{doc_id}
Authorization: Bearer <token>
```

#### Response

```json
{
  "status": "success",
  "message": "Document deleted successfully",
  "details": {
    "doc_id": "abc123",
    "chunks_deleted": 15,
    "entities_affected": 23,
    "relations_affected": 12
  }
}
```

---

### DELETE `/documents`

Clear all documents (dangerous operation).

```http
DELETE /documents
Authorization: Bearer <token>
```

---

### GET `/documents/status/{track_id}`

Get processing status for a tracking ID.

```http
GET /documents/status/{track_id}
Authorization: Bearer <token>
```

#### Response

```json
{
  "track_id": "track_20251204_123456_xyz",
  "status": "processing",
  "progress": 0.65,
  "current_step": "entity_extraction",
  "documents_processed": 5,
  "documents_total": 8,
  "errors": [],
  "started_at": "2025-12-04T10:30:00Z",
  "updated_at": "2025-12-04T10:35:00Z"
}
```

---

## Graph Endpoints

### GET `/graph/label/list`

Get all entity labels in the knowledge graph.

```http
GET /graph/label/list
Authorization: Bearer <token>
```

#### Response

```json
["Person", "Organization", "Location", "Event", "Concept"]
```

---

### GET `/graph/label/popular`

Get popular labels sorted by node degree.

```http
GET /graph/label/popular?limit=100
Authorization: Bearer <token>
```

#### Query Parameters

| Parameter | Type | Default | Max | Description |
|-----------|------|---------|-----|-------------|
| `limit` | int | `300` | `1000` | Max labels to return |

---

### GET `/graph/label/search`

Search labels with fuzzy matching.

```http
GET /graph/label/search?q=apple&limit=50
Authorization: Bearer <token>
```

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | string | - | Search query (required) |
| `limit` | int | `50` | Max results |

---

### GET `/graphs`

Get knowledge graph subgraph for a label.

```http
GET /graphs?label=Apple&max_depth=3&max_nodes=1000
Authorization: Bearer <token>
```

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label` | string | - | Starting node label (required) |
| `max_depth` | int | `3` | Maximum traversal depth |
| `max_nodes` | int | `1000` | Maximum nodes to return |

#### Response

```json
{
  "nodes": [
    {
      "id": "apple-inc",
      "labels": ["Organization"],
      "properties": {
        "description": "Technology company...",
        "entity_type": "organization"
      }
    }
  ],
  "edges": [
    {
      "id": "edge-001",
      "source": "apple-inc",
      "target": "iphone",
      "type": "produces",
      "properties": {
        "description": "Apple produces iPhone",
        "weight": 5.0
      }
    }
  ],
  "is_truncated": false
}
```

---

### PUT `/graph/entity`

Update an entity in the knowledge graph.

```http
PUT /graph/entity
Content-Type: application/json
Authorization: Bearer <token>
```

#### Request Body

```json
{
  "entity_name": "Apple Inc.",
  "updated_data": {
    "description": "Updated description...",
    "entity_type": "technology_company"
  },
  "allow_rename": false
}
```

---

### PUT `/graph/relation`

Update a relationship in the knowledge graph.

```http
PUT /graph/relation
Content-Type: application/json
Authorization: Bearer <token>
```

#### Request Body

```json
{
  "source_id": "Apple Inc.",
  "target_id": "iPhone",
  "updated_data": {
    "description": "Designs and manufactures",
    "keywords": "technology, smartphone"
  }
}
```

---

### DELETE `/graph/entity/{entity_name}`

Delete an entity from the knowledge graph.

```http
DELETE /graph/entity/{entity_name}
Authorization: Bearer <token>
```

---

### DELETE `/graph/relation`

Delete a relationship from the knowledge graph.

```http
DELETE /graph/relation?source_id=Apple&target_id=iPhone
Authorization: Bearer <token>
```

---

## Admin Endpoints

### GET `/health`

Health check endpoint.

```http
GET /health
```

#### Response

```json
{
  "status": "healthy",
  "version": "1.4.9.1",
  "storage_status": "initialized",
  "llm_status": "connected",
  "embedding_status": "connected"
}
```

---

### GET `/health/storage`

Detailed storage health check.

```http
GET /health/storage
Authorization: Bearer <token>
```

---

### POST `/admin/drop`

Drop all data from storages (destructive).

```http
POST /admin/drop
Authorization: Bearer <token>
```

---

### POST `/documents/rebuild-index`

Rebuild knowledge graph from cached extractions.

```http
POST /documents/rebuild-index
Authorization: Bearer <token>
```

---

## Multi-Tenant Headers

For multi-tenant deployments, include these headers:

| Header | Description | Required |
|--------|-------------|----------|
| `X-Tenant-ID` | Tenant UUID | When multi-tenant enabled |
| `X-KB-ID` | Knowledge Base UUID | When multi-tenant enabled |
| `X-User-ID` | User identifier | For audit logging |

### Example Multi-Tenant Request

```http
POST /query
Content-Type: application/json
Authorization: Bearer <jwt_token>
X-Tenant-ID: 550e8400-e29b-41d4-a716-446655440000
X-KB-ID: 6ba7b810-9dad-11d1-80b4-00c04fd430c8

{
  "query": "What are our Q3 results?",
  "mode": "mix"
}
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error description",
  "error_code": "ERROR_CODE",
  "request_id": "req-12345"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `201` | Created |
| `202` | Accepted (async processing started) |
| `400` | Bad Request (invalid input) |
| `401` | Unauthorized (missing/invalid auth) |
| `403` | Forbidden (insufficient permissions) |
| `404` | Not Found |
| `422` | Validation Error |
| `429` | Too Many Requests |
| `500` | Internal Server Error |

### Common Error Codes

| Error Code | Description |
|------------|-------------|
| `INVALID_QUERY` | Query text too short or invalid |
| `DOCUMENT_NOT_FOUND` | Requested document doesn't exist |
| `TENANT_NOT_FOUND` | Invalid tenant ID |
| `KB_NOT_FOUND` | Invalid knowledge base ID |
| `PROCESSING_FAILED` | Document processing failed |
| `STORAGE_ERROR` | Database connection error |
| `LLM_ERROR` | LLM provider error |
| `RATE_LIMITED` | Too many requests |

---

## Rate Limiting

Default rate limits (configurable via environment):

| Endpoint Category | Limit |
|-------------------|-------|
| Query endpoints | 60/minute |
| Document upload | 20/minute |
| Batch operations | 10/minute |
| Admin endpoints | 5/minute |

---

## Ollama-Compatible API

LightRAG provides Ollama-compatible endpoints for integration with Ollama clients.

### POST `/api/generate`

Ollama-compatible generation endpoint.

```http
POST /api/generate
Content-Type: application/json
```

#### Request Body

```json
{
  "model": "lightrag",
  "prompt": "What is the capital of France?",
  "stream": false
}
```

### POST `/api/chat`

Ollama-compatible chat endpoint.

```http
POST /api/chat
Content-Type: application/json
```

#### Request Body

```json
{
  "model": "lightrag",
  "messages": [
    {"role": "user", "content": "Tell me about Paris"}
  ],
  "stream": true
}
```

### GET `/api/tags`

List available models.

```http
GET /api/tags
```

---

## WebUI

Access the built-in web interface at:

```
http://localhost:9621/webui
```

Features:
- Document management
- Knowledge graph visualization
- Query interface
- System configuration

---

## OpenAPI Documentation

Interactive API documentation available at:

- **Swagger UI**: `http://localhost:9621/docs`
- **ReDoc**: `http://localhost:9621/redoc`
- **OpenAPI JSON**: `http://localhost:9621/openapi.json`

---

**Version**: 1.4.9.1 | **License**: MIT
