# API Contract: Workspace Routing

**Date**: 2025-12-01
**Feature**: 001-multi-workspace-server

## Overview

This feature adds workspace routing via HTTP headers. No new API endpoints are introduced; existing endpoints are enhanced to support multi-workspace operation through header-based routing.

## Contract Changes

### New Request Headers

All existing API endpoints now accept these optional headers:

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `LIGHTRAG-WORKSPACE` | `string` | No* | Primary workspace identifier |
| `X-Workspace-ID` | `string` | No* | Fallback workspace identifier |

\* Required when `LIGHTRAG_ALLOW_DEFAULT_WORKSPACE=false`

**Header Priority**:
1. `LIGHTRAG-WORKSPACE` (if present and non-empty)
2. `X-Workspace-ID` (if present and non-empty)
3. Default workspace from config (if headers missing)

### Workspace Identifier Format

Valid workspace identifiers must match:
- Pattern: `^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$`
- Length: 1-64 characters
- First character: alphanumeric
- Subsequent characters: alphanumeric, hyphen, underscore

**Valid Examples**:
- `tenant-123`
- `my_workspace`
- `ProjectAlpha`
- `user42_prod`

**Invalid Examples**:
- `_hidden` (starts with underscore)
- `-invalid` (starts with hyphen)
- `a` repeated 100 times (too long)
- `path/traversal` (contains slash)

### Error Responses

New error responses for workspace-related issues:

#### 400 Bad Request - Missing Workspace Header

**Condition**: No workspace header provided and `LIGHTRAG_ALLOW_DEFAULT_WORKSPACE=false`

```json
{
  "detail": "Missing LIGHTRAG-WORKSPACE header. Workspace identification is required."
}
```

#### 400 Bad Request - Invalid Workspace Identifier

**Condition**: Workspace identifier fails validation

```json
{
  "detail": "Invalid workspace identifier 'bad/id': must be 1-64 alphanumeric characters (hyphens and underscores allowed, must start with alphanumeric)"
}
```

#### 503 Service Unavailable - Workspace Initialization Failed

**Condition**: Failed to initialize workspace instance (storage unavailable, etc.)

```json
{
  "detail": "Failed to initialize workspace 'tenant-123': Storage connection failed"
}
```

## Affected Endpoints

All existing endpoints are affected. The workspace header determines which LightRAG instance processes the request.

### Document Endpoints
- `POST /documents/scan`
- `POST /documents/upload`
- `POST /documents/text`
- `POST /documents/batch`
- `DELETE /documents/{doc_id}`
- `GET /documents`
- `GET /documents/{doc_id}`

### Query Endpoints
- `POST /query`
- `POST /query/stream`

### Graph Endpoints
- `GET /graph/label/list`
- `POST /graph/label/entities`
- `GET /graphs`

### Ollama-Compatible Endpoints
- `POST /api/chat`
- `POST /api/generate`
- `GET /api/tags`

### Unaffected Endpoints

These endpoints operate at server level (not workspace-scoped):
- `GET /health`
- `GET /auth-status`
- `POST /login`
- `GET /docs`

## Example Usage

### Single-Workspace Mode (Backward Compatible)

No changes required. Requests without workspace headers use the default workspace.

```bash
# Uses default workspace (from WORKSPACE env var)
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "What is LightRAG?"}'
```

### Multi-Workspace Mode

Include workspace header to target specific workspace:

```bash
# Target tenant-a workspace
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -H "LIGHTRAG-WORKSPACE: tenant-a" \
  -d '{"query": "What is in this workspace?"}'

# Target tenant-b workspace
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -H "LIGHTRAG-WORKSPACE: tenant-b" \
  -d '{"query": "What is in this workspace?"}'
```

### Strict Multi-Tenant Mode

When `LIGHTRAG_ALLOW_DEFAULT_WORKSPACE=false`:

```bash
# This will return 400 error
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "Missing workspace header"}'

# Response:
# {"detail": "Missing LIGHTRAG-WORKSPACE header. Workspace identification is required."}
```

## Response Headers

No new response headers are added. The workspace used for processing is logged server-side but not returned to the client (to avoid information leakage in error cases).

## Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| Existing client, no workspace header | Uses default workspace (unchanged behavior) |
| Existing config, new server version | Works unchanged (default workspace = `WORKSPACE` env var) |
| New config vars not set | Falls back to existing `WORKSPACE` env var |
