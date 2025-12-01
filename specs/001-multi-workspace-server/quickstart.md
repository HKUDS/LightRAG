# Quickstart: Multi-Workspace LightRAG Server

**Date**: 2025-12-01
**Feature**: 001-multi-workspace-server

## Overview

This guide shows how to deploy LightRAG Server with multi-workspace support, enabling a single server instance to serve multiple isolated tenants.

## Configuration

### Environment Variables

Add these new environment variables to your deployment:

```bash
# Multi-workspace configuration
LIGHTRAG_DEFAULT_WORKSPACE=default          # Workspace for requests without header
LIGHTRAG_ALLOW_DEFAULT_WORKSPACE=true       # Allow requests without workspace header
LIGHTRAG_MAX_WORKSPACES_IN_POOL=50          # Max concurrent workspace instances

# Existing configuration (unchanged)
WORKSPACE=default                           # Backward compatible, used if DEFAULT_WORKSPACE not set
WORKING_DIR=/data/rag_storage               # Base directory for all workspace data
INPUT_DIR=/data/inputs                      # Base directory for workspace input files
```

### Configuration Modes

#### Mode 1: Backward Compatible (Default)

No changes needed. Existing deployments work unchanged.

```bash
# .env file
WORKSPACE=my_workspace
```

All requests use `my_workspace` regardless of headers.

#### Mode 2: Multi-Workspace with Default

Allow multiple workspaces, with a fallback for headerless requests.

```bash
# .env file
LIGHTRAG_DEFAULT_WORKSPACE=default
LIGHTRAG_ALLOW_DEFAULT_WORKSPACE=true
LIGHTRAG_MAX_WORKSPACES_IN_POOL=50
```

- Requests with `LIGHTRAG-WORKSPACE` header → use specified workspace
- Requests without header → use `default` workspace

#### Mode 3: Strict Multi-Tenant

Require workspace header on all requests. Prevents accidental data leakage.

```bash
# .env file
LIGHTRAG_ALLOW_DEFAULT_WORKSPACE=false
LIGHTRAG_MAX_WORKSPACES_IN_POOL=100
```

- Requests with `LIGHTRAG-WORKSPACE` header → use specified workspace
- Requests without header → return `400 Bad Request`

## Usage Examples

### Starting the Server

```bash
# Standard startup (works the same as before)
lightrag-server --host 0.0.0.0 --port 9621

# Or with environment variables
export LIGHTRAG_DEFAULT_WORKSPACE=default
export LIGHTRAG_ALLOW_DEFAULT_WORKSPACE=true
lightrag-server
```

### Making Requests

#### Single-Workspace (No Header)

```bash
# Uses default workspace
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "What is LightRAG?"}'
```

#### Multi-Workspace (With Header)

```bash
# Ingest document to tenant-a
curl -X POST http://localhost:9621/documents/text \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -H "LIGHTRAG-WORKSPACE: tenant-a" \
  -d '{"text": "Tenant A confidential document about AI."}'

# Query from tenant-a (finds the document)
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -H "LIGHTRAG-WORKSPACE: tenant-a" \
  -d '{"query": "What is this workspace about?"}'

# Query from tenant-b (does NOT find tenant-a's document)
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -H "LIGHTRAG-WORKSPACE: tenant-b" \
  -d '{"query": "What is this workspace about?"}'
```

### Python Client Example

```python
import httpx

class LightRAGClient:
    def __init__(self, base_url: str, api_key: str, workspace: str | None = None):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if workspace:
            self.headers["LIGHTRAG-WORKSPACE"] = workspace

    async def query(self, query: str) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/query",
                headers=self.headers,
                json={"query": query}
            )
            response.raise_for_status()
            return response.json()

# Usage
tenant_a_client = LightRAGClient(
    "http://localhost:9621",
    api_key="your-api-key",
    workspace="tenant-a"
)
tenant_b_client = LightRAGClient(
    "http://localhost:9621",
    api_key="your-api-key",
    workspace="tenant-b"
)

# Each client accesses only its own workspace
result_a = await tenant_a_client.query("What documents do I have?")
result_b = await tenant_b_client.query("What documents do I have?")
```

## Data Isolation

Each workspace has completely isolated:

- **Documents**: Files ingested in one workspace are invisible to others
- **Embeddings**: Vector indices are workspace-scoped
- **Knowledge Graph**: Entities and relationships are workspace-specific
- **Query Results**: Queries only return data from the specified workspace

### Directory Structure

```
/data/rag_storage/
├── tenant-a/                 # Workspace: tenant-a
│   ├── kv_store_*.json
│   ├── vdb_*.json
│   └── graph_*.json
├── tenant-b/                 # Workspace: tenant-b
│   ├── kv_store_*.json
│   ├── vdb_*.json
│   └── graph_*.json
└── default/                  # Default workspace
    └── ...

/data/inputs/
├── tenant-a/                 # Input files for tenant-a
├── tenant-b/                 # Input files for tenant-b
└── default/                  # Input files for default workspace
```

## Memory Management

The workspace pool uses LRU (Least Recently Used) eviction:

- First request to a workspace initializes its LightRAG instance
- Instances stay loaded for fast subsequent requests
- When pool reaches `LIGHTRAG_MAX_WORKSPACES_IN_POOL`, least recently used workspace is evicted
- Evicted workspaces are re-initialized on next request (data persists in storage)

### Tuning Pool Size

| Deployment Size | Recommended Pool Size | Notes |
|-----------------|----------------------|-------|
| Development | 5-10 | Minimal memory usage |
| Small SaaS | 20-50 | Handles typical multi-tenant load |
| Large SaaS | 100+ | Depends on available memory |

**Memory Estimate**: Each workspace instance uses approximately 50-200MB depending on LLM/embedding bindings and cache settings.

## Troubleshooting

### "Missing LIGHTRAG-WORKSPACE header"

**Cause**: `LIGHTRAG_ALLOW_DEFAULT_WORKSPACE=false` and no header provided

**Solution**: Either:
- Add `LIGHTRAG-WORKSPACE` header to all requests
- Set `LIGHTRAG_ALLOW_DEFAULT_WORKSPACE=true`

### "Invalid workspace identifier"

**Cause**: Workspace ID contains invalid characters

**Solution**: Use only alphanumeric characters, hyphens, and underscores. Must start with alphanumeric, max 64 characters.

### "Failed to initialize workspace"

**Cause**: Storage backend unavailable or misconfigured

**Solution**: Check storage backend connectivity (Postgres, Neo4j, etc.) and verify configuration.

### Slow First Request to New Workspace

**Expected Behavior**: First request to a workspace initializes storage connections.

**Mitigation**: Pre-warm frequently used workspaces at startup (implementation-specific).
