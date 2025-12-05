# LightRAG Multi-Tenancy Guide

> **🚀 Enterprise Feature** | Multi-tenancy is the first enterprise feature added to this fork of LightRAG.
>
> This feature was developed by [Raphaël MANSUY](https://www.elitizon.com/) as part of the enterprise-ready LightRAG initiative.

## Multi-Tenancy Overview

LightRAG Enterprise provides a complete multi-tenant architecture for isolating data across organizations, teams, or applications. This is essential for SaaS deployments, enterprise environments, and any scenario requiring data isolation between different user groups.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-TENANCY ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         API GATEWAY                                   │   │
│  │                    (JWT Authentication)                               │   │
│  └────────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      TENANT RAG MANAGER                               │   │
│  │              (LRU Cache + Per-Tenant Instances)                       │   │
│  └────────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                              │
│       ┌───────────────────────┼───────────────────────┐                     │
│       │                       │                       │                     │
│       ▼                       ▼                       ▼                     │
│  ┌────────────┐         ┌────────────┐         ┌────────────┐              │
│  │  Tenant A  │         │  Tenant B  │         │  Tenant C  │              │
│  ├────────────┤         ├────────────┤         ├────────────┤              │
│  │ ┌────────┐ │         │ ┌────────┐ │         │ ┌────────┐ │              │
│  │ │  KB 1  │ │         │ │  KB 1  │ │         │ │  KB 1  │ │              │
│  │ └────────┘ │         │ └────────┘ │         │ └────────┘ │              │
│  │ ┌────────┐ │         │ ┌────────┐ │         │            │              │
│  │ │  KB 2  │ │         │ │  KB 2  │ │         │            │              │
│  │ └────────┘ │         │ └────────┘ │         │            │              │
│  └────────────┘         └────────────┘         └────────────┘              │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     ISOLATED STORAGE                                  │   │
│  │  tenant_a/kb_1/      tenant_b/kb_1/      tenant_c/kb_1/              │   │
│  │  tenant_a/kb_2/      tenant_b/kb_2/                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### Tenant

A tenant represents an organization or isolated environment. Each tenant:
- Has unique configuration (models, thresholds, quotas)
- Contains multiple knowledge bases
- Manages its own users and roles
- Is isolated from other tenants

### Knowledge Base (KB)

A knowledge base is a document collection within a tenant:
- Stores documents, entities, and relationships
- Has isolated storage (KV, vector, graph)
- Can override tenant-level configuration
- Tracks statistics (document count, storage)

### Roles & Permissions

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         RBAC HIERARCHY                                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ADMIN           ─────────────────────────────────────────────────────┐    │
│  │ All permissions                                                     │    │
│  │ ├── tenant:manage           ← Manage tenant settings               │    │
│  │ ├── tenant:manage_members   ← Add/remove users                     │    │
│  │ ├── tenant:manage_billing   ← Billing access                       │    │
│  │ └── All KB/Document/Query permissions                              │    │
│  │                                                                     │    │
│  EDITOR          ─────────────────────────────────────────────────────┤    │
│  │ Content management                                                  │    │
│  │ ├── kb:create               ← Create knowledge bases               │    │
│  │ ├── kb:delete               ← Delete knowledge bases               │    │
│  │ ├── document:create         ← Upload documents                     │    │
│  │ ├── document:update         ← Edit documents                       │    │
│  │ ├── document:delete         ← Remove documents                     │    │
│  │ ├── document:read           ← Read documents                       │    │
│  │ └── query:run               ← Execute queries                      │    │
│  │                                                                     │    │
│  VIEWER          ─────────────────────────────────────────────────────┤    │
│  │ Read + query                                                        │    │
│  │ ├── document:read           ← Read documents                       │    │
│  │ └── query:run               ← Execute queries                      │    │
│  │                                                                     │    │
│  VIEWER_READONLY ─────────────────────────────────────────────────────┘    │
│    Query only                                                               │
│    └── query:run               ← Execute queries only                      │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Enable Multi-Tenancy

```bash
# Environment variables
ENABLE_MULTI_TENANTS=true
LIGHTRAG_MULTI_TENANT_STRICT=true
LIGHTRAG_REQUIRE_USER_AUTH=true
LIGHTRAG_SUPER_ADMIN_USERS=admin
```

### Security Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MULTI_TENANTS` | `false` | Enable multi-tenant mode |
| `LIGHTRAG_MULTI_TENANT_STRICT` | `true` | Enforce tenant isolation on data endpoints |
| `LIGHTRAG_REQUIRE_USER_AUTH` | `true` | Require user auth for tenant access |
| `LIGHTRAG_SUPER_ADMIN_USERS` | `admin` | Comma-separated list of super admins |

---

## TenantRAGManager

The `TenantRAGManager` handles LightRAG instance lifecycle:

### Features

- **Instance Caching**: LRU cache for tenant/KB instances
- **Lazy Initialization**: Instances created on-demand
- **Resource Cleanup**: Automatic finalization on eviction
- **Async-Safe**: Double-check locking for concurrent access
- **Security Validation**: User access verification

### Implementation

```python
from lightrag.tenant_rag_manager import TenantRAGManager
from lightrag.services.tenant_service import TenantService

# Initialize manager
manager = TenantRAGManager(
    base_working_dir="./rag_storage",
    tenant_service=tenant_service,
    template_rag=global_rag_instance,
    max_cached_instances=100  # LRU limit
)

# Get tenant-specific instance
rag = await manager.get_rag_instance(
    tenant_id="uuid-tenant-1",
    kb_id="uuid-kb-1",
    user_id="user@example.com"  # For access control
)

# Use normally
await rag.ainsert("Document content...")
result = await rag.aquery("Query?")

# Cleanup
await manager.cleanup_instance(tenant_id, kb_id)
await manager.cleanup_all()  # Shutdown
```

### Storage Isolation

Each tenant/KB combination gets isolated storage:

```
rag_storage/
├── tenant_abc123/
│   ├── kb_xyz789/
│   │   ├── kv_store_*.json
│   │   ├── vector_db/
│   │   └── graph_db/
│   └── kb_def456/
│       ├── kv_store_*.json
│       ├── vector_db/
│       └── graph_db/
└── tenant_ghi012/
    └── kb_jkl345/
        ├── kv_store_*.json
        ├── vector_db/
        └── graph_db/
```

---

## Tenant Service

The `TenantService` manages tenant and KB metadata:

### Create Tenant

```python
from lightrag.services.tenant_service import TenantService
from lightrag.models.tenant import TenantConfig, ResourceQuota

tenant = await tenant_service.create_tenant(
    tenant_name="Acme Corp",
    description="Production tenant",
    config=TenantConfig(
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-ada-002",
        top_k=50,
        enable_rerank=True
    ),
    created_by="admin@acme.com"
)
print(f"Created tenant: {tenant.tenant_id}")
```

### Create Knowledge Base

```python
kb = await tenant_service.create_knowledge_base(
    tenant_id=tenant.tenant_id,
    kb_name="Product Documentation",
    description="Internal product docs",
    created_by="admin@acme.com"
)
print(f"Created KB: {kb.kb_id}")
```

### User Management

```python
# Add user to tenant
await tenant_service.add_user_to_tenant(
    user_id="user@acme.com",
    tenant_id=tenant.tenant_id,
    role="editor",
    created_by="admin@acme.com"
)

# Verify access
has_access = await tenant_service.verify_user_access(
    user_id="user@acme.com",
    tenant_id=tenant.tenant_id,
    required_role="viewer"
)
```

---

## REST API Endpoints

### Tenant Management

```bash
# List tenants (public for selection)
GET /api/v1/tenants?page=1&page_size=10

# Create tenant
POST /api/v1/tenants
{
    "name": "Acme Corp",
    "description": "Production tenant",
    "metadata": {"plan": "enterprise"}
}

# Get tenant
GET /api/v1/tenants/{tenant_id}

# Update tenant
PUT /api/v1/tenants/{tenant_id}
{
    "name": "Acme Corporation",
    "description": "Updated description"
}

# Delete tenant
DELETE /api/v1/tenants/{tenant_id}
```

### Knowledge Base Management

```bash
# List KBs
GET /api/v1/tenants/{tenant_id}/kbs

# Create KB
POST /api/v1/tenants/{tenant_id}/kbs
{
    "name": "Product Docs",
    "description": "Documentation KB"
}

# Get KB
GET /api/v1/tenants/{tenant_id}/kbs/{kb_id}

# Update KB
PUT /api/v1/tenants/{tenant_id}/kbs/{kb_id}
{
    "name": "Updated Name"
}

# Delete KB
DELETE /api/v1/tenants/{tenant_id}/kbs/{kb_id}
```

### Member Management

```bash
# List members
GET /api/v1/tenants/{tenant_id}/members

# Add member
POST /api/v1/tenants/{tenant_id}/members
{
    "user_id": "user@example.com",
    "role": "editor"
}

# Update role
PUT /api/v1/tenants/{tenant_id}/members/{user_id}
{
    "role": "admin"
}

# Remove member
DELETE /api/v1/tenants/{tenant_id}/members/{user_id}
```

---

## Data Models

### Tenant

```python
@dataclass
class Tenant:
    tenant_id: str                      # UUID
    tenant_name: str                    # Display name
    description: Optional[str]          # Description
    config: TenantConfig                # Model/query configuration
    quota: ResourceQuota                # Resource limits
    is_active: bool = True              # Active status
    created_at: datetime                # Creation timestamp
    updated_at: datetime                # Last update
    created_by: Optional[str]           # Creator user ID
    metadata: Dict[str, Any]            # Custom metadata

    # Statistics
    kb_count: int = 0
    total_documents: int = 0
    total_storage_mb: float = 0.0
```

### TenantConfig

```python
@dataclass
class TenantConfig:
    # Model selection
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "bge-m3:latest"
    rerank_model: Optional[str] = None

    # LLM parameters
    llm_model_kwargs: Dict = {}
    llm_temperature: float = 1.0
    llm_max_tokens: int = 4096

    # Embedding
    embedding_dim: int = 1024
    embedding_batch_num: int = 10

    # Query defaults
    top_k: int = 40
    chunk_top_k: int = 20
    cosine_threshold: float = 0.2
    enable_llm_cache: bool = True
    enable_rerank: bool = True

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 100

    # Custom metadata (storage backends, etc.)
    custom_metadata: Dict = {}
```

### ResourceQuota

```python
@dataclass
class ResourceQuota:
    max_documents: int = 10000
    max_storage_gb: float = 100.0
    max_concurrent_queries: int = 10
    max_monthly_api_calls: int = 100000
    max_kb_per_tenant: int = 50
    max_entities_per_kb: int = 100000
    max_relationships_per_kb: int = 500000
```

### KnowledgeBase

```python
@dataclass
class KnowledgeBase:
    kb_id: str                          # UUID
    tenant_id: str                      # Parent tenant
    kb_name: str                        # Display name
    description: Optional[str]          # Description
    config: KBConfig                    # KB-specific config overrides
    is_active: bool = True              # Active status
    created_at: datetime                # Creation timestamp
    updated_at: datetime                # Last update
    created_by: Optional[str]           # Creator user ID

    # Statistics
    document_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    storage_size_mb: float = 0.0
```

---

## TenantContext

Request context carrying tenant/KB information:

```python
@dataclass
class TenantContext:
    tenant_id: str                      # Current tenant
    kb_id: Optional[str]                # Current KB (if scoped)
    user_id: str                        # Authenticated user
    role: Role                          # User's role
    permissions: List[Permission]       # Effective permissions

    def has_permission(self, permission: Permission) -> bool:
        """Check if context has specific permission."""
        return permission in self.permissions
```

---

## Request Headers

Multi-tenant requests require these headers:

```bash
# Authentication
Authorization: Bearer <jwt_token>

# Tenant context
X-Tenant-ID: <tenant_uuid>      # Required
X-KB-ID: <kb_uuid>              # Required for KB operations
```

### Example Request

```bash
curl -X POST "http://localhost:9621/api/v1/tenants/{tenant_id}/kbs/{kb_id}/documents/text" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -H "X-Tenant-ID: abc123-tenant-uuid" \
  -H "X-KB-ID: xyz789-kb-uuid" \
  -H "Content-Type: application/json" \
  -d '{"text": "Document content..."}'
```

---

## Security Best Practices

### 1. Enable Strict Mode

```bash
LIGHTRAG_MULTI_TENANT_STRICT=true
LIGHTRAG_REQUIRE_USER_AUTH=true
```

### 2. Use Strong JWT Secrets

```bash
TOKEN_SECRET=your-32-byte-cryptographic-secret
JWT_ALGORITHM=HS256
TOKEN_EXPIRE_HOURS=24
```

### 3. Limit Super Admins

```bash
LIGHTRAG_SUPER_ADMIN_USERS=admin@company.com
```

### 4. Audit Access

```python
# TenantService logs all access
logger.info(f"User {user_id} accessed tenant {tenant_id}")
logger.warning(f"Access denied: user={user_id} tenant={tenant_id}")
```

### 5. Resource Quotas

```python
quota = ResourceQuota(
    max_documents=5000,
    max_storage_gb=50.0,
    max_concurrent_queries=5,
    max_monthly_api_calls=50000
)
```

---

## Database Schema (PostgreSQL)

Multi-tenancy uses these tables:

```sql
-- Tenants table
CREATE TABLE tenants (
    tenant_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Knowledge bases table
CREATE TABLE knowledge_bases (
    kb_id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(tenant_id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tenant memberships
CREATE TABLE tenant_memberships (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(tenant_id),
    user_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(255),
    UNIQUE(tenant_id, user_id)
);

-- Create indexes
CREATE INDEX idx_kb_tenant ON knowledge_bases(tenant_id);
CREATE INDEX idx_membership_tenant ON tenant_memberships(tenant_id);
CREATE INDEX idx_membership_user ON tenant_memberships(user_id);
```

---

## Example: Complete Multi-Tenant Setup

```python
import asyncio
from lightrag import LightRAG
from lightrag.tenant_rag_manager import TenantRAGManager
from lightrag.services.tenant_service import TenantService
from lightrag.models.tenant import TenantConfig, Role

async def setup_multi_tenant():
    # 1. Initialize global components
    from lightrag.kg.postgres_impl import PGKVStorage

    kv_storage = PGKVStorage(
        namespace="system",
        global_config={"postgres_url": "postgresql://..."}
    )
    await kv_storage.initialize()

    # 2. Initialize tenant service
    tenant_service = TenantService(kv_storage)

    # 3. Create template RAG (for configuration inheritance)
    template_rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_name="gpt-4o-mini",
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="Neo4JStorage"
    )

    # 4. Initialize tenant manager
    manager = TenantRAGManager(
        base_working_dir="./rag_storage",
        tenant_service=tenant_service,
        template_rag=template_rag,
        max_cached_instances=100
    )

    # 5. Create tenant
    tenant = await tenant_service.create_tenant(
        tenant_name="Acme Corp",
        config=TenantConfig(
            llm_model="gpt-4o",
            top_k=50
        ),
        created_by="admin@acme.com"
    )

    # 6. Create knowledge base
    kb = await tenant_service.create_knowledge_base(
        tenant_id=tenant.tenant_id,
        kb_name="Product Docs",
        created_by="admin@acme.com"
    )

    # 7. Add user
    await tenant_service.add_user_to_tenant(
        user_id="user@acme.com",
        tenant_id=tenant.tenant_id,
        role="editor"
    )

    # 8. Get tenant-specific RAG instance
    rag = await manager.get_rag_instance(
        tenant_id=tenant.tenant_id,
        kb_id=kb.kb_id,
        user_id="user@acme.com"
    )

    # 9. Use normally
    await rag.ainsert("Product documentation content...")
    result = await rag.aquery("How do I use the product?")

    print(f"Answer: {result}")

    # 10. Cleanup
    await manager.cleanup_all()

asyncio.run(setup_multi_tenant())
```

---

**Related Documentation:**
- [Architecture Overview](0002-architecture-overview.md)
- [API Reference](0003-api-reference.md)
- [Configuration Reference](0007-configuration-reference.md)
- [Deployment Guide](0006-deployment-guide.md)

---

*Multi-tenancy feature developed by [Raphaël MANSUY](https://www.elitizon.com/) for LightRAG Enterprise Edition*
