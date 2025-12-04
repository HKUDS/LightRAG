# ADR 003: Data Models and Storage Design

## Status: Proposed

## Overview
This document details the data models for tenants, knowledge bases, and the storage architecture for complete data isolation.

## Data Models

### 1. Core Entity Models

#### 1.1 Tenant Model
```python
@dataclass
class Tenant:
    """
    Represents a tenant in the multi-tenant system.
    A tenant is the top-level isolation boundary.
    """
    tenant_id: str  # UUID: e.g., "550e8400-e29b-41d4-a716-446655440000"
    tenant_name: str  # Display name: e.g., "Acme Corp"
    description: Optional[str]  # Free-text description
    
    # Configuration
    config: TenantConfig
    quota: ResourceQuota
    
    # Lifecycle
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]
    updated_by: Optional[str]
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    kb_count: int = 0
    total_documents: int = 0
    total_storage_mb: float = 0.0
```

#### 1.2 Knowledge Base Model
```python
@dataclass
class KnowledgeBase:
    """
    Represents a knowledge base within a tenant.
    Contains documents, entities, and relationships for a specific domain.
    """
    kb_id: str  # UUID: e.g., "660e8400-e29b-41d4-a716-446655440000"
    tenant_id: str  # Foreign key to Tenant
    kb_name: str  # Display name: e.g., "Product Documentation"
    description: Optional[str]
    
    # Status and lifecycle
    is_active: bool = True
    status: str = "ready"  # ready | indexing | error
    
    # Statistics
    document_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    chunk_count: int = 0
    storage_used_mb: float = 0.0
    
    # Indexing info
    last_indexed_at: Optional[datetime] = None
    index_version: int = 1
    
    # Configuration (can override tenant defaults)
    config: Optional[KBConfig] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### 1.3 Configuration Models
```python
@dataclass
class TenantConfig:
    """Per-tenant model and parameter configuration"""
    # Model selection
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "bge-m3:latest"
    rerank_model: Optional[str] = None
    
    # LLM parameters
    llm_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    llm_temperature: float = 1.0
    llm_max_tokens: int = 4096
    
    # Embedding parameters
    embedding_dim: int = 1024
    embedding_batch_num: int = 10
    
    # Query defaults
    top_k: int = 40
    chunk_top_k: int = 20
    cosine_threshold: float = 0.2
    enable_llm_cache: bool = True
    enable_rerank: bool = True
    
    # Chunking defaults
    chunk_size: int = 1200
    chunk_overlap: int = 100
    
    # Custom tenant metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KBConfig:
    """Per-knowledge-base configuration (overrides tenant defaults)"""
    # Only include fields that override tenant config
    top_k: Optional[int] = None
    chunk_size: Optional[int] = None
    cosine_threshold: Optional[float] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResourceQuota:
    """Resource limits for a tenant"""
    max_documents: int = 10000
    max_storage_gb: float = 100.0
    max_concurrent_queries: int = 10
    max_monthly_api_calls: int = 100000
    max_kb_per_tenant: int = 50
    max_entities_per_kb: int = 100000
    max_relationships_per_kb: int = 500000
```

#### 1.4 Request Context
```python
@dataclass
class TenantContext:
    """
    Request-scoped tenant context.
    Injected into all request handlers and passed through the call stack.
    """
    tenant_id: str
    kb_id: str
    user_id: str
    role: str  # admin | editor | viewer | viewer:read-only
    
    # Authorization
    permissions: Dict[str, bool] = field(default_factory=dict)
    knowledge_base_ids: List[str] = field(default_factory=list)  # Accessible KBs
    
    # Request tracking
    request_id: str = field(default_factory=lambda: str(uuid4()))
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Computed properties
    @property
    def workspace_namespace(self) -> str:
        """Backward compatible workspace namespace"""
        return f"{self.tenant_id}_{self.kb_id}"
    
    def can_access_kb(self, kb_id: str) -> bool:
        """Check if user can access specific KB"""
        return kb_id in self.knowledge_base_ids or "*" in self.knowledge_base_ids
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return self.permissions.get(permission, False)
```

## Storage Architecture

### 2. Storage Isolation Strategy

#### 2.1 Composite Key Design
All data items are identified using composite keys that enforce tenant/KB isolation:

```
<tenant_id>:<kb_id>:<entity_id>
```

**Examples**:
- Document: `acme:prod-docs:doc-12345`
- Entity: `acme:prod-docs:ent-company-apple`
- Chunk: `acme:prod-docs:chunk-doc-12345-001`
- Relationship: `acme:prod-docs:rel-apple-ceo-tim_cook`

#### 2.2 Storage-Specific Implementation

### 2.3 PostgreSQL Storage

#### Schema Design
```sql
-- Tenants table
CREATE TABLE tenants (
    tenant_id UUID PRIMARY KEY,
    tenant_name VARCHAR(255) NOT NULL,
    description TEXT,
    llm_model VARCHAR(255) DEFAULT 'gpt-4o-mini',
    embedding_model VARCHAR(255) DEFAULT 'bge-m3:latest',
    rerank_model VARCHAR(255),
    chunk_size INTEGER DEFAULT 1200,
    chunk_overlap INTEGER DEFAULT 100,
    top_k INTEGER DEFAULT 40,
    cosine_threshold FLOAT DEFAULT 0.2,
    max_documents INTEGER DEFAULT 10000,
    max_storage_gb FLOAT DEFAULT 100.0,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    CONSTRAINT valid_tenant_name CHECK (length(tenant_name) > 0)
);

-- Knowledge bases table
CREATE TABLE knowledge_bases (
    kb_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    kb_name VARCHAR(255) NOT NULL,
    description TEXT,
    doc_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,
    relationship_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    storage_used_mb FLOAT DEFAULT 0.0,
    is_active BOOLEAN DEFAULT TRUE,
    status VARCHAR(50) DEFAULT 'ready',
    last_indexed_at TIMESTAMP,
    index_version INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    UNIQUE(tenant_id, kb_name),
    CONSTRAINT valid_kb_name CHECK (length(kb_name) > 0)
);

-- Documents table (updated with tenant/kb)
CREATE TABLE documents (
    doc_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id),
    kb_id UUID NOT NULL REFERENCES knowledge_bases(kb_id),
    doc_name VARCHAR(255) NOT NULL,
    doc_path TEXT,
    file_type VARCHAR(50),
    file_size INTEGER,
    chunk_count INTEGER DEFAULT 0,
    content_hash VARCHAR(64),  -- SHA256 for deduplication
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    CONSTRAINT fk_tenant_kb UNIQUE (tenant_id, kb_id, doc_id)
);

-- Chunks table (text chunks with tenant/kb filtering)
CREATE TABLE chunks (
    chunk_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id),
    kb_id UUID NOT NULL REFERENCES knowledge_bases(kb_id),
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER,
    content TEXT NOT NULL,
    token_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_tenant_kb_chunk UNIQUE (tenant_id, kb_id, chunk_id)
);

-- Entities table (knowledge graph entities)
CREATE TABLE entities (
    entity_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id),
    kb_id UUID NOT NULL REFERENCES knowledge_bases(kb_id),
    entity_name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100),
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_tenant_kb_entity UNIQUE (tenant_id, kb_id, entity_id)
);

-- Relationships table (knowledge graph relationships)
CREATE TABLE relationships (
    rel_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id),
    kb_id UUID NOT NULL REFERENCES knowledge_bases(kb_id),
    source_entity_id UUID NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    relation_type VARCHAR(100) NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_tenant_kb_rel UNIQUE (tenant_id, kb_id, rel_id)
);

-- Vector embeddings table
CREATE TABLE vector_embeddings (
    vector_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id),
    kb_id UUID NOT NULL REFERENCES knowledge_bases(kb_id),
    entity_id UUID NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    embedding vector(1024),  -- pgvector extension required
    embedding_model VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_tenant_kb_vector UNIQUE (tenant_id, kb_id, vector_id)
);

-- Create indexes for tenant/kb filtering on all tables
CREATE INDEX idx_documents_tenant_kb ON documents(tenant_id, kb_id);
CREATE INDEX idx_chunks_tenant_kb ON chunks(tenant_id, kb_id, doc_id);
CREATE INDEX idx_entities_tenant_kb ON entities(tenant_id, kb_id);
CREATE INDEX idx_relationships_tenant_kb ON relationships(tenant_id, kb_id);
CREATE INDEX idx_vectors_tenant_kb ON vector_embeddings(tenant_id, kb_id);

-- Full-text search index
CREATE INDEX idx_chunks_fts ON chunks USING GIN(to_tsvector('english', content));

-- Composite indexes for common queries
CREATE INDEX idx_docs_tenant_active ON documents(tenant_id, kb_id, is_active);
CREATE INDEX idx_entities_tenant_type ON entities(tenant_id, kb_id, entity_type);
CREATE INDEX idx_rel_tenant_source ON relationships(tenant_id, kb_id, source_entity_id);
```

#### Query Examples

```sql
-- Get all documents for a tenant/KB
SELECT * FROM documents 
WHERE tenant_id = $1 AND kb_id = $2 AND is_active = true;

-- Get all chunks for a document (with tenant isolation)
SELECT * FROM chunks 
WHERE tenant_id = $1 AND kb_id = $2 AND doc_id = $3
ORDER BY chunk_index;

-- Search entities by name and type (tenant-scoped)
SELECT * FROM entities 
WHERE tenant_id = $1 AND kb_id = $2 
AND entity_name ILIKE '%' || $3 || '%'
AND entity_type = $4;

-- Find related chunks for an entity (tenant-scoped)
SELECT DISTINCT c.* FROM chunks c
WHERE c.tenant_id = $1 AND c.kb_id = $2
AND c.chunk_id IN (
    SELECT chunk_id FROM chunk_entity_links
    WHERE tenant_id = $1 AND kb_id = $2
    AND entity_id = $3
);
```

### 2.4 Neo4j Storage

#### Schema Design
```cypher
// Tenant node
CREATE CONSTRAINT unique_tenant_id IF NOT EXISTS
  FOR (t:Tenant) REQUIRE t.tenant_id IS UNIQUE;

// Knowledge base node
CREATE CONSTRAINT unique_kb_id IF NOT EXISTS
  FOR (k:KnowledgeBase) REQUIRE k.kb_id IS UNIQUE;

// Entity node with tenant/kb scope
CREATE CONSTRAINT unique_entity IF NOT EXISTS
  FOR (e:Entity) REQUIRE (e.tenant_id, e.kb_id, e.entity_id) IS UNIQUE;

// Create nodes with tenant/kb properties
CREATE (t:Tenant {
  tenant_id: 'tenant-uuid',
  tenant_name: 'Acme Corp',
  created_at: timestamp()
});

CREATE (kb:KnowledgeBase {
  kb_id: 'kb-uuid',
  tenant_id: 'tenant-uuid',
  kb_name: 'Product Docs',
  created_at: timestamp()
}) -[:BELONGS_TO]-> (t:Tenant {tenant_id: 'tenant-uuid'});

// Entity with tenant/kb scope
CREATE (e:Entity {
  entity_id: 'entity-uuid',
  tenant_id: 'tenant-uuid',
  kb_id: 'kb-uuid',
  name: 'Apple Inc',
  type: 'Organization'
}) -[:IN_KB]-> (kb:KnowledgeBase {kb_id: 'kb-uuid'});
```

#### Query Examples
```cypher
// Get all entities in a KB
MATCH (e:Entity {tenant_id: $tenant_id, kb_id: $kb_id})
RETURN e;

// Get entities connected to another entity (tenant-scoped)
MATCH (e1:Entity {tenant_id: $tenant_id, kb_id: $kb_id, entity_id: $entity_id})
-[r:RELATES_TO]-
(e2:Entity {tenant_id: $tenant_id, kb_id: $kb_id})
RETURN e1, r, e2;

// Prevent cross-tenant queries
MATCH (e:Entity)
WHERE e.tenant_id = $tenant_id AND e.kb_id = $kb_id
RETURN e;

// Enforce scope in relationship queries
MATCH (e1:Entity {tenant_id: $tenant_id, kb_id: $kb_id})
-[r:RELATES_TO]->
(e2:Entity {tenant_id: $tenant_id, kb_id: $kb_id})
RETURN e1, r, e2;
```

### 2.5 Vector Database Storage (Milvus/Qdrant)

#### Collection Schema
```python
# Milvus collection
collection_schema = {
    "fields": [
        {"name": "id", "type": "VARCHAR", "params": {"max_length": 512}},
        {"name": "tenant_id", "type": "VARCHAR", "params": {"max_length": 36}},
        {"name": "kb_id", "type": "VARCHAR", "params": {"max_length": 36}},
        {"name": "entity_id", "type": "VARCHAR", "params": {"max_length": 512}},
        {"name": "entity_type", "type": "VARCHAR", "params": {"max_length": 100}},
        {"name": "embedding", "type": "FLOAT_VECTOR", "params": {"dim": 1024}},
        {"name": "text", "type": "VARCHAR", "params": {"max_length": 4096}},
        {"name": "metadata", "type": "JSON"},
        {"name": "created_at", "type": "INT64"},
    ],
    "primary_field": "id",
    "vector_field": "embedding"
}

# Create index with tenant/kb partitioning
index_params = {
    "metric_type": "L2",  # or "IP" for inner product
    "index_type": "HNSW",
    "params": {"efConstruction": 200, "M": 16}
}

# Partition by tenant for better performance
collection.create_partition(partition_name=f"{tenant_id}_{kb_id}")
```

#### Query Examples
```python
# Search with tenant/kb filter
expr = f'tenant_id == "{tenant_id}" AND kb_id == "{kb_id}"'
results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"ef": 100}},
    limit=10,
    expr=expr,
    output_fields=["entity_id", "text", "metadata"]
)

# Prevent cross-tenant queries
# Always include tenant/kb filter in expr
```

## Access Control Lists (ACL)

### 3.1 Role Definitions

```python
class Role(str, Enum):
    ADMIN = "admin"           # Full control
    EDITOR = "editor"         # Create/update/delete documents and KBs
    VIEWER = "viewer"         # Query and read-only access
    VIEWER_READONLY = "viewer:read-only"  # Query access only

class Permission(str, Enum):
    # Tenant-level permissions
    MANAGE_TENANT = "tenant:manage"
    MANAGE_MEMBERS = "tenant:manage_members"
    MANAGE_BILLING = "tenant:manage_billing"
    
    # KB-level permissions
    CREATE_KB = "kb:create"
    DELETE_KB = "kb:delete"
    MANAGE_KB = "kb:manage"
    
    # Document-level permissions
    CREATE_DOCUMENT = "document:create"
    UPDATE_DOCUMENT = "document:update"
    DELETE_DOCUMENT = "document:delete"
    READ_DOCUMENT = "document:read"
    
    # Query permissions
    RUN_QUERY = "query:run"
    ACCESS_KB = "kb:access"

ROLE_PERMISSIONS = {
    Role.ADMIN: [Permission.value for Permission in Permission],
    Role.EDITOR: [
        Permission.CREATE_KB,
        Permission.DELETE_KB,
        Permission.CREATE_DOCUMENT,
        Permission.UPDATE_DOCUMENT,
        Permission.DELETE_DOCUMENT,
        Permission.READ_DOCUMENT,
        Permission.RUN_QUERY,
        Permission.ACCESS_KB,
    ],
    Role.VIEWER: [
        Permission.READ_DOCUMENT,
        Permission.RUN_QUERY,
        Permission.ACCESS_KB,
    ],
    Role.VIEWER_READONLY: [
        Permission.RUN_QUERY,
        Permission.ACCESS_KB,
    ],
}
```

### 3.2 JWT Token Payload with Permissions

```python
{
    "sub": "user-123",
    "tenant_id": "acme-corp",
    "knowledge_base_ids": ["kb-1", "kb-2"],  # Accessible KBs
    "role": "admin",  # or editor, viewer
    "permissions": {
        "kb:create": true,
        "kb:delete": true,
        "document:create": true,
        "query:run": true,
        ...
    },
    "exp": 1703123456,
    "iat": 1703100000,
    "iss": "lightrag-server",
    "metadata": {
        "department": "engineering",
        "cost_center": "cc-123"
    }
}
```

## Backward Compatibility

### 4.1 Legacy Workspace to Tenant Migration

For existing single-workspace deployments:

1. **Auto-create tenant on startup** if not exists:
   ```python
   async def initialize_tenant_from_workspace(workspace: str) -> Tenant:
       """Create tenant from legacy workspace name"""
       tenant_id = workspace if workspace else "default"
       tenant = Tenant(
           tenant_id=tenant_id,
           tenant_name=workspace or "default",
           metadata={"legacy_workspace": True}
       )
       return tenant
   ```

2. **Transparent workspace → tenant mapping**:
   ```python
   def get_workspace_namespace(tenant_id: str, kb_id: str) -> str:
       """Backward compatible workspace string"""
       return f"{tenant_id}_{kb_id}"
   ```

3. **Migration script** provided to convert existing data

## Data Validation & Constraints

### 5.1 Validation Rules

```python
class TenantValidator:
    @staticmethod
    def validate_tenant_id(tenant_id: str) -> bool:
        """Validate tenant ID format (UUID)"""
        return bool(UUID(tenant_id))
    
    @staticmethod
    def validate_tenant_name(name: str) -> bool:
        """Validate tenant name"""
        return 1 <= len(name) <= 255

class KBValidator:
    @staticmethod
    def validate_kb_id(kb_id: str) -> bool:
        """Validate KB ID format"""
        return bool(UUID(kb_id))
    
    @staticmethod
    def validate_kb_name(name: str, tenant_id: str) -> bool:
        """Validate KB name is unique within tenant"""
        # Check with database
        pass

class EntityValidator:
    @staticmethod
    def validate_entity_id(entity_id: str, tenant_id: str, kb_id: str) -> bool:
        """Validate entity belongs to tenant/KB"""
        # Parse composite key
        parts = entity_id.split(':')
        return len(parts) == 3 and parts[0] == tenant_id and parts[1] == kb_id
```

## Summary Table

| Component | Single-Tenant | Multi-Tenant |
|-----------|---------------|--------------|
| **Isolation Boundary** | Workspace | Tenant + KB |
| **Data Sharing** | N/A | Cross-KB within tenant possible |
| **Configuration** | Global | Per-tenant + per-KB |
| **Storage Model** | Shared | Tenant-scoped queries |
| **Authentication** | Simple JWT | Tenant-aware JWT |
| **Complexity** | Low | Medium |
| **Performance** | Baseline | +5-10% overhead |

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-20  
**Related Files**: 002-implementation-strategy.md, 004-api-design.md
