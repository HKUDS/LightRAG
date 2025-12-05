# ADR 002: Implementation Strategy - Multi-Tenant, Multi-Knowledge-Base Architecture

## Status: Proposed

## Overview
This document provides a detailed, step-by-step implementation strategy for the multi-tenant, multi-knowledge-base (MT-MKB) architecture. It includes specific code changes, file modifications, new components, and testing strategies.

## Phase 1: Core Infrastructure (Weeks 1-3)

### 1.1 Database Schema Changes

#### Files to Create/Modify
- **New**: `lightrag/models/tenant.py` - Tenant and KnowledgeBase models
- **New**: `lightrag/models/__init__.py` - Model exports
- **Modify**: All storage implementations (PostgreSQL, Neo4j, MongoDB, etc.)

#### 1.1.1 Tenant and KnowledgeBase Models

**File**: `lightrag/models/tenant.py`
```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

@dataclass
class ResourceQuota:
    """Resource limits for a tenant"""
    max_documents: int = 10000
    max_storage_gb: float = 100.0
    max_concurrent_queries: int = 10
    max_monthly_api_calls: int = 100000
    max_kb_per_tenant: int = 50

@dataclass
class TenantConfig:
    """Per-tenant configuration for models and parameters"""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "bge-m3:latest"
    rerank_model: Optional[str] = None
    chunk_size: int = 1200
    chunk_overlap: int = 100
    top_k: int = 40
    cosine_threshold: float = 0.2
    enable_llm_cache: bool = True
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Tenant:
    """Tenant representation"""
    tenant_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_name: str = ""
    description: Optional[str] = None
    config: TenantConfig = field(default_factory=TenantConfig)
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeBase:
    """Knowledge Base representation"""
    kb_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = ""  # Foreign key to Tenant
    kb_name: str = ""
    description: Optional[str] = None
    is_active: bool = True
    doc_count: int = 0
    storage_used_mb: float = 0.0
    last_indexed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TenantContext:
    """Request-scoped tenant context"""
    tenant_id: str
    kb_id: str
    user_id: str
    role: str  # admin, editor, viewer
    permissions: Dict[str, bool] = field(default_factory=dict)

    @property
    def workspace_namespace(self) -> str:
        """Backward compatible workspace namespace"""
        return f"{self.tenant_id}_{self.kb_id}"
```

#### 1.1.2 PostgreSQL Schema Migration

**File**: `lightrag/kg/migrations/001_add_tenant_schema.sql`
```sql
-- Create tenants table
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_name VARCHAR(255) NOT NULL,
    description TEXT,
    llm_model VARCHAR(255) DEFAULT 'gpt-4o-mini',
    embedding_model VARCHAR(255) DEFAULT 'bge-m3:latest',
    rerank_model VARCHAR(255),
    chunk_size INTEGER DEFAULT 1200,
    chunk_overlap INTEGER DEFAULT 100,
    top_k INTEGER DEFAULT 40,
    cosine_threshold FLOAT DEFAULT 0.2,
    enable_llm_cache BOOLEAN DEFAULT TRUE,
    max_documents INTEGER DEFAULT 10000,
    max_storage_gb FLOAT DEFAULT 100.0,
    max_concurrent_queries INTEGER DEFAULT 10,
    max_monthly_api_calls INTEGER DEFAULT 100000,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

-- Create knowledge_bases table
CREATE TABLE IF NOT EXISTS knowledge_bases (
    kb_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    kb_name VARCHAR(255) NOT NULL,
    description TEXT,
    doc_count INTEGER DEFAULT 0,
    storage_used_mb FLOAT DEFAULT 0.0,
    is_active BOOLEAN DEFAULT TRUE,
    last_indexed_at TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    updated_by VARCHAR(255),
    UNIQUE(tenant_id, kb_name),
    INDEX idx_tenant_kb (tenant_id, kb_id)
);

-- Create api_keys table (for per-tenant API keys)
CREATE TABLE IF NOT EXISTS api_keys (
    api_key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    key_name VARCHAR(255) NOT NULL,
    hashed_key VARCHAR(255) NOT NULL UNIQUE,
    knowledge_base_ids UUID[] DEFAULT '{}',  -- NULL = all KBs
    permissions TEXT[] DEFAULT ARRAY['query', 'document:read'],
    is_active BOOLEAN DEFAULT TRUE,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    created_by VARCHAR(255)
);

-- Add tenant/kb columns to existing tables with defaults for backward compatibility
ALTER TABLE IF EXISTS kv_store_full_docs
ADD COLUMN IF NOT EXISTS tenant_id UUID DEFAULT NULL,
ADD COLUMN IF NOT EXISTS kb_id UUID DEFAULT NULL;

ALTER TABLE IF EXISTS kv_store_text_chunks
ADD COLUMN IF NOT EXISTS tenant_id UUID DEFAULT NULL,
ADD COLUMN IF NOT EXISTS kb_id UUID DEFAULT NULL;

ALTER TABLE IF EXISTS vector_store_entities
ADD COLUMN IF NOT EXISTS tenant_id UUID DEFAULT NULL,
ADD COLUMN IF NOT EXISTS kb_id UUID DEFAULT NULL;

-- Create indexes for tenant/kb filtering
CREATE INDEX IF NOT EXISTS idx_kv_store_tenant_kb ON kv_store_full_docs(tenant_id, kb_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_kb ON kv_store_text_chunks(tenant_id, kb_id);
CREATE INDEX IF NOT EXISTS idx_vectors_tenant_kb ON vector_store_entities(tenant_id, kb_id);
```

#### 1.1.3 MongoDB Schema

**File**: `lightrag/kg/migrations/mongo_001_add_tenant_collections.py`
```python
from typing import Any
import motor.motor_asyncio  # type: ignore

async def migrate_add_tenant_collections(client: motor.motor_asyncio.AsyncMotorClient):
    """Add tenant and knowledge base collections to MongoDB"""
    db = client.lightrag

    # Create tenants collection with schema validation
    await db.create_collection("tenants", validator={
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["tenant_id", "tenant_name", "created_at"],
            "properties": {
                "tenant_id": {"bsonType": "string"},
                "tenant_name": {"bsonType": "string"},
                "description": {"bsonType": "string"},
                "llm_model": {"bsonType": "string", "default": "gpt-4o-mini"},
                "embedding_model": {"bsonType": "string", "default": "bge-m3:latest"},
                "is_active": {"bsonType": "bool", "default": True},
                "metadata": {"bsonType": "object"},
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
            }
        }
    })

    # Create knowledge_bases collection
    await db.create_collection("knowledge_bases", validator={
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["kb_id", "tenant_id", "kb_name"],
            "properties": {
                "kb_id": {"bsonType": "string"},
                "tenant_id": {"bsonType": "string"},
                "kb_name": {"bsonType": "string"},
                "description": {"bsonType": "string"},
                "is_active": {"bsonType": "bool", "default": True},
                "metadata": {"bsonType": "object"},
                "created_at": {"bsonType": "date"},
            }
        }
    })

    # Create indexes
    await db.tenants.create_index("tenant_id", unique=True)
    await db.knowledge_bases.create_index([("tenant_id", 1), ("kb_id", 1)], unique=True)
    await db.knowledge_bases.create_index([("tenant_id", 1)])

    # Add tenant_id and kb_id indexes to existing collections
    for collection_name in ["documents", "chunks", "entities"]:
        col = db[collection_name]
        await col.create_index([("tenant_id", 1), ("kb_id", 1)])
```

### 1.2 Create Tenant Management Service

**File**: `lightrag/services/tenant_service.py`
```python
from typing import Optional, List, Dict, Any
from uuid import UUID
from lightrag.models.tenant import Tenant, KnowledgeBase, TenantContext, TenantConfig
from lightrag.base import BaseKVStorage

class TenantService:
    """Service for managing tenants and knowledge bases"""

    def __init__(self, kv_storage: BaseKVStorage):
        self.kv_storage = kv_storage
        self.tenant_namespace = "__tenants__"
        self.kb_namespace = "__knowledge_bases__"

    async def create_tenant(self, tenant_name: str, config: Optional[TenantConfig] = None) -> Tenant:
        """Create a new tenant"""
        tenant = Tenant(tenant_name=tenant_name, config=config or TenantConfig())
        await self.kv_storage.upsert({
            f"{self.tenant_namespace}:{tenant.tenant_id}": {
                "id": tenant.tenant_id,
                "name": tenant.tenant_name,
                "config": asdict(tenant.config),
                "quota": asdict(tenant.quota),
                "is_active": tenant.is_active,
                "created_at": tenant.created_at.isoformat(),
                "updated_at": tenant.updated_at.isoformat(),
            }
        })
        return tenant

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Retrieve a tenant by ID"""
        data = await self.kv_storage.get_by_id(f"{self.tenant_namespace}:{tenant_id}")
        if not data:
            return None
        return self._deserialize_tenant(data)

    async def create_knowledge_base(self, tenant_id: str, kb_name: str, description: Optional[str] = None) -> KnowledgeBase:
        """Create a new knowledge base for a tenant"""
        # Verify tenant exists
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        kb = KnowledgeBase(
            tenant_id=tenant_id,
            kb_name=kb_name,
            description=description
        )
        await self.kv_storage.upsert({
            f"{self.kb_namespace}:{tenant_id}:{kb.kb_id}": {
                "id": kb.kb_id,
                "tenant_id": kb.tenant_id,
                "kb_name": kb.kb_name,
                "description": kb.description,
                "is_active": kb.is_active,
                "created_at": kb.created_at.isoformat(),
            }
        })
        return kb

    async def list_knowledge_bases(self, tenant_id: str) -> List[KnowledgeBase]:
        """List all knowledge bases for a tenant"""
        # Implementation depends on storage backend
        pass

    def _deserialize_tenant(self, data: Dict[str, Any]) -> Tenant:
        """Convert stored data to Tenant object"""
        pass
```

### 1.3 Update Storage Base Classes

**File**: `lightrag/base.py` (Modifications)

Add tenant context to all StorageNameSpace classes:
```python
@dataclass
class StorageNameSpace(ABC):
    namespace: str
    workspace: str  # Keep for backward compatibility
    global_config: dict[str, Any]
    tenant_id: Optional[str] = None  # NEW
    kb_id: Optional[str] = None      # NEW

    async def initialize(self):
        """Initialize the storage"""
        pass

    # Helper method to build composite workspace key
    def _get_composite_workspace(self) -> str:
        """Build workspace key with tenant/kb isolation"""
        if self.tenant_id and self.kb_id:
            return f"{self.tenant_id}_{self.kb_id}"
        elif self.workspace:
            return self.workspace
        else:
            return "_"  # Default for backward compatibility
```

### 1.4 Update Storage Implementations

#### PostgreSQL Storage Update

**File**: `lightrag/kg/postgres_impl.py` (Key modifications)

```python
# Modify all queries to include tenant/kb filters
class PGKVStorage(BaseKVStorage):
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        # Add tenant/kb columns when upserting
        for key, value in data.items():
            if self.tenant_id and self.kb_id:
                value['tenant_id'] = self.tenant_id
                value['kb_id'] = self.kb_id

        # Original upsert logic with tenant/kb in WHERE clause
        # ... existing code ...

    async def query_with_tenant_filter(self, query: str) -> List[Any]:
        """Execute query with automatic tenant/kb filtering"""
        if self.tenant_id and self.kb_id:
            # Add WHERE clause filters
            if "WHERE" in query:
                query += f" AND tenant_id = $1 AND kb_id = $2"
            else:
                query += f" WHERE tenant_id = $1 AND kb_id = $2"
            return await self._execute(query, [self.tenant_id, self.kb_id])
        return await self._execute(query)

class PGVectorStorage(BaseVectorStorage):
    async def query(self, query: str, top_k: int, query_embedding: list[float] = None) -> list[dict[str, Any]]:
        # Add tenant/kb filtering
        sql = """
            SELECT * FROM vector_store_entities
            WHERE tenant_id = $1 AND kb_id = $2
            AND vector <-> $3 < $4
            ORDER BY vector <-> $3
            LIMIT $5
        """
        # Filter results by tenant/kb
        results = await self._execute(sql, [self.tenant_id, self.kb_id, query_embedding, threshold, top_k])
        return results
```

#### JSON Storage Update

**File**: `lightrag/kg/json_kv_impl.py` (Key modifications)

```python
@dataclass
class JsonKVStorage(BaseKVStorage):
    async def _get_file_path(self) -> str:
        """Get file path with tenant/kb isolation"""
        working_dir = self.global_config["working_dir"]

        # Build tenant/kb specific directory
        if self.tenant_id and self.kb_id:
            dir_path = os.path.join(working_dir, self.tenant_id, self.kb_id)
            file_name = f"kv_store_{self.namespace}.json"
        elif self.workspace:
            dir_path = os.path.join(working_dir, self.workspace)
            file_name = f"kv_store_{self.namespace}.json"
        else:
            dir_path = working_dir
            file_name = f"kv_store_{self.namespace}.json"

        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, file_name)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert with tenant/kb context"""
        # Add tenant/kb to metadata
        for key, value in data.items():
            if self.tenant_id:
                value['__tenant_id__'] = self.tenant_id
            if self.kb_id:
                value['__kb_id__'] = self.kb_id

        # Original upsert logic
        # ... existing code ...
```

## Phase 2: API Layer (Weeks 2-3)

### 2.1 Create Tenant-Aware Request Models

**File**: `lightrag/api/models/requests.py` (New)

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from uuid import UUID

class TenantRequest(BaseModel):
    """Base model for tenant-scoped requests"""
    tenant_id: str = Field(..., description="Tenant identifier")
    kb_id: str = Field(..., description="Knowledge base identifier")

class CreateTenantRequest(BaseModel):
    tenant_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None

class CreateKnowledgeBaseRequest(BaseModel):
    kb_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None

class DocumentAddRequest(TenantRequest):
    """Request to add documents to a knowledge base"""
    document_path: str = Field(..., description="Path to document")
    metadata: Optional[dict] = None

class QueryRequest(TenantRequest):
    """Request to query a knowledge base"""
    query: str = Field(..., min_length=3)
    mode: str = Field(default="mix", regex="local|global|hybrid|naive|mix|bypass")
    top_k: Optional[int] = None
    stream: Optional[bool] = None
```

### 2.2 Create Tenant-Aware Dependency Injection

**File**: `lightrag/api/dependencies.py` (New)

```python
from fastapi import Depends, HTTPException, status, Path, Header
from typing import Optional
from lightrag.models.tenant import TenantContext
from lightrag.services.tenant_service import TenantService
from lightrag.api.auth import validate_token, get_tenant_from_token

async def get_tenant_context(
    tenant_id: str = Path(..., description="Tenant ID"),
    kb_id: str = Path(..., description="Knowledge Base ID"),
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    tenant_service: TenantService = Depends(get_tenant_service),
) -> TenantContext:
    """
    Dependency to extract and validate tenant context from request.
    Verifies user has access to the specified tenant/KB.
    """

    # Determine authentication method
    if authorization and authorization.startswith("Bearer "):
        # JWT token authentication
        token = authorization[7:]
        try:
            token_data = await validate_token(token)
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid token")

        user_id = token_data.get("sub")
        token_tenant_id = token_data.get("tenant_id")

        # Verify user's tenant matches request tenant
        if token_tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied: tenant mismatch")

        # Verify user can access this KB
        accessible_kbs = token_data.get("knowledge_base_ids", [])
        if kb_id not in accessible_kbs and "*" not in accessible_kbs:
            raise HTTPException(status_code=403, detail="Access denied: KB not accessible")

    elif api_key:
        # API key authentication
        user_id = await validate_api_key(api_key, tenant_id, kb_id)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

    else:
        raise HTTPException(status_code=401, detail="Missing authentication")

    # Verify tenant and KB exist
    tenant = await tenant_service.get_tenant(tenant_id)
    if not tenant or not tenant.is_active:
        raise HTTPException(status_code=404, detail="Tenant not found")

    # Return validated context
    return TenantContext(
        tenant_id=tenant_id,
        kb_id=kb_id,
        user_id=user_id,
        role=token_data.get("role", "viewer"),
        permissions=token_data.get("permissions", {})
    )

async def get_tenant_service() -> TenantService:
    """Get singleton tenant service"""
    # This should be initialized at app startup
    pass
```

### 2.3 Create Tenant-Aware API Routes

**File**: `lightrag/api/routers/tenant_routes.py` (New)

```python
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from lightrag.api.models.requests import CreateTenantRequest, CreateKnowledgeBaseRequest
from lightrag.api.dependencies import get_tenant_context, get_tenant_service
from lightrag.models.tenant import TenantContext

router = APIRouter(prefix="/api/v1/tenants", tags=["tenants"])

@router.post("")
async def create_tenant(
    request: CreateTenantRequest,
    tenant_service = Depends(get_tenant_service),
) -> dict:
    """Create a new tenant"""
    tenant = await tenant_service.create_tenant(
        tenant_name=request.tenant_name,
        config=request.dict(exclude_none=True)
    )
    return {"status": "success", "data": tenant}

@router.get("/{tenant_id}")
async def get_tenant(
    tenant_context: TenantContext = Depends(get_tenant_context),
    tenant_service = Depends(get_tenant_service),
) -> dict:
    """Get tenant details"""
    tenant = await tenant_service.get_tenant(tenant_context.tenant_id)
    return {"status": "success", "data": tenant}

@router.post("/{tenant_id}/knowledge-bases")
async def create_knowledge_base(
    request: CreateKnowledgeBaseRequest,
    tenant_context: TenantContext = Depends(get_tenant_context),
    tenant_service = Depends(get_tenant_service),
) -> dict:
    """Create a knowledge base in a tenant"""
    kb = await tenant_service.create_knowledge_base(
        tenant_id=tenant_context.tenant_id,
        kb_name=request.kb_name,
        description=request.description
    )
    return {"status": "success", "data": kb}

@router.get("/{tenant_id}/knowledge-bases")
async def list_knowledge_bases(
    tenant_context: TenantContext = Depends(get_tenant_context),
    tenant_service = Depends(get_tenant_service),
) -> dict:
    """List all knowledge bases in a tenant"""
    kbs = await tenant_service.list_knowledge_bases(tenant_context.tenant_id)
    return {"status": "success", "data": kbs}
```

### 2.4 Update Query Routes for Multi-Tenancy

**File**: `lightrag/api/routers/query_routes.py` (Modifications)

```python
@router.post("/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/query")
async def query_knowledge_base(
    request: QueryRequest,
    tenant_context: TenantContext = Depends(get_tenant_context),
    rag_manager = Depends(get_rag_instance_manager),
) -> QueryResponse:
    """
    Query a specific knowledge base with tenant isolation.

    The request context is automatically scoped to the tenant/KB
    via dependency injection.
    """

    # Get tenant-specific RAG instance (with per-tenant config)
    rag = await rag_manager.get_rag_instance(
        tenant_id=tenant_context.tenant_id,
        kb_id=tenant_context.kb_id
    )

    # Execute query with tenant context
    result = await rag.aquery(
        query=request.query,
        param=QueryParam(mode=request.mode, top_k=request.top_k or 40),
        # Inject tenant context into query execution
        tenant_context=tenant_context
    )

    return QueryResponse(response=result["response"])
```

### 2.5 Update Document Routes for Multi-Tenancy

**File**: `lightrag/api/routers/document_routes.py` (Modifications)

```python
@router.post("/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/documents/add")
async def add_document(
    file: UploadFile = File(...),
    tenant_context: TenantContext = Depends(get_tenant_context),
    rag_manager = Depends(get_rag_instance_manager),
) -> dict:
    """
    Add a document to a specific knowledge base.

    Tenant/KB context is enforced through dependency injection.
    """

    # Get tenant-specific RAG instance
    rag = await rag_manager.get_rag_instance(
        tenant_id=tenant_context.tenant_id,
        kb_id=tenant_context.kb_id
    )

    # Insert document with tenant/KB context automatically
    result = await rag.ainsert(
        file_path=file.filename,
        tenant_id=tenant_context.tenant_id,
        kb_id=tenant_context.kb_id
    )

    return {"status": "success", "data": result}

@router.delete("/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    tenant_context: TenantContext = Depends(get_tenant_context),
    rag_manager = Depends(get_rag_instance_manager),
) -> dict:
    """Delete document with tenant isolation"""

    rag = await rag_manager.get_rag_instance(
        tenant_id=tenant_context.tenant_id,
        kb_id=tenant_context.kb_id
    )

    # Verify document belongs to this tenant/KB before deletion
    result = await rag.adelete_by_doc_id(
        doc_id=doc_id,
        tenant_id=tenant_context.tenant_id,
        kb_id=tenant_context.kb_id
    )

    return {"status": "success", "message": "Document deleted"}
```

## Phase 3: LightRAG Integration (Weeks 2-4)

### 3.1 Create Tenant-Aware LightRAG Instance Manager

**File**: `lightrag/tenant_rag_manager.py` (New)

```python
from typing import Dict, Optional, Tuple
from lightrag import LightRAG
from lightrag.models.tenant import TenantContext, TenantConfig
from lightrag.services.tenant_service import TenantService
import asyncio
from functools import lru_cache

class TenantRAGManager:
    """
    Manages LightRAG instances per tenant/KB combination.
    Handles caching, initialization, and cleanup of instances.
    """

    def __init__(
        self,
        base_working_dir: str,
        tenant_service: TenantService,
        max_cached_instances: int = 100,
    ):
        self.base_working_dir = base_working_dir
        self.tenant_service = tenant_service
        self.max_cached_instances = max_cached_instances
        self._instances: Dict[Tuple[str, str], LightRAG] = {}
        self._lock = asyncio.Lock()

    async def get_rag_instance(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> LightRAG:
        """
        Get or create a LightRAG instance for a tenant/KB combination.

        Instances are cached to avoid repeated initialization.
        Each instance uses a separate namespace for complete isolation.
        """
        cache_key = (tenant_id, kb_id)

        # Return cached instance if exists
        if cache_key in self._instances:
            instance = self._instances[cache_key]
            if instance._storages_status.value >= 1:  # INITIALIZED
                return instance

        async with self._lock:
            # Double-check locking pattern
            if cache_key in self._instances:
                return self._instances[cache_key]

            # Get tenant config
            tenant = await self.tenant_service.get_tenant(tenant_id)
            if not tenant:
                raise ValueError(f"Tenant {tenant_id} not found")

            # Create tenant-specific working directory
            tenant_working_dir = os.path.join(
                self.base_working_dir,
                tenant_id,
                kb_id
            )

            # Create LightRAG instance with tenant-specific config and workspace
            instance = LightRAG(
                working_dir=tenant_working_dir,
                workspace=f"{tenant_id}_{kb_id}",  # Backward compatible workspace
                # Use tenant-specific models and settings
                llm_model_name=tenant.config.llm_model,
                embedding_func=self._get_embedding_func(tenant),
                llm_model_func=self._get_llm_func(tenant),
                # ... other tenant-specific configurations ...
            )

            # Initialize storages
            await instance.initialize_storages()

            # Cache the instance
            if len(self._instances) >= self.max_cached_instances:
                # Evict oldest entry
                oldest_key = next(iter(self._instances))
                await self._instances[oldest_key].finalize_storages()
                del self._instances[oldest_key]

            self._instances[cache_key] = instance
            return instance

    async def cleanup_instance(self, tenant_id: str, kb_id: str) -> None:
        """Clean up and remove a cached instance"""
        cache_key = (tenant_id, kb_id)
        if cache_key in self._instances:
            await self._instances[cache_key].finalize_storages()
            del self._instances[cache_key]

    async def cleanup_all(self) -> None:
        """Clean up all cached instances"""
        for instance in self._instances.values():
            await instance.finalize_storages()
        self._instances.clear()

    def _get_embedding_func(self, tenant: TenantConfig):
        """Create embedding function with tenant-specific model"""
        # Use tenant's embedding model configuration
        # Can be overridden from global config
        pass

    def _get_llm_func(self, tenant: TenantConfig):
        """Create LLM function with tenant-specific model"""
        # Use tenant's LLM model configuration
        pass
```

### 3.2 Modify LightRAG Query Methods

**File**: `lightrag/lightrag.py` (Key modifications)

```python
async def aquery(
    self,
    query: str,
    param: QueryParam,
    tenant_context: Optional[TenantContext] = None,  # NEW
) -> QueryResult:
    """
    Query with optional tenant context for filtering.

    Args:
        query: The query string
        param: Query parameters
        tenant_context: Tenant context for data isolation (NEW)
    """

    # If tenant context provided, inject it into all storage operations
    if tenant_context:
        # Temporarily set tenant/kb context on storages
        original_tenant = getattr(self, '_tenant_id', None)
        original_kb = getattr(self, '_kb_id', None)

        self._tenant_id = tenant_context.tenant_id
        self._kb_id = tenant_context.kb_id

    try:
        # Existing query logic
        # All storage operations will now respect tenant/kb context
        result = await self._execute_query(query, param)
        return result
    finally:
        # Restore original context
        if tenant_context:
            self._tenant_id = original_tenant
            self._kb_id = original_kb

async def ainsert(
    self,
    file_path: str,
    tenant_id: Optional[str] = None,  # NEW
    kb_id: Optional[str] = None,      # NEW
    **kwargs,
) -> InsertionResult:
    """Insert documents with optional tenant/KB context"""

    if tenant_id:
        self._tenant_id = tenant_id
    if kb_id:
        self._kb_id = kb_id

    # Existing insertion logic
    # Documents will be stored with tenant/kb metadata
    result = await self._process_documents(file_path, **kwargs)
    return result
```

## Phase 4: Testing & Deployment (Week 4)

### 4.1 Unit Tests

**File**: `tests/test_tenant_isolation.py` (New)

```python
import pytest
from lightrag.models.tenant import Tenant, KnowledgeBase, TenantContext
from lightrag.services.tenant_service import TenantService

@pytest.mark.asyncio
class TestTenantIsolation:

    async def test_tenant_creation(self, tenant_service):
        """Test creating a tenant"""
        tenant = await tenant_service.create_tenant("Test Tenant")
        assert tenant.tenant_name == "Test Tenant"
        assert tenant.is_active is True

    async def test_knowledge_base_creation(self, tenant_service):
        """Test creating KB in a tenant"""
        tenant = await tenant_service.create_tenant("Tenant 1")
        kb = await tenant_service.create_knowledge_base(
            tenant.tenant_id,
            "KB 1"
        )
        assert kb.tenant_id == tenant.tenant_id

    async def test_cross_tenant_data_isolation(self, tenant_service, rag_manager):
        """Test that data from one tenant cannot be accessed by another"""
        # Create two tenants
        tenant1 = await tenant_service.create_tenant("Tenant 1")
        tenant2 = await tenant_service.create_tenant("Tenant 2")

        # Create KBs
        kb1 = await tenant_service.create_knowledge_base(tenant1.tenant_id, "KB1")
        kb2 = await tenant_service.create_knowledge_base(tenant2.tenant_id, "KB2")

        # Add documents to each KB
        rag1 = await rag_manager.get_rag_instance(tenant1.tenant_id, kb1.kb_id)
        rag2 = await rag_manager.get_rag_instance(tenant2.tenant_id, kb2.kb_id)

        # Verify documents are isolated
        # Query in tenant2 should not return documents from tenant1
        pass

    async def test_query_with_tenant_context(self, rag_manager):
        """Test queries include tenant context"""
        context = TenantContext(
            tenant_id="tenant1",
            kb_id="kb1",
            user_id="user1",
            role="admin"
        )
        # Execute query with context
        # Verify only tenant1/kb1 data returned
        pass
```

### 4.2 Integration Tests

**File**: `tests/test_api_tenant_routes.py` (New)

```python
import pytest
from fastapi.testclient import TestClient

@pytest.mark.asyncio
class TestTenantAPIs:

    async def test_create_tenant_endpoint(self, client: TestClient, auth_token):
        """Test POST /api/v1/tenants"""
        response = client.post(
            "/api/v1/tenants",
            json={"tenant_name": "New Tenant"},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert "tenant_id" in data["data"]

    async def test_create_knowledge_base_endpoint(self, client: TestClient, tenant_id, auth_token):
        """Test POST /api/v1/tenants/{tenant_id}/knowledge-bases"""
        response = client.post(
            f"/api/v1/tenants/{tenant_id}/knowledge-bases",
            json={"kb_name": "KB 1"},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 201
        data = response.json()
        assert "kb_id" in data["data"]

    async def test_cross_tenant_access_denied(self, client: TestClient, tenant1_token, tenant2_id):
        """Test accessing tenant2 with tenant1 token fails"""
        response = client.get(
            f"/api/v1/tenants/{tenant2_id}",
            headers={"Authorization": f"Bearer {tenant1_token}"}
        )
        assert response.status_code == 403

    async def test_query_with_tenant_isolation(self, client: TestClient, tenant_id, kb_id, auth_token):
        """Test query is isolated to tenant/KB"""
        # Add document to KB
        # Query should only search that KB
        pass
```

### 4.3 Migration Script

**File**: `scripts/migrate_workspace_to_tenant.py` (New)

```python
"""
Migration script to convert existing workspaces to multi-tenant architecture.
Creates a default tenant for each workspace.
"""

import asyncio
import argparse
from lightrag.services.tenant_service import TenantService
from lightrag.models.tenant import Tenant
import uuid

async def migrate_workspaces_to_tenants(
    working_dir: str,
    storage_config: dict
):
    """
    Migrate existing workspace-based deployments to multi-tenant.

    For each workspace directory:
    1. Create a tenant with that workspace name
    2. Create a default KB
    3. Map workspace data to tenant/KB
    """

    tenant_service = TenantService(storage_config)

    # Scan working directory for existing workspaces
    workspaces = []  # Get from directory structure

    for workspace_name in workspaces:
        print(f"Migrating workspace: {workspace_name}")

        # Create tenant from workspace
        tenant = await tenant_service.create_tenant(
            tenant_name=workspace_name or "default",
            metadata={"migrated_from_workspace": workspace_name}
        )

        # Create default KB
        kb = await tenant_service.create_knowledge_base(
            tenant.tenant_id,
            kb_name="default",
            description="Default knowledge base (migrated from workspace)"
        )

        # Migrate data from workspace files to tenant/KB storage
        # Update storage paths and metadata

        print(f"  ✓ Created tenant {tenant.tenant_id}")
        print(f"  ✓ Created KB {kb.kb_id}")

    print("\nMigration complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate workspaces to multi-tenant")
    parser.add_argument("--working-dir", required=True)
    args = parser.parse_args()

    asyncio.run(migrate_workspaces_to_tenants(args.working_dir, {}))
```

### 4.4 Deployment Checklist

```markdown
## Pre-Deployment Checklist

### Database & Schema
- [ ] Database migration scripts tested on staging
- [ ] Backup of production database created
- [ ] Index creation verified on prod-like data volume
- [ ] Schema rollback scripts prepared

### Code Changes
- [ ] All unit tests passing (100% coverage of new code)
- [ ] Integration tests passing
- [ ] Load testing completed (1000+ tenant/KB combinations)
- [ ] Security audit completed
- [ ] Code review approved by 2+ team members

### Documentation
- [ ] API documentation updated
- [ ] Migration guide prepared
- [ ] Tenant management guide written
- [ ] Troubleshooting guide created

### Deployment
- [ ] Feature flag to enable multi-tenancy (default: off)
- [ ] Gradual rollout: 10% → 50% → 100%
- [ ] Health checks monitor tenant isolation
- [ ] Rollback plan tested
- [ ] Team trained on new architecture
- [ ] On-call engineer assigned for release window

### Post-Deployment
- [ ] Monitor error rates and latency
- [ ] Verify tenant data isolation (spot checks)
- [ ] Collect feedback from early adopters
- [ ] Performance baseline established
```

## Configuration Examples

### Environment Variables

```bash
# Tenant Manager Configuration
TENANT_ENABLED=true
MAX_CACHED_INSTANCES=100
TENANT_CONFIG_SYNC_INTERVAL=300

# Storage Configuration (remains the same)
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage

# Tenant Service Configuration
TENANT_SERVICE_STORAGE=PostgreSQL
TENANT_DB_HOST=localhost
TENANT_DB_PORT=5432
TENANT_DB_NAME=lightrag_tenants
```

### Python Configuration

```python
# In config.py or app initialization
class TenantConfig:
    ENABLED = os.getenv("TENANT_ENABLED", "false").lower() == "true"
    MAX_CACHED_INSTANCES = int(os.getenv("MAX_CACHED_INSTANCES", "100"))
    SYNC_INTERVAL = int(os.getenv("TENANT_CONFIG_SYNC_INTERVAL", "300"))

    # Storage for tenant metadata
    STORAGE_TYPE = os.getenv("TENANT_SERVICE_STORAGE", "PostgreSQL")
    STORAGE_CONFIG = {
        "host": os.getenv("TENANT_DB_HOST"),
        "port": int(os.getenv("TENANT_DB_PORT", "5432")),
        "database": os.getenv("TENANT_DB_NAME", "lightrag_tenants"),
    }
```

## Testing Strategy

### Unit Testing (40% of tests)
- Tenant service operations
- Storage isolation logic
- Configuration management
- Authentication/authorization

### Integration Testing (40% of tests)
- API endpoint functionality
- Cross-component data flow
- Tenant context propagation
- Error handling

### System Testing (20% of tests)
- End-to-end workflows per tenant
- Multi-tenant concurrent operations
- Resource quota enforcement
- Performance under load

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Query latency | <10ms overhead | Per query with/without tenant filtering |
| API response time | <200ms p99 | Single query endpoint |
| Storage overhead | <3% | Per-tenant metadata vs. data |
| Memory per instance | <500MB | Per cached LightRAG instance |
| Tenant isolation overhead | <15% | Compare to single-tenant baseline |

## Known Limitations & Future Work

### Phase 1 Limitations
1. No cross-tenant queries or data sharing
2. No tenant-to-tenant access delegation
3. No per-tenant storage encryption
4. No real-time multi-region replication
5. No automatic tenant data backup management

### Future Enhancements (Phase 2)
1. **Cross-tenant sharing**: Allow tenants to share specific KB data
2. **Advanced RBAC**: Support custom roles and fine-grained permissions
3. **Encryption at rest**: Per-tenant data encryption
4. **Audit logging**: Comprehensive audit trail with retention policies
5. **Multi-region**: Replicate tenant data across regions
6. **Tenant quotas**: Storage, API call, and compute quotas with enforcement
7. **SSO integration**: Enterprise SSO (SAML, OIDC) support

---

**Document Version**: 1.0
**Last Updated**: 2025-11-20
**Phase Duration**: 3-4 weeks
**Estimated Effort**: 160 developer hours
**Team Size**: 2-3 backend engineers
