"""Data models for tenant, knowledge base, and related configurations in LightRAG."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4
from enum import Enum


class Role(str, Enum):
    """User roles in the multi-tenant system."""
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    VIEWER_READONLY = "viewer:read-only"


class Permission(str, Enum):
    """Permissions in the multi-tenant system."""
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


# Role-to-permissions mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [p.value for p in Permission],
    Role.EDITOR: [
        Permission.CREATE_KB.value,
        Permission.DELETE_KB.value,
        Permission.CREATE_DOCUMENT.value,
        Permission.UPDATE_DOCUMENT.value,
        Permission.DELETE_DOCUMENT.value,
        Permission.READ_DOCUMENT.value,
        Permission.RUN_QUERY.value,
        Permission.ACCESS_KB.value,
    ],
    Role.VIEWER: [
        Permission.READ_DOCUMENT.value,
        Permission.RUN_QUERY.value,
        Permission.ACCESS_KB.value,
    ],
    Role.VIEWER_READONLY: [
        Permission.RUN_QUERY.value,
        Permission.ACCESS_KB.value,
    ],
}


@dataclass
class ResourceQuota:
    """Resource limits for a tenant."""
    max_documents: int = 10000
    max_storage_gb: float = 100.0
    max_concurrent_queries: int = 10
    max_monthly_api_calls: int = 100000
    max_kb_per_tenant: int = 50
    max_entities_per_kb: int = 100000
    max_relationships_per_kb: int = 500000


@dataclass
class TenantConfig:
    """Per-tenant configuration for models and parameters."""
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
    """Per-knowledge-base configuration (overrides tenant defaults)."""
    # Only include fields that override tenant config
    top_k: Optional[int] = None
    chunk_size: Optional[int] = None
    cosine_threshold: Optional[float] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tenant:
    """Represents a tenant in the multi-tenant system."""
    tenant_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_name: str = ""
    description: Optional[str] = None
    
    # Configuration
    config: TenantConfig = field(default_factory=TenantConfig)
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    
    # Lifecycle
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    kb_count: int = 0
    total_documents: int = 0
    total_storage_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tenant_id": self.tenant_id,
            "tenant_name": self.tenant_name,
            "description": self.description,
            "config": {
                "llm_model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
                "rerank_model": self.config.rerank_model,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "top_k": self.config.top_k,
                "cosine_threshold": self.config.cosine_threshold,
                "enable_llm_cache": self.config.enable_llm_cache,
                "custom_metadata": self.config.custom_metadata,
            },
            "quota": {
                "max_documents": self.quota.max_documents,
                "max_storage_gb": self.quota.max_storage_gb,
                "max_concurrent_queries": self.quota.max_concurrent_queries,
                "max_monthly_api_calls": self.quota.max_monthly_api_calls,
                "max_kb_per_tenant": self.quota.max_kb_per_tenant,
            },
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "metadata": self.metadata,
            "kb_count": self.kb_count,
            "total_documents": self.total_documents,
            "total_storage_mb": self.total_storage_mb,
        }


@dataclass
class KnowledgeBase:
    """Represents a knowledge base within a tenant."""
    kb_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = ""
    kb_name: str = ""
    description: Optional[str] = None
    
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
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "kb_id": self.kb_id,
            "tenant_id": self.tenant_id,
            "kb_name": self.kb_name,
            "description": self.description,
            "is_active": self.is_active,
            "status": self.status,
            "document_count": self.document_count,
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
            "chunk_count": self.chunk_count,
            "storage_used_mb": self.storage_used_mb,
            "last_indexed_at": self.last_indexed_at.isoformat() if self.last_indexed_at else None,
            "index_version": self.index_version,
            "config": self.config.__dict__ if self.config else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "metadata": self.metadata,
        }


@dataclass
class TenantContext:
    """Request-scoped tenant context injected into all request handlers."""
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
        """Backward compatible workspace namespace."""
        return f"{self.tenant_id}_{self.kb_id}"
    
    def can_access_kb(self, kb_id: str) -> bool:
        """Check if user can access specific KB."""
        return kb_id in self.knowledge_base_ids or "*" in self.knowledge_base_ids
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return self.permissions.get(permission, False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tenant_id": self.tenant_id,
            "kb_id": self.kb_id,
            "user_id": self.user_id,
            "role": self.role,
            "permissions": self.permissions,
            "knowledge_base_ids": self.knowledge_base_ids,
            "request_id": self.request_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "workspace_namespace": self.workspace_namespace,
        }
