"""
API request and response models for multi-tenant LightRAG.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


# Tenant Management Models


class TenantConfigRequest(BaseModel):
    """Request model for tenant configuration."""

    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    rerank_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k: Optional[int] = None
    cosine_threshold: Optional[float] = None


class CreateTenantRequest(BaseModel):
    """Request to create a new tenant."""

    tenant_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    config: Optional[TenantConfigRequest] = None


class UpdateTenantRequest(BaseModel):
    """Request to update a tenant."""

    tenant_name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[TenantConfigRequest] = None


class TenantResponse(BaseModel):
    """Response model for tenant information."""

    tenant_id: str
    tenant_name: str
    description: Optional[str] = None
    is_active: bool
    created_at: str
    updated_at: str
    kb_count: int = 0


class CreateKBRequest(BaseModel):
    """Request to create a knowledge base."""

    kb_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class KBResponse(BaseModel):
    """Response model for knowledge base information."""

    kb_id: str
    kb_name: str
    description: Optional[str] = None
    tenant_id: str
    is_active: bool
    document_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    chunk_count: int = 0
    storage_used_mb: float = 0.0
    created_at: str
    updated_at: str


class PaginatedKBResponse(BaseModel):
    """Paginated response for knowledge bases."""

    items: List[KBResponse]
    total: int
    skip: int
    limit: int


# Membership Management Models


class UserRole(str, Enum):
    """User roles for tenant access control."""

    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


class TenantMembership(BaseModel):
    """Tenant membership information."""

    id: str
    user_id: str
    tenant_id: str
    role: UserRole
    created_at: str
    created_by: str
    updated_at: str


class AddMemberRequest(BaseModel):
    """Request to add a user to a tenant."""

    user_id: str = Field(..., min_length=1, max_length=255)
    role: UserRole = UserRole.VIEWER


class UpdateMemberRoleRequest(BaseModel):
    """Request to update a member's role."""

    role: UserRole


class MemberResponse(BaseModel):
    """Response model for tenant member."""

    user_id: str
    role: UserRole
    created_at: str
    created_by: str


class PaginatedMembersResponse(BaseModel):
    """Paginated response for tenant members."""

    items: List[MemberResponse]
    total: int
    skip: int
    limit: int


# Document Models


class DocumentAddRequest(BaseModel):
    """Request to add a document (file path is passed as form data)."""

    metadata: Optional[str] = None  # JSON string


class DocumentStatusResponse(BaseModel):
    """Response for document processing status."""

    doc_id: str
    status: str  # ready | processing | error
    chunks_processed: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    error_message: Optional[str] = None


class DocumentResponse(BaseModel):
    """Response model for document information."""

    doc_id: str
    kb_id: str
    tenant_id: str
    doc_name: str
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    chunk_count: int = 0
    created_at: str
    updated_at: str


# Query Models


class QueryRequest(BaseModel):
    """Request to query a knowledge base."""

    query: str = Field(..., min_length=3, max_length=2000)
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    top_k: Optional[int] = Field(None, ge=1, le=100)
    include_references: bool = True
    stream: bool = False


class QueryReference(BaseModel):
    """Reference to a source document/chunk."""

    doc_id: str
    doc_name: str
    chunk_id: Optional[str] = None
    content: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for query results."""

    response: str
    references: Optional[List[QueryReference]] = None
    metadata: Dict[str, Any] = {}


class QueryDataResponse(BaseModel):
    """Response model for query with full context data."""

    response: str
    references: Optional[List[QueryReference]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    chunks: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = {}


# API Key Models


class CreateAPIKeyRequest(BaseModel):
    """Request to create an API key."""

    key_name: str = Field(..., min_length=1, max_length=255)
    kb_id: Optional[str] = None  # None = all KBs
    permissions: Optional[List[str]] = None  # Default: ['query', 'document:read']
    expires_at: Optional[str] = None


class APIKeyResponse(BaseModel):
    """Response when creating an API key (includes the key itself)."""

    key_id: str
    key: str  # Only returned once on creation
    key_name: str
    created_at: str
    message: str = "Save this key securely - it won't be shown again"


class APIKeyMetadata(BaseModel):
    """Metadata for an API key (without the key itself)."""

    key_id: str
    key_name: str
    created_at: str
    last_used_at: Optional[str] = None
    permissions: List[str]
    is_active: bool = True


# Error Response Models


class ErrorResponse(BaseModel):
    """Standard error response."""

    status: str = "error"
    code: str  # e.g., "ACCESS_DENIED", "NOT_FOUND", "INVALID_REQUEST"
    message: str
    request_id: Optional[str] = None


class ValidationErrorResponse(BaseModel):
    """Response for validation errors."""

    status: str = "error"
    code: str = "VALIDATION_ERROR"
    message: str
    details: List[Dict[str, Any]]  # List of field-level errors
    request_id: Optional[str] = None
