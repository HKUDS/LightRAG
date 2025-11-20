# ADR 004: API Design and Routing

## Status: Proposed

## Overview
This document specifies the API design for the multi-tenant, multi-knowledge-base architecture, including endpoint structure, request/response models, authentication, and error handling.

## API Versioning and Structure

### Base URL
```
https://lightrag.example.com/api/v1
```

### URL Path Structure
```
/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/{resource_type}/{operation}
```

### Example Endpoints
```
POST   /api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/documents/add
GET    /api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/documents/{doc_id}
POST   /api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/query
DELETE /api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/documents/{doc_id}
GET    /api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/graph
POST   /api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/entities/{entity_id}/delete
```

## Authentication Mechanisms

### 1. JWT Bearer Token Authentication

#### Token Creation
```python
class TokenPayload(BaseModel):
    sub: str  # User ID
    tenant_id: str  # Assigned tenant
    knowledge_base_ids: List[str]  # Accessible KBs (or ["*"] for all)
    role: str  # admin | editor | viewer
    permissions: Dict[str, bool]  # Specific permissions
    exp: int  # Expiration time (Unix timestamp)
    iat: int  # Issued at time
    jti: str  # JWT ID (for revocation)
```

#### Usage
```bash
# Request with JWT token
curl -X POST https://lightrag.example.com/api/v1/tenants/acme/knowledge-bases/docs/query \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the product roadmap?"}'
```

#### Token Validation
```python
async def validate_token(token: str) -> TokenPayload:
    """Validate JWT token and return payload"""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )
        # Verify expiration
        exp_time = datetime.fromtimestamp(payload["exp"])
        if datetime.utcnow() > exp_time:
            raise HTTPException(status_code=401, detail="Token expired")
        
        return TokenPayload(**payload)
    except jwt.DecodeError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### 2. API Key Authentication

#### API Key Format
```
X-API-Key: sk-tenant_12345_kb_67890_randomstring1234567890
```

#### API Key Structure
```
sk-{tenant_id}_{kb_id}_{random_bytes}
```

#### Usage
```bash
curl -X POST https://lightrag.example.com/api/v1/tenants/acme/knowledge-bases/docs/query \
  -H "X-API-Key: sk-acme_docs_xyz123..." \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the product roadmap?"}'
```

#### API Key Management Endpoints
```python
@router.post("/api/v1/tenants/{tenant_id}/api-keys")
async def create_api_key(
    request: CreateAPIKeyRequest,
    tenant_context: TenantContext = Depends(get_tenant_context),
) -> APIKeyResponse:
    """Create a new API key for a tenant"""
    # Generate hashed key
    api_key = APIKeyService.generate_api_key(
        tenant_id=tenant_context.tenant_id,
        kb_id=request.kb_id,
        permissions=request.permissions
    )
    # Store hashed version
    await api_key_service.store_api_key(api_key)
    # Return key (only once, must be saved by client)
    return APIKeyResponse(
        key_id=api_key.key_id,
        key=api_key.unhashed_key,  # Only returned once
        created_at=api_key.created_at
    )

@router.get("/api/v1/tenants/{tenant_id}/api-keys")
async def list_api_keys(
    tenant_context: TenantContext = Depends(get_tenant_context),
) -> List[APIKeyMetadata]:
    """List API keys (without revealing the key itself)"""
    keys = await api_key_service.list_keys(tenant_context.tenant_id)
    return [
        APIKeyMetadata(
            key_id=k.key_id,
            key_name=k.key_name,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
            permissions=k.permissions
        )
        for k in keys
    ]

@router.delete("/api/v1/tenants/{tenant_id}/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    tenant_context: TenantContext = Depends(get_tenant_context),
) -> dict:
    """Revoke an API key"""
    await api_key_service.revoke_key(key_id)
    return {"status": "success", "message": "API key revoked"}
```

## Tenant Management Endpoints

### Create Tenant
```python
@router.post("/api/v1/tenants")
async def create_tenant(
    request: CreateTenantRequest,
    admin_token: str = Depends(validate_admin_token),
) -> TenantResponse:
    """Create a new tenant (admin only)"""
    tenant = await tenant_service.create_tenant(
        tenant_name=request.tenant_name,
        description=request.description,
        config=request.config or TenantConfig()
    )
    return TenantResponse(
        tenant_id=tenant.tenant_id,
        tenant_name=tenant.tenant_name,
        description=tenant.description,
        created_at=tenant.created_at,
        is_active=tenant.is_active
    )

# Request model
class CreateTenantRequest(BaseModel):
    tenant_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    config: Optional[TenantConfigRequest] = None

class TenantConfigRequest(BaseModel):
    llm_model: Optional[str] = "gpt-4o-mini"
    embedding_model: Optional[str] = "bge-m3:latest"
    chunk_size: Optional[int] = 1200
    top_k: Optional[int] = 40
```

### Get Tenant
```python
@router.get("/api/v1/tenants/{tenant_id}")
async def get_tenant(
    tenant_context: TenantContext = Depends(get_tenant_context),
) -> TenantResponse:
    """Get tenant details"""
    tenant = await tenant_service.get_tenant(tenant_context.tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return TenantResponse.from_tenant(tenant)
```

### Update Tenant
```python
@router.put("/api/v1/tenants/{tenant_id}")
async def update_tenant(
    request: UpdateTenantRequest,
    tenant_context: TenantContext = Depends(get_tenant_context),
) -> TenantResponse:
    """Update tenant configuration"""
    if not has_permission(tenant_context, "tenant:manage"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    tenant = await tenant_service.update_tenant(
        tenant_id=tenant_context.tenant_id,
        **request.dict(exclude_none=True)
    )
    return TenantResponse.from_tenant(tenant)
```

## Knowledge Base Endpoints

### Create Knowledge Base
```python
@router.post("/api/v1/tenants/{tenant_id}/knowledge-bases")
async def create_knowledge_base(
    request: CreateKBRequest,
    tenant_context: TenantContext = Depends(get_tenant_context),
) -> KBResponse:
    """Create a knowledge base in a tenant"""
    if not has_permission(tenant_context, "kb:create"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    kb = await tenant_service.create_knowledge_base(
        tenant_id=tenant_context.tenant_id,
        kb_name=request.kb_name,
        description=request.description
    )
    return KBResponse.from_kb(kb)

class CreateKBRequest(BaseModel):
    kb_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
```

### List Knowledge Bases
```python
@router.get("/api/v1/tenants/{tenant_id}/knowledge-bases")
async def list_knowledge_bases(
    tenant_context: TenantContext = Depends(get_tenant_context),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> PaginatedKBResponse:
    """List all KBs accessible to the user"""
    kbs = await tenant_service.list_knowledge_bases(
        tenant_id=tenant_context.tenant_id,
        accessible_kb_ids=tenant_context.knowledge_base_ids,
        skip=skip,
        limit=limit
    )
    return PaginatedKBResponse(
        items=[KBResponse.from_kb(kb) for kb in kbs],
        total=kbs.total,
        skip=skip,
        limit=limit
    )
```

### Delete Knowledge Base
```python
@router.delete("/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    tenant_context: TenantContext = Depends(get_tenant_context),
) -> dict:
    """Delete a knowledge base"""
    if not has_permission(tenant_context, "kb:delete"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    await tenant_service.delete_knowledge_base(
        tenant_id=tenant_context.tenant_id,
        kb_id=kb_id
    )
    return {"status": "success", "message": "Knowledge base deleted"}
```

## Document Endpoints

### Add Document
```python
@router.post("/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/documents/add")
async def add_document(
    tenant_id: str = Path(...),
    kb_id: str = Path(...),
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),  # JSON string
    tenant_context: TenantContext = Depends(get_tenant_context),
    rag_manager = Depends(get_rag_manager),
) -> DocumentAddResponse:
    """
    Add a document to a knowledge base.
    
    Returns a track_id for monitoring progress via websocket or polling.
    """
    if not has_permission(tenant_context, "document:create"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Validate file
    if not is_allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    # Get tenant-specific RAG instance
    rag = await rag_manager.get_rag_instance(tenant_id, kb_id)
    
    # Start document processing (async)
    track_id = generate_track_id()
    asyncio.create_task(
        process_document(
            rag=rag,
            file=file,
            metadata=metadata,
            track_id=track_id,
            tenant_context=tenant_context
        )
    )
    
    return DocumentAddResponse(
        status="processing",
        track_id=track_id,
        message="Document is being processed"
    )

class DocumentAddResponse(BaseModel):
    status: str  # processing | success | error
    track_id: str
    message: Optional[str] = None
    doc_id: Optional[str] = None
```

### Get Document Status
```python
@router.get("/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/documents/{doc_id}/status")
async def get_document_status(
    doc_id: str,
    tenant_context: TenantContext = Depends(get_tenant_context),
) -> DocumentStatusResponse:
    """Get document processing status"""
    status = await doc_status_service.get_status(
        doc_id=doc_id,
        tenant_id=tenant_context.tenant_id,
        kb_id=tenant_context.kb_id
    )
    return DocumentStatusResponse(
        doc_id=doc_id,
        status=status.status,  # ready | processing | error
        chunks_processed=status.chunks_processed,
        entities_extracted=status.entities_extracted,
        relationships_extracted=status.relationships_extracted,
        error_message=status.error_message
    )
```

### Delete Document
```python
@router.delete("/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    tenant_context: TenantContext = Depends(get_tenant_context),
    rag_manager = Depends(get_rag_manager),
) -> dict:
    """Delete a document from knowledge base"""
    if not has_permission(tenant_context, "document:delete"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Verify document belongs to this tenant/KB
    doc = await doc_service.get_document(doc_id, tenant_context.tenant_id, tenant_context.kb_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from RAG
    rag = await rag_manager.get_rag_instance(
        tenant_context.tenant_id,
        tenant_context.kb_id
    )
    await rag.adelete_by_doc_id(doc_id)
    
    return {"status": "success", "message": "Document deleted"}
```

## Query Endpoints

### Standard Query
```python
@router.post("/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/query")
async def query_knowledge_base(
    request: QueryRequest,
    tenant_context: TenantContext = Depends(get_tenant_context),
    rag_manager = Depends(get_rag_manager),
) -> QueryResponse:
    """
    Execute a query against a knowledge base.
    
    Returns the generated response with optional references.
    """
    if not has_permission(tenant_context, "query:run"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Validate query
    if len(request.query) < 3:
        raise HTTPException(status_code=400, detail="Query too short")
    
    # Get tenant-specific RAG instance
    rag = await rag_manager.get_rag_instance(
        tenant_context.tenant_id,
        tenant_context.kb_id
    )
    
    # Execute query with tenant context
    result = await rag.aquery(
        query=request.query,
        param=QueryParam(
            mode=request.mode or "mix",
            top_k=request.top_k or 40,
            stream=False
        )
    )
    
    return QueryResponse(
        response=result.response,
        references=result.references if request.include_references else None,
        metadata={
            "mode": request.mode,
            "top_k": request.top_k,
            "processing_time_ms": result.processing_time
        }
    )

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    mode: Optional[str] = Field("mix", regex="local|global|hybrid|naive|mix|bypass")
    top_k: Optional[int] = Field(None, ge=1, le=100)
    include_references: bool = Field(True)
    stream: bool = Field(False)

class QueryResponse(BaseModel):
    response: str
    references: Optional[List[Dict[str, str]]] = None
    metadata: Dict[str, Any] = {}
```

### Streaming Query
```python
@router.post("/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/query/stream")
async def query_knowledge_base_stream(
    request: QueryRequest,
    tenant_context: TenantContext = Depends(get_tenant_context),
    rag_manager = Depends(get_rag_manager),
) -> StreamingResponse:
    """
    Execute a query with streaming response.
    
    Returns Server-Sent Events (SSE) with streamed tokens and metadata.
    """
    if not has_permission(tenant_context, "query:run"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    async def stream_response():
        # Get RAG instance
        rag = await rag_manager.get_rag_instance(
            tenant_context.tenant_id,
            tenant_context.kb_id
        )
        
        # Stream the response
        async for chunk in rag.aquery_stream(
            query=request.query,
            param=QueryParam(
                mode=request.mode or "mix",
                top_k=request.top_k or 40,
                stream=True
            )
        ):
            # Emit Server-Sent Event
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )
```

### Query with Data
```python
@router.post("/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/query/data")
async def query_knowledge_base_data(
    request: QueryRequest,
    tenant_context: TenantContext = Depends(get_tenant_context),
    rag_manager = Depends(get_rag_manager),
) -> QueryDataResponse:
    """
    Execute a query and return full context data.
    
    Returns entities, relationships, chunks, and references.
    """
    if not has_permission(tenant_context, "query:run"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    rag = await rag_manager.get_rag_instance(
        tenant_context.tenant_id,
        tenant_context.kb_id
    )
    
    result = await rag.aquery_with_data(
        query=request.query,
        param=QueryParam(mode=request.mode or "mix", top_k=request.top_k or 40)
    )
    
    return QueryDataResponse(
        status="success",
        message="Query executed successfully",
        data={
            "entities": result.entities,
            "relationships": result.relationships,
            "chunks": result.chunks,
            "response": result.response
        },
        metadata={
            "mode": request.mode,
            "entity_count": len(result.entities),
            "relationship_count": len(result.relationships),
            "chunk_count": len(result.chunks)
        }
    )

class QueryDataResponse(BaseModel):
    status: str
    message: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
```

## Graph Endpoints

### Get Graph
```python
@router.get("/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/graph")
async def get_graph(
    tenant_context: TenantContext = Depends(get_tenant_context),
    rag_manager = Depends(get_rag_manager),
    max_nodes: int = Query(100, ge=10, le=1000),
    entity_type: Optional[str] = None,
) -> GraphResponse:
    """Get knowledge graph visualization data"""
    if not has_permission(tenant_context, "kb:access"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    rag = await rag_manager.get_rag_instance(
        tenant_context.tenant_id,
        tenant_context.kb_id
    )
    
    graph_data = await rag.get_graph(
        max_nodes=max_nodes,
        entity_type=entity_type
    )
    
    return GraphResponse(
        nodes=graph_data.nodes,
        edges=graph_data.edges,
        metadata={
            "node_count": len(graph_data.nodes),
            "edge_count": len(graph_data.edges)
        }
    )
```

## Error Responses

### Standard Error Response
```python
class ErrorResponse(BaseModel):
    status: str = "error"
    code: str  # error code for client handling
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: str  # For tracking

# Example error codes
ERROR_CODES = {
    "INVALID_TENANT": "Specified tenant does not exist",
    "INVALID_KB": "Specified knowledge base does not exist",
    "UNAUTHORIZED": "Authentication failed",
    "FORBIDDEN": "User does not have permission",
    "INVALID_REQUEST": "Request validation failed",
    "INTERNAL_ERROR": "Internal server error",
    "RATE_LIMITED": "Too many requests",
    "QUOTA_EXCEEDED": "Resource quota exceeded"
}
```

### Example Error Response
```json
{
    "status": "error",
    "code": "FORBIDDEN",
    "message": "You do not have permission to access this knowledge base",
    "details": {
        "required_permission": "kb:access",
        "user_permissions": ["query:run"]
    },
    "request_id": "req-12345"
}
```

## Request/Response Headers

### Request Headers
```
Authorization: Bearer <jwt_token>
or
X-API-Key: <api_key>

X-Request-ID: <unique_request_id>  (optional, generated if not provided)
X-Tenant-ID: <tenant_id>           (optional, extracted from path)
X-KB-ID: <kb_id>                   (optional, extracted from path)
```

### Response Headers
```
X-Request-ID: <unique_request_id>
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1703123456
Content-Type: application/json
```

## Rate Limiting

### Per-Tenant Rate Limits
```python
class RateLimitConfig:
    # Per tenant
    QUERIES_PER_MINUTE = 100
    DOCUMENTS_PER_HOUR = 50
    API_CALLS_PER_MONTH = 100000
    
    # Global
    GLOBAL_QPS = 10000  # Queries per second

# Implement with Redis
@router.post("/api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/query")
async def query_with_rate_limit(
    request: QueryRequest,
    tenant_context: TenantContext = Depends(get_tenant_context),
    rate_limiter = Depends(get_rate_limiter)
):
    # Check rate limit
    await rate_limiter.check_limit(
        key=f"{tenant_context.tenant_id}:queries",
        limit=RateLimitConfig.QUERIES_PER_MINUTE,
        window=60
    )
    
    # Execute query
    # ...
```

## API Documentation

### OpenAPI/Swagger
```python
app = FastAPI(
    title="LightRAG Multi-Tenant API",
    description="API for multi-tenant RAG system",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)
```

### Example cURL Commands
```bash
# Create tenant (admin)
curl -X POST https://lightrag.example.com/api/v1/tenants \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_name": "Acme Corp",
    "description": "Our main tenant"
  }'

# Create knowledge base
curl -X POST https://lightrag.example.com/api/v1/tenants/acme/knowledge-bases \
  -H "Authorization: Bearer <tenant_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "kb_name": "Product Docs",
    "description": "Product documentation"
  }'

# Add document
curl -X POST https://lightrag.example.com/api/v1/tenants/acme/knowledge-bases/docs/documents/add \
  -H "Authorization: Bearer <tenant_token>" \
  -F "file=@document.pdf"

# Query knowledge base
curl -X POST https://lightrag.example.com/api/v1/tenants/acme/knowledge-bases/docs/query \
  -H "Authorization: Bearer <tenant_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the product roadmap?",
    "mode": "mix",
    "top_k": 10,
    "include_references": true
  }'

# Stream query
curl -X POST https://lightrag.example.com/api/v1/tenants/acme/knowledge-bases/docs/query/stream \
  -H "Authorization: Bearer <tenant_token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "Product roadmap?"}' \
  --stream
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-20  
**Related Files**: 001-multi-tenant-architecture-overview.md, 002-implementation-strategy.md
