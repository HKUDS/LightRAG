"""Tenant management routes for multi-tenant LightRAG API.

Provides CRUD endpoints for managing tenants and knowledge bases.
"""

import logging
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from pydantic import BaseModel

from lightrag.models.tenant import TenantContext, Permission
from lightrag.services.tenant_service import TenantService
from lightrag.api.dependencies import get_tenant_context, check_permission, get_admin_context, get_tenant_context_no_kb, resolve_default_tenant

logger = logging.getLogger(__name__)

# Request/Response Models
class TenantCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    metadata: Optional[dict] = None

class TenantUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[dict] = None

class TenantResponse(BaseModel):
    tenant_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    num_knowledge_bases: int
    num_documents: int
    storage_used_gb: float

class KBCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    metadata: Optional[dict] = None

class KBUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[dict] = None

class KBResponse(BaseModel):
    kb_id: str
    tenant_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    num_documents: int
    num_entities: int
    num_relations: int

# Pagination Models
class PaginatedKBResponse(BaseModel):
    """Paginated response for knowledge bases."""
    items: List[KBResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool

class PaginatedTenantResponse(BaseModel):
    """Paginated response for tenants."""
    items: List[TenantResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


def create_tenant_routes(tenant_service: TenantService) -> APIRouter:
    """Create tenant management routes.
    
    Args:
        tenant_service: Service instance for tenant operations
        
    Returns:
        APIRouter with tenant routes
    """
    router = APIRouter(prefix="/api/v1", tags=["tenants"])
    
    # Tenant management endpoints
    
    @router.get("/tenants", response_model=PaginatedTenantResponse)
    async def list_tenants(
        page: int = 1,
        page_size: int = 10,
        search: Optional[str] = None,
        authorization: Optional[str] = Header(None),
    ):
        """List all available tenants with pagination.
        
        Useful for tenant selection. This endpoint is public to allow
        unauthenticated access for tenant selection on the login page.
        """
        # Note: This endpoint is intentionally public to support tenant selection
        # on the login page before authentication. In production, you may want to
        # restrict this to specific IPs or use rate limiting.
             
        try:
            # Validate pagination parameters
            page = max(1, page)
            page_size = min(max(1, page_size), 100)  # Max 100 per page
            
            # Get tenants from service
            tenants_data = await tenant_service.list_tenants(
                skip=(page - 1) * page_size,
                limit=page_size,
                search=search,
                tenant_id_filter=None
            )
            
            total_count = tenants_data.get("total", 0)
            tenants_list = tenants_data.get("items", [])
            
            # Convert to response models
            items = [
                TenantResponse(
                    tenant_id=t.tenant_id,
                    name=t.tenant_name,
                    description=t.description,
                    created_at=t.created_at.isoformat(),
                    updated_at=t.updated_at.isoformat(),
                    num_knowledge_bases=t.kb_count,
                    num_documents=t.total_documents,
                    storage_used_gb=t.total_storage_mb / 1024.0,
                )
                for t in tenants_list
            ]
            
            # Calculate pagination metadata
            total_pages = (total_count + page_size - 1) // page_size
            
            return PaginatedTenantResponse(
                items=items,
                total=total_count,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_prev=page > 1
            )
        except Exception as e:
            logger.error(f"Error listing tenants: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list tenants"
            )

    @router.get("/tenants/me", response_model=TenantResponse)
    async def get_current_tenant(
        context: TenantContext = Depends(get_tenant_context_no_kb)
    ):
        """Get current tenant details based on context.
        
        The tenant is identified by the X-Tenant-ID header or authentication token.
        """
        try:
            tenant = await tenant_service.get_tenant(context.tenant_id)
            if not tenant:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Tenant not found"
                )
            
            return TenantResponse(
                tenant_id=tenant.tenant_id,
                name=tenant.tenant_name,
                description=tenant.description,
                created_at=tenant.created_at.isoformat(),
                updated_at=tenant.updated_at.isoformat(),
                num_knowledge_bases=tenant.kb_count,
                num_documents=tenant.total_documents,
                storage_used_gb=tenant.total_storage_mb / 1024.0,
            )
        except HTTPException:
            # Let explicit HTTPExceptions (404 etc.) bubble up unchanged
            raise
        except Exception as e:
            logger.error(f"Error getting current tenant {context.tenant_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get tenant details"
            )

    @router.post("/tenants", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
    async def create_tenant(
        request: TenantCreateRequest,
        admin_context: dict = Depends(get_admin_context)
    ):
        """Create a new tenant.

        Requires admin authentication (no tenant_id required).
        This allows creating the initial tenant(s) for new organizations.
        """
        try:
            username = admin_context.get("username")
            
            tenant = await tenant_service.create_tenant(
                tenant_name=request.name,
                description=request.description or "",
                created_by=username
            )
            
            # Add creator as owner
            if username:
                try:
                    await tenant_service.add_user_to_tenant(
                        user_id=username,
                        tenant_id=tenant.tenant_id,
                        role="owner",
                        created_by=username
                    )
                    logger.info(f"Added user {username} as owner of new tenant {tenant.tenant_id}")
                except Exception as e:
                    logger.error(f"Failed to add creator as owner: {e}")
                    # Continue anyway, as tenant was created
            
            return TenantResponse(
                tenant_id=tenant.tenant_id,
                name=tenant.tenant_name,
                description=tenant.description,
                created_at=tenant.created_at.isoformat(),
                updated_at=tenant.updated_at.isoformat(),
                num_knowledge_bases=tenant.kb_count,
                num_documents=tenant.total_documents,
                storage_used_gb=tenant.total_storage_mb / 1024.0,
            )
        except Exception as e:
            logger.error(f"Error creating tenant: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create tenant"
            )
    

    # Knowledge base management endpoints
    
    @router.post("/knowledge-bases", response_model=KBResponse, status_code=status.HTTP_201_CREATED)
    async def create_knowledge_base_context(
        request: KBCreateRequest,
        context: TenantContext = Depends(get_tenant_context_no_kb)
    ):
        """Create a new knowledge base within the current tenant.
        
        The tenant is identified by the X-Tenant-ID header or authentication token.
        """
        try:
            kb = await tenant_service.create_knowledge_base(
                tenant_id=context.tenant_id,
                kb_name=request.name,
                description=request.description or "",
            )
            
            return KBResponse(
                kb_id=kb.kb_id,
                tenant_id=kb.tenant_id,
                name=kb.kb_name,
                description=kb.description,
                created_at=kb.created_at.isoformat(),
                updated_at=kb.updated_at.isoformat(),
                num_documents=kb.document_count,
                num_entities=kb.entity_count,
                num_relations=kb.relationship_count,
            )
        except Exception as e:
            logger.error(f"Error creating KB for tenant {context.tenant_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create knowledge base"
            )

    @router.get("/knowledge-bases", response_model=PaginatedKBResponse)
    async def list_knowledge_bases_context(
        page: int = 1,
        page_size: int = 10,
        search: Optional[str] = None,
        context: TenantContext = Depends(get_tenant_context_no_kb)
    ):
        """List all knowledge bases in the current tenant with pagination.
        
        The tenant is identified by the X-Tenant-ID header or authentication token.
        
        Query Parameters:
            page: Page number (1-indexed), defaults to 1
            page_size: Number of items per page, defaults to 10 (max 100)
            search: Optional search string to filter KBs by name or description
        """
        try:
            # Validate pagination parameters
            page = max(1, page)
            page_size = min(max(1, page_size), 100)  # Max 100 per page
            
            # Get KBs from service
            kbs_data = await tenant_service.list_knowledge_bases(
                tenant_id=context.tenant_id,
                skip=(page - 1) * page_size,
                limit=page_size,
                search=search
            )
            
            total_count = kbs_data.get("total", 0)
            kbs_list = kbs_data.get("items", [])
            
            # Convert to response models
            items = [
                KBResponse(
                    kb_id=kb.kb_id,
                    tenant_id=kb.tenant_id,
                    name=kb.kb_name,
                    description=kb.description,
                    created_at=kb.created_at.isoformat(),
                    updated_at=kb.updated_at.isoformat(),
                    num_documents=kb.document_count,
                    num_entities=kb.entity_count,
                    num_relations=kb.relationship_count,
                )
                for kb in kbs_list
            ]
            
            # Calculate pagination metadata
            total_pages = (total_count + page_size - 1) // page_size
            
            return PaginatedKBResponse(
                items=items,
                total=total_count,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_prev=page > 1
            )
        except Exception as e:
            logger.error(f"Error listing KBs for tenant {context.tenant_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list knowledge bases"
            )

    @router.get("/knowledge-bases/{kb_id}", response_model=KBResponse)
    async def get_knowledge_base_context(
        kb_id: str,
        context: TenantContext = Depends(get_tenant_context)
    ):
        """Get knowledge base details.
        
        The tenant is identified by the X-Tenant-ID header or authentication token.
        """
        # Note: get_tenant_context already validates that context.kb_id matches kb_id if kb_id is in context
        # But here kb_id is a path param, so we should double check if context has a specific kb_id
        if context.kb_id and context.kb_id != kb_id:
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot access other knowledge bases"
            )

        try:
            kb = await tenant_service.get_knowledge_base(context.tenant_id, kb_id)
            if not kb or kb.tenant_id != context.tenant_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Knowledge base not found"
                )
            
            return KBResponse(
                kb_id=kb.kb_id,
                tenant_id=kb.tenant_id,
                name=kb.kb_name,
                description=kb.description,
                created_at=kb.created_at.isoformat(),
                updated_at=kb.updated_at.isoformat(),
                num_documents=kb.document_count,
                num_entities=kb.entity_count,
                num_relations=kb.relationship_count,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting KB {kb_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get knowledge base"
            )

    @router.put("/knowledge-bases/{kb_id}", response_model=KBResponse)
    async def update_knowledge_base_context(
        kb_id: str,
        request: KBUpdateRequest,
        context: TenantContext = Depends(
            check_permission(Permission.MANAGE_KB.value)
        )
    ):
        """Update knowledge base settings.
        
        The tenant is identified by the X-Tenant-ID header or authentication token.
        """
        if context.kb_id and context.kb_id != kb_id:
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot update other knowledge bases"
            )

        try:
            kb = await tenant_service.update_knowledge_base(
                tenant_id=context.tenant_id,
                kb_id=kb_id,
                name=request.name,
                description=request.description,
                metadata=request.metadata
            )
            
            if not kb or kb.tenant_id != context.tenant_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Knowledge base not found"
                )
            
            return KBResponse(
                kb_id=kb.kb_id,
                tenant_id=kb.tenant_id,
                name=kb.kb_name,
                description=kb.description,
                created_at=kb.created_at.isoformat(),
                updated_at=kb.updated_at.isoformat(),
                num_documents=kb.document_count,
                num_entities=kb.entity_count,
                num_relations=kb.relationship_count,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating KB {kb_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update knowledge base"
            )

    @router.delete("/knowledge-bases/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_knowledge_base_context(
        kb_id: str,
        context: TenantContext = Depends(
            check_permission(Permission.DELETE_KB.value)
        )
    ):
        """Delete a knowledge base.
        
        The tenant is identified by the X-Tenant-ID header or authentication token.
        """
        if context.kb_id and context.kb_id != kb_id:
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete other knowledge bases"
            )

        try:
            success = await tenant_service.delete_knowledge_base(context.tenant_id, kb_id)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Knowledge base not found"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting KB {kb_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete knowledge base"
            )


    @router.get("/tenants/{tenant_id}/knowledge-bases", response_model=PaginatedKBResponse)
    async def list_knowledge_bases_path(
        tenant_id: str,
        request: Request,
        page: int = 1,
        page_size: int = 10,
        search: Optional[str] = None,
        authorization: Optional[str] = Header(None),
    ):
        """List all knowledge bases for a specific tenant (path param).
        
        Legacy endpoint support.
        """
        if not authorization:
             raise HTTPException(status_code=401, detail="Missing authorization header")
             
        try:
            # Extract username from token
            from lightrag.api.auth import auth_handler
            try:
                print(f"DEBUG: Validating token: {authorization[:30]}...")
                logger.info(f"Validating token: {authorization[:30]}...")
                scheme, token = authorization.split()
                token_data = auth_handler.validate_token(token)
                username = token_data.get("username")
                print(f"DEBUG: Token valid for user: {username}")
                logger.info(f"Token valid for user: {username}")
            except Exception as e:
                print(f"DEBUG: Token validation failed: {e}")
                logger.error(f"Token validation failed: {e}")
                username = None
            
            # Resolve default tenant
            resolved_tenant_id = await resolve_default_tenant(request, tenant_id, user_id=username)
            
            # Validate pagination parameters
            page = max(1, page)
            page_size = min(max(1, page_size), 100)  # Max 100 per page
            
            # Get KBs from service
            kbs_data = await tenant_service.list_knowledge_bases(
                tenant_id=resolved_tenant_id,
                skip=(page - 1) * page_size,
                limit=page_size,
                search=search
            )
            
            total_count = kbs_data.get("total", 0)
            kbs_list = kbs_data.get("items", [])
            
            # Convert to response models
            items = [
                KBResponse(
                    kb_id=kb.kb_id,
                    tenant_id=kb.tenant_id,
                    name=kb.kb_name,
                    description=kb.description,
                    created_at=kb.created_at.isoformat(),
                    updated_at=kb.updated_at.isoformat(),
                    num_documents=kb.document_count,
                    num_entities=kb.entity_count,
                    num_relations=kb.relationship_count,
                )
                for kb in kbs_list
            ]
            
            # Calculate pagination metadata
            total_pages = (total_count + page_size - 1) // page_size
            
            return PaginatedKBResponse(
                items=items,
                total=total_count,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_prev=page > 1
            )
        except Exception as e:
            logger.error(f"Error listing KBs for tenant {tenant_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list knowledge bases"
            )

    return router
