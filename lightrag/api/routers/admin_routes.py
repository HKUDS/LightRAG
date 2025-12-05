"""Admin management routes for LightRAG API.

Provides endpoints for system administration, including tenant creation and listing.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from lightrag.services.tenant_service import TenantService
from lightrag.api.dependencies import get_admin_context

logger = logging.getLogger(__name__)


# Request/Response Models
class TenantCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
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


class PaginatedTenantResponse(BaseModel):
    """Paginated response for tenants."""

    items: List[TenantResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


def create_admin_routes(tenant_service: TenantService) -> APIRouter:
    """Create admin management routes.

    Args:
        tenant_service: Service instance for tenant operations

    Returns:
        APIRouter with admin routes
    """
    router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

    @router.post(
        "/tenants", response_model=TenantResponse, status_code=status.HTTP_201_CREATED
    )
    async def create_tenant(
        request: TenantCreateRequest, admin_context: dict = Depends(get_admin_context)
    ):
        """Create a new tenant.

        Requires admin authentication.
        """
        try:
            tenant = await tenant_service.create_tenant(
                tenant_name=request.name,
                description=request.description or "",
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
        except Exception as e:
            logger.error(f"Error creating tenant: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create tenant",
            )

    @router.get("/tenants", response_model=PaginatedTenantResponse)
    async def list_tenants(
        page: int = 1,
        page_size: int = 10,
        search: Optional[str] = None,
        admin_context: dict = Depends(get_admin_context),
    ):
        """List all available tenants with pagination.

        Useful for tenant selection.
        """
        try:
            # Validate pagination parameters
            page = max(1, page)
            page_size = min(max(1, page_size), 100)  # Max 100 per page

            # Get tenants from service
            tenants_data = await tenant_service.list_tenants(
                skip=(page - 1) * page_size,
                limit=page_size,
                search=search,
                tenant_id_filter=None,
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
                has_prev=page > 1,
            )
        except Exception as e:
            logger.error(f"Error listing tenants: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list tenants",
            )

    return router
