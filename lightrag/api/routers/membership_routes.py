"""
API routes for tenant membership management.

Handles user invitations, role management, and member listing.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional

from lightrag.api.models import (
    AddMemberRequest,
    UpdateMemberRoleRequest,
    MemberResponse,
    PaginatedMembersResponse,
    UserRole
)
from lightrag.api.dependencies import get_tenant_context_no_kb
from lightrag.models.tenant import TenantContext
from lightrag.api.utils_api import get_combined_auth_dependency
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["membership"])


@router.post(
    "/tenants/{tenant_id}/members",
    response_model=MemberResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(get_combined_auth_dependency())]
)
async def add_member_to_tenant(
    tenant_id: str,
    request: AddMemberRequest,
    context: TenantContext = Depends(get_tenant_context_no_kb)
):
    """
    Add a user to a tenant with specified role.
    
    Requires admin or owner role in the tenant.
    """
    from fastapi import Request as FastAPIRequest
    from starlette.requests import Request
    
    # Get tenant service from app state
    try:
        # Access via starlette request
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'request' in frame.f_locals and isinstance(frame.f_locals['request'], Request):
                req = frame.f_locals['request']
                break
            frame = frame.f_back
        
        if not req or not hasattr(req.app.state, 'rag_manager'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not initialized"
            )
        
        tenant_service = req.app.state.rag_manager.tenant_service
        
        # Verify requester has admin or owner role
        has_permission = await tenant_service.verify_user_access(
            user_id=context.user_id,
            tenant_id=tenant_id,
            required_role="admin"
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions. Admin or owner role required."
            )
        
        # Add member
        membership = await tenant_service.add_user_to_tenant(
            user_id=request.user_id,
            tenant_id=tenant_id,
            role=request.role.value,
            created_by=context.user_id
        )
        
        return MemberResponse(
            user_id=membership["user_id"],
            role=UserRole(membership["role"]),
            created_at=membership["created_at"],
            created_by=membership["created_by"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding member to tenant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/tenants/{tenant_id}/members",
    response_model=PaginatedMembersResponse,
    dependencies=[Depends(get_combined_auth_dependency())]
)
async def list_tenant_members(
    tenant_id: str,
    context: TenantContext = Depends(get_tenant_context_no_kb),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    """
    List all members of a tenant.
    
    Requires at least viewer role in the tenant.
    """
    from fastapi import Request as FastAPIRequest
    from starlette.requests import Request
    
    try:
        # Get tenant service
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'request' in frame.f_locals and isinstance(frame.f_locals['request'], Request):
                req = frame.f_locals['request']
                break
            frame = frame.f_back
        
        if not req or not hasattr(req.app.state, 'rag_manager'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not initialized"
            )
        
        tenant_service = req.app.state.rag_manager.tenant_service
        
        # Verify requester has access
        has_access = await tenant_service.verify_user_access(
            user_id=context.user_id,
            tenant_id=tenant_id,
            required_role="viewer"
        )
        
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant"
            )
        
        # Get members
        result = await tenant_service.get_tenant_members(
            tenant_id=tenant_id,
            skip=skip,
            limit=limit
        )
        
        members = [
            MemberResponse(
                user_id=m["user_id"],
                role=UserRole(m["role"]),
                created_at=m["created_at"],
                created_by=m["created_by"]
            )
            for m in result["items"]
        ]
        
        return PaginatedMembersResponse(
            items=members,
            total=result["total"],
            skip=skip,
            limit=limit
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing tenant members: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.put(
    "/tenants/{tenant_id}/members/{user_id}",
    response_model=MemberResponse,
    dependencies=[Depends(get_combined_auth_dependency())]
)
async def update_member_role(
    tenant_id: str,
    user_id: str,
    request: UpdateMemberRoleRequest,
    context: TenantContext = Depends(get_tenant_context_no_kb)
):
    """
    Update a member's role in a tenant.
    
    Requires admin or owner role in the tenant.
    """
    from fastapi import Request as FastAPIRequest
    from starlette.requests import Request
    
    try:
        # Get tenant service
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'request' in frame.f_locals and isinstance(frame.f_locals['request'], Request):
                req = frame.f_locals['request']
                break
            frame = frame.f_back
        
        if not req or not hasattr(req.app.state, 'rag_manager'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not initialized"
            )
        
        tenant_service = req.app.state.rag_manager.tenant_service
        
        # Verify requester has admin or owner role
        has_permission = await tenant_service.verify_user_access(
            user_id=context.user_id,
            tenant_id=tenant_id,
            required_role="admin"
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions. Admin or owner role required."
            )
        
        # Update role
        membership = await tenant_service.update_user_role(
            user_id=user_id,
            tenant_id=tenant_id,
            new_role=request.role.value
        )
        
        return MemberResponse(
            user_id=membership["user_id"],
            role=UserRole(membership["role"]),
            created_at=membership["created_at"],
            created_by=membership["created_by"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating member role: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete(
    "/tenants/{tenant_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(get_combined_auth_dependency())]
)
async def remove_member_from_tenant(
    tenant_id: str,
    user_id: str,
    context: TenantContext = Depends(get_tenant_context_no_kb)
):
    """
    Remove a member from a tenant.
    
    Requires admin or owner role in the tenant.
    """
    from fastapi import Request as FastAPIRequest
    from starlette.requests import Request
    
    try:
        # Get tenant service
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'request' in frame.f_locals and isinstance(frame.f_locals['request'], Request):
                req = frame.f_locals['request']
                break
            frame = frame.f_back
        
        if not req or not hasattr(req.app.state, 'rag_manager'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not initialized"
            )
        
        tenant_service = req.app.state.rag_manager.tenant_service
        
        # Verify requester has admin or owner role
        has_permission = await tenant_service.verify_user_access(
            user_id=context.user_id,
            tenant_id=tenant_id,
            required_role="admin"
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions. Admin or owner role required."
            )
        
        # Remove member
        removed = await tenant_service.remove_user_from_tenant(
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} is not a member of tenant {tenant_id}"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing member from tenant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/users/me/tenants",
    dependencies=[Depends(get_combined_auth_dependency())]
)
async def get_my_tenants(
    context: TenantContext = Depends(get_tenant_context_no_kb),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get all tenants the current user has access to.
    """
    from fastapi import Request as FastAPIRequest
    from starlette.requests import Request
    
    try:
        # Get tenant service
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'request' in frame.f_locals and isinstance(frame.f_locals['request'], Request):
                req = frame.f_locals['request']
                break
            frame = frame.f_back
        
        if not req or not hasattr(req.app.state, 'rag_manager'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not initialized"
            )
        
        tenant_service = req.app.state.rag_manager.tenant_service
        
        # Get user's tenants
        result = await tenant_service.get_user_tenants(
            user_id=context.user_id,
            skip=skip,
            limit=limit
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user tenants: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
