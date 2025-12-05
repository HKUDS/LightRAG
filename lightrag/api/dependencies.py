"""FastAPI dependencies for multi-tenant request handling.

This module provides dependency injection for tenant context extraction and validation.
"""

from fastapi import Depends, HTTPException, status, Header, Request
from typing import Optional
import logging
import os

from lightrag.models.tenant import TenantContext
from lightrag.api.auth import auth_handler
from lightrag.api.config import global_args

logger = logging.getLogger(__name__)


async def resolve_default_tenant(
    request: Request, tenant_id: str, user_id: str = None
) -> str:
    """Resolve 'default' tenant_id to user's first accessible tenant.

    Args:
        request: FastAPI request object
        tenant_id: Tenant ID (may be "default")
        user_id: User identifier from token

    Returns:
        Resolved tenant ID

    Raises:
        HTTPException: If user has no accessible tenants
    """
    if tenant_id != "default":
        return tenant_id

    try:
        # Access rag_manager from app state
        if not hasattr(request.app.state, "rag_manager"):
            logger.warning("rag_manager not found in app state")
            return tenant_id

        rag_manager = request.app.state.rag_manager
        tenant_service = rag_manager.tenant_service

        # If no user_id provided, try to get first public tenant
        if not user_id:
            tenants_result = await tenant_service.list_tenants(limit=1)
            if tenants_result["items"]:
                resolved_id = tenants_result["items"][0].tenant_id
                logger.info(f"Resolved 'default' tenant_id to {resolved_id} (no user)")
                return resolved_id
            return tenant_id

        # Get user's accessible tenants
        user_tenants = await tenant_service.get_user_tenants(user_id=user_id, limit=1)

        if user_tenants["items"] and len(user_tenants["items"]) > 0:
            resolved_id = user_tenants["items"][0].tenant_id
            logger.info(
                f"Resolved 'default' tenant_id to {resolved_id} for user {user_id}"
            )
            return resolved_id
        else:
            # No accessible tenants - create a default one for the user
            logger.warning(
                f"User {user_id} has no accessible tenants. Creating default tenant."
            )
            try:
                new_tenant = await tenant_service.create_tenant(
                    tenant_name=f"Default Tenant for {user_id}",
                    description="Automatically created default tenant",
                    created_by=user_id,
                    metadata={},
                )
                # Add user as owner
                await tenant_service.add_user_to_tenant(
                    user_id=user_id,
                    tenant_id=new_tenant.tenant_id,
                    role="owner",
                    created_by="system",
                )
                logger.info(
                    f"Created default tenant: {new_tenant.tenant_id} for user {user_id}"
                )
                return new_tenant.tenant_id
            except Exception as create_error:
                logger.error(f"Failed to create default tenant: {create_error}")
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User has no accessible tenants and failed to create default tenant",
                )

    except Exception as e:
        logger.error(f"Error resolving default tenant: {e}")
        raise

    return tenant_id


async def resolve_default_kb(
    request: Request, tenant_id: str, kb_id: str, user_id: str = None
) -> str:
    """Resolve 'default' kb_id to first available KB for the tenant.

    Args:
        request: FastAPI request object
        tenant_id: Tenant ID
        kb_id: KB ID (may be "default")
        user_id: User identifier from token

    Returns:
        Resolved KB ID
    """
    if kb_id != "default":
        return kb_id

    try:
        # Access rag_manager from app state
        if not hasattr(request.app.state, "rag_manager"):
            return kb_id

        rag_manager = request.app.state.rag_manager
        tenant_service = rag_manager.tenant_service

        # Verify user has access to tenant if user_id provided
        if user_id:
            has_access = await tenant_service.verify_user_access(
                user_id=user_id, tenant_id=tenant_id, required_role="viewer"
            )
            if not has_access:
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User {user_id} has no access to tenant {tenant_id}",
                )

        # Get first available KB for tenant
        kbs_result = await tenant_service.list_knowledge_bases(tenant_id, limit=1)
        if kbs_result["items"]:
            resolved_id = kbs_result["items"][0].kb_id
            logger.info(f"Resolved 'default' kb_id to {resolved_id}")
            return resolved_id
        else:
            logger.warning(
                f"No KBs found for tenant {tenant_id} to resolve 'default'. Creating default KB."
            )
            try:
                new_kb = await tenant_service.create_knowledge_base(
                    tenant_id=tenant_id,
                    kb_name="Default Knowledge Base",
                    description="Automatically created default knowledge base",
                    created_by=user_id or "system",
                )
                logger.info(f"Created default KB: {new_kb.kb_id}")
                return new_kb.kb_id
            except Exception as create_error:
                logger.error(f"Failed to create default KB: {create_error}")
                return kb_id

    except Exception as e:
        logger.error(f"Error resolving default KB: {e}")
        raise

    return kb_id


async def get_tenant_context(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_kb_id: Optional[str] = Header(None, alias="X-KB-ID"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> TenantContext:
    """Extract and validate tenant context from request headers.

    Multi-tenant requests must include:
    - Authorization header with JWT token containing tenant_id
    - OR X-API-Key header with valid API key
    - OR no auth if auth is not configured (auth_mode=disabled)
    - X-Tenant-ID header (optional, if not in token)
    - X-KB-ID header (optional, if not in token)

    Args:
        request: FastAPI request object
        authorization: Authorization header with Bearer token
        x_tenant_id: Tenant ID from custom header
        x_kb_id: Knowledge base ID from custom header
        x_api_key: API Key from custom header

    Returns:
        TenantContext: Validated tenant context for the request

    Raises:
        HTTPException: If tenant context cannot be extracted or validated
    """
    username = None
    role_str = "viewer"
    metadata = {}

    # Check if authentication is configured
    api_key = os.getenv("LIGHTRAG_API_KEY") or global_args.key
    api_key_configured = bool(api_key)
    auth_configured = bool(auth_handler.accounts)

    # Check API Key first
    if api_key_configured and x_api_key and x_api_key == api_key:
        username = "system_admin"
        role_str = "admin"
    elif authorization:
        # Extract token from "Bearer <token>" format
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                raise ValueError("Invalid auth scheme")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format. Use: Bearer <token>",
            )

        # Validate token
        try:
            token_data = auth_handler.validate_token(token)
        except Exception as e:
            # Log the reason for failure to help debug 401s in server logs
            logger.warning(
                f"Token validation error while processing request from {getattr(request.client, 'host', 'unknown')}: {e}"
            )
            # Re-raise so FastAPI converts it to an HTTPException response
            raise
        username = token_data.get("username")
        metadata = token_data.get("metadata", {})
        role_str = token_data.get("role", "viewer")
    elif not auth_configured and not api_key_configured:
        # No auth configured - allow unauthenticated access with guest user
        username = "guest"
        role_str = "viewer"
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header or API Key",
        )

    # Determine tenant_id
    # Priority:
    # 1. Middleware state (Subdomain/JWT extracted early)
    # 2. Token metadata
    # 3. X-Tenant-ID (Fallback)

    middleware_tenant_id = getattr(request.state, "tenant_id", None)
    token_tenant_id = metadata.get("tenant_id")

    if middleware_tenant_id:
        tenant_id = middleware_tenant_id
        # Ensure token doesn't contradict middleware (if token has tenant_id)
        if token_tenant_id and token_tenant_id != middleware_tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID mismatch between subdomain/context and token",
            )
    else:
        tenant_id = token_tenant_id or x_tenant_id

    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing tenant_id in token, subdomain, or X-Tenant-ID header",
        )

    # Resolve default tenant
    tenant_id = await resolve_default_tenant(request, tenant_id, user_id=username)

    # Extract kb_id from token metadata or header
    kb_id = metadata.get("kb_id") or x_kb_id
    if not kb_id:
        # If tenant_id is present, default to "default"
        if tenant_id:
            kb_id = "default"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing kb_id in token or X-KB-ID header",
            )

    # Resolve default kb
    kb_id = await resolve_default_kb(request, tenant_id, kb_id, user_id=username)

    # Create and return tenant context
    context = TenantContext(
        tenant_id=tenant_id, kb_id=kb_id, user_id=username, role=role_str
    )

    logger.debug(
        f"Extracted TenantContext: tenant={tenant_id}, kb={kb_id}, user={username}, role={role_str}"
    )

    return context


async def get_tenant_context_optional(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_kb_id: Optional[str] = Header(None, alias="X-KB-ID"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[TenantContext]:
    """Extract tenant context from request headers (optional).

    In strict multi-tenant mode (LIGHTRAG_MULTI_TENANT_STRICT=true), this function
    will raise an error if tenant context cannot be extracted, preventing fallback
    to global RAG which could cause data leakage.

    In non-strict mode (default), returns None if tenant context is missing,
    allowing backward compatibility with single-tenant deployments.

    Returns:
        TenantContext or None: Validated tenant context, or None if not provided

    Raises:
        HTTPException: In strict mode, if tenant context is missing
    """
    logger.debug(
        f"get_tenant_context_optional: auth={bool(authorization)}, tenant_id={x_tenant_id}, kb_id={x_kb_id}"
    )

    # SEC-001 FIX: Check if strict multi-tenant mode is enabled
    try:
        from lightrag.api.config import MULTI_TENANT_STRICT_MODE

        strict_mode = MULTI_TENANT_STRICT_MODE
    except ImportError:
        strict_mode = False

    # If X-Tenant-ID is explicitly provided, we must try to resolve the tenant context
    # and propagate any errors (like missing auth or invalid KB) instead of falling back to global RAG.
    if x_tenant_id:
        return await get_tenant_context(
            request, authorization, x_tenant_id, x_kb_id, x_api_key
        )

    try:
        return await get_tenant_context(
            request, authorization, x_tenant_id, x_kb_id, x_api_key
        )
    except HTTPException as e:
        if strict_mode:
            # SEC-001 FIX: In strict mode, don't allow fallback to global RAG
            logger.error(
                f"Tenant context required in strict mode but not provided: {e.detail}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant context required. Provide X-Tenant-ID and X-KB-ID headers.",
            )
        logger.warning(f"Failed to extract tenant context (optional): {e.detail}")
        return None


async def get_tenant_context_no_kb(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> TenantContext:
    """Extract tenant context without requiring kb_id.

    Used for tenant-level operations that don't require a specific KB.

    Args:
        request: FastAPI request object
        authorization: Authorization header with Bearer token
        x_tenant_id: Tenant ID from custom header

    Returns:
        TenantContext: Validated tenant context (kb_id set to placeholder)

    Raises:
        HTTPException: If authentication or tenant_id is missing
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )

    # Extract token from "Bearer <token>" format
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError("Invalid auth scheme")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use: Bearer <token>",
        )

    # Validate token
    try:
        token_data = auth_handler.validate_token(token)
    except Exception as e:
        logger.warning(
            f"Token validation error (no-kb) while processing request from {getattr(request.client, 'host', 'unknown')}: {e}"
        )
        raise
    username = token_data.get("username")
    metadata = token_data.get("metadata", {})
    role_str = token_data.get("role", "viewer")

    # Determine tenant_id
    # Priority:
    # 1. Middleware state (Subdomain/JWT extracted early)
    # 2. Token metadata
    # 3. X-Tenant-ID (Fallback)

    middleware_tenant_id = getattr(request.state, "tenant_id", None)
    token_tenant_id = metadata.get("tenant_id")

    if middleware_tenant_id:
        tenant_id = middleware_tenant_id
        # Ensure token doesn't contradict middleware (if token has tenant_id)
        if token_tenant_id and token_tenant_id != middleware_tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID mismatch between subdomain/context and token",
            )
    else:
        tenant_id = token_tenant_id or x_tenant_id

    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing tenant_id in token, subdomain, or X-Tenant-ID header",
        )

    # Resolve default tenant
    tenant_id = await resolve_default_tenant(request, tenant_id, user_id=username)

    # Create context with placeholder kb_id for tenant-level operations
    context = TenantContext(
        tenant_id=tenant_id,
        kb_id="",  # Placeholder for tenant-level operations
        user_id=username,
        role=role_str,
    )

    logger.debug(
        f"Extracted TenantContext (no KB): tenant={tenant_id}, user={username}, role={role_str}"
    )

    return context


def check_permission(permission_required: str):
    """Factory function to create a dependency that checks for specific permission.

    Args:
        permission_required: The permission to check (e.g., "query:run")

    Returns:
        Async function that can be used as FastAPI dependency
    """

    async def verify_permission(
        context: TenantContext = Depends(get_tenant_context),
    ) -> TenantContext:
        """Verify that user has required permission."""
        from lightrag.models.tenant import Permission

        try:
            perm = Permission(permission_required)
        except ValueError:
            logger.error(f"Invalid permission: {permission_required}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid permission check configuration",
            )

        if not context.has_permission(perm):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have permission: {permission_required}",
            )

        return context

    return verify_permission


async def get_admin_context(
    authorization: Optional[str] = Header(None),
) -> dict:
    """Extract and validate admin context from request headers.

    For admin-only operations like creating initial tenants.
    Only requires valid authentication, no tenant_id needed.

    Args:
        authorization: Authorization header with Bearer token

    Returns:
        dict: Token data with user information

    Raises:
        HTTPException: If authentication fails
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )

    # Extract token from "Bearer <token>" format
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError("Invalid auth scheme")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use: Bearer <token>",
        )

    # Validate token
    token_data = auth_handler.validate_token(token)
    print(f"DEBUG: get_admin_context token_data={token_data}")

    # Check for admin role
    if token_data.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )

    logger.debug(f"Admin context extracted for user: {token_data.get('username')}")

    return token_data
