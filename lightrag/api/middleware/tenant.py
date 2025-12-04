from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from lightrag.api.auth import auth_handler
from lightrag.utils_context import set_current_tenant_id
import logging

logger = logging.getLogger(__name__)

class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract and set tenant context from request.
    
    Priority of tenant identification:
    1. Subdomain (e.g., tenant.app.com)
    2. JWT Token (metadata.tenant_id)
    
    Sets request.state.tenant_id if found.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Skip for public endpoints
        if request.url.path in [
            "/health", 
            "/docs", 
            "/openapi.json", 
            "/redoc", 
            "/auth-status", 
            "/login",
            "/",
            "/webui"
        ] or request.url.path.startswith("/assets"):
            return await call_next(request)

        tenant_id = None
        
        # 1. Attempt Subdomain Extraction
        host = request.headers.get("host", "")
        # Simple logic: if 3 parts, first is subdomain. 
        # Adjust based on actual domain config (e.g. if using localhost)
        if "." in host:
            parts = host.split(".")
            # e.g. tenant.localhost:8000 or tenant.example.com
            if len(parts) >= 2 and not parts[0].isdigit(): 
                # Avoid IP addresses. 
                # For localhost (localhost:8000), it's just "localhost", no subdomain usually unless configured
                # For tenant.localhost, parts=['tenant', 'localhost:8000']
                if parts[0] != "www":
                    # Potential subdomain
                    # In a real app, check against a list of allowed domains or Redis
                    pass

        # 2. Attempt JWT Extraction
        # We peek at the token to get tenant_id. 
        # Full validation happens in dependencies, but we want to set state early.
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ")[1]
                # We use validate_token which verifies signature too
                token_data = auth_handler.validate_token(token)
                jwt_tenant_id = token_data.get("metadata", {}).get("tenant_id")
                
                if jwt_tenant_id:
                    tenant_id = jwt_tenant_id
            except Exception as e:
                # Token invalid or expired. 
                # We don't block here, we let dependencies.py handle 401 if auth is required.
                logger.debug(f"TenantMiddleware: Token validation failed: {e}")
                pass

        # 3. Set State
        if tenant_id:
            request.state.tenant_id = tenant_id
            # Set ContextVar for deep integration (DB layer)
            token = set_current_tenant_id(tenant_id)
            logger.debug(f"TenantMiddleware: Set tenant_id={tenant_id}")
        else:
            logger.debug("TenantMiddleware: No tenant_id found")

        response = await call_next(request)
        return response
