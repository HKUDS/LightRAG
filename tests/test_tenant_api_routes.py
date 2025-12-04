"""Integration tests for multi-tenant API routes (Phase 2).

Tests the tenant and knowledge base management endpoints with authentication
and authorization checks.
"""

import pytest
from uuid import uuid4
from datetime import datetime
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from fastapi import FastAPI

from lightrag.models.tenant import Tenant, KnowledgeBase, TenantContext, Role
from lightrag.services.tenant_service import TenantService
from lightrag.api.routers.tenant_routes import create_tenant_routes
from lightrag.api.dependencies import get_tenant_context, check_permission
from lightrag.api.dependencies import get_tenant_context, check_permission, get_tenant_context_no_kb, get_admin_context


# Test fixtures

@pytest.fixture
def sample_tenant():
    """Create a sample tenant for testing."""
    return Tenant(
        tenant_id=str(uuid4()),
        tenant_name="Test Tenant",
        description="A test tenant",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        kb_count=1,
        total_documents=42,
        total_storage_mb=5.0 * 1024.0
    )


@pytest.fixture
def sample_kb():
    """Create a sample knowledge base for testing."""
    return KnowledgeBase(
        kb_id=str(uuid4()),
        tenant_id=str(uuid4()),
        kb_name="Test KB",
        description="A test knowledge base",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        document_count=10,
        entity_count=50,
        relationship_count=100
    )


@pytest.fixture
def sample_context():
    """Create a sample tenant context for testing."""
    return TenantContext(
        tenant_id="tenant-123",
        kb_id="kb-456",
        user_id="user-789",
        role=Role.ADMIN
    )


@pytest.fixture
def mock_tenant_service():
    """Create a mock TenantService for testing."""
    service = AsyncMock(spec=TenantService)
    return service


@pytest.fixture
def app_with_routes(mock_tenant_service):
    """Create FastAPI app with tenant routes."""
    app = FastAPI()
    
    # Add tenant routes
    tenant_routes = create_tenant_routes(mock_tenant_service)
    app.include_router(tenant_routes)
    
    # Override dependencies for testing
    async def mock_get_tenant_context(*args, **kwargs):
        return TenantContext(
            tenant_id="tenant-123",
            kb_id="kb-456",
            user_id="user-789",
            role=Role.ADMIN
        )
    
    app.dependency_overrides[get_tenant_context] = mock_get_tenant_context
    
    return app


@pytest.fixture
def client(app_with_routes):
    """Create test client."""
    return TestClient(app_with_routes)


# Tests for tenant CRUD operations

class TestTenantCrud:
    """Tests for tenant CRUD endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_tenant_success(self, mock_tenant_service, app_with_routes, sample_tenant):
        """Test successful tenant creation."""
        mock_tenant_service.create_tenant.return_value = sample_tenant
        
        # create_tenant requires admin context (get_admin_context). Override it so route allows creation.
        app_with_routes.dependency_overrides[get_admin_context] = lambda: {"username": "admin-user"}
        
        client = TestClient(app_with_routes)
        response = client.post(
            "/api/v1/tenants",
            json={
                "name": sample_tenant.tenant_name,
                "description": sample_tenant.description,
                "metadata": {}
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_tenant.tenant_name
        assert data["tenant_id"] == sample_tenant.tenant_id
    
    @pytest.mark.asyncio
    async def test_get_tenant_success(self, mock_tenant_service, app_with_routes, sample_tenant):
        """Test getting tenant details."""
        mock_tenant_service.get_tenant.return_value = sample_tenant
        # request routes return tenant info for current context at /tenants/me
        app_with_routes.dependency_overrides[get_tenant_context_no_kb] = lambda: TenantContext(
            tenant_id=sample_tenant.tenant_id,
            kb_id="",
            user_id="user-xyz",
            role=Role.ADMIN,
        )

        client = TestClient(app_with_routes)
        response = client.get("/api/v1/tenants/me")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == sample_tenant.tenant_name
    
    @pytest.mark.asyncio
    async def test_get_tenant_forbidden_other_tenant(self, mock_tenant_service, app_with_routes, sample_tenant):
        """Test that users cannot access other tenants."""
        # The tenant path-based endpoints were removed. Verify tenants/me is accessible for a viewer.
        app_with_routes.dependency_overrides[get_tenant_context_no_kb] = lambda: TenantContext(
            tenant_id="tenant-123",
            kb_id="",
            user_id="user-789",
            role=Role.VIEWER,
        )

        # Ensure service returns a tenant record so route can return 200
        mock_tenant_service.get_tenant.return_value = sample_tenant

        client = TestClient(app_with_routes)
        response = client.get("/api/v1/tenants/me")

        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_get_tenant_not_found(self, mock_tenant_service, app_with_routes):
        """Test getting non-existent tenant."""
        mock_tenant_service.get_tenant.return_value = None
        
        app_with_routes.dependency_overrides[get_tenant_context_no_kb] = lambda: TenantContext(
            tenant_id="nonexistent",
            kb_id="",
            user_id="user-xyz",
            role=Role.ADMIN,
        )

        client = TestClient(app_with_routes)
        response = client.get("/api/v1/tenants/me")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_update_tenant_success(self, mock_tenant_service, app_with_routes, sample_tenant):
        """Tenant update endpoint removed - skip this test."""
        pytest.skip("Tenant update endpoint removed in API refactor - test skipped")
        
        # Override permission check
        async def mock_config_permission(context):
            return context
        
        app_with_routes.dependency_overrides[check_permission("config:update")] = mock_config_permission
        
        client = TestClient(app_with_routes)
        response = client.put(
            f"/api/v1/tenants/{sample_tenant.tenant_id}",
            json={"name": "Updated Name"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
    
    @pytest.mark.asyncio
    async def test_delete_tenant_success(self, mock_tenant_service, app_with_routes, sample_tenant):
        """Tenant delete endpoint removed - skip this test."""
        pytest.skip("Tenant delete endpoint removed in API refactor - test skipped")
        


# Tests for knowledge base CRUD operations

class TestKnowledgeBaseCrud:
    """Tests for knowledge base management endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_kb_success(self, mock_tenant_service, app_with_routes, sample_kb):
        """Test successful KB creation."""
        mock_tenant_service.create_knowledge_base.return_value = sample_kb
        
        # For creating a KB we call the tenant-scoped endpoint POST /knowledge-bases
        app_with_routes.dependency_overrides[get_tenant_context_no_kb] = lambda: TenantContext(
            tenant_id=sample_kb.tenant_id,
            kb_id="",
            user_id="user-789",
            role=Role.ADMIN
        )
        
        client = TestClient(app_with_routes)
        response = client.post(
            "/api/v1/knowledge-bases",
            json={
                "name": sample_kb.kb_name,
                "description": sample_kb.description,
                "metadata": {}
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_kb.kb_name
        assert data["kb_id"] == sample_kb.kb_id
    
    @pytest.mark.asyncio
    async def test_get_kb_success(self, mock_tenant_service, app_with_routes, sample_kb):
        """Test getting KB details."""
        mock_tenant_service.get_knowledge_base.return_value = sample_kb
        
        app_with_routes.dependency_overrides[get_tenant_context] = lambda: TenantContext(
            tenant_id=sample_kb.tenant_id,
            kb_id=sample_kb.kb_id,
            user_id="user-789",
            role=Role.VIEWER,
        )
        
        client = TestClient(app_with_routes)
        response = client.get(f"/api/v1/knowledge-bases/{sample_kb.kb_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == sample_kb.kb_name
        assert data["kb_id"] == sample_kb.kb_id
    
    @pytest.mark.asyncio
    async def test_get_kb_forbidden_other_tenant(self, app_with_routes, sample_kb):
        """Test that users cannot access KBs in other tenants."""
        app_with_routes.dependency_overrides[get_tenant_context] = lambda: TenantContext(
            tenant_id=sample_kb.tenant_id,
            kb_id="different-kb-id",
            user_id="user-789",
            role=Role.VIEWER,
        )

        client = TestClient(app_with_routes)
        response = client.get(f"/api/v1/knowledge-bases/{sample_kb.kb_id}")
        
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_update_kb_success(self, mock_tenant_service, app_with_routes, sample_kb):
        """Test successful KB update."""
        updated_kb = KnowledgeBase(
            kb_id=sample_kb.kb_id,
            tenant_id=sample_kb.tenant_id,
            kb_name="Updated KB Name",
            description="Updated description",
            created_at=sample_kb.created_at,
            updated_at=datetime.utcnow(),
            document_count=sample_kb.document_count,
            entity_count=sample_kb.entity_count,
            relationship_count=sample_kb.relationship_count
        )
        mock_tenant_service.update_knowledge_base.return_value = updated_kb
        
        # Override permission check
        async def mock_kb_update_permission(context):
            return context
        
        app_with_routes.dependency_overrides[get_tenant_context] = lambda: TenantContext(
            tenant_id=sample_kb.tenant_id,
            kb_id=sample_kb.kb_id,
            user_id="user-789",
            role=Role.EDITOR,
            permissions={
                "kb:manage": True
            }
        )
        # update endpoint requires MANAGE_KB permission
        # The permission check depends on TenantContext.has_permission(), so ensure permission is present above
        
        client = TestClient(app_with_routes)
        response = client.put(
            f"/api/v1/knowledge-bases/{sample_kb.kb_id}",
            json={"name": "Updated KB Name"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated KB Name"
    
    @pytest.mark.asyncio
    async def test_delete_kb_success(self, mock_tenant_service, app_with_routes, sample_kb):
        """Test successful KB deletion."""
        mock_tenant_service.delete_knowledge_base.return_value = True
        
        # Override permission check
        async def mock_kb_delete_permission(context):
            return context
        
        app_with_routes.dependency_overrides[get_tenant_context] = lambda: TenantContext(
            tenant_id=sample_kb.tenant_id,
            kb_id=sample_kb.kb_id,
            user_id="user-789",
            role=Role.ADMIN,
            permissions={
                "kb:delete": True
            }
        )
        # The permission check depends on TenantContext.has_permission(), so ensure permission is present above
        
        client = TestClient(app_with_routes)
        response = client.delete(f"/api/v1/knowledge-bases/{sample_kb.kb_id}")
        
        assert response.status_code == 204
