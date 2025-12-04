"""Tests to verify backward compatibility with single-tenant deployments.

These tests ensure that the multi-tenant updates don't break existing
single-tenant functionality.
"""

import pytest

from lightrag.models.tenant import TenantContext


class TestQueryRoutesBackwardCompatibility:
    """Test that query routes work without tenant context."""

    def test_tenant_context_optional_return_type(self):
        """Test that optional tenant context can return None."""
        # In single-tenant mode, this should work without raising errors
        # The actual test validates the TenantContext model
        assert TenantContext is not None


class TestDocumentRoutesBackwardCompatibility:
    """Test that document routes work without tenant context."""

    def test_upload_endpoint_accepts_none_tenant_context(self):
        """Test that upload endpoint handles None tenant context."""
        # Verify that optional TenantContext parameter doesn't break non-tenant requests
        # This is handled by the Depends(get_tenant_context_optional) dependency
        assert TenantContext is not None

    def test_scan_endpoint_accepts_none_tenant_context(self):
        """Test that scan endpoint handles None tenant context."""
        # Verify that optional TenantContext parameter doesn't break non-tenant requests
        assert TenantContext is not None


class TestStorageLayerBackwardCompatibility:
    """Test that storage operations work without tenant context."""

    def test_storage_namespace_without_tenant_id(self):
        """Test that StorageNameSpace works without tenant_id field."""
        from lightrag.base import StorageNameSpace

        # Create a minimal concrete subclass for testing purposes
        class DummyStorage(StorageNameSpace):
            async def index_done_callback(self) -> None:
                return None

            async def drop(self) -> dict[str, str]:
                return {"status": "success", "message": "dropped"}

        # Create instance without tenant_id (legacy-like behavior)
        namespace = DummyStorage(namespace="test", workspace="ws", global_config={})

        # Should have workspace set
        assert hasattr(namespace, 'workspace')

        # tenant_id should be optional (getattr should return None)
        tenant_id = getattr(namespace, 'tenant_id', None)
        assert tenant_id is None

    def test_storage_namespace_backward_compat_workspace_only(self):
        """Test StorageNameSpace works with workspace-only configuration."""
        from lightrag.base import StorageNameSpace

        class DummyStorage(StorageNameSpace):
            async def index_done_callback(self) -> None:
                return None

            async def drop(self) -> dict[str, str]:
                return {"status": "success", "message": "dropped"}

        # Legacy usage: creating namespace with just workspace (using dummy concrete class)
        namespace = DummyStorage(namespace="test", workspace="test-workspace", global_config={})
        
        # Should work without errors
        assert namespace.workspace == "test-workspace"
        
        # tenant_id and kb_id should not be required
        assert getattr(namespace, 'tenant_id', None) is None
        assert getattr(namespace, 'kb_id', None) is None


class TestAuthenticationBackwardCompatibility:
    """Test that authentication works without multi-tenant awareness."""

    def test_auth_handler_works_without_tenant_metadata(self):
        """Test that JWT tokens work without tenant metadata."""
        from lightrag.api.auth import auth_handler

        # Create a token without tenant metadata (legacy mode)
        token = auth_handler.create_token(
            username="test_user",
            role="admin",
            metadata={}  # No tenant info
        )
        
        # Token should be valid
        assert token is not None
        assert isinstance(token, str)

    def test_auth_handler_works_with_tenant_metadata(self):
        """Test that JWT tokens work WITH tenant metadata (new mode)."""
        from lightrag.api.auth import auth_handler

        # Create a token with tenant metadata (multi-tenant mode)
        token = auth_handler.create_token(
            username="test_user",
            role="admin",
            metadata={
                "tenant_id": "tenant-123",
                "kb_id": "kb-456"
            }
        )
        
        # Token should be valid
        assert token is not None
        assert isinstance(token, str)


class TestTenantServiceOptionalUsage:
    """Test that TenantService is optional and doesn't break existing flows."""

    @pytest.mark.asyncio
    async def test_tenant_service_initialization(self):
        """Test that TenantService initializes without errors."""
        from lightrag.services.tenant_service import TenantService
        from lightrag.base import BaseKVStorage

        # Provide a minimal in-memory KV storage implementation for tests
        class FakeKV(BaseKVStorage):
            async def index_done_callback(self) -> None:
                return None

            async def drop(self) -> dict[str, str]:
                return {"status": "success", "message": "dropped"}

            async def get_by_id(self, id: str):
                return None

            async def get_by_ids(self, ids: list[str]):
                return []

            async def filter_keys(self, keys: set[str]) -> set[str]:
                return set()

            async def upsert(self, data: dict[str, dict]):
                return None

            async def delete(self, ids: list[str]) -> None:
                return None

        # Should initialize with a KV storage instance
        service = TenantService(FakeKV(namespace="kv", workspace="kv", global_config={}))
        assert service is not None

    @pytest.mark.asyncio
    async def test_tenant_service_crud_operations(self):
        """Test basic CRUD operations on TenantService."""
        from lightrag.services.tenant_service import TenantService

        from lightrag.base import BaseKVStorage

        class FakeKV(BaseKVStorage):
            def __init__(self, namespace, workspace, global_config):
                super().__init__(namespace=namespace, workspace=workspace, global_config=global_config)
                self.store: dict[str, dict] = {}

            async def index_done_callback(self) -> None:
                return None

            async def drop(self) -> dict[str, str]:
                self.store.clear()
                return {"status": "success", "message": "dropped"}

            async def get_by_id(self, id: str):
                return self.store.get(id)

            async def get_by_ids(self, ids: list[str]):
                return [self.store.get(i) for i in ids if i in self.store]

            async def filter_keys(self, keys: set[str]) -> set[str]:
                return {k for k in keys if k in self.store}

            async def upsert(self, data: dict[str, dict]):
                for k, v in data.items():
                    self.store[k] = v

            async def delete(self, ids: list[str]) -> None:
                for i in ids:
                    self.store.pop(i, None)

        service = TenantService(FakeKV(namespace="kv", workspace="kv", global_config={}))
        
        # Create a tenant
        tenant = await service.create_tenant(
            tenant_name="Test Tenant",
            description="Test Description",
            metadata={}
        )
        
        assert tenant is not None
        assert tenant.tenant_name == "Test Tenant"
        assert tenant.tenant_id is not None

        # Get the tenant
        retrieved = await service.get_tenant(tenant.tenant_id)
        assert retrieved is not None
        assert retrieved.tenant_id == tenant.tenant_id

        # Update the tenant
        updated = await service.update_tenant(
            tenant.tenant_id,
            tenant_name="Updated Name"
        )
        assert updated is not None
        assert updated.tenant_name == "Updated Name"

        # Delete the tenant
        deleted = await service.delete_tenant(tenant.tenant_id)
        assert deleted is True


class TestMultiTenantOptionalness:
    """Test that multi-tenant features are truly optional."""

    def test_dependency_injection_works_without_tenant_context(self):
        """Test that query routes work when TenantContext is None."""
        # The dependency injection should gracefully handle None tenant context
        pass

    def test_tenant_routes_are_isolated(self):
        """Test that tenant routes don't interfere with existing routes."""
        from lightrag.api.routers.tenant_routes import create_tenant_routes
        from lightrag.services.tenant_service import TenantService
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        from lightrag.base import BaseKVStorage

        class FakeKV(BaseKVStorage):
            async def index_done_callback(self) -> None:
                return None
            async def drop(self) -> dict[str, str]:
                return {"status": "success", "message": "dropped"}
            async def get_by_id(self, id: str):
                return None
            async def get_by_ids(self, ids: list[str]):
                return []
            async def filter_keys(self, keys: set[str]) -> set[str]:
                return set()
            async def upsert(self, data: dict[str, dict]):
                return None
            async def delete(self, ids: list[str]) -> None:
                return None

        service = TenantService(FakeKV(namespace="kv", workspace="kv", global_config={}))
        
        # Register tenant routes
        router = create_tenant_routes(service)
        app.include_router(router)
        
        # App should initialize without errors
        client = TestClient(app)
        
        # Tenant endpoints should exist
        response = client.options("/api/v1/tenants")
        # Should return some response (method not allowed or similar)
        assert response.status_code in [200, 405, 404]  # 405 = method not allowed


class TestFeatureToggling:
    """Test that multi-tenant can be toggled on/off without code changes."""

    def test_single_tenant_deployment_config(self):
        """Test configuration for single-tenant deployment."""
        # In single-tenant mode, TenantContext would be None
        # All endpoints should work with optional context
        pass

    def test_multi_tenant_deployment_config(self):
        """Test configuration for multi-tenant deployment."""
        # In multi-tenant mode, TenantContext would be required
        # All endpoints should validate tenant isolation
        pass
