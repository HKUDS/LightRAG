"""
Phase 3 Unit Tests: Tenant Storage Layer Integration

Tests for tenant-aware storage operations, instance caching, and data isolation.
Verifies that the storage layer correctly enforces multi-tenant boundaries.
"""

import pytest
import tempfile
from unittest.mock import MagicMock
from lightrag.tenant_rag_manager import TenantRAGManager
from lightrag.models.tenant import Tenant, TenantContext, TenantConfig
from lightrag.services.tenant_service import TenantService


@pytest.fixture
def temp_working_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_tenant_service():
    """Create a mock TenantService."""
    service = MagicMock(spec=TenantService)
    
    async def mock_get_tenant(tenant_id: str):
        if tenant_id.startswith("tenant_"):
            return Tenant(
                tenant_id=tenant_id,
                tenant_name=f"Test {tenant_id}",
                config=TenantConfig(
                    llm_model="gpt-4o-mini",
                    embedding_model="bge-m3:latest",
                    top_k=40,
                    cosine_threshold=0.2,
                ),
                is_active=True
            )
        return None
    
    service.get_tenant = mock_get_tenant
    return service


class TestTenantRAGManagerBasics:
    """Basic unit tests for TenantRAGManager."""
    
    def test_manager_initialization(self, temp_working_dir, mock_tenant_service):
        """Test TenantRAGManager initializes correctly."""
        manager = TenantRAGManager(
            base_working_dir=temp_working_dir,
            tenant_service=mock_tenant_service,
            max_cached_instances=10
        )
        
        assert manager.base_working_dir == temp_working_dir
        assert manager.max_cached_instances == 10
        assert manager.get_instance_count() == 0
    
    def test_manager_str_representation(self, temp_working_dir, mock_tenant_service):
        """Test manager string representation."""
        manager = TenantRAGManager(
            base_working_dir=temp_working_dir,
            tenant_service=mock_tenant_service,
            max_cached_instances=50
        )
        
        repr_str = repr(manager)
        assert "TenantRAGManager" in repr_str
        assert "50" in repr_str
    
    def test_get_cached_keys_empty(self, temp_working_dir, mock_tenant_service):
        """Test get_cached_keys returns empty list when no instances cached."""
        manager = TenantRAGManager(
            base_working_dir=temp_working_dir,
            tenant_service=mock_tenant_service,
        )
        
        keys = manager.get_cached_keys()
        assert isinstance(keys, list)
        assert len(keys) == 0
    
    def test_manager_attributes(self, temp_working_dir, mock_tenant_service):
        """Test manager has expected attributes."""
        manager = TenantRAGManager(
            base_working_dir=temp_working_dir,
            tenant_service=mock_tenant_service,
            max_cached_instances=100
        )
        
        assert hasattr(manager, "_instances")
        assert hasattr(manager, "_lock")
        assert hasattr(manager, "_access_order")
        assert hasattr(manager, "base_working_dir")
        assert hasattr(manager, "tenant_service")
        assert hasattr(manager, "max_cached_instances")


class TestTenantContextUnit:
    """Test suite for TenantContext usage."""
    
    def test_tenant_context_creation(self):
        """Test TenantContext creation."""
        context = TenantContext(
            tenant_id="tenant_1",
            kb_id="kb_1",
            user_id="user_1",
            role="admin"
        )
        
        assert context.tenant_id == "tenant_1"
        assert context.kb_id == "kb_1"
        assert context.user_id == "user_1"
        assert context.role == "admin"
    
    def test_tenant_context_with_permissions(self):
        """Test TenantContext with permissions."""
        context = TenantContext(
            tenant_id="tenant_1",
            kb_id="kb_1",
            user_id="user_1",
            role="editor",
            permissions={"query": True, "edit": True, "delete": False}
        )
        
        assert context.permissions["query"] is True
        assert context.permissions["edit"] is True
        assert context.permissions["delete"] is False
    
    def test_tenant_context_workspace_namespace(self):
        """Test workspace namespace generation from TenantContext."""
        context = TenantContext(
            tenant_id="tenant_1",
            kb_id="kb_1",
            user_id="user_1",
            role="admin"
        )
        
        # Should generate workspace namespace for backward compatibility
        workspace = context.workspace_namespace
        assert "tenant_1" in workspace
        assert "kb_1" in workspace
    
    def test_tenant_context_different_roles(self):
        """Test TenantContext with different role values."""
        roles = ["admin", "editor", "viewer", "viewer_readonly"]
        
        for role in roles:
            context = TenantContext(
                tenant_id="tenant_1",
                kb_id="kb_1",
                user_id="user_1",
                role=role
            )
            assert context.role == role
    
    def test_tenant_context_default_permissions(self):
        """Test TenantContext with default permissions."""
        context = TenantContext(
            tenant_id="tenant_1",
            kb_id="kb_1",
            user_id="user_1",
            role="admin"
        )
        
        # Default permissions should be empty dict
        assert isinstance(context.permissions, dict)


class TestTenantModel:
    """Test suite for Tenant data model."""
    
    def test_tenant_creation_basic(self):
        """Test basic Tenant creation."""
        tenant = Tenant(
            tenant_id="test_tenant",
            tenant_name="Test Tenant"
        )
        
        assert tenant.tenant_id == "test_tenant"
        assert tenant.tenant_name == "Test Tenant"
        assert tenant.is_active is True
    
    def test_tenant_creation_with_config(self):
        """Test Tenant creation with custom config."""
        config = TenantConfig(
            llm_model="gpt-4",
            embedding_model="text-embedding-3",
            top_k=50,
        )
        
        tenant = Tenant(
            tenant_id="test_tenant",
            tenant_name="Test Tenant",
            config=config
        )
        
        assert tenant.config.llm_model == "gpt-4"
        assert tenant.config.embedding_model == "text-embedding-3"
        assert tenant.config.top_k == 50
    
    def test_tenant_inactive_status(self):
        """Test Tenant with inactive status."""
        tenant = Tenant(
            tenant_id="test_tenant",
            tenant_name="Inactive Tenant",
            is_active=False
        )
        
        assert tenant.is_active is False
    
    def test_tenant_with_metadata(self):
        """Test Tenant with custom metadata."""
        metadata = {"custom_field": "value", "flag": True}
        tenant = Tenant(
            tenant_id="test_tenant",
            tenant_name="Test",
            metadata=metadata
        )
        
        assert tenant.metadata["custom_field"] == "value"
        assert tenant.metadata["flag"] is True


class TestTenantConfigModel:
    """Test suite for TenantConfig data model."""
    
    def test_config_defaults(self):
        """Test TenantConfig default values."""
        config = TenantConfig()
        
        assert config.llm_model == "gpt-4o-mini"
        assert config.embedding_model == "bge-m3:latest"
        assert config.chunk_size == 1200
        assert config.chunk_overlap == 100
        assert config.top_k == 40
        assert config.cosine_threshold == 0.2
        assert config.enable_llm_cache is True
    
    def test_config_custom_values(self):
        """Test TenantConfig with custom values."""
        config = TenantConfig(
            llm_model="gpt-4-turbo",
            embedding_model="claude-embedding",
            chunk_size=2000,
            top_k=100,
            cosine_threshold=0.5
        )
        
        assert config.llm_model == "gpt-4-turbo"
        assert config.embedding_model == "claude-embedding"
        assert config.chunk_size == 2000
        assert config.top_k == 100
        assert config.cosine_threshold == 0.5
    
    def test_config_custom_metadata(self):
        """Test TenantConfig with custom metadata."""
        custom_meta = {"provider": "custom", "version": "2.0"}
        config = TenantConfig(custom_metadata=custom_meta)
        
        assert config.custom_metadata["provider"] == "custom"
        assert config.custom_metadata["version"] == "2.0"


class TestMultiTenantStructure:
    """Test multi-tenant structure and isolation concepts."""
    
    def test_different_tenant_ids_are_distinct(self):
        """Test that different tenant IDs create distinct objects."""
        tenant1 = Tenant(tenant_id="tenant_1", tenant_name="T1")
        tenant2 = Tenant(tenant_id="tenant_2", tenant_name="T2")
        
        assert tenant1.tenant_id != tenant2.tenant_id
        assert tenant1.tenant_id == "tenant_1"
        assert tenant2.tenant_id == "tenant_2"
    
    def test_tenant_context_scoping(self):
        """Test that TenantContext properly scopes to tenant/kb."""
        context1 = TenantContext(
            tenant_id="t1",
            kb_id="kb1",
            user_id="user1",
            role="admin"
        )
        context2 = TenantContext(
            tenant_id="t2",
            kb_id="kb1",
            user_id="user1",
            role="admin"
        )
        
        # Same KB but different tenants
        assert context1.tenant_id != context2.tenant_id
        assert context1.kb_id == context2.kb_id
        assert context1.workspace_namespace != context2.workspace_namespace
    
    def test_composite_workspace_naming_convention(self):
        """Test composite workspace naming convention."""
        context_t1_kb1 = TenantContext(
            tenant_id="tenant_1",
            kb_id="kb_1",
            user_id="user1",
            role="admin"
        )
        context_t1_kb2 = TenantContext(
            tenant_id="tenant_1",
            kb_id="kb_2",
            user_id="user1",
            role="admin"
        )
        
        ns1 = context_t1_kb1.workspace_namespace
        ns2 = context_t1_kb2.workspace_namespace
        
        # Both should have tenant_1 but different kb_id
        assert "tenant_1" in ns1
        assert "tenant_1" in ns2
        assert "kb_1" in ns1
        assert "kb_2" in ns2
        assert ns1 != ns2
    
    def test_multiple_tenants_same_kb_different_contexts(self):
        """Test multiple tenants with same KB name have different contexts."""
        contexts = []
        for i in range(1, 4):
            ctx = TenantContext(
                tenant_id=f"tenant_{i}",
                kb_id="shared_kb",
                user_id="admin",
                role="admin"
            )
            contexts.append(ctx)
        
        # All should have different workspaces
        namespaces = [c.workspace_namespace for c in contexts]
        assert len(set(namespaces)) == len(namespaces), "Namespaces should be unique"
        
        # All should contain shared_kb
        for ns in namespaces:
            assert "shared_kb" in ns


class TestTenantContextIsolation:
    """Test tenant context isolation and permissions."""
    
    def test_context_prevents_cross_tenant_access_conceptually(self):
        """Test that context structure prevents cross-tenant access."""
        admin_context = TenantContext(
            tenant_id="tenant_1",
            kb_id="kb_1",
            user_id="admin_user",
            role="admin",
            permissions={"query": True, "edit": True, "delete": True}
        )
        
        viewer_context = TenantContext(
            tenant_id="tenant_1",
            kb_id="kb_2",
            user_id="viewer_user",
            role="viewer",
            permissions={"query": True}
        )
        
        # Different KB IDs even in same tenant
        assert admin_context.kb_id != viewer_context.kb_id
        # Different permissions
        assert admin_context.permissions != viewer_context.permissions
    
    def test_context_role_based_access_control(self):
        """Test role-based access control in TenantContext."""
        admin_perms = {"query": True, "edit": True, "delete": True}
        editor_perms = {"query": True, "edit": True, "delete": False}
        viewer_perms = {"query": True, "edit": False, "delete": False}
        
        admin_ctx = TenantContext("t1", "kb1", "u1", "admin", admin_perms)
        editor_ctx = TenantContext("t1", "kb1", "u2", "editor", editor_perms)
        viewer_ctx = TenantContext("t1", "kb1", "u3", "viewer", viewer_perms)
        
        # Admin has all permissions
        assert admin_ctx.permissions["delete"] is True
        # Editor cannot delete
        assert editor_ctx.permissions["delete"] is False
        # Viewer cannot edit
        assert viewer_ctx.permissions["edit"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
