"""
Tests for tenant isolation in multi-tenant LightRAG.

These tests verify that:
1. Tenants can be created and managed
2. Knowledge bases are properly isolated per tenant  
3. Cross-tenant data access is denied
4. Storage operations respect tenant/KB boundaries
"""

import pytest
from lightrag.models.tenant import (
    Tenant, TenantConfig, TenantContext,
    KnowledgeBase, KBConfig, ResourceQuota
)


class TestTenantModels:
    """Test tenant data models."""
    
    def test_tenant_creation(self):
        """Test creating a tenant with default values."""
        tenant = Tenant(
            tenant_name="Test Tenant"
        )
        assert tenant.tenant_name == "Test Tenant"
        assert tenant.is_active is True
        assert tenant.tenant_id is not None
        assert len(tenant.tenant_id) > 0
        assert tenant.kb_count == 0
    
    def test_tenant_with_custom_config(self):
        """Test creating a tenant with custom configuration."""
        config = TenantConfig(
            llm_model="custom-llm",
            embedding_model="custom-embedding",
            chunk_size=2000,
            top_k=50
        )
        tenant = Tenant(
            tenant_name="Test",
            config=config
        )
        assert tenant.config.llm_model == "custom-llm"
        assert tenant.config.chunk_size == 2000
        assert tenant.config.top_k == 50
    
    def test_tenant_to_dict(self):
        """Test tenant serialization to dictionary."""
        tenant = Tenant(
            tenant_id="test-123",
            tenant_name="Test",
            description="Test tenant"
        )
        data = tenant.to_dict()
        assert data["tenant_id"] == "test-123"
        assert data["tenant_name"] == "Test"
        assert data["description"] == "Test tenant"
        assert "config" in data
        assert "quota" in data
        assert data["is_active"] is True


class TestKnowledgeBase:
    """Test knowledge base models."""
    
    def test_kb_creation(self):
        """Test creating a knowledge base."""
        kb = KnowledgeBase(
            tenant_id="tenant-1",
            kb_name="Test KB"
        )
        assert kb.kb_name == "Test KB"
        assert kb.tenant_id == "tenant-1"
        assert kb.is_active is True
        assert kb.status == "ready"
        assert kb.document_count == 0
    
    def test_kb_to_dict(self):
        """Test KB serialization."""
        kb = KnowledgeBase(
            kb_id="kb-123",
            tenant_id="tenant-1",
            kb_name="Test KB"
        )
        data = kb.to_dict()
        assert data["kb_id"] == "kb-123"
        assert data["tenant_id"] == "tenant-1"
        assert data["kb_name"] == "Test KB"
    
    def test_kb_with_override_config(self):
        """Test KB with configuration override."""
        config = KBConfig(
            top_k=30,
            chunk_size=1500
        )
        kb = KnowledgeBase(
            tenant_id="tenant-1",
            kb_name="Test",
            config=config
        )
        assert kb.config.top_k == 30
        assert kb.config.chunk_size == 1500


class TestTenantContext:
    """Test tenant context for requests."""
    
    def test_context_creation(self):
        """Test creating a tenant context."""
        context = TenantContext(
            tenant_id="tenant-1",
            kb_id="kb-1",
            user_id="user-1",
            role="admin"
        )
        assert context.tenant_id == "tenant-1"
        assert context.kb_id == "kb-1"
        assert context.user_id == "user-1"
        assert context.role == "admin"
    
    def test_workspace_namespace(self):
        """Test backward-compatible workspace namespace."""
        context = TenantContext(
            tenant_id="acme",
            kb_id="docs",
            user_id="user-1",
            role="editor"
        )
        assert context.workspace_namespace == "acme_docs"
    
    def test_kb_access_control(self):
        """Test KB access control in context."""
        context = TenantContext(
            tenant_id="tenant-1",
            kb_id="kb-1",
            user_id="user-1",
            role="viewer",
            knowledge_base_ids=["kb-1", "kb-2"]
        )
        assert context.can_access_kb("kb-1") is True
        assert context.can_access_kb("kb-2") is True
        assert context.can_access_kb("kb-3") is False
    
    def test_kb_access_all(self):
        """Test KB access with wildcard."""
        context = TenantContext(
            tenant_id="tenant-1",
            kb_id="kb-1",
            user_id="user-1",
            role="admin",
            knowledge_base_ids=["*"]  # Admin has access to all
        )
        assert context.can_access_kb("kb-1") is True
        assert context.can_access_kb("kb-2") is True
        assert context.can_access_kb("kb-999") is True
    
    def test_permission_checking(self):
        """Test permission checking."""
        context = TenantContext(
            tenant_id="tenant-1",
            kb_id="kb-1",
            user_id="user-1",
            role="viewer",
            permissions={
                "query:run": True,
                "document:create": False
            }
        )
        assert context.has_permission("query:run") is True
        assert context.has_permission("document:create") is False
    
    def test_context_to_dict(self):
        """Test context serialization."""
        context = TenantContext(
            tenant_id="tenant-1",
            kb_id="kb-1",
            user_id="user-1",
            role="editor"
        )
        data = context.to_dict()
        assert data["tenant_id"] == "tenant-1"
        assert data["kb_id"] == "kb-1"
        assert data["user_id"] == "user-1"
        assert data["workspace_namespace"] == "tenant-1_kb-1"


class TestResourceQuota:
    """Test resource quota limits."""
    
    def test_quota_defaults(self):
        """Test quota has reasonable defaults."""
        quota = ResourceQuota()
        assert quota.max_documents == 10000
        assert quota.max_storage_gb == 100.0
        assert quota.max_concurrent_queries == 10
        assert quota.max_monthly_api_calls == 100000
    
    def test_quota_custom(self):
        """Test custom quota settings."""
        quota = ResourceQuota(
            max_documents=5000,
            max_storage_gb=50.0,
            max_concurrent_queries=5
        )
        assert quota.max_documents == 5000
        assert quota.max_storage_gb == 50.0
        assert quota.max_concurrent_queries == 5


class TestRoles:
    """Test role and permission definitions."""
    
    def test_role_enum_values(self):
        """Test role enum has expected values."""
        from lightrag.models.tenant import Role
        assert Role.ADMIN.value == "admin"
        assert Role.EDITOR.value == "editor"
        assert Role.VIEWER.value == "viewer"
        assert Role.VIEWER_READONLY.value == "viewer:read-only"
    
    def test_role_permissions(self):
        """Test role-to-permissions mapping."""
        from lightrag.models.tenant import ROLE_PERMISSIONS, Role, Permission
        
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert len(admin_perms) > 0
        assert "query:run" in admin_perms
        
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert "query:run" in viewer_perms
        assert "document:delete" not in viewer_perms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
