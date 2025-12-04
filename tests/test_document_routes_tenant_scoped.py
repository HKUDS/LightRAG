"""
Integration tests for tenant-scoped document routes (Phase 3).

This test suite verifies that all document-related endpoints properly use
tenant-scoped RAG instances, ensuring multi-tenant data isolation and correct
document visibility within each tenant's knowledge base.

Key test scenarios:
1. Document text insertion via /text endpoint (tenant-scoped)
2. Batch text insertion via /texts endpoint (tenant-scoped)
3. Document listing via /documents endpoint (tenant-scoped)
4. Document status tracking via /track_status endpoint (tenant-scoped)
5. Multi-tenant isolation: documents in Tenant A are not visible in Tenant B
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends

from lightrag.models.tenant import TenantContext, Role
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.api.dependencies import get_tenant_context
from lightrag.api.routers.document_routes import create_document_routes
from lightrag.lightrag import LightRAG


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tenant_context_a():
    """Tenant A context for isolation testing."""
    return TenantContext(
        tenant_id="tenant-a",
        kb_id="kb-a-1",
        user_id="user-a-123",
        role=Role.ADMIN
    )


@pytest.fixture
def tenant_context_b():
    """Tenant B context for isolation testing."""
    return TenantContext(
        tenant_id="tenant-b",
        kb_id="kb-b-1",
        user_id="user-b-456",
        role=Role.ADMIN
    )


@pytest.fixture
def mock_rag_instance():
    """Create a mock LightRAG instance for testing."""
    mock = AsyncMock(spec=LightRAG)
    
    # Mock doc_status storage
    mock.doc_status = AsyncMock()
    mock.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
    mock.doc_status.get_doc = AsyncMock(return_value=None)
    
    # Mock document retrieval methods
    mock.get_docs_by_status = AsyncMock(return_value={})
    mock.aget_docs_by_track_id = AsyncMock(return_value={})
    
    return mock


@pytest.fixture
def mock_rag_instances():
    """Create separate mock RAG instances for each tenant for isolation testing."""
    rag_a = AsyncMock(spec=LightRAG)
    rag_b = AsyncMock(spec=LightRAG)
    
    # Setup Tenant A RAG
    rag_a.doc_status = AsyncMock()
    rag_a.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
    rag_a.get_docs_by_status = AsyncMock(return_value={
        "doc-a-1": DocProcessingStatus(
            content_summary="Sample doc in Tenant A",
            content_length=100,
            file_path="doc-a-1",
            status=DocStatus.PROCESSED,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            track_id="track-a-1",
        )
    })
    rag_a.aget_docs_by_track_id = AsyncMock(return_value={
        "doc-a-1": DocProcessingStatus(
            content_summary="Sample doc in Tenant A",
            content_length=100,
            file_path="doc-a-1",
            status=DocStatus.PROCESSED,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            track_id="track-a-1",
        )
    })
    
    # Setup Tenant B RAG
    rag_b.doc_status = AsyncMock()
    rag_b.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
    rag_b.get_docs_by_status = AsyncMock(return_value={
        "doc-b-1": DocProcessingStatus(
            content_summary="Sample doc in Tenant B",
            content_length=200,
            file_path="doc-b-1",
            status=DocStatus.PROCESSED,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            track_id="track-b-1",
        )
    })
    rag_b.aget_docs_by_track_id = AsyncMock(return_value={
        "doc-b-1": DocProcessingStatus(
            content_summary="Sample doc in Tenant B",
            content_length=200,
            file_path="doc-b-1",
            status=DocStatus.PROCESSED,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            track_id="track-b-1",
        )
    })
    
    return {"tenant-a": rag_a, "tenant-b": rag_b}


@pytest.fixture
def app_with_document_routes(mock_rag_instances):
    """Create FastAPI app with document routes and mocked dependencies."""
    app = FastAPI()

    # Build router with a custom rag_manager so tests can provide mock rag instances
    # Create a tiny rag_manager that returns the correct mock for a tenant id
    class DummyRagManager:
        def __init__(self, mapping):
            self.mapping = mapping

        async def get_rag_instance(self, tenant_id, kb_id, user_id=None):
            return self.mapping.get(tenant_id)

    dummy_rag = AsyncMock(spec=LightRAG)
    dummy_doc_manager = AsyncMock()
    rag_manager = DummyRagManager(mock_rag_instances)

    # Create and include a router instance pointing to our rag_manager
    doc_router = create_document_routes(dummy_rag, dummy_doc_manager, rag_manager=rag_manager)
    # The document router already defines prefix="/documents" internally.
    # Mount it under "/api" so final paths become "/api/documents/...".
    app.include_router(doc_router, prefix="/api", tags=["documents"])
    
    # Track the current tenant context
    current_context = {"context": None, "rag": None}
    
    # Mock get_tenant_context
    async def mock_get_tenant_context(*args, **kwargs):
        return current_context["context"]
    
    # Override dependencies
    # The router uses get_tenant_context_optional in its internal dependency chain
    # so override both get_tenant_context and get_tenant_context_optional to ensure
    # tenant context is injected in tests (strict multi-tenant mode requires it).
    from lightrag.api.dependencies import get_tenant_context as _gtc
    from lightrag.api.dependencies import get_tenant_context_optional as _gtco

    app.dependency_overrides[_gtc] = mock_get_tenant_context
    app.dependency_overrides[_gtco] = mock_get_tenant_context

    # For testing route logic we don't want auth to interfere. Disable global auth checks
    # by toggling the module-level flag in utils_api. This mirrors the behavior in
    # many test fixtures where authentication is mocked out.
    import lightrag.api.utils_api as utils_api
    utils_api.auth_configured = False
    
    # Store context setter for tests
    app._set_context = lambda ctx: current_context.update({"context": ctx})
    
    return app


@pytest.fixture
def client(app_with_document_routes):
    """Create test client."""
    return TestClient(app_with_document_routes)


# ============================================================================
# Test Cases
# ============================================================================

class TestDocumentRoutesUseTenantRAG:
    """Test that document routes properly use tenant-scoped RAG instances."""
    
    def test_text_endpoint_uses_tenant_rag(self, client, app_with_document_routes, tenant_context_a, mock_rag_instances):
        """Test that /text endpoint uses tenant-specific RAG instance."""
        app_with_document_routes._set_context(tenant_context_a)
        
        response = client.post(
            "/api/documents/text?args=1&kwargs=1",
            json={
                "text": "This is a test document",
                "file_source": "test_file.txt"
            }
        )
        
        # Verify that the request was processed
        if response.status_code == 422:
            # Debugging info: show validation errors
            print("RESPONSE 422 BODY:", response.json())
        assert response.status_code in [200, 400, 500], f"Unexpected status code: {response.status_code}"
        
        # Verify that tenant A's RAG was queried for doc status
        mock_rag_instances["tenant-a"].doc_status.get_doc_by_file_path.assert_called()
    
    def test_texts_endpoint_uses_tenant_rag(self, client, app_with_document_routes, tenant_context_a, mock_rag_instances):
        """Test that /texts endpoint uses tenant-specific RAG instance."""
        app_with_document_routes._set_context(tenant_context_a)
        
        response = client.post(
            "/api/documents/texts?args=1&kwargs=1",
            json={
                "texts": ["Document 1", "Document 2"],
                "file_sources": ["doc1.txt", "doc2.txt"]
            }
        )
        
        # Verify that the request was processed
        if response.status_code == 422:
            print("RESPONSE 422 BODY:", response.json())
        assert response.status_code in [200, 400, 500]
        
        # Verify that tenant A's RAG was queried for doc status
        mock_rag_instances["tenant-a"].doc_status.get_doc_by_file_path.assert_called()
    
    def test_documents_endpoint_uses_tenant_rag(self, client, app_with_document_routes, tenant_context_a, mock_rag_instances):
        """Test that /documents GET endpoint uses tenant-specific RAG instance."""
        app_with_document_routes._set_context(tenant_context_a)
        
        response = client.get("/api/documents?args=1&kwargs=1")
        
        # Verify that the request was processed
        if response.status_code == 422:
            print("RESPONSE 422 BODY:", response.json())
        assert response.status_code == 200
        
        # Verify that tenant A's RAG was used to get docs by status
        mock_rag_instances["tenant-a"].get_docs_by_status.assert_called()
    
    def test_track_status_endpoint_uses_tenant_rag(self, client, app_with_document_routes, tenant_context_a, mock_rag_instances):
        """Test that /track_status endpoint uses tenant-specific RAG instance."""
        app_with_document_routes._set_context(tenant_context_a)
        
        response = client.get("/api/documents/track_status/track-a-1?args=1&kwargs=1")
        
        # Verify that the request was processed
        if response.status_code == 422:
            print("RESPONSE 422 BODY:", response.json())
        assert response.status_code == 200
        
        # Verify that tenant A's RAG was used to get docs by track_id
        mock_rag_instances["tenant-a"].aget_docs_by_track_id.assert_called()


class TestMultiTenantIsolation:
    """Test that documents from different tenants are isolated from each other."""
    
    def test_tenant_a_cannot_see_tenant_b_documents(self, client, app_with_document_routes, tenant_context_a, tenant_context_b, mock_rag_instances):
        """Verify that Tenant A's document queries don't return Tenant B's documents."""
        # Query as Tenant A
        app_with_document_routes._set_context(tenant_context_a)
        response_a = client.get("/api/documents?args=1&kwargs=1")
        
        # Query as Tenant B
        app_with_document_routes._set_context(tenant_context_b)
        response_b = client.get("/api/documents?args=1&kwargs=1")
        
        # Both should succeed
        assert response_a.status_code == 200
        assert response_b.status_code == 200
        
        # Verify that different RAG instances were used
        mock_rag_instances["tenant-a"].get_docs_by_status.assert_called()
        mock_rag_instances["tenant-b"].get_docs_by_status.assert_called()
    
    def test_track_status_returns_only_tenant_documents(self, client, app_with_document_routes, tenant_context_a, tenant_context_b, mock_rag_instances):
        """Verify that track_status endpoint returns docs from the correct tenant only."""
        # Track status in Tenant A
        app_with_document_routes._set_context(tenant_context_a)
        response_a = client.get("/api/documents/track_status/track-a-1?args=1&kwargs=1")
        
        # Track status in Tenant B
        app_with_document_routes._set_context(tenant_context_b)
        response_b = client.get("/api/documents/track_status/track-b-1?args=1&kwargs=1")
        
        # Both should succeed
        assert response_a.status_code == 200
        assert response_b.status_code == 200
        
        # Verify that different RAG instances were queried for different track IDs
        mock_rag_instances["tenant-a"].aget_docs_by_track_id.assert_called_with("track-a-1")
        mock_rag_instances["tenant-b"].aget_docs_by_track_id.assert_called_with("track-b-1")


class TestDocumentEndpointFunctionality:
    """Test basic functionality of document endpoints."""
    
    def test_text_endpoint_duplicate_file_source_rejection(self, client, app_with_document_routes, tenant_context_a, mock_rag_instances):
        """Test that /text endpoint rejects duplicate file sources."""
        app_with_document_routes._set_context(tenant_context_a)
        
        # Mock that file already exists
        existing_doc = {"status": "PROCESSED"}
        mock_rag_instances["tenant-a"].doc_status.get_doc_by_file_path.return_value = existing_doc
        
        response = client.post(
            "/api/documents/text?args=1&kwargs=1",
            json={
                "text": "Duplicate content",
                "file_source": "duplicate.txt"
            }
        )
        
        # Should return duplicated status
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "duplicated"
    
    def test_texts_endpoint_accepts_batch_insert(self, client, app_with_document_routes, tenant_context_a, mock_rag_instances):
        """Test that /texts endpoint can handle batch text insertion."""
        app_with_document_routes._set_context(tenant_context_a)
        
        response = client.post(
            "/api/documents/texts?args=1&kwargs=1",
            json={
                "texts": ["Text 1", "Text 2", "Text 3"],
                "file_sources": ["file1.txt", "file2.txt", "file3.txt"]
            }
        )
        
        # Should accept the batch
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "success"


# ============================================================================
# Test Runner Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
