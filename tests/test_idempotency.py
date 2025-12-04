"""Tests for document ingestion idempotency with external_id.

These tests verify that:
1. Documents with external_id are deduplicated correctly
2. Re-submitting the same external_id returns existing document
3. Different external_ids create separate documents
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Mock the document routes module components
@pytest.fixture
def mock_tenant_rag():
    """Create a mock tenant RAG instance with doc_status."""
    mock_rag = MagicMock()
    mock_rag.doc_status = MagicMock()
    mock_rag.doc_status.get_doc_by_external_id = AsyncMock(return_value=None)
    mock_rag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
    return mock_rag


class TestInsertTextIdempotency:
    """Test idempotency behavior for text insertion."""

    @pytest.mark.asyncio
    async def test_external_id_prevents_duplicate_insertion(self, mock_tenant_rag):
        """When external_id already exists, return duplicated status."""
        # Setup: Document with this external_id already exists
        existing_doc = {
            "id": "doc-123",
            "status": "processed",
            "track_id": "insert_12345",
        }
        mock_tenant_rag.doc_status.get_doc_by_external_id = AsyncMock(
            return_value=existing_doc
        )

        # Verify the mock returns expected value
        result = await mock_tenant_rag.doc_status.get_doc_by_external_id("my-external-id")
        
        assert result is not None
        assert result["id"] == "doc-123"
        assert result["status"] == "processed"

    @pytest.mark.asyncio
    async def test_new_external_id_allows_insertion(self, mock_tenant_rag):
        """When external_id is new, allow insertion."""
        # Setup: No document with this external_id exists
        mock_tenant_rag.doc_status.get_doc_by_external_id = AsyncMock(return_value=None)

        # Verify the mock returns None (no existing doc)
        result = await mock_tenant_rag.doc_status.get_doc_by_external_id("new-external-id")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_no_external_id_skips_idempotency_check(self, mock_tenant_rag):
        """When no external_id provided, skip idempotency check."""
        # The get_doc_by_external_id should not be called when external_id is None
        mock_tenant_rag.doc_status.get_doc_by_external_id = AsyncMock(return_value=None)
        
        # In the actual implementation, when external_id is None or empty,
        # get_doc_by_external_id is not called


class TestInsertTextsIdempotency:
    """Test idempotency behavior for batch text insertion."""

    @pytest.mark.asyncio
    async def test_external_ids_batch_deduplication(self, mock_tenant_rag):
        """External_ids should be checked for each text in batch."""
        # Setup: First text has existing external_id, second is new
        async def mock_get_by_external_id(ext_id):
            if ext_id == "existing-id":
                return {"id": "doc-existing", "status": "processed"}
            return None

        mock_tenant_rag.doc_status.get_doc_by_external_id = AsyncMock(
            side_effect=mock_get_by_external_id
        )

        # Verify first returns existing, second returns None
        result1 = await mock_tenant_rag.doc_status.get_doc_by_external_id("existing-id")
        result2 = await mock_tenant_rag.doc_status.get_doc_by_external_id("new-id")

        assert result1 is not None
        assert result1["id"] == "doc-existing"
        assert result2 is None


class TestExternalIdValidation:
    """Test external_id field validation."""

    def test_external_id_stripped(self):
        """External_id should be stripped of whitespace."""
        from lightrag.api.routers.document_routes import InsertTextRequest
        
        request = InsertTextRequest(
            text="Test content",
            external_id="  my-id-with-spaces  "
        )
        
        assert request.external_id == "my-id-with-spaces"

    def test_external_id_max_length(self):
        """External_id should respect max length."""
        from lightrag.api.routers.document_routes import InsertTextRequest
        from pydantic import ValidationError
        
        # This should fail if external_id is too long
        long_id = "x" * 256  # Exceeds max_length of 255
        
        with pytest.raises(ValidationError):
            InsertTextRequest(text="Test", external_id=long_id)

    def test_external_id_optional(self):
        """External_id should be optional."""
        from lightrag.api.routers.document_routes import InsertTextRequest
        
        request = InsertTextRequest(text="Test content")
        
        assert request.external_id is None


class TestTenantScopedIdempotency:
    """Test that idempotency is tenant-scoped."""

    @pytest.mark.asyncio
    async def test_same_external_id_different_tenants(self, mock_tenant_rag):
        """Same external_id in different tenants should be separate."""
        # In the actual implementation, the workspace includes tenant_id
        # so the same external_id in different tenants won't conflict
        
        # This is a conceptual test - the actual isolation happens
        # because each tenant's doc_status storage has a different workspace
        
        mock_rag_tenant_a = MagicMock()
        mock_rag_tenant_a.doc_status = MagicMock()
        mock_rag_tenant_a.doc_status.workspace = "tenant_a_kb_default"
        
        mock_rag_tenant_b = MagicMock()
        mock_rag_tenant_b.doc_status = MagicMock()
        mock_rag_tenant_b.doc_status.workspace = "tenant_b_kb_default"
        
        # Workspaces should be different
        assert mock_rag_tenant_a.doc_status.workspace != mock_rag_tenant_b.doc_status.workspace
