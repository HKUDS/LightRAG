"""
Test suite for Knowledge Base document count functionality.

Tests the document count calculation for knowledge bases in multi-tenant mode.
This was a bug fix where the document count was showing 0 even when documents existed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class MockKBRow:
    """Mock database row for knowledge base."""
    kb_id: str
    tenant_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    document_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)


class TestKBDocumentCount:
    """Test document count is correctly computed for knowledge bases."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database connection."""
        db = AsyncMock()
        return db
    
    @pytest.fixture
    def mock_kv_storage(self, mock_db):
        """Create a mock KV storage with database."""
        storage = MagicMock()
        storage.db = mock_db
        return storage
    
    @pytest.mark.asyncio
    async def test_list_kbs_returns_document_count(self, mock_kv_storage):
        """Test that list_knowledge_bases returns document count from database."""
        from lightrag.services.tenant_service import TenantService
        
        # Setup mock response with document counts
        mock_row = {
            'kb_id': 'kb-main',
            'tenant_id': 'techstart',
            'name': 'Main KB',
            'description': 'Test KB',
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'document_count': 5,
            'entity_count': 100,
            'relationship_count': 50,
        }
        
        mock_kv_storage.db.query = AsyncMock(return_value=[mock_row])
        
        # Create service and call list_knowledge_bases
        service = TenantService(kv_storage=mock_kv_storage)
        result = await service.list_knowledge_bases(tenant_id='techstart')
        
        # Verify document count is returned
        assert len(result['items']) == 1
        kb = result['items'][0]
        assert kb.document_count == 5
        assert kb.entity_count == 100
        assert kb.relationship_count == 50
    
    @pytest.mark.asyncio
    async def test_list_kbs_with_zero_documents(self, mock_kv_storage):
        """Test that list_knowledge_bases correctly shows 0 documents when empty."""
        from lightrag.services.tenant_service import TenantService
        
        mock_row = {
            'kb_id': 'kb-empty',
            'tenant_id': 'techstart',
            'name': 'Empty KB',
            'description': 'Empty KB',
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'document_count': 0,
            'entity_count': 0,
            'relationship_count': 0,
        }
        
        mock_kv_storage.db.query = AsyncMock(return_value=[mock_row])
        
        service = TenantService(kv_storage=mock_kv_storage)
        result = await service.list_knowledge_bases(tenant_id='techstart')
        
        assert len(result['items']) == 1
        kb = result['items'][0]
        assert kb.document_count == 0
    
    @pytest.mark.asyncio
    async def test_get_kb_returns_document_count(self, mock_kv_storage):
        """Test that get_knowledge_base returns document count from database."""
        from lightrag.services.tenant_service import TenantService
        
        mock_row = {
            'kb_id': 'kb-main',
            'tenant_id': 'techstart',
            'name': 'Main KB',
            'description': 'Test KB',
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'document_count': 10,
            'entity_count': 200,
            'relationship_count': 100,
        }
        
        mock_kv_storage.db.query = AsyncMock(return_value=mock_row)
        
        service = TenantService(kv_storage=mock_kv_storage)
        kb = await service.get_knowledge_base(tenant_id='techstart', kb_id='kb-main')
        
        assert kb is not None
        assert kb.document_count == 10
        assert kb.entity_count == 200
        assert kb.relationship_count == 100
    
    @pytest.mark.asyncio
    async def test_list_kbs_multiple_kbs_different_counts(self, mock_kv_storage):
        """Test document counts are correct for multiple KBs."""
        from lightrag.services.tenant_service import TenantService
        
        mock_rows = [
            {
                'kb_id': 'kb-main',
                'tenant_id': 'techstart',
                'name': 'Main KB',
                'description': 'Main',
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'document_count': 5,
                'entity_count': 50,
                'relationship_count': 25,
            },
            {
                'kb_id': 'kb-backup',
                'tenant_id': 'techstart',
                'name': 'Backup KB',
                'description': 'Backup',
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'document_count': 10,
                'entity_count': 100,
                'relationship_count': 50,
            },
        ]
        
        mock_kv_storage.db.query = AsyncMock(return_value=mock_rows)
        
        service = TenantService(kv_storage=mock_kv_storage)
        result = await service.list_knowledge_bases(tenant_id='techstart')
        
        assert len(result['items']) == 2
        
        # Find KBs by ID
        kb_main = next((kb for kb in result['items'] if kb.kb_id == 'kb-main'), None)
        kb_backup = next((kb for kb in result['items'] if kb.kb_id == 'kb-backup'), None)
        
        assert kb_main is not None
        assert kb_main.document_count == 5
        
        assert kb_backup is not None
        assert kb_backup.document_count == 10
    
    @pytest.mark.asyncio
    async def test_db_query_failure_fallback(self, mock_kv_storage):
        """Test that when DB query fails, we fallback gracefully."""
        from lightrag.services.tenant_service import TenantService
        
        # Make db query raise an exception
        mock_kv_storage.db.query = AsyncMock(side_effect=Exception("DB Error"))
        mock_kv_storage.get_by_id = AsyncMock(return_value=None)
        
        service = TenantService(kv_storage=mock_kv_storage)
        result = await service.list_knowledge_bases(tenant_id='techstart')
        
        # Should return empty list gracefully
        assert result['items'] == []
        assert result['total'] == 0


class TestDocumentCountIsolation:
    """Test document counts are properly isolated between tenants."""
    
    @pytest.mark.asyncio
    async def test_different_tenants_independent_counts(self):
        """Test that document counts are independent per tenant."""
        from lightrag.services.tenant_service import TenantService
        
        # Mock for tenant A
        mock_storage_a = MagicMock()
        mock_storage_a.db = AsyncMock()
        mock_storage_a.db.query = AsyncMock(return_value=[{
            'kb_id': 'kb-main',
            'tenant_id': 'tenant-a',
            'name': 'KB A',
            'description': '',
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'document_count': 100,
            'entity_count': 0,
            'relationship_count': 0,
        }])
        
        service_a = TenantService(kv_storage=mock_storage_a)
        result_a = await service_a.list_knowledge_bases(tenant_id='tenant-a')
        
        # Mock for tenant B
        mock_storage_b = MagicMock()
        mock_storage_b.db = AsyncMock()
        mock_storage_b.db.query = AsyncMock(return_value=[{
            'kb_id': 'kb-main',
            'tenant_id': 'tenant-b',
            'name': 'KB B',
            'description': '',
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'document_count': 50,
            'entity_count': 0,
            'relationship_count': 0,
        }])
        
        service_b = TenantService(kv_storage=mock_storage_b)
        result_b = await service_b.list_knowledge_bases(tenant_id='tenant-b')
        
        # Verify counts are independent
        assert result_a['items'][0].document_count == 100
        assert result_b['items'][0].document_count == 50


class TestNullHandling:
    """Test handling of NULL values from database."""
    
    @pytest.mark.asyncio
    async def test_null_counts_default_to_zero(self):
        """Test that NULL counts default to 0."""
        from lightrag.services.tenant_service import TenantService
        
        mock_storage = MagicMock()
        mock_storage.db = AsyncMock()
        mock_storage.db.query = AsyncMock(return_value=[{
            'kb_id': 'kb-main',
            'tenant_id': 'techstart',
            'name': 'KB',
            'description': '',
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'document_count': None,  # NULL from DB
            'entity_count': None,
            'relationship_count': None,
        }])
        
        service = TenantService(kv_storage=mock_storage)
        result = await service.list_knowledge_bases(tenant_id='techstart')
        
        kb = result['items'][0]
        assert kb.document_count == 0  # NULL should become 0
        assert kb.entity_count == 0
        assert kb.relationship_count == 0
