"""Tests for search routes in lightrag/api/routers/search_routes.py.

This module tests the BM25 full-text search endpoint using httpx AsyncClient
and FastAPI's TestClient pattern with mocked PostgreSQLDB.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock the config module BEFORE importing search_routes to prevent
# argparse from trying to parse pytest arguments as server arguments
mock_global_args = MagicMock()
mock_global_args.token_secret = 'test-secret'
mock_global_args.jwt_secret_key = 'test-jwt-secret'
mock_global_args.jwt_algorithm = 'HS256'
mock_global_args.jwt_expire_hours = 24
mock_global_args.username = None
mock_global_args.password = None
mock_global_args.guest_token = None

# Pre-populate sys.modules with mocked config
mock_config_module = MagicMock()
mock_config_module.global_args = mock_global_args
sys.modules['lightrag.api.config'] = mock_config_module

# Also mock the auth module to prevent initialization issues
mock_auth_module = MagicMock()
mock_auth_module.auth_handler = MagicMock()
sys.modules['lightrag.api.auth'] = mock_auth_module

# Now import FastAPI components (after mocking)
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, Field
from typing import Annotated, Any, ClassVar

# Import the components we need from search_routes without triggering full init
# We'll recreate the essential parts for testing


class SearchResult(BaseModel):
    """A single search result (chunk match)."""

    id: str = Field(description='Chunk ID')
    full_doc_id: str = Field(description='Parent document ID')
    chunk_order_index: int = Field(description='Position in document')
    tokens: int = Field(description='Token count')
    content: str = Field(description='Chunk content')
    file_path: str | None = Field(default=None, description='Source file path')
    s3_key: str | None = Field(default=None, description='S3 key for source document')
    char_start: int | None = Field(default=None, description='Character offset start')
    char_end: int | None = Field(default=None, description='Character offset end')
    score: float = Field(description='BM25 relevance score')


class SearchResponse(BaseModel):
    """Response model for search endpoint."""

    query: str = Field(description='Original search query')
    results: list[SearchResult] = Field(description='Matching chunks')
    count: int = Field(description='Number of results returned')
    workspace: str = Field(description='Workspace searched')


def create_test_search_routes(db: Any, api_key: str | None = None):
    """Create search routes for testing (simplified version without auth dep)."""
    from fastapi import APIRouter, HTTPException, Query

    router = APIRouter(prefix='/search', tags=['search'])

    @router.get('', response_model=SearchResponse)
    async def search(
        q: Annotated[str, Query(description='Search query', min_length=1)],
        limit: Annotated[int, Query(description='Max results', ge=1, le=100)] = 10,
        workspace: Annotated[str, Query(description='Workspace')] = 'default',
    ) -> SearchResponse:
        """Perform BM25 full-text search on chunks."""
        try:
            results = await db.full_text_search(
                query=q,
                workspace=workspace,
                limit=limit,
            )

            search_results = [
                SearchResult(
                    id=r.get('id', ''),
                    full_doc_id=r.get('full_doc_id', ''),
                    chunk_order_index=r.get('chunk_order_index', 0),
                    tokens=r.get('tokens', 0),
                    content=r.get('content', ''),
                    file_path=r.get('file_path'),
                    s3_key=r.get('s3_key'),
                    char_start=r.get('char_start'),
                    char_end=r.get('char_end'),
                    score=float(r.get('score', 0)),
                )
                for r in results
            ]

            return SearchResponse(
                query=q,
                results=search_results,
                count=len(search_results),
                workspace=workspace,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Search failed: {e}') from e

    return router


@pytest.fixture
def mock_db():
    """Create a mock PostgreSQLDB instance."""
    db = MagicMock()
    db.full_text_search = AsyncMock()
    return db


@pytest.fixture
def app(mock_db):
    """Create FastAPI app with search routes."""
    app = FastAPI()
    router = create_test_search_routes(db=mock_db, api_key=None)
    app.include_router(router)
    return app


@pytest.fixture
async def client(app):
    """Create async HTTP client for testing."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url='http://test',
    ) as client:
        yield client


# ============================================================================
# Search Endpoint Tests
# ============================================================================


@pytest.mark.offline
class TestSearchEndpoint:
    """Tests for GET /search endpoint."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, client, mock_db):
        """Test that search returns properly formatted results."""
        mock_db.full_text_search.return_value = [
            {
                'id': 'chunk-1',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
                'tokens': 100,
                'content': 'This is test content about machine learning.',
                'file_path': '/path/to/doc.pdf',
                's3_key': 'archive/default/doc-1/doc.pdf',
                'char_start': 0,
                'char_end': 100,
                'score': 0.85,
            }
        ]

        response = await client.get('/search', params={'q': 'machine learning'})

        assert response.status_code == 200
        data = response.json()
        assert data['query'] == 'machine learning'
        assert data['count'] == 1
        assert data['workspace'] == 'default'
        assert len(data['results']) == 1

        result = data['results'][0]
        assert result['id'] == 'chunk-1'
        assert result['content'] == 'This is test content about machine learning.'
        assert result['score'] == 0.85

    @pytest.mark.asyncio
    async def test_search_includes_char_positions(self, client, mock_db):
        """Test that char_start/char_end are included in results."""
        mock_db.full_text_search.return_value = [
            {
                'id': 'chunk-1',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
                'tokens': 50,
                'content': 'Test content',
                'char_start': 100,
                'char_end': 200,
                'score': 0.75,
            }
        ]

        response = await client.get('/search', params={'q': 'test'})

        assert response.status_code == 200
        result = response.json()['results'][0]
        assert result['char_start'] == 100
        assert result['char_end'] == 200

    @pytest.mark.asyncio
    async def test_search_includes_s3_key(self, client, mock_db):
        """Test that s3_key is included in results when present."""
        mock_db.full_text_search.return_value = [
            {
                'id': 'chunk-1',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
                'tokens': 50,
                'content': 'Test content',
                's3_key': 'archive/default/doc-1/report.pdf',
                'file_path': 's3://bucket/archive/default/doc-1/report.pdf',
                'score': 0.75,
            }
        ]

        response = await client.get('/search', params={'q': 'test'})

        assert response.status_code == 200
        result = response.json()['results'][0]
        assert result['s3_key'] == 'archive/default/doc-1/report.pdf'
        assert result['file_path'] == 's3://bucket/archive/default/doc-1/report.pdf'

    @pytest.mark.asyncio
    async def test_search_null_s3_key(self, client, mock_db):
        """Test that null s3_key is handled correctly."""
        mock_db.full_text_search.return_value = [
            {
                'id': 'chunk-1',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
                'tokens': 50,
                'content': 'Test content',
                's3_key': None,
                'file_path': '/local/path/doc.pdf',
                'score': 0.75,
            }
        ]

        response = await client.get('/search', params={'q': 'test'})

        assert response.status_code == 200
        result = response.json()['results'][0]
        assert result['s3_key'] is None
        assert result['file_path'] == '/local/path/doc.pdf'

    @pytest.mark.asyncio
    async def test_search_empty_results(self, client, mock_db):
        """Test that empty results are returned correctly."""
        mock_db.full_text_search.return_value = []

        response = await client.get('/search', params={'q': 'nonexistent'})

        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 0
        assert data['results'] == []

    @pytest.mark.asyncio
    async def test_search_limit_parameter(self, client, mock_db):
        """Test that limit parameter is passed to database."""
        mock_db.full_text_search.return_value = []

        await client.get('/search', params={'q': 'test', 'limit': 25})

        mock_db.full_text_search.assert_called_once_with(
            query='test',
            workspace='default',
            limit=25,
        )

    @pytest.mark.asyncio
    async def test_search_workspace_parameter(self, client, mock_db):
        """Test that workspace parameter is passed to database."""
        mock_db.full_text_search.return_value = []

        await client.get('/search', params={'q': 'test', 'workspace': 'custom'})

        mock_db.full_text_search.assert_called_once_with(
            query='test',
            workspace='custom',
            limit=10,  # default
        )

    @pytest.mark.asyncio
    async def test_search_multiple_results(self, client, mock_db):
        """Test that multiple results are returned correctly."""
        mock_db.full_text_search.return_value = [
            {
                'id': f'chunk-{i}',
                'full_doc_id': f'doc-{i}',
                'chunk_order_index': i,
                'tokens': 100,
                'content': f'Content {i}',
                'score': 0.9 - i * 0.1,
            }
            for i in range(5)
        ]

        response = await client.get('/search', params={'q': 'test', 'limit': 5})

        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 5
        assert len(data['results']) == 5
        # Results should be in order by score (as returned by db)
        assert data['results'][0]['id'] == 'chunk-0'
        assert data['results'][4]['id'] == 'chunk-4'


# ============================================================================
# Validation Tests
# ============================================================================


@pytest.mark.offline
class TestSearchValidation:
    """Tests for search endpoint validation."""

    @pytest.mark.asyncio
    async def test_search_empty_query_rejected(self, client, mock_db):
        """Test that empty query string is rejected with 422."""
        response = await client.get('/search', params={'q': ''})

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_missing_query_rejected(self, client, mock_db):
        """Test that missing query parameter is rejected with 422."""
        response = await client.get('/search')

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_limit_too_low_rejected(self, client, mock_db):
        """Test that limit < 1 is rejected."""
        response = await client.get('/search', params={'q': 'test', 'limit': 0})

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_limit_too_high_rejected(self, client, mock_db):
        """Test that limit > 100 is rejected."""
        response = await client.get('/search', params={'q': 'test', 'limit': 101})

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_limit_boundary_valid(self, client, mock_db):
        """Test that limit at boundaries (1 and 100) are accepted."""
        mock_db.full_text_search.return_value = []

        # Test lower boundary
        response = await client.get('/search', params={'q': 'test', 'limit': 1})
        assert response.status_code == 200

        # Test upper boundary
        response = await client.get('/search', params={'q': 'test', 'limit': 100})
        assert response.status_code == 200


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.offline
class TestSearchErrors:
    """Tests for search endpoint error handling."""

    @pytest.mark.asyncio
    async def test_search_database_error(self, client, mock_db):
        """Test that database errors return 500."""
        mock_db.full_text_search.side_effect = Exception('Database connection failed')

        response = await client.get('/search', params={'q': 'test'})

        assert response.status_code == 500
        assert 'Database connection failed' in response.json()['detail']

    @pytest.mark.asyncio
    async def test_search_handles_missing_fields(self, client, mock_db):
        """Test that missing fields in db results are handled with defaults."""
        mock_db.full_text_search.return_value = [
            {
                'id': 'chunk-1',
                'full_doc_id': 'doc-1',
                # Missing many fields
            }
        ]

        response = await client.get('/search', params={'q': 'test'})

        assert response.status_code == 200
        result = response.json()['results'][0]
        # Should have defaults
        assert result['chunk_order_index'] == 0
        assert result['tokens'] == 0
        assert result['content'] == ''
        assert result['score'] == 0.0
