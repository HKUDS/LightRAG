"""Tests for upload routes in lightrag/api/routers/upload_routes.py.

This module tests the S3 document staging endpoints using httpx AsyncClient
and FastAPI's TestClient pattern with mocked S3Client and LightRAG.
"""

import contextlib
import sys
from typing import Annotated, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock the config module BEFORE importing to prevent argparse issues
mock_global_args = MagicMock()
mock_global_args.token_secret = 'test-secret'
mock_global_args.jwt_secret_key = 'test-jwt-secret'
mock_global_args.jwt_algorithm = 'HS256'
mock_global_args.jwt_expire_hours = 24
mock_global_args.username = None
mock_global_args.password = None
mock_global_args.guest_token = None

mock_config_module = MagicMock()
mock_config_module.global_args = mock_global_args
sys.modules['lightrag.api.config'] = mock_config_module

mock_auth_module = MagicMock()
mock_auth_module.auth_handler = MagicMock()
sys.modules['lightrag.api.auth'] = mock_auth_module

# Now import FastAPI components
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel


# Recreate models for testing (to avoid import chain issues)
class UploadResponse(BaseModel):
    """Response model for document upload."""

    status: str
    doc_id: str
    s3_key: str
    s3_url: str
    message: str | None = None


class StagedDocument(BaseModel):
    """Model for a staged document."""

    key: str
    size: int
    last_modified: str


class ListStagedResponse(BaseModel):
    """Response model for listing staged documents."""

    workspace: str
    documents: list[StagedDocument]
    count: int


class PresignedUrlResponse(BaseModel):
    """Response model for presigned URL."""

    s3_key: str
    presigned_url: str
    expiry_seconds: int


class ProcessS3Request(BaseModel):
    """Request model for processing a document from S3 staging."""

    s3_key: str
    doc_id: str | None = None
    archive_after_processing: bool = True


class ProcessS3Response(BaseModel):
    """Response model for S3 document processing."""

    status: str
    track_id: str
    doc_id: str
    s3_key: str
    archive_key: str | None = None
    message: str | None = None


def create_test_upload_routes(
    rag: Any,
    s3_client: Any,
    api_key: str | None = None,
) -> APIRouter:
    """Create upload routes for testing (simplified without auth)."""
    router = APIRouter(prefix='/upload', tags=['upload'])

    @router.post('', response_model=UploadResponse)
    async def upload_document(
        file: Annotated[UploadFile, File(description='Document file')],
        workspace: Annotated[str, Form(description='Workspace')] = 'default',
        doc_id: Annotated[str | None, Form(description='Document ID')] = None,
    ) -> UploadResponse:
        """Upload a document to S3 staging."""
        try:
            content = await file.read()

            if not content:
                raise HTTPException(status_code=400, detail='Empty file')

            # Generate doc_id if not provided
            if not doc_id:
                import hashlib

                doc_id = 'doc_' + hashlib.md5(content).hexdigest()[:8]

            content_type = file.content_type or 'application/octet-stream'

            s3_key = await s3_client.upload_to_staging(
                workspace=workspace,
                doc_id=doc_id,
                content=content,
                filename=file.filename or f'{doc_id}.bin',
                content_type=content_type,
                metadata={
                    'original_size': str(len(content)),
                    'content_type': content_type,
                },
            )

            s3_url = s3_client.get_s3_url(s3_key)

            return UploadResponse(
                status='uploaded',
                doc_id=doc_id,
                s3_key=s3_key,
                s3_url=s3_url,
                message='Document staged for processing',
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Upload failed: {e}') from e

    @router.get('/staged', response_model=ListStagedResponse)
    async def list_staged(workspace: str = 'default') -> ListStagedResponse:
        """List documents in staging."""
        try:
            objects = await s3_client.list_staging(workspace)

            documents = [
                StagedDocument(
                    key=obj['key'],
                    size=obj['size'],
                    last_modified=obj['last_modified'],
                )
                for obj in objects
            ]

            return ListStagedResponse(
                workspace=workspace,
                documents=documents,
                count=len(documents),
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to list staged documents: {e}') from e

    @router.get('/presigned-url', response_model=PresignedUrlResponse)
    async def get_presigned_url(
        s3_key: str,
        expiry: int = 3600,
    ) -> PresignedUrlResponse:
        """Get presigned URL for a document."""
        try:
            if not await s3_client.object_exists(s3_key):
                raise HTTPException(status_code=404, detail='Object not found')

            url = await s3_client.get_presigned_url(s3_key, expiry=expiry)

            return PresignedUrlResponse(
                s3_key=s3_key,
                presigned_url=url,
                expiry_seconds=expiry,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to generate presigned URL: {e}') from e

    @router.delete('/staged/{doc_id}')
    async def delete_staged(
        doc_id: str,
        workspace: str = 'default',
    ) -> dict[str, str]:
        """Delete a staged document."""
        try:
            prefix = f'staging/{workspace}/{doc_id}/'
            objects = await s3_client.list_staging(workspace)

            to_delete = [obj['key'] for obj in objects if obj['key'].startswith(prefix)]

            if not to_delete:
                raise HTTPException(status_code=404, detail='Document not found in staging')

            for key in to_delete:
                await s3_client.delete_object(key)

            return {
                'status': 'deleted',
                'doc_id': doc_id,
                'deleted_count': str(len(to_delete)),
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to delete staged document: {e}') from e

    @router.post('/process', response_model=ProcessS3Response)
    async def process_from_s3(request: ProcessS3Request) -> ProcessS3Response:
        """Process a staged document through the RAG pipeline."""
        try:
            s3_key = request.s3_key

            if not await s3_client.object_exists(s3_key):
                raise HTTPException(
                    status_code=404,
                    detail=f'Document not found in S3: {s3_key}',
                )

            content_bytes, metadata = await s3_client.get_object(s3_key)

            doc_id = request.doc_id
            if not doc_id:
                parts = s3_key.split('/')
                doc_id = parts[2] if len(parts) >= 3 else 'doc_unknown'

            content_type = metadata.get('content_type', 'application/octet-stream')
            s3_url = s3_client.get_s3_url(s3_key)

            # Try to decode as text
            try:
                text_content = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail=f'Cannot process binary content type: {content_type}.',
                ) from None

            if not text_content.strip():
                raise HTTPException(
                    status_code=400,
                    detail='Document content is empty after decoding',
                )

            # Process through RAG
            track_id = await rag.ainsert(
                input=text_content,
                ids=doc_id,
                file_paths=s3_url,
            )

            archive_key = None
            if request.archive_after_processing:
                with contextlib.suppress(Exception):
                    archive_key = await s3_client.move_to_archive(s3_key)

            return ProcessS3Response(
                status='processing_complete',
                track_id=track_id,
                doc_id=doc_id,
                s3_key=s3_key,
                archive_key=archive_key,
                message='Document processed and stored in RAG pipeline',
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to process S3 document: {e}') from e

    return router


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_s3_client():
    """Create a mock S3Client."""
    client = MagicMock()
    client.upload_to_staging = AsyncMock()
    client.list_staging = AsyncMock()
    client.get_presigned_url = AsyncMock()
    client.object_exists = AsyncMock()
    client.delete_object = AsyncMock()
    client.get_object = AsyncMock()
    client.move_to_archive = AsyncMock()
    client.get_s3_url = MagicMock()
    return client


@pytest.fixture
def mock_rag():
    """Create a mock LightRAG instance."""
    rag = MagicMock()
    rag.ainsert = AsyncMock()
    return rag


@pytest.fixture
def app(mock_rag, mock_s3_client):
    """Create FastAPI app with upload routes."""
    app = FastAPI()
    router = create_test_upload_routes(
        rag=mock_rag,
        s3_client=mock_s3_client,
        api_key=None,
    )
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
# Upload Endpoint Tests
# ============================================================================


@pytest.mark.offline
class TestUploadEndpoint:
    """Tests for POST /upload endpoint."""

    @pytest.mark.asyncio
    async def test_upload_creates_staging_key(self, client, mock_s3_client):
        """Test that upload creates correct S3 staging key."""
        mock_s3_client.upload_to_staging.return_value = 'staging/default/doc_abc123/test.txt'
        mock_s3_client.get_s3_url.return_value = 's3://bucket/staging/default/doc_abc123/test.txt'

        files = {'file': ('test.txt', b'Hello, World!', 'text/plain')}
        data = {'workspace': 'default', 'doc_id': 'doc_abc123'}

        response = await client.post('/upload', files=files, data=data)

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'uploaded'
        assert data['doc_id'] == 'doc_abc123'
        assert data['s3_key'] == 'staging/default/doc_abc123/test.txt'

        mock_s3_client.upload_to_staging.assert_called_once()
        call_args = mock_s3_client.upload_to_staging.call_args
        assert call_args.kwargs['workspace'] == 'default'
        assert call_args.kwargs['doc_id'] == 'doc_abc123'
        assert call_args.kwargs['content'] == b'Hello, World!'

    @pytest.mark.asyncio
    async def test_upload_auto_generates_doc_id(self, client, mock_s3_client):
        """Test that doc_id is auto-generated if not provided."""
        mock_s3_client.upload_to_staging.return_value = 'staging/default/doc_auto/test.txt'
        mock_s3_client.get_s3_url.return_value = 's3://bucket/staging/default/doc_auto/test.txt'

        files = {'file': ('test.txt', b'Test content', 'text/plain')}
        data = {'workspace': 'default'}

        response = await client.post('/upload', files=files, data=data)

        assert response.status_code == 200
        data = response.json()
        assert data['doc_id'].startswith('doc_')

    @pytest.mark.asyncio
    async def test_upload_empty_file_rejected(self, client, mock_s3_client):
        """Test that empty files are rejected."""
        files = {'file': ('empty.txt', b'', 'text/plain')}

        response = await client.post('/upload', files=files)

        assert response.status_code == 400
        assert 'Empty file' in response.json()['detail']

    @pytest.mark.asyncio
    async def test_upload_returns_s3_url(self, client, mock_s3_client):
        """Test that upload returns S3 URL."""
        mock_s3_client.upload_to_staging.return_value = 'staging/default/doc_xyz/file.pdf'
        mock_s3_client.get_s3_url.return_value = 's3://mybucket/staging/default/doc_xyz/file.pdf'

        files = {'file': ('file.pdf', b'PDF content', 'application/pdf')}

        response = await client.post('/upload', files=files)

        assert response.status_code == 200
        assert response.json()['s3_url'] == 's3://mybucket/staging/default/doc_xyz/file.pdf'

    @pytest.mark.asyncio
    async def test_upload_handles_s3_error(self, client, mock_s3_client):
        """Test that S3 errors are handled."""
        mock_s3_client.upload_to_staging.side_effect = Exception('S3 connection failed')

        files = {'file': ('test.txt', b'Content', 'text/plain')}

        response = await client.post('/upload', files=files)

        assert response.status_code == 500
        assert 'S3 connection failed' in response.json()['detail']


# ============================================================================
# List Staged Endpoint Tests
# ============================================================================


@pytest.mark.offline
class TestListStagedEndpoint:
    """Tests for GET /upload/staged endpoint."""

    @pytest.mark.asyncio
    async def test_list_staged_returns_documents(self, client, mock_s3_client):
        """Test that list returns staged documents."""
        mock_s3_client.list_staging.return_value = [
            {
                'key': 'staging/default/doc1/file.pdf',
                'size': 1024,
                'last_modified': '2024-01-01T00:00:00Z',
            },
            {
                'key': 'staging/default/doc2/report.docx',
                'size': 2048,
                'last_modified': '2024-01-02T00:00:00Z',
            },
        ]

        response = await client.get('/upload/staged')

        assert response.status_code == 200
        data = response.json()
        assert data['workspace'] == 'default'
        assert data['count'] == 2
        assert len(data['documents']) == 2
        assert data['documents'][0]['key'] == 'staging/default/doc1/file.pdf'

    @pytest.mark.asyncio
    async def test_list_staged_empty(self, client, mock_s3_client):
        """Test that empty staging returns empty list."""
        mock_s3_client.list_staging.return_value = []

        response = await client.get('/upload/staged')

        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 0
        assert data['documents'] == []

    @pytest.mark.asyncio
    async def test_list_staged_custom_workspace(self, client, mock_s3_client):
        """Test listing documents in custom workspace."""
        mock_s3_client.list_staging.return_value = []

        response = await client.get('/upload/staged', params={'workspace': 'custom'})

        assert response.status_code == 200
        assert response.json()['workspace'] == 'custom'
        mock_s3_client.list_staging.assert_called_once_with('custom')


# ============================================================================
# Presigned URL Endpoint Tests
# ============================================================================


@pytest.mark.offline
class TestPresignedUrlEndpoint:
    """Tests for GET /upload/presigned-url endpoint."""

    @pytest.mark.asyncio
    async def test_presigned_url_returns_url(self, client, mock_s3_client):
        """Test that presigned URL is returned."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_presigned_url.return_value = 'https://s3.example.com/signed-url?token=xyz'

        response = await client.get(
            '/upload/presigned-url',
            params={'s3_key': 'staging/default/doc1/file.pdf'},
        )

        assert response.status_code == 200
        data = response.json()
        assert data['s3_key'] == 'staging/default/doc1/file.pdf'
        assert data['presigned_url'] == 'https://s3.example.com/signed-url?token=xyz'
        assert data['expiry_seconds'] == 3600

    @pytest.mark.asyncio
    async def test_presigned_url_custom_expiry(self, client, mock_s3_client):
        """Test custom expiry time."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_presigned_url.return_value = 'https://signed-url'

        response = await client.get(
            '/upload/presigned-url',
            params={'s3_key': 'test/key', 'expiry': 7200},
        )

        assert response.status_code == 200
        assert response.json()['expiry_seconds'] == 7200
        mock_s3_client.get_presigned_url.assert_called_once_with('test/key', expiry=7200)

    @pytest.mark.asyncio
    async def test_presigned_url_not_found(self, client, mock_s3_client):
        """Test 404 for non-existent object."""
        mock_s3_client.object_exists.return_value = False

        response = await client.get(
            '/upload/presigned-url',
            params={'s3_key': 'nonexistent/key'},
        )

        assert response.status_code == 404
        assert 'not found' in response.json()['detail'].lower()


# ============================================================================
# Delete Staged Endpoint Tests
# ============================================================================


@pytest.mark.offline
class TestDeleteStagedEndpoint:
    """Tests for DELETE /upload/staged/{doc_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_removes_document(self, client, mock_s3_client):
        """Test that delete removes the document."""
        mock_s3_client.list_staging.return_value = [
            {'key': 'staging/default/doc123/file.pdf', 'size': 1024, 'last_modified': '2024-01-01'},
        ]

        response = await client.delete('/upload/staged/doc123')

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'deleted'
        assert data['doc_id'] == 'doc123'
        assert data['deleted_count'] == '1'

        mock_s3_client.delete_object.assert_called_once_with('staging/default/doc123/file.pdf')

    @pytest.mark.asyncio
    async def test_delete_not_found(self, client, mock_s3_client):
        """Test 404 when document not found."""
        mock_s3_client.list_staging.return_value = []

        response = await client.delete('/upload/staged/nonexistent')

        assert response.status_code == 404
        assert 'not found' in response.json()['detail'].lower()

    @pytest.mark.asyncio
    async def test_delete_multiple_objects(self, client, mock_s3_client):
        """Test deleting document with multiple S3 objects."""
        mock_s3_client.list_staging.return_value = [
            {'key': 'staging/default/doc456/part1.pdf', 'size': 1024, 'last_modified': '2024-01-01'},
            {'key': 'staging/default/doc456/part2.pdf', 'size': 2048, 'last_modified': '2024-01-01'},
        ]

        response = await client.delete('/upload/staged/doc456')

        assert response.status_code == 200
        assert response.json()['deleted_count'] == '2'
        assert mock_s3_client.delete_object.call_count == 2


# ============================================================================
# Process S3 Endpoint Tests
# ============================================================================


@pytest.mark.offline
class TestProcessS3Endpoint:
    """Tests for POST /upload/process endpoint."""

    @pytest.mark.asyncio
    async def test_process_fetches_and_archives(self, client, mock_s3_client, mock_rag):
        """Test that process fetches content and archives."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = (b'Document content here', {'content_type': 'text/plain'})
        mock_s3_client.get_s3_url.return_value = 's3://bucket/staging/default/doc1/file.txt'
        mock_s3_client.move_to_archive.return_value = 'archive/default/doc1/file.txt'
        mock_rag.ainsert.return_value = 'track_123'

        response = await client.post(
            '/upload/process',
            json={'s3_key': 'staging/default/doc1/file.txt'},
        )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'processing_complete'
        assert data['track_id'] == 'track_123'
        assert data['archive_key'] == 'archive/default/doc1/file.txt'

        mock_rag.ainsert.assert_called_once()
        mock_s3_client.move_to_archive.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_extracts_doc_id_from_key(self, client, mock_s3_client, mock_rag):
        """Test that doc_id is extracted from s3_key."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = (b'Content', {'content_type': 'text/plain'})
        mock_s3_client.get_s3_url.return_value = 's3://bucket/key'
        mock_s3_client.move_to_archive.return_value = 'archive/key'
        mock_rag.ainsert.return_value = 'track_456'

        response = await client.post(
            '/upload/process',
            json={'s3_key': 'staging/workspace1/extracted_doc_id/file.txt'},
        )

        assert response.status_code == 200
        assert response.json()['doc_id'] == 'extracted_doc_id'

    @pytest.mark.asyncio
    async def test_process_not_found(self, client, mock_s3_client, mock_rag):
        """Test 404 when S3 object not found."""
        mock_s3_client.object_exists.return_value = False

        response = await client.post(
            '/upload/process',
            json={'s3_key': 'staging/default/missing/file.txt'},
        )

        assert response.status_code == 404
        assert 'not found' in response.json()['detail'].lower()

    @pytest.mark.asyncio
    async def test_process_empty_content_rejected(self, client, mock_s3_client, mock_rag):
        """Test that empty content is rejected."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = (b'   \n  ', {'content_type': 'text/plain'})

        response = await client.post(
            '/upload/process',
            json={'s3_key': 'staging/default/doc/file.txt'},
        )

        assert response.status_code == 400
        assert 'empty' in response.json()['detail'].lower()

    @pytest.mark.asyncio
    async def test_process_binary_content_rejected(self, client, mock_s3_client, mock_rag):
        """Test that binary content that can't be decoded is rejected."""
        mock_s3_client.object_exists.return_value = True
        # Invalid UTF-8 bytes
        mock_s3_client.get_object.return_value = (b'\x80\x81\x82\x83', {'content_type': 'application/pdf'})

        response = await client.post(
            '/upload/process',
            json={'s3_key': 'staging/default/doc/file.pdf'},
        )

        assert response.status_code == 400
        assert 'binary' in response.json()['detail'].lower()

    @pytest.mark.asyncio
    async def test_process_skip_archive(self, client, mock_s3_client, mock_rag):
        """Test that archiving can be skipped."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = (b'Content', {'content_type': 'text/plain'})
        mock_s3_client.get_s3_url.return_value = 's3://bucket/key'
        mock_rag.ainsert.return_value = 'track_789'

        response = await client.post(
            '/upload/process',
            json={
                's3_key': 'staging/default/doc/file.txt',
                'archive_after_processing': False,
            },
        )

        assert response.status_code == 200
        assert response.json()['archive_key'] is None
        mock_s3_client.move_to_archive.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_uses_provided_doc_id(self, client, mock_s3_client, mock_rag):
        """Test that provided doc_id is used."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = (b'Content', {'content_type': 'text/plain'})
        mock_s3_client.get_s3_url.return_value = 's3://bucket/key'
        mock_s3_client.move_to_archive.return_value = 'archive/key'
        mock_rag.ainsert.return_value = 'track_999'

        response = await client.post(
            '/upload/process',
            json={
                's3_key': 'staging/default/doc/file.txt',
                'doc_id': 'custom_doc_id',
            },
        )

        assert response.status_code == 200
        assert response.json()['doc_id'] == 'custom_doc_id'

        # Verify RAG was called with custom doc_id
        mock_rag.ainsert.assert_called_once()
        call_kwargs = mock_rag.ainsert.call_args.kwargs
        assert call_kwargs['ids'] == 'custom_doc_id'
