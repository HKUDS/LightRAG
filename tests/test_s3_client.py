"""Tests for S3 client functionality in lightrag/storage/s3_client.py.

This module tests S3 operations by mocking the aioboto3 session layer,
avoiding the moto/aiobotocore async incompatibility issue.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Note: The S3Client in lightrag uses aioboto3 which requires proper async mocking


@pytest.fixture
def aws_credentials(monkeypatch):
    """Set mock AWS credentials."""
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'testing')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'testing')
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-east-1')
    monkeypatch.setenv('S3_ACCESS_KEY_ID', 'testing')
    monkeypatch.setenv('S3_SECRET_ACCESS_KEY', 'testing')
    monkeypatch.setenv('S3_BUCKET_NAME', 'test-bucket')
    monkeypatch.setenv('S3_REGION', 'us-east-1')
    monkeypatch.setenv('S3_ENDPOINT_URL', '')


@pytest.fixture
def s3_config(aws_credentials):
    """Create S3Config for testing."""
    from lightrag.storage.s3_client import S3Config

    return S3Config(
        endpoint_url='',
        access_key_id='testing',
        secret_access_key='testing',
        bucket_name='test-bucket',
        region='us-east-1',
    )


def create_mock_s3_client():
    """Create a mock S3 client with common operations."""
    mock_client = MagicMock()

    # Storage for mock objects
    mock_client._objects = {}

    # head_bucket - succeeds (bucket exists)
    mock_client.head_bucket = AsyncMock(return_value={})

    # put_object
    async def mock_put_object(**kwargs):
        key = kwargs['Key']
        body = kwargs['Body']
        metadata = kwargs.get('Metadata', {})
        content_type = kwargs.get('ContentType', 'application/octet-stream')

        # Read body if it's a file-like object
        content = body.read() if hasattr(body, 'read') else body

        mock_client._objects[key] = {
            'Body': content,
            'Metadata': metadata,
            'ContentType': content_type,
        }
        return {'ETag': '"mock-etag"'}

    mock_client.put_object = AsyncMock(side_effect=mock_put_object)

    # get_object
    async def mock_get_object(**kwargs):
        key = kwargs['Key']
        if key not in mock_client._objects:
            from botocore.exceptions import ClientError

            raise ClientError({'Error': {'Code': 'NoSuchKey', 'Message': 'Not found'}}, 'GetObject')

        obj = mock_client._objects[key]
        body_mock = MagicMock()
        body_mock.read = AsyncMock(return_value=obj['Body'])

        return {
            'Body': body_mock,
            'Metadata': obj['Metadata'],
            'ContentType': obj['ContentType'],
        }

    mock_client.get_object = AsyncMock(side_effect=mock_get_object)

    # head_object (for object_exists)
    async def mock_head_object(**kwargs):
        key = kwargs['Key']
        if key not in mock_client._objects:
            from botocore.exceptions import ClientError

            raise ClientError({'Error': {'Code': '404', 'Message': 'Not found'}}, 'HeadObject')
        return {'ContentLength': len(mock_client._objects[key]['Body'])}

    mock_client.head_object = AsyncMock(side_effect=mock_head_object)

    # delete_object
    async def mock_delete_object(**kwargs):
        key = kwargs['Key']
        if key in mock_client._objects:
            del mock_client._objects[key]
        return {}

    mock_client.delete_object = AsyncMock(side_effect=mock_delete_object)

    # copy_object
    async def mock_copy_object(**kwargs):
        source = kwargs['CopySource']
        dest_key = kwargs['Key']

        # CopySource is like {'Bucket': 'bucket', 'Key': 'key'}
        source_key = source['Key']

        if source_key not in mock_client._objects:
            from botocore.exceptions import ClientError

            raise ClientError({'Error': {'Code': 'NoSuchKey', 'Message': 'Not found'}}, 'CopyObject')

        mock_client._objects[dest_key] = mock_client._objects[source_key].copy()
        return {}

    mock_client.copy_object = AsyncMock(side_effect=mock_copy_object)

    # list_objects_v2
    async def mock_list_objects_v2(**kwargs):
        prefix = kwargs.get('Prefix', '')
        contents = []

        for key, obj in mock_client._objects.items():
            if key.startswith(prefix):
                contents.append(
                    {
                        'Key': key,
                        'Size': len(obj['Body']),
                        'LastModified': '2024-01-01T00:00:00Z',
                    }
                )

        return {'Contents': contents} if contents else {}

    mock_client.list_objects_v2 = AsyncMock(side_effect=mock_list_objects_v2)

    # get_paginator for list_staging - returns async paginator
    class MockPaginator:
        def __init__(self, objects_dict):
            self._objects = objects_dict

        def paginate(self, **kwargs):
            return MockPaginatorIterator(self._objects, kwargs.get('Prefix', ''))

    class MockPaginatorIterator:
        def __init__(self, objects_dict, prefix):
            self._objects = objects_dict
            self._prefix = prefix
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration

            self._done = True
            from datetime import datetime

            contents = []
            for key, obj in self._objects.items():
                if key.startswith(self._prefix):
                    contents.append(
                        {
                            'Key': key,
                            'Size': len(obj['Body']),
                            'LastModified': datetime(2024, 1, 1),
                        }
                    )
            return {'Contents': contents} if contents else {}

    def mock_get_paginator(operation_name):
        return MockPaginator(mock_client._objects)

    mock_client.get_paginator = MagicMock(side_effect=mock_get_paginator)

    # generate_presigned_url - the code awaits this, so return an awaitable
    async def mock_generate_presigned_url(ClientMethod, Params, ExpiresIn=3600):
        key = Params.get('Key', 'unknown')
        bucket = Params.get('Bucket', 'bucket')
        return f'https://{bucket}.s3.amazonaws.com/{key}?signature=mock'

    mock_client.generate_presigned_url = mock_generate_presigned_url

    return mock_client


@pytest.fixture
def mock_s3_session():
    """Create a mock aioboto3 session that returns a mock S3 client."""
    mock_session = MagicMock()
    mock_client = create_mock_s3_client()

    @asynccontextmanager
    async def mock_client_context(*args, **kwargs):
        yield mock_client

    # Return a NEW context manager each time client() is called
    mock_session.client = MagicMock(side_effect=lambda *args, **kwargs: mock_client_context())

    return mock_session, mock_client


# ============================================================================
# Unit Tests for Key Generation (no mocking needed)
# ============================================================================


class TestKeyGeneration:
    """Tests for S3 key generation methods."""

    @pytest.mark.offline
    def test_make_staging_key(self, s3_config):
        """Test staging key format."""
        from lightrag.storage.s3_client import S3Client

        client = S3Client(config=s3_config)
        key = client._make_staging_key('default', 'doc123', 'report.pdf')
        assert key == 'staging/default/doc123/report.pdf'

    @pytest.mark.offline
    def test_make_staging_key_sanitizes_slashes(self, s3_config):
        """Test that slashes in filename are sanitized."""
        from lightrag.storage.s3_client import S3Client

        client = S3Client(config=s3_config)
        key = client._make_staging_key('default', 'doc123', 'path/to/file.pdf')
        assert key == 'staging/default/doc123/path_to_file.pdf'
        assert '//' not in key

    @pytest.mark.offline
    def test_make_staging_key_sanitizes_backslashes(self, s3_config):
        """Test that backslashes in filename are sanitized."""
        from lightrag.storage.s3_client import S3Client

        client = S3Client(config=s3_config)
        key = client._make_staging_key('default', 'doc123', 'path\\to\\file.pdf')
        assert key == 'staging/default/doc123/path_to_file.pdf'

    @pytest.mark.offline
    def test_make_archive_key(self, s3_config):
        """Test archive key format."""
        from lightrag.storage.s3_client import S3Client

        client = S3Client(config=s3_config)
        key = client._make_archive_key('workspace1', 'doc456', 'data.json')
        assert key == 'archive/workspace1/doc456/data.json'

    @pytest.mark.offline
    def test_staging_to_archive_key(self, s3_config):
        """Test staging to archive key transformation."""
        from lightrag.storage.s3_client import S3Client

        client = S3Client(config=s3_config)
        staging_key = 'staging/default/doc123/report.pdf'
        archive_key = client._staging_to_archive_key(staging_key)
        assert archive_key == 'archive/default/doc123/report.pdf'

    @pytest.mark.offline
    def test_staging_to_archive_key_non_staging(self, s3_config):
        """Test that non-staging keys are returned unchanged."""
        from lightrag.storage.s3_client import S3Client

        client = S3Client(config=s3_config)
        key = 'archive/default/doc123/report.pdf'
        result = client._staging_to_archive_key(key)
        assert result == key

    @pytest.mark.offline
    def test_get_s3_url(self, s3_config):
        """Test S3 URL generation."""
        from lightrag.storage.s3_client import S3Client

        client = S3Client(config=s3_config)
        url = client.get_s3_url('archive/default/doc123/report.pdf')
        assert url == 's3://test-bucket/archive/default/doc123/report.pdf'


# ============================================================================
# Integration Tests with Mocked S3 Session
# ============================================================================


@pytest.mark.offline
class TestS3ClientOperations:
    """Tests for S3 client operations using mocked session."""

    @pytest.mark.asyncio
    async def test_initialize_creates_bucket(self, s3_config, mock_s3_session):
        """Test that initialize checks bucket exists."""
        from lightrag.storage.s3_client import S3Client, S3ClientManager

        mock_session, mock_client = mock_s3_session

        with patch.object(S3ClientManager, 'get_session', return_value=mock_session):
            client = S3Client(config=s3_config)
            await client.initialize()

            assert client._initialized is True
            mock_client.head_bucket.assert_called_once()

            await client.finalize()

    @pytest.mark.asyncio
    async def test_upload_to_staging(self, s3_config, mock_s3_session):
        """Test uploading content to staging."""
        from lightrag.storage.s3_client import S3Client, S3ClientManager

        mock_session, mock_client = mock_s3_session

        with patch.object(S3ClientManager, 'get_session', return_value=mock_session):
            client = S3Client(config=s3_config)
            await client.initialize()

            s3_key = await client.upload_to_staging(
                workspace='default',
                doc_id='doc123',
                content=b'Hello, World!',
                filename='test.txt',
                content_type='text/plain',
            )

            assert s3_key == 'staging/default/doc123/test.txt'
            mock_client.put_object.assert_called_once()

            await client.finalize()

    @pytest.mark.asyncio
    async def test_upload_string_content(self, s3_config, mock_s3_session):
        """Test uploading string content (should be encoded to bytes)."""
        from lightrag.storage.s3_client import S3Client, S3ClientManager

        mock_session, _mock_client = mock_s3_session

        with patch.object(S3ClientManager, 'get_session', return_value=mock_session):
            client = S3Client(config=s3_config)
            await client.initialize()

            s3_key = await client.upload_to_staging(
                workspace='default',
                doc_id='doc123',
                content='String content',  # String, not bytes
                filename='test.txt',
            )

            # Verify we can retrieve it
            content, _metadata = await client.get_object(s3_key)
            assert content == b'String content'

            await client.finalize()

    @pytest.mark.asyncio
    async def test_get_object(self, s3_config, mock_s3_session):
        """Test retrieving uploaded object."""
        from lightrag.storage.s3_client import S3Client, S3ClientManager

        mock_session, _mock_client = mock_s3_session

        with patch.object(S3ClientManager, 'get_session', return_value=mock_session):
            client = S3Client(config=s3_config)
            await client.initialize()

            # Upload
            test_content = b'Test content for retrieval'
            s3_key = await client.upload_to_staging(
                workspace='default',
                doc_id='doc123',
                content=test_content,
                filename='test.txt',
            )

            # Retrieve
            content, metadata = await client.get_object(s3_key)

            assert content == test_content
            assert metadata.get('workspace') == 'default'
            assert metadata.get('doc_id') == 'doc123'
            assert 'content_hash' in metadata

            await client.finalize()

    @pytest.mark.asyncio
    async def test_move_to_archive(self, s3_config, mock_s3_session):
        """Test moving object from staging to archive."""
        from lightrag.storage.s3_client import S3Client, S3ClientManager

        mock_session, _mock_client = mock_s3_session

        with patch.object(S3ClientManager, 'get_session', return_value=mock_session):
            client = S3Client(config=s3_config)
            await client.initialize()

            # Upload to staging
            test_content = b'Content to archive'
            staging_key = await client.upload_to_staging(
                workspace='default',
                doc_id='doc123',
                content=test_content,
                filename='test.txt',
            )

            # Move to archive
            archive_key = await client.move_to_archive(staging_key)

            assert archive_key == 'archive/default/doc123/test.txt'

            # Verify staging key no longer exists
            assert not await client.object_exists(staging_key)

            # Verify archive key exists and has correct content
            assert await client.object_exists(archive_key)
            content, _ = await client.get_object(archive_key)
            assert content == test_content

            await client.finalize()

    @pytest.mark.asyncio
    async def test_delete_object(self, s3_config, mock_s3_session):
        """Test deleting an object."""
        from lightrag.storage.s3_client import S3Client, S3ClientManager

        mock_session, _mock_client = mock_s3_session

        with patch.object(S3ClientManager, 'get_session', return_value=mock_session):
            client = S3Client(config=s3_config)
            await client.initialize()

            # Upload
            s3_key = await client.upload_to_staging(
                workspace='default',
                doc_id='doc123',
                content=b'Content to delete',
                filename='test.txt',
            )

            assert await client.object_exists(s3_key)

            # Delete
            await client.delete_object(s3_key)

            # Verify deleted
            assert not await client.object_exists(s3_key)

            await client.finalize()

    @pytest.mark.asyncio
    async def test_list_staging(self, s3_config, mock_s3_session):
        """Test listing objects in staging."""
        from lightrag.storage.s3_client import S3Client, S3ClientManager

        mock_session, _mock_client = mock_s3_session

        with patch.object(S3ClientManager, 'get_session', return_value=mock_session):
            client = S3Client(config=s3_config)
            await client.initialize()

            # Upload multiple objects
            await client.upload_to_staging('default', 'doc1', b'Content 1', 'file1.txt')
            await client.upload_to_staging('default', 'doc2', b'Content 2', 'file2.txt')
            await client.upload_to_staging('other', 'doc3', b'Content 3', 'file3.txt')

            # List only 'default' workspace
            objects = await client.list_staging('default')

            assert len(objects) == 2
            keys = [obj['key'] for obj in objects]
            assert 'staging/default/doc1/file1.txt' in keys
            assert 'staging/default/doc2/file2.txt' in keys
            assert 'staging/other/doc3/file3.txt' not in keys

            await client.finalize()

    @pytest.mark.asyncio
    async def test_object_exists_true(self, s3_config, mock_s3_session):
        """Test object_exists returns True for existing object."""
        from lightrag.storage.s3_client import S3Client, S3ClientManager

        mock_session, _mock_client = mock_s3_session

        with patch.object(S3ClientManager, 'get_session', return_value=mock_session):
            client = S3Client(config=s3_config)
            await client.initialize()

            s3_key = await client.upload_to_staging(
                workspace='default',
                doc_id='doc123',
                content=b'Test',
                filename='test.txt',
            )

            assert await client.object_exists(s3_key) is True

            await client.finalize()

    @pytest.mark.asyncio
    async def test_object_exists_false(self, s3_config, mock_s3_session):
        """Test object_exists returns False for non-existing object."""
        from lightrag.storage.s3_client import S3Client, S3ClientManager

        mock_session, _mock_client = mock_s3_session

        with patch.object(S3ClientManager, 'get_session', return_value=mock_session):
            client = S3Client(config=s3_config)
            await client.initialize()

            assert await client.object_exists('nonexistent/key') is False

            await client.finalize()

    @pytest.mark.asyncio
    async def test_get_presigned_url(self, s3_config, mock_s3_session):
        """Test generating presigned URL."""
        from lightrag.storage.s3_client import S3Client, S3ClientManager

        mock_session, _mock_client = mock_s3_session

        with patch.object(S3ClientManager, 'get_session', return_value=mock_session):
            client = S3Client(config=s3_config)
            await client.initialize()

            s3_key = await client.upload_to_staging(
                workspace='default',
                doc_id='doc123',
                content=b'Test',
                filename='test.txt',
            )

            url = await client.get_presigned_url(s3_key)

            # URL should be a string containing the bucket
            assert isinstance(url, str)
            assert 'test-bucket' in url

            await client.finalize()

    @pytest.mark.asyncio
    async def test_upload_with_metadata(self, s3_config, mock_s3_session):
        """Test uploading with custom metadata."""
        from lightrag.storage.s3_client import S3Client, S3ClientManager

        mock_session, _mock_client = mock_s3_session

        with patch.object(S3ClientManager, 'get_session', return_value=mock_session):
            client = S3Client(config=s3_config)
            await client.initialize()

            custom_metadata = {'author': 'test-user', 'version': '1.0'}

            s3_key = await client.upload_to_staging(
                workspace='default',
                doc_id='doc123',
                content=b'Test',
                filename='test.txt',
                metadata=custom_metadata,
            )

            _, metadata = await client.get_object(s3_key)

            # Custom metadata should be included
            assert metadata.get('author') == 'test-user'
            assert metadata.get('version') == '1.0'
            # Built-in metadata should also be present
            assert metadata.get('workspace') == 'default'

            await client.finalize()


# ============================================================================
# S3Config Tests
# ============================================================================


class TestS3Config:
    """Tests for S3Config validation."""

    @pytest.mark.offline
    def test_config_requires_credentials(self, monkeypatch):
        """Test that S3Config raises error without credentials."""
        from lightrag.storage.s3_client import S3Config

        monkeypatch.setenv('S3_ACCESS_KEY_ID', '')
        monkeypatch.setenv('S3_SECRET_ACCESS_KEY', '')

        with pytest.raises(ValueError, match='S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be set'):
            S3Config(
                access_key_id='',
                secret_access_key='',
            )

    @pytest.mark.offline
    def test_config_with_valid_credentials(self, aws_credentials):
        """Test that S3Config initializes with valid credentials."""
        from lightrag.storage.s3_client import S3Config

        config = S3Config(
            access_key_id='valid-key',
            secret_access_key='valid-secret',
        )

        assert config.access_key_id == 'valid-key'
        assert config.secret_access_key == 'valid-secret'
