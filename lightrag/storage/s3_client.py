"""
Async S3 client wrapper for RustFS/MinIO/AWS S3 compatible object storage.

This module provides staging and archive functionality for documents:
- Upload to staging: s3://bucket/staging/{workspace}/{doc_id}
- Move to archive: s3://bucket/archive/{workspace}/{doc_id}
- Generate presigned URLs for citations
"""

import hashlib
import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pipmaster as pm

if not pm.is_installed('aioboto3'):
    pm.install('aioboto3')

import aioboto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.utils import logger

# Constants with environment variable support
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', '')
S3_ACCESS_KEY_ID = os.getenv('S3_ACCESS_KEY_ID', '')
S3_SECRET_ACCESS_KEY = os.getenv('S3_SECRET_ACCESS_KEY', '')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'lightrag')
S3_REGION = os.getenv('S3_REGION', 'us-east-1')
S3_RETRY_ATTEMPTS = int(os.getenv('S3_RETRY_ATTEMPTS', '3'))
S3_CONNECT_TIMEOUT = int(os.getenv('S3_CONNECT_TIMEOUT', '10'))
S3_READ_TIMEOUT = int(os.getenv('S3_READ_TIMEOUT', '30'))
S3_PRESIGNED_URL_EXPIRY = int(os.getenv('S3_PRESIGNED_URL_EXPIRY', '3600'))  # 1 hour


# Retry decorator for S3 operations
s3_retry = retry(
    stop=stop_after_attempt(S3_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(ClientError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


@dataclass
class S3Config:
    """Configuration for S3 client."""

    endpoint_url: str = field(default_factory=lambda: S3_ENDPOINT_URL)
    access_key_id: str = field(default_factory=lambda: S3_ACCESS_KEY_ID)
    secret_access_key: str = field(default_factory=lambda: S3_SECRET_ACCESS_KEY)
    bucket_name: str = field(default_factory=lambda: S3_BUCKET_NAME)
    region: str = field(default_factory=lambda: S3_REGION)
    connect_timeout: int = field(default_factory=lambda: S3_CONNECT_TIMEOUT)
    read_timeout: int = field(default_factory=lambda: S3_READ_TIMEOUT)
    presigned_url_expiry: int = field(default_factory=lambda: S3_PRESIGNED_URL_EXPIRY)

    def __post_init__(self):
        if not self.access_key_id or not self.secret_access_key:
            raise ValueError('S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be set')


class S3ClientManager:
    """Shared S3 session manager to avoid creating multiple sessions."""

    _sessions: ClassVar[dict[str, aioboto3.Session]] = {}
    _session_refs: ClassVar[dict[str, int]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_session(cls, config: S3Config) -> aioboto3.Session:
        """Get or create a session for the given S3 config."""
        # Use endpoint + access_key as session key
        session_key = f'{config.endpoint_url}:{config.access_key_id}'

        with cls._lock:
            if session_key not in cls._sessions:
                cls._sessions[session_key] = aioboto3.Session(
                    aws_access_key_id=config.access_key_id,
                    aws_secret_access_key=config.secret_access_key,
                    region_name=config.region,
                )
                cls._session_refs[session_key] = 0
                logger.info(f'Created shared S3 session for {config.endpoint_url}')

            cls._session_refs[session_key] += 1
            logger.debug(f'S3 session {session_key} reference count: {cls._session_refs[session_key]}')

        return cls._sessions[session_key]

    @classmethod
    def release_session(cls, config: S3Config):
        """Release a reference to the session."""
        session_key = f'{config.endpoint_url}:{config.access_key_id}'

        with cls._lock:
            if session_key in cls._session_refs:
                cls._session_refs[session_key] -= 1
                logger.debug(f'S3 session {session_key} reference count: {cls._session_refs[session_key]}')


@dataclass
class S3Client:
    """
    Async S3 client for document staging and archival.

    Usage:
        config = S3Config()
        client = S3Client(config)
        await client.initialize()

        # Upload to staging
        s3_key = await client.upload_to_staging(workspace, doc_id, content, filename)

        # Move to archive after processing
        archive_key = await client.move_to_archive(s3_key)

        # Get presigned URL for citations
        url = await client.get_presigned_url(archive_key)

        await client.finalize()
    """

    config: S3Config
    _session: aioboto3.Session | None = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    async def initialize(self):
        """Initialize the S3 client."""
        if self._initialized:
            return

        self._session = S3ClientManager.get_session(self.config)

        # Ensure bucket exists
        await self._ensure_bucket_exists()

        self._initialized = True
        logger.info(f'S3 client initialized for bucket: {self.config.bucket_name}')

    async def finalize(self):
        """Release resources."""
        if self._initialized:
            S3ClientManager.release_session(self.config)
            self._initialized = False
            logger.info('S3 client finalized')

    @asynccontextmanager
    async def _get_client(self):
        """Get an S3 client from the session."""
        if self._session is None:
            raise RuntimeError('S3Client not initialized')

        boto_config = BotoConfig(
            connect_timeout=self.config.connect_timeout,
            read_timeout=self.config.read_timeout,
            retries={'max_attempts': S3_RETRY_ATTEMPTS},
        )

        async with self._session.client(  # type: ignore
            's3',
            endpoint_url=self.config.endpoint_url if self.config.endpoint_url else None,
            config=boto_config,
        ) as client:
            yield client

    async def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        async with self._get_client() as client:
            try:
                await client.head_bucket(Bucket=self.config.bucket_name)
                logger.debug(f'Bucket {self.config.bucket_name} exists')
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code in ('404', 'NoSuchBucket'):
                    logger.info(f'Creating bucket: {self.config.bucket_name}')
                    create_kwargs: dict[str, Any] = {'Bucket': self.config.bucket_name}
                    if self.config.region and self.config.region != 'us-east-1':
                        create_kwargs['CreateBucketConfiguration'] = {'LocationConstraint': self.config.region}
                    await client.create_bucket(**create_kwargs)
                else:
                    raise

    def _make_staging_key(self, workspace: str, doc_id: str, filename: str) -> str:
        """Generate S3 key for staging area."""
        safe_filename = filename.replace('/', '_').replace('\\', '_')
        return f'staging/{workspace}/{doc_id}/{safe_filename}'

    def _make_archive_key(self, workspace: str, doc_id: str, filename: str) -> str:
        """Generate S3 key for archive area."""
        safe_filename = filename.replace('/', '_').replace('\\', '_')
        return f'archive/{workspace}/{doc_id}/{safe_filename}'

    def _staging_to_archive_key(self, staging_key: str) -> str:
        """Convert staging key to archive key."""
        if staging_key.startswith('staging/'):
            return 'archive/' + staging_key[8:]
        return staging_key

    @s3_retry
    async def upload_to_staging(
        self,
        workspace: str,
        doc_id: str,
        content: bytes | str,
        filename: str,
        content_type: str = 'application/octet-stream',
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Upload document to staging area.

        Args:
            workspace: Workspace/tenant identifier
            doc_id: Document ID
            content: File content (bytes or string)
            filename: Original filename
            content_type: MIME type
            metadata: Optional metadata dict

        Returns:
            S3 key for the uploaded object
        """
        s3_key = self._make_staging_key(workspace, doc_id, filename)

        if isinstance(content, str):
            content = content.encode('utf-8')

        # Calculate content hash for deduplication
        content_hash = hashlib.sha256(content).hexdigest()

        upload_metadata = {
            'workspace': workspace,
            'doc_id': doc_id,
            'original_filename': filename,
            'content_hash': content_hash,
            **(metadata or {}),
        }

        async with self._get_client() as client:
            await client.put_object(
                Bucket=self.config.bucket_name,
                Key=s3_key,
                Body=content,
                ContentType=content_type,
                Metadata=upload_metadata,
            )

        logger.info(f'Uploaded to staging: {s3_key} ({len(content)} bytes)')
        return s3_key

    @s3_retry
    async def get_object(self, s3_key: str) -> tuple[bytes, dict[str, Any]]:
        """
        Get object content and metadata.

        Returns:
            Tuple of (content_bytes, metadata_dict)
        """
        async with self._get_client() as client:
            response = await client.get_object(
                Bucket=self.config.bucket_name,
                Key=s3_key,
            )
            content = await response['Body'].read()
            metadata = response.get('Metadata', {})

        logger.debug(f'Retrieved object: {s3_key} ({len(content)} bytes)')
        return content, metadata

    @s3_retry
    async def move_to_archive(self, staging_key: str) -> str:
        """
        Move object from staging to archive.

        Args:
            staging_key: Current S3 key in staging/

        Returns:
            New S3 key in archive/
        """
        archive_key = self._staging_to_archive_key(staging_key)

        async with self._get_client() as client:
            # Copy to archive
            await client.copy_object(
                Bucket=self.config.bucket_name,
                CopySource={'Bucket': self.config.bucket_name, 'Key': staging_key},
                Key=archive_key,
            )

            # Delete from staging
            try:
                await client.delete_object(
                    Bucket=self.config.bucket_name,
                    Key=staging_key,
                )
            except Exception as e:
                logger.error(
                    'Copied %s to %s but failed to delete staging object: %s',
                    staging_key,
                    archive_key,
                    e,
                )
                raise

        logger.info(f'Moved to archive: {staging_key} -> {archive_key}')
        return archive_key

    @s3_retry
    async def delete_object(self, s3_key: str):
        """Delete an object."""
        async with self._get_client() as client:
            await client.delete_object(
                Bucket=self.config.bucket_name,
                Key=s3_key,
            )
        logger.info(f'Deleted object: {s3_key}')

    @s3_retry
    async def list_staging(self, workspace: str) -> list[dict[str, Any]]:
        """
        List all objects in staging for a workspace.

        Returns:
            List of dicts with key, size, last_modified
        """
        prefix = f'staging/{workspace}/'
        objects = []

        async with self._get_client() as client:
            paginator = client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=self.config.bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    objects.append(
                        {
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                        }
                    )

        return objects

    @s3_retry
    async def get_presigned_url(self, s3_key: str, expiry: int | None = None) -> str:
        """
        Generate a presigned URL for direct access.

        Args:
            s3_key: S3 object key
            expiry: URL expiry in seconds (default from config)

        Returns:
            Presigned URL string
        """
        expiry = expiry or self.config.presigned_url_expiry

        async with self._get_client() as client:
            url = await client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.config.bucket_name, 'Key': s3_key},
                ExpiresIn=expiry,
            )

        return url

    @s3_retry
    async def object_exists(self, s3_key: str) -> bool:
        """Check if an object exists."""
        async with self._get_client() as client:
            try:
                await client.head_object(
                    Bucket=self.config.bucket_name,
                    Key=s3_key,
                )
                return True
            except ClientError as e:
                if e.response.get('Error', {}).get('Code') == '404':
                    return False
                raise

    def get_s3_url(self, s3_key: str) -> str:
        """Get the S3 URL for an object (not presigned, for reference)."""
        return f's3://{self.config.bucket_name}/{s3_key}'

    @s3_retry
    async def list_objects(self, prefix: str = '', delimiter: str = '/') -> dict[str, Any]:
        """
        List objects and common prefixes (virtual folders) under a prefix.

        Uses delimiter to group objects into virtual folders. This enables
        folder-style navigation in the bucket browser.

        Args:
            prefix: S3 prefix to list under (e.g., "staging/default/")
            delimiter: Delimiter for grouping (default "/" for folder navigation)

        Returns:
            Dict with:
                - bucket: Bucket name
                - prefix: The prefix that was listed
                - folders: List of common prefixes (virtual folders)
                - objects: List of dicts with key, size, last_modified, content_type
        """
        folders: list[str] = []
        objects: list[dict[str, Any]] = []

        async with self._get_client() as client:
            paginator = client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(
                Bucket=self.config.bucket_name,
                Prefix=prefix,
                Delimiter=delimiter,
            ):
                # Get common prefixes (virtual folders)
                for cp in page.get('CommonPrefixes', []):
                    folders.append(cp['Prefix'])

                # Get objects at this level
                for obj in page.get('Contents', []):
                    # Skip the prefix itself if it's a "folder marker"
                    if obj['Key'] == prefix:
                        continue
                    objects.append(
                        {
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'content_type': None,  # Would need HEAD request for each
                        }
                    )

        return {
            'bucket': self.config.bucket_name,
            'prefix': prefix,
            'folders': folders,
            'objects': objects,
        }

    @s3_retry
    async def upload_object(
        self,
        key: str,
        data: bytes,
        content_type: str = 'application/octet-stream',
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Upload object to an arbitrary key path.

        Unlike upload_to_staging which enforces a path structure, this method
        allows uploading to any path in the bucket.

        Args:
            key: Full S3 key path (e.g., "staging/workspace/doc_id/file.txt")
            data: File content as bytes
            content_type: MIME type (default: application/octet-stream)
            metadata: Optional metadata dict

        Returns:
            The S3 key where the object was uploaded
        """
        async with self._get_client() as client:
            await client.put_object(
                Bucket=self.config.bucket_name,
                Key=key,
                Body=data,
                ContentType=content_type,
                Metadata=metadata or {},
            )

        logger.info(f'Uploaded object: {key} ({len(data)} bytes)')
        return key
