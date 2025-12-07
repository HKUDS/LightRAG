"""
S3 Browser routes for object storage management.

This module provides endpoints for browsing, uploading, downloading,
and deleting objects in the S3/RustFS bucket.

Endpoints:
- GET /s3/list - List objects and folders under a prefix
- GET /s3/download/{key:path} - Get presigned download URL
- POST /s3/upload - Upload file to a prefix
- DELETE /s3/object/{key:path} - Delete an object
"""

import mimetypes
import traceback
from typing import Annotated, Any, ClassVar

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import BaseModel, Field

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.storage.s3_client import S3Client
from lightrag.utils import logger


class S3ObjectInfo(BaseModel):
    """Information about an S3 object."""

    key: str = Field(description='S3 object key')
    size: int = Field(description='File size in bytes')
    last_modified: str = Field(description='Last modified timestamp (ISO format)')
    content_type: str | None = Field(default=None, description='MIME content type')


class S3ListResponse(BaseModel):
    """Response model for listing S3 objects."""

    bucket: str = Field(description='Bucket name')
    prefix: str = Field(description='Current prefix path')
    folders: list[str] = Field(description='Virtual folders (common prefixes)')
    objects: list[S3ObjectInfo] = Field(description='Objects at this level')

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            'example': {
                'bucket': 'lightrag',
                'prefix': 'staging/default/',
                'folders': ['doc_abc123/', 'doc_def456/'],
                'objects': [
                    {
                        'key': 'staging/default/readme.txt',
                        'size': 1024,
                        'last_modified': '2025-12-06T10:30:00Z',
                        'content_type': 'text/plain',
                    }
                ],
            }
        }


class S3DownloadResponse(BaseModel):
    """Response model for download URL generation."""

    key: str = Field(description='S3 object key')
    url: str = Field(description='Presigned download URL')
    expiry_seconds: int = Field(description='URL expiry time in seconds')


class S3UploadResponse(BaseModel):
    """Response model for file upload."""

    key: str = Field(description='S3 object key where file was uploaded')
    size: int = Field(description='File size in bytes')
    url: str = Field(description='Presigned URL for immediate access')


class S3DeleteResponse(BaseModel):
    """Response model for object deletion."""

    key: str = Field(description='Deleted S3 object key')
    status: str = Field(description='Deletion status')


router = APIRouter(tags=['S3 Storage'])


def create_s3_routes(s3_client: S3Client, api_key: str | None = None) -> APIRouter:
    """
    Create S3 browser routes with the given S3 client.

    Args:
        s3_client: Initialized S3Client instance
        api_key: Optional API key for authentication

    Returns:
        APIRouter with S3 browser endpoints
    """
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get('/list', response_model=S3ListResponse, dependencies=[Depends(combined_auth)])
    async def list_objects(
        prefix: str = Query(default='', description='S3 prefix to list (e.g., "staging/default/")'),
    ) -> S3ListResponse:
        """
        List objects and folders under a prefix.

        This endpoint enables folder-style navigation of the S3 bucket by using
        the delimiter to group objects into virtual folders (common prefixes).

        Args:
            prefix: S3 prefix to list under. Use empty string for root.
                   Example: "staging/default/" lists contents of that folder.

        Returns:
            S3ListResponse with folders (common prefixes) and objects at this level
        """
        try:
            result = await s3_client.list_objects(prefix=prefix, delimiter='/')

            # Convert to response model
            objects = [
                S3ObjectInfo(
                    key=obj['key'],
                    size=obj['size'],
                    last_modified=obj['last_modified'],
                    content_type=obj.get('content_type'),
                )
                for obj in result['objects']
            ]

            return S3ListResponse(
                bucket=result['bucket'],
                prefix=result['prefix'],
                folders=result['folders'],
                objects=objects,
            )
        except Exception as e:
            logger.error(f'Error listing S3 objects at prefix "{prefix}": {e!s}')
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f'Error listing objects: {e!s}',
            ) from e

    @router.get(
        '/download/{key:path}',
        response_model=S3DownloadResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_download_url(
        key: str,
        expiry: int = Query(default=3600, description='URL expiry in seconds', ge=60, le=86400),
    ) -> S3DownloadResponse:
        """
        Generate a presigned URL for downloading an object.

        The presigned URL allows direct download from S3 without going through
        the API server, which is efficient for large files.

        Args:
            key: Full S3 object key (e.g., "staging/default/doc_123/file.pdf")
            expiry: URL expiry time in seconds (default: 3600, max: 86400)

        Returns:
            S3DownloadResponse with presigned URL
        """
        try:
            # Check if object exists first
            exists = await s3_client.object_exists(key)
            if not exists:
                raise HTTPException(
                    status_code=404,
                    detail=f'Object not found: {key}',
                )

            url = await s3_client.get_presigned_url(key, expiry=expiry)

            return S3DownloadResponse(
                key=key,
                url=url,
                expiry_seconds=expiry,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f'Error generating download URL for "{key}": {e!s}')
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f'Error generating download URL: {e!s}',
            ) from e

    @router.post('/upload', response_model=S3UploadResponse, dependencies=[Depends(combined_auth)])
    async def upload_file(
        file: Annotated[UploadFile, File(description='File to upload')],
        prefix: Annotated[str, Form(description='S3 prefix path (e.g., "staging/default/")')] = '',
    ) -> S3UploadResponse:
        """
        Upload a file to the specified prefix.

        The file will be uploaded to: {prefix}{filename}
        If prefix is empty, file is uploaded to bucket root.

        Args:
            file: File to upload (multipart form data)
            prefix: S3 prefix path. Should end with "/" for folder-like structure.

        Returns:
            S3UploadResponse with the key where file was uploaded
        """
        try:
            # Read file content
            content = await file.read()
            if not content:
                raise HTTPException(
                    status_code=400,
                    detail='Empty file uploaded',
                )

            # Sanitize filename
            filename = file.filename or 'unnamed'
            safe_filename = filename.replace('/', '_').replace('\\', '_')

            # Construct key
            key = f'{prefix}{safe_filename}' if prefix else safe_filename

            # Detect content type
            content_type = file.content_type
            if not content_type or content_type == 'application/octet-stream':
                guessed_type, _ = mimetypes.guess_type(filename)
                content_type = guessed_type or 'application/octet-stream'

            # Upload to S3
            await s3_client.upload_object(
                key=key,
                data=content,
                content_type=content_type,
            )

            # Generate presigned URL for immediate access
            url = await s3_client.get_presigned_url(key)

            logger.info(f'Uploaded file to S3: {key} ({len(content)} bytes)')

            return S3UploadResponse(
                key=key,
                size=len(content),
                url=url,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f'Error uploading file to S3: {e!s}')
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f'Error uploading file: {e!s}',
            ) from e

    @router.delete(
        '/object/{key:path}',
        response_model=S3DeleteResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_object(key: str) -> S3DeleteResponse:
        """
        Delete an object from S3.

        This operation is permanent and cannot be undone.

        Args:
            key: Full S3 object key to delete (e.g., "staging/default/doc_123/file.pdf")

        Returns:
            S3DeleteResponse confirming deletion
        """
        try:
            # Check if object exists first
            exists = await s3_client.object_exists(key)
            if not exists:
                raise HTTPException(
                    status_code=404,
                    detail=f'Object not found: {key}',
                )

            await s3_client.delete_object(key)

            logger.info(f'Deleted S3 object: {key}')

            return S3DeleteResponse(
                key=key,
                status='deleted',
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f'Error deleting S3 object "{key}": {e!s}')
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f'Error deleting object: {e!s}',
            ) from e

    return router
