"""
Upload routes for S3/RustFS document staging.

This module provides endpoints for:
- Uploading documents to S3 staging
- Listing staged documents
- Getting presigned URLs
"""

import mimetypes
from typing import Annotated, Any, ClassVar, cast

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field

from lightrag import LightRAG
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.kg.postgres_impl import PGDocStatusStorage, PGKVStorage
from lightrag.storage.s3_client import S3Client
from lightrag.utils import compute_mdhash_id, logger


class UploadResponse(BaseModel):
    """Response model for document upload."""

    status: str = Field(description='Upload status')
    doc_id: str = Field(description='Document ID')
    s3_key: str = Field(description='S3 object key')
    s3_url: str = Field(description='S3 URL (s3://bucket/key)')
    message: str | None = Field(default=None, description='Additional message')

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            'example': {
                'status': 'uploaded',
                'doc_id': 'doc_abc123',
                's3_key': 'staging/default/doc_abc123/report.pdf',
                's3_url': 's3://lightrag/staging/default/doc_abc123/report.pdf',
                'message': 'Document staged for processing',
            }
        }


class StagedDocument(BaseModel):
    """Model for a staged document."""

    key: str = Field(description='S3 object key')
    size: int = Field(description='File size in bytes')
    last_modified: str = Field(description='Last modified timestamp')


class ListStagedResponse(BaseModel):
    """Response model for listing staged documents."""

    workspace: str = Field(description='Workspace name')
    documents: list[StagedDocument] = Field(description='List of staged documents')
    count: int = Field(description='Number of documents')


class PresignedUrlResponse(BaseModel):
    """Response model for presigned URL."""

    s3_key: str = Field(description='S3 object key')
    presigned_url: str = Field(description='Presigned URL for direct access')
    expiry_seconds: int = Field(description='URL expiry time in seconds')


class ProcessS3Request(BaseModel):
    """Request model for processing a document from S3 staging."""

    s3_key: str = Field(description='S3 key of the staged document')
    doc_id: str | None = Field(
        default=None,
        description='Document ID (extracted from s3_key if not provided)',
    )
    archive_after_processing: bool = Field(
        default=True,
        description='Move document to archive after successful processing',
    )

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            'example': {
                's3_key': 'staging/default/doc_abc123/report.pdf',
                'doc_id': 'doc_abc123',
                'archive_after_processing': True,
            }
        }


class ProcessS3Response(BaseModel):
    """Response model for S3 document processing."""

    status: str = Field(description='Processing status')
    track_id: str = Field(description='Track ID for monitoring processing progress')
    doc_id: str = Field(description='Document ID')
    s3_key: str = Field(description='Original S3 key')
    archive_key: str | None = Field(default=None, description='Archive S3 key (if archived)')
    message: str | None = Field(default=None, description='Additional message')

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            'example': {
                'status': 'processing_started',
                'track_id': 'insert_20250101_120000_abc123',
                'doc_id': 'doc_abc123',
                's3_key': 'staging/default/doc_abc123/report.pdf',
                'archive_key': 'archive/default/doc_abc123/report.pdf',
                'message': 'Document processing started',
            }
        }


def create_upload_routes(
    rag: LightRAG,
    s3_client: S3Client,
    api_key: str | None = None,
) -> APIRouter:
    """
    Create upload routes for S3 document staging.

    Args:
        rag: LightRAG instance
        s3_client: Initialized S3Client instance
        api_key: Optional API key for authentication

    Returns:
        FastAPI router with upload endpoints
    """
    router = APIRouter(
        prefix='/upload',
        tags=['upload'],
    )

    optional_api_key = get_combined_auth_dependency(api_key)

    @router.post(
        '',
        response_model=UploadResponse,
        summary='Upload document to S3 staging',
        description="""
        Upload a document to S3/RustFS staging area.

        The document will be staged at: s3://bucket/staging/{workspace}/{doc_id}/{filename}

        After upload, the document can be processed by calling the standard document
        processing endpoints, which will:
        1. Fetch the document from S3 staging
        2. Process it through the RAG pipeline
        3. Move it to S3 archive
        4. Store processed data in PostgreSQL
        """,
    )
    async def upload_document(
        file: Annotated[UploadFile, File(description='Document file to upload')],
        workspace: Annotated[str, Form(description='Workspace name')] = 'default',
        doc_id: Annotated[str | None, Form(description='Optional document ID (auto-generated if not provided)')] = None,
        _: Annotated[bool, Depends(optional_api_key)] = True,
    ) -> UploadResponse:
        """Upload a document to S3 staging."""
        try:
            # Read file content
            content = await file.read()

            if not content:
                raise HTTPException(status_code=400, detail='Empty file')

            # Generate doc_id if not provided
            if not doc_id:
                doc_id = compute_mdhash_id(content, prefix='doc_')

            # Determine content type
            final_content_type = file.content_type
            if not final_content_type:
                guessed_type, encoding = mimetypes.guess_type(file.filename or '')
                final_content_type = guessed_type or 'application/octet-stream'

            # Upload to S3 staging
            s3_key = await s3_client.upload_to_staging(
                workspace=workspace,
                doc_id=doc_id,
                content=content,
                filename=file.filename or f'{doc_id}.bin',
                content_type=final_content_type,
                metadata={
                    'original_size': str(len(content)),
                    'content_type': final_content_type,
                },
            )

            s3_url = s3_client.get_s3_url(s3_key)

            logger.info(f'Document uploaded to staging: {s3_key}')

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
            logger.error(f'Upload failed: {e}')
            raise HTTPException(status_code=500, detail=f'Upload failed: {e}') from e

    @router.get(
        '/staged',
        response_model=ListStagedResponse,
        summary='List staged documents',
        description='List all documents in the staging area for a workspace.',
    )
    async def list_staged(
        workspace: str = 'default',
        _: Annotated[bool, Depends(optional_api_key)] = True,
    ) -> ListStagedResponse:
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
            logger.error(f'Failed to list staged documents: {e}')
            raise HTTPException(status_code=500, detail=f'Failed to list staged documents: {e}') from e

    @router.get(
        '/presigned-url',
        response_model=PresignedUrlResponse,
        summary='Get presigned URL',
        description='Generate a presigned URL for direct access to a document in S3.',
    )
    async def get_presigned_url(
        s3_key: str,
        expiry: int = 3600,
        _: Annotated[bool, Depends(optional_api_key)] = True,
    ) -> PresignedUrlResponse:
        """Get presigned URL for a document."""
        try:
            # Verify object exists
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
            logger.error(f'Failed to generate presigned URL: {e}')
            raise HTTPException(status_code=500, detail=f'Failed to generate presigned URL: {e}') from e

    @router.delete(
        '/staged/{doc_id}',
        summary='Delete staged document',
        description='Delete a document from the staging area.',
    )
    async def delete_staged(
        doc_id: str,
        workspace: str = 'default',
        _: Annotated[bool, Depends(optional_api_key)] = True,
    ) -> dict[str, str]:
        """Delete a staged document."""
        try:
            # List objects with this doc_id prefix
            prefix = f'staging/{workspace}/{doc_id}/'
            objects = await s3_client.list_staging(workspace)

            # Filter to this doc_id
            to_delete = [obj['key'] for obj in objects if obj['key'].startswith(prefix)]

            if not to_delete:
                raise HTTPException(status_code=404, detail='Document not found in staging')

            # Delete each object
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
            logger.error(f'Failed to delete staged document: {e}')
            raise HTTPException(status_code=500, detail=f'Failed to delete staged document: {e}') from e

    @router.post(
        '/process',
        response_model=ProcessS3Response,
        summary='Process document from S3 staging',
        description="""
        Fetch a document from S3 staging and process it through the RAG pipeline.

        This endpoint:
        1. Fetches the document content from S3 staging
        2. Processes it through the RAG pipeline (chunking, entity extraction, embedding)
        3. Stores processed data in PostgreSQL with s3_key reference
        4. Optionally moves the document from staging to archive

        The s3_key should be the full key returned from the upload endpoint,
        e.g., "staging/default/doc_abc123/report.pdf"
        """,
    )
    async def process_from_s3(
        request: ProcessS3Request,
        _: Annotated[bool, Depends(optional_api_key)] = True,
    ) -> ProcessS3Response:
        """Process a staged document through the RAG pipeline."""
        try:
            s3_key = request.s3_key

            # Verify object exists
            if not await s3_client.object_exists(s3_key):
                raise HTTPException(
                    status_code=404,
                    detail=f'Document not found in S3: {s3_key}',
                )

            # Fetch content from S3
            content_bytes, metadata = await s3_client.get_object(s3_key)

            # Extract doc_id from s3_key if not provided
            # s3_key format: staging/{workspace}/{doc_id}/{filename}
            doc_id = request.doc_id
            if not doc_id:
                parts = s3_key.split('/')
                doc_id = parts[2] if len(parts) >= 3 else compute_mdhash_id(content_bytes, prefix='doc_')

            # Determine content type and decode appropriately
            content_type = metadata.get('content_type', 'application/octet-stream')
            s3_url = s3_client.get_s3_url(s3_key)

            # For text-based content, decode to string
            if content_type.startswith('text/') or content_type in (
                'application/json',
                'application/xml',
                'application/javascript',
            ):
                try:
                    text_content = content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    text_content = content_bytes.decode('latin-1')
            else:
                # For binary content (PDF, Word, etc.), we need document parsing
                # For now, attempt UTF-8 decode or fail gracefully
                try:
                    text_content = content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    raise HTTPException(
                        status_code=400,
                        detail=f'Cannot process binary content type: {content_type}. '
                        'Document parsing for PDF/Word not yet implemented.',
                    ) from None

            if not text_content.strip():
                raise HTTPException(
                    status_code=400,
                    detail='Document content is empty after decoding',
                )

            # Process through RAG pipeline
            # Use s3_url as file_path for citation reference
            logger.info(f'Processing S3 document: {s3_key} (doc_id: {doc_id})')

            track_id = await rag.ainsert(
                input=text_content,
                ids=doc_id,
                file_paths=s3_url,
            )

            # Move to archive if requested
            archive_key = None
            if request.archive_after_processing:
                try:
                    archive_key = await s3_client.move_to_archive(s3_key)
                    logger.info(f'Moved to archive: {s3_key} -> {archive_key}')

                    # Update database chunks with archive s3_key
                    archive_url = s3_client.get_s3_url(archive_key)
                    updated_count = await cast(PGKVStorage, rag.text_chunks).update_s3_key_by_doc_id(
                        full_doc_id=doc_id,
                        s3_key=archive_key,
                        archive_url=archive_url,
                    )
                    logger.info(f'Updated {updated_count} chunks with archive s3_key: {archive_key}')

                    # Update doc_status with archive s3_key
                    await cast(PGDocStatusStorage, rag.doc_status).update_s3_key(doc_id, archive_key)
                    logger.info(f'Updated doc_status with archive s3_key: {archive_key}')
                except Exception as e:
                    logger.warning(f'Failed to archive document: {e}')
                    # Don't fail the request, processing succeeded

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
            logger.error(f'Failed to process S3 document: {e}')
            raise HTTPException(
                status_code=500,
                detail=f'Failed to process S3 document: {e}',
            ) from e

    return router
