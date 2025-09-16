"""
This module contains all document-related routes for the LightRAG API.
"""

import asyncio
import json
import uuid
from lightrag.utils import logger, get_pinyin_sort_key
import aiofiles
import shutil
import traceback
import pipmaster as pm
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    Form,
)
from pydantic import BaseModel, Field, field_validator

from lightrag import LightRAG
from lightrag.base import DeletionResult, DocProcessingStatus, DocStatus
from lightrag.utils import generate_track_id
from lightrag.api.utils_api import get_combined_auth_dependency
from ..config import global_args
from raganything import RAGAnything


# Function to format datetime to ISO format string with timezone information
def format_datetime(dt: Any) -> Optional[str]:
    """Format datetime to ISO format string with timezone information

    Args:
        dt: Datetime object, string, or None

    Returns:
        ISO format string with timezone information, or None if input is None
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt

    # Check if datetime object has timezone information
    if isinstance(dt, datetime):
        # If datetime object has no timezone info (naive datetime), add UTC timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

    # Return ISO format string with timezone information
    return dt.isoformat()


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)

# Temporary file prefix
temp_prefix = "__tmp__"


def sanitize_filename(filename: str, input_dir: Path) -> str:
    """
    Sanitize uploaded filename to prevent Path Traversal attacks.

    Args:
        filename: The original filename from the upload
        input_dir: The target input directory

    Returns:
        str: Sanitized filename that is safe to use

    Raises:
        HTTPException: If the filename is unsafe or invalid
    """
    # Basic validation
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    # Remove path separators and traversal sequences
    clean_name = filename.replace("/", "").replace("\\", "")
    clean_name = clean_name.replace("..", "")

    # Remove control characters and null bytes
    clean_name = "".join(c for c in clean_name if ord(c) >= 32 and c != "\x7f")

    # Remove leading/trailing whitespace and dots
    clean_name = clean_name.strip().strip(".")

    # Check if anything is left after sanitization
    if not clean_name:
        raise HTTPException(
            status_code=400, detail="Invalid filename after sanitization"
        )

    # Verify the final path stays within the input directory
    try:
        final_path = (input_dir / clean_name).resolve()
        if not final_path.is_relative_to(input_dir.resolve()):
            raise HTTPException(status_code=400, detail="Unsafe filename detected")
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid filename")

    return clean_name


class SchemeConfig(BaseModel):
    """Configuration model for processing schemes.

    Defines the processing framework and optional extractor to use for document processing.

    Attributes:
        framework (Literal['lightrag', 'raganything']): Processing framework to use.
            - "lightrag": Standard LightRAG processing for text-based documents
            - "raganything": Advanced multimodal processing with image/table/equation support
        extractor (Literal['mineru', 'docling', '']): Document extraction tool to use.
            - "mineru": MinerU parser for comprehensive document parsing
            - "docling": Docling parser for office document processing
            - "": Default/automatic extractor selection
        modelSource (Literal["huggingface", "modelscope", "local", ""]): The model source used by Mineru.
            - "huggingface": Using pre-trained models from the Hugging Face model library
            - "modelscope": using model resources on ModelScope platform
            - "local": Use custom models deployed locally
            - "":Maintain the default model source configuration of the system (usually huggingface)
    """

    framework: Literal["lightrag", "raganything"]
    extractor: Literal["mineru", "docling", ""] = ""  # 默认值
    modelSource: Literal["huggingface", "modelscope", "local", ""] = ""


class Scheme(BaseModel):
    """Base model for processing schemes.

    Attributes:
        name (str): Human-readable name for the processing scheme
        config (SchemeConfig): Configuration settings for the scheme
    """

    name: str
    config: SchemeConfig


class Scheme_include_id(Scheme):
    """Scheme model with unique identifier included.

    Extends the base Scheme model to include a unique ID field for
    identification and management operations.

    Attributes:
        id (int): Unique identifier for the scheme
        name (str): Inherited from Scheme
        config (SchemeConfig): Inherited from Scheme
    """

    id: int


class SchemesResponse(BaseModel):
    """Response model for scheme management operations.

    Used for all scheme-related endpoints to provide consistent response format
    for scheme retrieval, creation, update, and deletion operations.

    Attributes:
        status (str): Operation status ("success", "error")
        message (Optional[str]): Additional message with operation details
        data (Optional[List[Dict[str, Any]]]): List of scheme objects when retrieving schemes
    """

    status: str = Field(..., description="Operation status")
    message: Optional[str] = Field(None, description="Additional message")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="List of schemes")


class ScanRequest(BaseModel):
    """Request model for document scanning operations."""

    schemeConfig: SchemeConfig = Field(..., description="Scanning scheme configuration")


class ScanResponse(BaseModel):
    """Response model for document scanning operation

    Attributes:
        status: Status of the scanning operation
        message: Optional message with additional details
        track_id: Tracking ID for monitoring scanning progress
    """

    status: Literal["scanning_started"] = Field(
        description="Status of the scanning operation"
    )
    message: Optional[str] = Field(
        default=None, description="Additional details about the scanning operation"
    )
    track_id: str = Field(description="Tracking ID for monitoring scanning progress")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "scanning_started",
                "message": "Scanning process has been initiated in the background",
                "track_id": "scan_20250729_170612_abc123",
            }
        }


class InsertTextRequest(BaseModel):
    """Request model for inserting a single text document

    Attributes:
        text: The text content to be inserted into the RAG system
        file_source: Source of the text (optional)
    """

    text: str = Field(
        min_length=1,
        description="The text to insert",
    )
    file_source: str = Field(default=None, min_length=0, description="File Source")

    @field_validator("text", mode="after")
    @classmethod
    def strip_text_after(cls, text: str) -> str:
        return text.strip()

    @field_validator("file_source", mode="after")
    @classmethod
    def strip_source_after(cls, file_source: str) -> str:
        return file_source.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a sample text to be inserted into the RAG system.",
                "file_source": "Source of the text (optional)",
            }
        }


class InsertTextsRequest(BaseModel):
    """Request model for inserting multiple text documents

    Attributes:
        texts: List of text contents to be inserted into the RAG system
        file_sources: Sources of the texts (optional)
    """

    texts: list[str] = Field(
        min_length=1,
        description="The texts to insert",
    )
    file_sources: list[str] = Field(
        default=None, min_length=0, description="Sources of the texts"
    )

    @field_validator("texts", mode="after")
    @classmethod
    def strip_texts_after(cls, texts: list[str]) -> list[str]:
        return [text.strip() for text in texts]

    @field_validator("file_sources", mode="after")
    @classmethod
    def strip_sources_after(cls, file_sources: list[str]) -> list[str]:
        return [file_source.strip() for file_source in file_sources]

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "This is the first text to be inserted.",
                    "This is the second text to be inserted.",
                ],
                "file_sources": [
                    "First file source (optional)",
                ],
            }
        }


class InsertResponse(BaseModel):
    """Response model for document insertion operations

    Attributes:
        status: Status of the operation (success, duplicated, partial_success, failure)
        message: Detailed message describing the operation result
        track_id: Tracking ID for monitoring processing status
    """

    status: Literal["success", "duplicated", "partial_success", "failure"] = Field(
        description="Status of the operation"
    )
    message: str = Field(description="Message describing the operation result")
    track_id: str = Field(description="Tracking ID for monitoring processing status")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "File 'document.pdf' uploaded successfully. Processing will continue in background.",
                "track_id": "upload_20250729_170612_abc123",
            }
        }


class ClearDocumentsResponse(BaseModel):
    """Response model for document clearing operation

    Attributes:
        status: Status of the clear operation
        message: Detailed message describing the operation result
    """

    status: Literal["success", "partial_success", "busy", "fail"] = Field(
        description="Status of the clear operation"
    )
    message: str = Field(description="Message describing the operation result")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "All documents cleared successfully. Deleted 15 files.",
            }
        }


class ClearCacheRequest(BaseModel):
    """Request model for clearing cache

    This model is kept for API compatibility but no longer accepts any parameters.
    All cache will be cleared regardless of the request content.
    """

    class Config:
        json_schema_extra = {"example": {}}


class ClearCacheResponse(BaseModel):
    """Response model for cache clearing operation

    Attributes:
        status: Status of the clear operation
        message: Detailed message describing the operation result
    """

    status: Literal["success", "fail"] = Field(
        description="Status of the clear operation"
    )
    message: str = Field(description="Message describing the operation result")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Successfully cleared cache for modes: ['default', 'naive']",
            }
        }


"""Response model for document status

Attributes:
    id: Document identifier
    content_summary: Summary of document content
    content_length: Length of document content
    status: Current processing status
    created_at: Creation timestamp (ISO format string)
    updated_at: Last update timestamp (ISO format string)
    chunks_count: Number of chunks (optional)
    error: Error message if any (optional)
    metadata: Additional metadata (optional)
    file_path: Path to the document file
"""


class DeleteDocRequest(BaseModel):
    doc_ids: List[str] = Field(..., description="The IDs of the documents to delete.")
    delete_file: bool = Field(
        default=False,
        description="Whether to delete the corresponding file in the upload directory.",
    )

    @field_validator("doc_ids", mode="after")
    @classmethod
    def validate_doc_ids(cls, doc_ids: List[str]) -> List[str]:
        if not doc_ids:
            raise ValueError("Document IDs list cannot be empty")

        validated_ids = []
        for doc_id in doc_ids:
            if not doc_id or not doc_id.strip():
                raise ValueError("Document ID cannot be empty")
            validated_ids.append(doc_id.strip())

        # Check for duplicates
        if len(validated_ids) != len(set(validated_ids)):
            raise ValueError("Document IDs must be unique")

        return validated_ids


class DeleteEntityRequest(BaseModel):
    entity_name: str = Field(..., description="The name of the entity to delete.")

    @field_validator("entity_name", mode="after")
    @classmethod
    def validate_entity_name(cls, entity_name: str) -> str:
        if not entity_name or not entity_name.strip():
            raise ValueError("Entity name cannot be empty")
        return entity_name.strip()


class DeleteRelationRequest(BaseModel):
    source_entity: str = Field(..., description="The name of the source entity.")
    target_entity: str = Field(..., description="The name of the target entity.")

    @field_validator("source_entity", "target_entity", mode="after")
    @classmethod
    def validate_entity_names(cls, entity_name: str) -> str:
        if not entity_name or not entity_name.strip():
            raise ValueError("Entity name cannot be empty")
        return entity_name.strip()


class DocStatusResponse(BaseModel):
    id: str = Field(description="Document identifier")
    content_summary: str = Field(description="Summary of document content")
    content_length: int = Field(description="Length of document content in characters")
    status: DocStatus = Field(description="Current processing status")
    created_at: str = Field(description="Creation timestamp (ISO format string)")
    updated_at: str = Field(description="Last update timestamp (ISO format string)")
    track_id: Optional[str] = Field(
        default=None, description="Tracking ID for monitoring progress"
    )
    chunks_count: Optional[int] = Field(
        default=None, description="Number of chunks the document was split into"
    )
    error_msg: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Additional metadata about the document"
    )
    file_path: str = Field(description="Path to the document file")
    scheme_name: str = Field(
        default=None, description="Name of the processing scheme used for this document"
    )
    multimodal_content: Optional[list[dict[str, Any]]] = Field(
        default=None, description="Multimodal content of the document"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_123456",
                "content_summary": "Research paper on machine learning",
                "scheme_name": "lightrag",
                "multimodal_content": [],
                "content_length": 15240,
                "status": "PROCESSED",
                "created_at": "2025-03-31T12:34:56",
                "updated_at": "2025-03-31T12:35:30",
                "track_id": "upload_20250729_170612_abc123",
                "chunks_count": 12,
                "error": None,
                "metadata": {"author": "John Doe", "year": 2025},
                "file_path": "research_paper.pdf",
            }
        }


class DocsStatusesResponse(BaseModel):
    """Response model for document statuses

    Attributes:
        statuses: Dictionary mapping document status to lists of document status responses
    """

    statuses: Dict[DocStatus, List[DocStatusResponse]] = Field(
        default_factory=dict,
        description="Dictionary mapping document status to lists of document status responses",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "statuses": {
                    "PENDING": [
                        {
                            "id": "doc_123",
                            "content_summary": "Pending document",
                            "scheme_name": "lightrag",
                            "multimodal_content": [],
                            "content_length": 5000,
                            "status": "PENDING",
                            "created_at": "2025-03-31T10:00:00",
                            "updated_at": "2025-03-31T10:00:00",
                            "track_id": "upload_20250331_100000_abc123",
                            "chunks_count": None,
                            "error": None,
                            "metadata": None,
                            "file_path": "pending_doc.pdf",
                        }
                    ],
                    "PROCESSED": [
                        {
                            "id": "doc_456",
                            "content_summary": "Processed document",
                            "scheme_name": "lightrag",
                            "multimodal_content": [],
                            "content_length": 8000,
                            "status": "PROCESSED",
                            "created_at": "2025-03-31T09:00:00",
                            "updated_at": "2025-03-31T09:05:00",
                            "track_id": "insert_20250331_090000_def456",
                            "chunks_count": 8,
                            "error": None,
                            "metadata": {"author": "John Doe"},
                            "file_path": "processed_doc.pdf",
                        }
                    ],
                }
            }
        }


class TrackStatusResponse(BaseModel):
    """Response model for tracking document processing status by track_id

    Attributes:
        track_id: The tracking ID
        documents: List of documents associated with this track_id
        total_count: Total number of documents for this track_id
        status_summary: Count of documents by status
    """

    track_id: str = Field(description="The tracking ID")
    documents: List[DocStatusResponse] = Field(
        description="List of documents associated with this track_id"
    )
    total_count: int = Field(description="Total number of documents for this track_id")
    status_summary: Dict[str, int] = Field(description="Count of documents by status")

    class Config:
        json_schema_extra = {
            "example": {
                "track_id": "upload_20250729_170612_abc123",
                "documents": [
                    {
                        "id": "doc_123456",
                        "content_summary": "Research paper on machine learning",
                        "content_length": 15240,
                        "status": "PROCESSED",
                        "created_at": "2025-03-31T12:34:56",
                        "updated_at": "2025-03-31T12:35:30",
                        "track_id": "upload_20250729_170612_abc123",
                        "chunks_count": 12,
                        "error": None,
                        "metadata": {"author": "John Doe", "year": 2025},
                        "file_path": "research_paper.pdf",
                    }
                ],
                "total_count": 1,
                "status_summary": {"PROCESSED": 1},
            }
        }


class DocumentsRequest(BaseModel):
    """Request model for paginated document queries

    Attributes:
        status_filter: Filter by document status, None for all statuses
        page: Page number (1-based)
        page_size: Number of documents per page (10-200)
        sort_field: Field to sort by ('created_at', 'updated_at', 'id', 'file_path')
        sort_direction: Sort direction ('asc' or 'desc')
    """

    status_filter: Optional[DocStatus] = Field(
        default=None, description="Filter by document status, None for all statuses"
    )
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(
        default=50, ge=10, le=200, description="Number of documents per page (10-200)"
    )
    sort_field: Literal["created_at", "updated_at", "id", "file_path"] = Field(
        default="updated_at", description="Field to sort by"
    )
    sort_direction: Literal["asc", "desc"] = Field(
        default="desc", description="Sort direction"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status_filter": "PROCESSED",
                "page": 1,
                "page_size": 50,
                "sort_field": "updated_at",
                "sort_direction": "desc",
            }
        }


class PaginationInfo(BaseModel):
    """Pagination information

    Attributes:
        page: Current page number
        page_size: Number of items per page
        total_count: Total number of items
        total_pages: Total number of pages
        has_next: Whether there is a next page
        has_prev: Whether there is a previous page
    """

    page: int = Field(description="Current page number")
    page_size: int = Field(description="Number of items per page")
    total_count: int = Field(description="Total number of items")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there is a next page")
    has_prev: bool = Field(description="Whether there is a previous page")

    class Config:
        json_schema_extra = {
            "example": {
                "page": 1,
                "page_size": 50,
                "total_count": 150,
                "total_pages": 3,
                "has_next": True,
                "has_prev": False,
            }
        }


class PaginatedDocsResponse(BaseModel):
    """Response model for paginated document queries

    Attributes:
        documents: List of documents for the current page
        pagination: Pagination information
        status_counts: Count of documents by status for all documents
    """

    documents: List[DocStatusResponse] = Field(
        description="List of documents for the current page"
    )
    pagination: PaginationInfo = Field(description="Pagination information")
    status_counts: Dict[str, int] = Field(
        description="Count of documents by status for all documents"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "id": "doc_123456",
                        "content_summary": "Research paper on machine learning",
                        "content_length": 15240,
                        "status": "PROCESSED",
                        "created_at": "2025-03-31T12:34:56",
                        "updated_at": "2025-03-31T12:35:30",
                        "track_id": "upload_20250729_170612_abc123",
                        "chunks_count": 12,
                        "error_msg": None,
                        "metadata": {"author": "John Doe", "year": 2025},
                        "file_path": "research_paper.pdf",
                    }
                ],
                "pagination": {
                    "page": 1,
                    "page_size": 50,
                    "total_count": 150,
                    "total_pages": 3,
                    "has_next": True,
                    "has_prev": False,
                },
                "status_counts": {
                    "PENDING": 10,
                    "PROCESSING": 5,
                    "PROCESSED": 130,
                    "FAILED": 5,
                },
            }
        }


class StatusCountsResponse(BaseModel):
    """Response model for document status counts

    Attributes:
        status_counts: Count of documents by status
    """

    status_counts: Dict[str, int] = Field(description="Count of documents by status")

    class Config:
        json_schema_extra = {
            "example": {
                "status_counts": {
                    "PENDING": 10,
                    "PROCESSING": 5,
                    "PROCESSED": 130,
                    "FAILED": 5,
                }
            }
        }


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status

    Attributes:
        autoscanned: Whether auto-scan has started
        busy: Whether the pipeline is currently busy
        job_name: Current job name (e.g., indexing files/indexing texts)
        job_start: Job start time as ISO format string with timezone (optional)
        docs: Total number of documents to be indexed
        batchs: Number of batches for processing documents
        cur_batch: Current processing batch
        request_pending: Flag for pending request for processing
        latest_message: Latest message from pipeline processing
        history_messages: List of history messages
        update_status: Status of update flags for all namespaces
    """

    autoscanned: bool = False
    busy: bool = False
    job_name: str = "Default Job"
    job_start: Optional[str] = None
    docs: int = 0
    batchs: int = 0
    cur_batch: int = 0
    request_pending: bool = False
    latest_message: str = ""
    history_messages: Optional[List[str]] = None
    update_status: Optional[dict] = None

    @field_validator("job_start", mode="before")
    @classmethod
    def parse_job_start(cls, value):
        """Process datetime and return as ISO format string with timezone"""
        return format_datetime(value)

    class Config:
        extra = "allow"  # Allow additional fields from the pipeline status


class DocumentManager:
    def __init__(
        self,
        input_dir: str,
        workspace: str = "",  # New parameter for workspace isolation
        supported_extensions: tuple = (
            ".txt",
            ".md",
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".rtf",  # Rich Text Format
            ".odt",  # OpenDocument Text
            ".tex",  # LaTeX
            ".epub",  # Electronic Publication
            ".html",  # HyperText Markup Language
            ".htm",  # HyperText Markup Language
            ".csv",  # Comma-Separated Values
            ".json",  # JavaScript Object Notation
            ".xml",  # eXtensible Markup Language
            ".yaml",  # YAML Ain't Markup Language
            ".yml",  # YAML
            ".log",  # Log files
            ".conf",  # Configuration files
            ".ini",  # Initialization files
            ".properties",  # Java properties files
            ".sql",  # SQL scripts
            ".bat",  # Batch files
            ".sh",  # Shell scripts
            ".c",  # C source code
            ".cpp",  # C++ source code
            ".py",  # Python source code
            ".java",  # Java source code
            ".js",  # JavaScript source code
            ".ts",  # TypeScript source code
            ".swift",  # Swift source code
            ".go",  # Go source code
            ".rb",  # Ruby source code
            ".php",  # PHP source code
            ".css",  # Cascading Style Sheets
            ".scss",  # Sassy CSS
            ".less",  # LESS CSS
        ),
    ):
        # Store the base input directory and workspace
        self.base_input_dir = Path(input_dir)
        self.workspace = workspace
        self.supported_extensions = supported_extensions
        self.indexed_files = set()

        # Create workspace-specific input directory
        # If workspace is provided, create a subdirectory for data isolation
        if workspace:
            self.input_dir = self.base_input_dir / workspace
        else:
            self.input_dir = self.base_input_dir

        # Create input directory if it doesn't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def scan_directory_for_new_files(self) -> List[Path]:
        """Scan input directory for new files"""
        new_files = []
        for ext in self.supported_extensions:
            logger.debug(f"Scanning for {ext} files in {self.input_dir}")
            for file_path in self.input_dir.glob(f"*{ext}"):
                if file_path not in self.indexed_files:
                    new_files.append(file_path)
        return new_files

    def mark_as_indexed(self, file_path: Path):
        self.indexed_files.add(file_path)

    def is_supported_file(self, filename: str) -> bool:
        return any(filename.lower().endswith(ext) for ext in self.supported_extensions)


def get_unique_filename_in_enqueued(target_dir: Path, original_name: str) -> str:
    """Generate a unique filename in the target directory by adding numeric suffixes if needed

    Args:
        target_dir: Target directory path
        original_name: Original filename

    Returns:
        str: Unique filename (may have numeric suffix added)
    """
    from pathlib import Path
    import time

    original_path = Path(original_name)
    base_name = original_path.stem
    extension = original_path.suffix

    # Try original name first
    if not (target_dir / original_name).exists():
        return original_name

    # Try with numeric suffixes 001-999
    for i in range(1, 1000):
        suffix = f"{i:03d}"
        new_name = f"{base_name}_{suffix}{extension}"
        if not (target_dir / new_name).exists():
            return new_name

    # Fallback with timestamp if all 999 slots are taken
    timestamp = int(time.time())
    return f"{base_name}_{timestamp}{extension}"


async def pipeline_enqueue_file(
    rag: LightRAG, file_path: Path, track_id: str = None, scheme_name: str = None
) -> tuple[bool, str]:
    """Add a file to the queue for processing

    Args:
        rag: LightRAG instance
        file_path: Path to the saved file
        track_id: Optional tracking ID, if not provided will be generated
        scheme_name (str, optional): Processing scheme name for categorization.
            Defaults to None
    Returns:
        tuple: (success: bool, track_id: str)
    """

    # Generate track_id if not provided
    if track_id is None:
        track_id = generate_track_id("unknown")

    try:
        content = ""
        ext = file_path.suffix.lower()
        file_size = 0

        # Get file size for error reporting
        try:
            file_size = file_path.stat().st_size
        except Exception:
            file_size = 0

        file = None
        try:
            async with aiofiles.open(file_path, "rb") as f:
                file = await f.read()
        except PermissionError as e:
            error_files = [
                {
                    "file_path": str(file_path.name),
                    "error_description": "[File Extraction]Permission denied - cannot read file",
                    "original_error": str(e),
                    "file_size": file_size,
                }
            ]
            await rag.apipeline_enqueue_error_documents(error_files, track_id)
            logger.error(
                f"[File Extraction]Permission denied reading file: {file_path.name}"
            )
            return False, track_id
        except FileNotFoundError as e:
            error_files = [
                {
                    "file_path": str(file_path.name),
                    "error_description": "[File Extraction]File not found",
                    "original_error": str(e),
                    "file_size": file_size,
                }
            ]
            await rag.apipeline_enqueue_error_documents(error_files, track_id)
            logger.error(f"[File Extraction]File not found: {file_path.name}")
            return False, track_id
        except Exception as e:
            error_files = [
                {
                    "file_path": str(file_path.name),
                    "error_description": "[File Extraction]File reading error",
                    "original_error": str(e),
                    "file_size": file_size,
                }
            ]
            await rag.apipeline_enqueue_error_documents(error_files, track_id)
            logger.error(
                f"[File Extraction]Error reading file {file_path.name}: {str(e)}"
            )
            return False, track_id

        # Process based on file type
        try:
            match ext:
                case (
                    ".txt"
                    | ".md"
                    | ".html"
                    | ".htm"
                    | ".tex"
                    | ".json"
                    | ".xml"
                    | ".yaml"
                    | ".yml"
                    | ".rtf"
                    | ".odt"
                    | ".epub"
                    | ".csv"
                    | ".log"
                    | ".conf"
                    | ".ini"
                    | ".properties"
                    | ".sql"
                    | ".bat"
                    | ".sh"
                    | ".c"
                    | ".cpp"
                    | ".py"
                    | ".java"
                    | ".js"
                    | ".ts"
                    | ".swift"
                    | ".go"
                    | ".rb"
                    | ".php"
                    | ".css"
                    | ".scss"
                    | ".less"
                ):
                    try:
                        # Try to decode as UTF-8
                        content = file.decode("utf-8")

                        # Validate content
                        if not content or len(content.strip()) == 0:
                            error_files = [
                                {
                                    "file_path": str(file_path.name),
                                    "error_description": "[File Extraction]Empty file content",
                                    "original_error": "File contains no content or only whitespace",
                                    "file_size": file_size,
                                }
                            ]
                            await rag.apipeline_enqueue_error_documents(
                                error_files, track_id
                            )
                            logger.error(
                                f"[File Extraction]Empty content in file: {file_path.name}"
                            )
                            return False, track_id

                        # Check if content looks like binary data string representation
                        if content.startswith("b'") or content.startswith('b"'):
                            error_files = [
                                {
                                    "file_path": str(file_path.name),
                                    "error_description": "[File Extraction]Binary data in text file",
                                    "original_error": "File appears to contain binary data representation instead of text",
                                    "file_size": file_size,
                                }
                            ]
                            await rag.apipeline_enqueue_error_documents(
                                error_files, track_id
                            )
                            logger.error(
                                f"[File Extraction]File {file_path.name} appears to contain binary data representation instead of text"
                            )
                            return False, track_id

                    except UnicodeDecodeError as e:
                        error_files = [
                            {
                                "file_path": str(file_path.name),
                                "error_description": "[File Extraction]UTF-8 encoding error, please convert it to UTF-8 before processing",
                                "original_error": f"File is not valid UTF-8 encoded text: {str(e)}",
                                "file_size": file_size,
                            }
                        ]
                        await rag.apipeline_enqueue_error_documents(
                            error_files, track_id
                        )
                        logger.error(
                            f"[File Extraction]File {file_path.name} is not valid UTF-8 encoded text. Please convert it to UTF-8 before processing."
                        )
                        return False, track_id

                case ".pdf":
                    try:
                        if global_args.document_loading_engine == "DOCLING":
                            if not pm.is_installed("docling"):  # type: ignore
                                pm.install("docling")
                            from docling.document_converter import DocumentConverter  # type: ignore

                            converter = DocumentConverter()
                            result = converter.convert(file_path)
                            content = result.document.export_to_markdown()
                        else:
                            if not pm.is_installed("pypdf2"):  # type: ignore
                                pm.install("pypdf2")
                            from PyPDF2 import PdfReader  # type: ignore
                            from io import BytesIO

                            pdf_file = BytesIO(file)
                            reader = PdfReader(pdf_file)
                            for page in reader.pages:
                                content += page.extract_text() + "\n"
                    except Exception as e:
                        error_files = [
                            {
                                "file_path": str(file_path.name),
                                "error_description": "[File Extraction]PDF processing error",
                                "original_error": f"Failed to extract text from PDF: {str(e)}",
                                "file_size": file_size,
                            }
                        ]
                        await rag.apipeline_enqueue_error_documents(
                            error_files, track_id
                        )
                        logger.error(
                            f"[File Extraction]Error processing PDF {file_path.name}: {str(e)}"
                        )
                        return False, track_id

                case ".docx":
                    try:
                        if global_args.document_loading_engine == "DOCLING":
                            if not pm.is_installed("docling"):  # type: ignore
                                pm.install("docling")
                            from docling.document_converter import DocumentConverter  # type: ignore

                            converter = DocumentConverter()
                            result = converter.convert(file_path)
                            content = result.document.export_to_markdown()
                        else:
                            if not pm.is_installed("python-docx"):  # type: ignore
                                try:
                                    pm.install("python-docx")
                                except Exception:
                                    pm.install("docx")
                            from docx import Document  # type: ignore
                            from io import BytesIO

                            docx_file = BytesIO(file)
                            doc = Document(docx_file)
                            content = "\n".join(
                                [paragraph.text for paragraph in doc.paragraphs]
                            )
                    except Exception as e:
                        error_files = [
                            {
                                "file_path": str(file_path.name),
                                "error_description": "[File Extraction]DOCX processing error",
                                "original_error": f"Failed to extract text from DOCX: {str(e)}",
                                "file_size": file_size,
                            }
                        ]
                        await rag.apipeline_enqueue_error_documents(
                            error_files, track_id
                        )
                        logger.error(
                            f"[File Extraction]Error processing DOCX {file_path.name}: {str(e)}"
                        )
                        return False, track_id

                case ".pptx":
                    try:
                        if global_args.document_loading_engine == "DOCLING":
                            if not pm.is_installed("docling"):  # type: ignore
                                pm.install("docling")
                            from docling.document_converter import DocumentConverter  # type: ignore

                            converter = DocumentConverter()
                            result = converter.convert(file_path)
                            content = result.document.export_to_markdown()
                        else:
                            if not pm.is_installed("python-pptx"):  # type: ignore
                                pm.install("pptx")
                            from pptx import Presentation  # type: ignore
                            from io import BytesIO

                            pptx_file = BytesIO(file)
                            prs = Presentation(pptx_file)
                            for slide in prs.slides:
                                for shape in slide.shapes:
                                    if hasattr(shape, "text"):
                                        content += shape.text + "\n"
                    except Exception as e:
                        error_files = [
                            {
                                "file_path": str(file_path.name),
                                "error_description": "[File Extraction]PPTX processing error",
                                "original_error": f"Failed to extract text from PPTX: {str(e)}",
                                "file_size": file_size,
                            }
                        ]
                        await rag.apipeline_enqueue_error_documents(
                            error_files, track_id
                        )
                        logger.error(
                            f"[File Extraction]Error processing PPTX {file_path.name}: {str(e)}"
                        )
                        return False, track_id

                case ".xlsx":
                    try:
                        if global_args.document_loading_engine == "DOCLING":
                            if not pm.is_installed("docling"):  # type: ignore
                                pm.install("docling")
                            from docling.document_converter import DocumentConverter  # type: ignore

                            converter = DocumentConverter()
                            result = converter.convert(file_path)
                            content = result.document.export_to_markdown()
                        else:
                            if not pm.is_installed("openpyxl"):  # type: ignore
                                pm.install("openpyxl")
                            from openpyxl import load_workbook  # type: ignore
                            from io import BytesIO

                            xlsx_file = BytesIO(file)
                            wb = load_workbook(xlsx_file)
                            for sheet in wb:
                                content += f"Sheet: {sheet.title}\n"
                                for row in sheet.iter_rows(values_only=True):
                                    content += (
                                        "\t".join(
                                            str(cell) if cell is not None else ""
                                            for cell in row
                                        )
                                        + "\n"
                                    )
                                content += "\n"
                    except Exception as e:
                        error_files = [
                            {
                                "file_path": str(file_path.name),
                                "error_description": "[File Extraction]XLSX processing error",
                                "original_error": f"Failed to extract text from XLSX: {str(e)}",
                                "file_size": file_size,
                            }
                        ]
                        await rag.apipeline_enqueue_error_documents(
                            error_files, track_id
                        )
                        logger.error(
                            f"[File Extraction]Error processing XLSX {file_path.name}: {str(e)}"
                        )
                        return False, track_id

                case _:
                    error_files = [
                        {
                            "file_path": str(file_path.name),
                            "error_description": f"[File Extraction]Unsupported file type: {ext}",
                            "original_error": f"File extension {ext} is not supported",
                            "file_size": file_size,
                        }
                    ]
                    await rag.apipeline_enqueue_error_documents(error_files, track_id)
                    logger.error(
                        f"[File Extraction]Unsupported file type: {file_path.name} (extension {ext})"
                    )
                    return False, track_id

        except Exception as e:
            error_files = [
                {
                    "file_path": str(file_path.name),
                    "error_description": "[File Extraction]File format processing error",
                    "original_error": f"Unexpected error during file extracting: {str(e)}",
                    "file_size": file_size,
                }
            ]
            await rag.apipeline_enqueue_error_documents(error_files, track_id)
            logger.error(
                f"[File Extraction]Unexpected error during {file_path.name} extracting: {str(e)}"
            )
            return False, track_id

        # Insert into the RAG queue
        if content:
            # Check if content contains only whitespace characters
            if not content.strip():
                error_files = [
                    {
                        "file_path": str(file_path.name),
                        "error_description": "[File Extraction]File contains only whitespace",
                        "original_error": "File content contains only whitespace characters",
                        "file_size": file_size,
                    }
                ]
                await rag.apipeline_enqueue_error_documents(error_files, track_id)
                logger.warning(
                    f"[File Extraction]File contains only whitespace characters: {file_path.name}"
                )
                return False, track_id

            try:
                await rag.apipeline_enqueue_documents(
                    content,
                    file_paths=file_path.name,
                    track_id=track_id,
                    scheme_name=scheme_name,
                )

                logger.info(
                    f"Successfully extracted and enqueued file: {file_path.name}"
                )

                # Move file to __enqueued__ directory after enqueuing
                try:
                    enqueued_dir = file_path.parent / "__enqueued__"
                    enqueued_dir.mkdir(exist_ok=True)

                    # Generate unique filename to avoid conflicts
                    unique_filename = get_unique_filename_in_enqueued(
                        enqueued_dir, file_path.name
                    )
                    target_path = enqueued_dir / unique_filename

                    # Move the file
                    file_path.rename(target_path)
                    logger.debug(
                        f"Moved file to enqueued directory: {file_path.name} -> {unique_filename}"
                    )

                except Exception as move_error:
                    logger.error(
                        f"Failed to move file {file_path.name} to __enqueued__ directory: {move_error}"
                    )
                    # Don't affect the main function's success status

                return True, track_id

            except Exception as e:
                error_files = [
                    {
                        "file_path": str(file_path.name),
                        "error_description": "Document enqueue error",
                        "original_error": f"Failed to enqueue document: {str(e)}",
                        "file_size": file_size,
                    }
                ]
                await rag.apipeline_enqueue_error_documents(error_files, track_id)
                logger.error(f"Error enqueueing document {file_path.name}: {str(e)}")
                return False, track_id
        else:
            error_files = [
                {
                    "file_path": str(file_path.name),
                    "error_description": "No content extracted",
                    "original_error": "No content could be extracted from file",
                    "file_size": file_size,
                }
            ]
            await rag.apipeline_enqueue_error_documents(error_files, track_id)
            logger.error(f"No content extracted from file: {file_path.name}")
            return False, track_id

    except Exception as e:
        # Catch-all for any unexpected errors
        try:
            file_size = file_path.stat().st_size if file_path.exists() else 0
        except Exception:
            file_size = 0

        error_files = [
            {
                "file_path": str(file_path.name),
                "error_description": "Unexpected processing error",
                "original_error": f"Unexpected error: {str(e)}",
                "file_size": file_size,
            }
        ]
        await rag.apipeline_enqueue_error_documents(error_files, track_id)
        logger.error(f"Enqueuing file {file_path.name} error: {str(e)}")
        logger.error(traceback.format_exc())
        return False, track_id
    finally:
        if file_path.name.startswith(temp_prefix):
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")


async def pipeline_index_file(
    rag: LightRAG, file_path: Path, track_id: str = None, scheme_name: str = None
):
    """Index a file with track_id

    Args:
        rag: LightRAG instance
        file_path: Path to the saved file
        track_id: Optional tracking ID
        scheme_name (str, optional): Processing scheme name for categorization.
            Defaults to None
    """
    try:
        success, returned_track_id = await pipeline_enqueue_file(
            rag, file_path, track_id, scheme_name
        )
        if success:
            await rag.apipeline_process_enqueue_documents()

    except Exception as e:
        logger.error(f"Error indexing file {file_path.name}: {str(e)}")
        logger.error(traceback.format_exc())


async def pipeline_index_files(
    rag: LightRAG, file_paths: List[Path], track_id: str = None, scheme_name: str = None
):
    """Index multiple files sequentially to avoid high CPU load

    Args:
        rag: LightRAG instance
        file_paths: Paths to the files to index
        track_id: Optional tracking ID to pass to all files
        scheme_name (str, optional): Processing scheme name for categorization.
            Defaults to None
    """
    if not file_paths:
        return
    try:
        enqueued = False

        # Use get_pinyin_sort_key for Chinese pinyin sorting
        sorted_file_paths = sorted(
            file_paths, key=lambda p: get_pinyin_sort_key(str(p))
        )

        # Process files sequentially with track_id
        for file_path in sorted_file_paths:
            success, _ = await pipeline_enqueue_file(
                rag, file_path, track_id, scheme_name
            )
            if success:
                enqueued = True

        # Process the queue only if at least one file was successfully enqueued
        if enqueued:
            await rag.apipeline_process_enqueue_documents()
    except Exception as e:
        logger.error(f"Error indexing files: {str(e)}")
        logger.error(traceback.format_exc())


async def pipeline_index_files_raganything(
    rag_anything: RAGAnything,
    file_paths: List[Path],
    scheme_name: str = None,
    parser: str = None,
    source: str = None,
):
    """Index multiple files using RAGAnything framework for multimodal processing.

    Args:
        rag_anything (RAGAnything): RAGAnything instance for multimodal document processing
        file_paths (List[Path]): List of file paths to be processed
        track_id (str, optional): Tracking ID for batch monitoring. Defaults to None.
        scheme_name (str, optional): Processing scheme name for categorization.
            Defaults to None.
        parser (str, optional): Document extraction tool to use.
            Defaults to None.
        source (str, optional): The model source used by Mineru.
            Defaults to None.

    Note:
        - Uses RAGAnything's process_document_complete_lightrag_api method for each file
        - Supports multimodal content processing (images, tables, equations)
        - Files are processed with "auto" parse method and "modelscope" source
        - Output is saved to "./output" directory
        - Errors are logged but don't stop processing of remaining files
    """
    if not file_paths:
        return

    try:
        # Use get_pinyin_sort_key for Chinese pinyin sorting
        sorted_file_paths = sorted(
            file_paths, key=lambda p: get_pinyin_sort_key(str(p))
        )

        # Process files sequentially with track_id
        for file_path in sorted_file_paths:
            success = await rag_anything.process_document_complete_lightrag_api(
                file_path=str(file_path),
                output_dir="./output",
                parse_method="auto",
                scheme_name=scheme_name,
                parser=parser,
                source=source,
            )
            if success:
                pass

    except Exception as e:
        error_msg = f"Error indexing files: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())


async def pipeline_index_texts(
    rag: LightRAG,
    texts: List[str],
    file_sources: List[str] = None,
    track_id: str = None,
):
    """Index a list of texts with track_id

    Args:
        rag: LightRAG instance
        texts: The texts to index
        file_sources: Sources of the texts
        track_id: Optional tracking ID
    """
    if not texts:
        return
    if file_sources is not None:
        if len(file_sources) != 0 and len(file_sources) != len(texts):
            [
                file_sources.append("unknown_source")
                for _ in range(len(file_sources), len(texts))
            ]
    await rag.apipeline_enqueue_documents(
        input=texts, file_paths=file_sources, track_id=track_id
    )
    await rag.apipeline_process_enqueue_documents()


async def run_scanning_process(
    rag: LightRAG,
    rag_anything: RAGAnything,
    doc_manager: DocumentManager,
    track_id: str = None,
    schemeConfig=None,
):
    """Background task to scan and index documents

    Args:
        rag: LightRAG instance
        rag_anythingL: RAGAnything instance
        doc_manager: DocumentManager instance
        track_id: Optional tracking ID to pass to all scanned files
        schemeConfig: Scanning scheme configuration.
            Defaults to None
    """
    try:
        new_files = doc_manager.scan_directory_for_new_files()
        total_files = len(new_files)
        logger.info(f"Found {total_files} files to index.")

        from lightrag.kg.shared_storage import get_namespace_data

        pipeline_status = await get_namespace_data("pipeline_status")
        is_pipeline_scan_busy = pipeline_status.get("scan_disabled", False)
        is_pipeline_busy = pipeline_status.get("busy", False)

        scheme_name = schemeConfig.framework
        extractor = schemeConfig.extractor
        modelSource = schemeConfig.modelSource

        if new_files:
            # Process all files at once with track_id
            if is_pipeline_busy:
                logger.info(
                    "Pipe is currently busy, skipping processing to avoid conflicts..."
                )
                return
            if is_pipeline_scan_busy:
                logger.info(
                    "Pipe is currently busy, skipping processing to avoid conflicts..."
                )
                return
            if scheme_name == "lightrag":
                await pipeline_index_files(
                    rag, new_files, track_id, scheme_name=scheme_name
                )
                logger.info(
                    f"Scanning process completed with lightrag: {total_files} files Processed."
                )
            elif scheme_name == "raganything":
                await pipeline_index_files_raganything(
                    rag_anything,
                    new_files,
                    scheme_name=scheme_name,
                    parser=extractor,
                    source=modelSource,
                )
                logger.info(
                    f"Scanning process completed with raganything: {total_files} files Processed."
                )
        else:
            # No new files to index, check if there are any documents in the queue
            logger.info(
                "No upload file found, check if there are any documents in the queue..."
            )
            await rag.apipeline_process_enqueue_documents()

    except Exception as e:
        logger.error(f"Error during scanning process: {str(e)}")
        logger.error(traceback.format_exc())


async def background_delete_documents(
    rag: LightRAG,
    doc_manager: DocumentManager,
    doc_ids: List[str],
    delete_file: bool = False,
):
    """Background task to delete multiple documents"""
    from lightrag.kg.shared_storage import (
        get_namespace_data,
        get_pipeline_status_lock,
    )

    pipeline_status = await get_namespace_data("pipeline_status")
    pipeline_status_lock = get_pipeline_status_lock()

    total_docs = len(doc_ids)
    successful_deletions = []
    failed_deletions = []

    # Double-check pipeline status before proceeding
    async with pipeline_status_lock:
        if pipeline_status.get("busy", False):
            logger.warning("Error: Unexpected pipeline busy state, aborting deletion.")
            return  # Abort deletion operation

        # Set pipeline status to busy for deletion
        pipeline_status.update(
            {
                "busy": True,
                "job_name": f"Deleting {total_docs} Documents",
                "job_start": datetime.now().isoformat(),
                "docs": total_docs,
                "batchs": total_docs,
                "cur_batch": 0,
                "latest_message": "Starting document deletion process",
            }
        )
        # Use slice assignment to clear the list in place
        pipeline_status["history_messages"][:] = ["Starting document deletion process"]

    try:
        # Loop through each document ID and delete them one by one
        for i, doc_id in enumerate(doc_ids, 1):
            async with pipeline_status_lock:
                start_msg = f"Deleting document {i}/{total_docs}: {doc_id}"
                logger.info(start_msg)
                pipeline_status["cur_batch"] = i
                pipeline_status["latest_message"] = start_msg
                pipeline_status["history_messages"].append(start_msg)

            file_path = "#"
            try:
                result = await rag.adelete_by_doc_id(doc_id)
                file_path = (
                    getattr(result, "file_path", "-") if "result" in locals() else "-"
                )
                if result.status == "success":
                    successful_deletions.append(doc_id)
                    success_msg = (
                        f"Document deleted {i}/{total_docs}: {doc_id}[{file_path}]"
                    )
                    logger.info(success_msg)
                    async with pipeline_status_lock:
                        pipeline_status["history_messages"].append(success_msg)

                    # Handle file deletion if requested and file_path is available
                    if (
                        delete_file
                        and result.file_path
                        and result.file_path != "unknown_source"
                    ):
                        try:
                            deleted_files = []
                            # check and delete files from input_dir directory
                            file_path = doc_manager.input_dir / result.file_path
                            if file_path.exists():
                                try:
                                    file_path.unlink()
                                    deleted_files.append(file_path.name)
                                    file_delete_msg = f"Successfully deleted input_dir file: {result.file_path}"
                                    logger.info(file_delete_msg)
                                    async with pipeline_status_lock:
                                        pipeline_status["latest_message"] = (
                                            file_delete_msg
                                        )
                                        pipeline_status["history_messages"].append(
                                            file_delete_msg
                                        )
                                except Exception as file_error:
                                    file_error_msg = f"Failed to delete input_dir file {result.file_path}: {str(file_error)}"
                                    logger.debug(file_error_msg)
                                    async with pipeline_status_lock:
                                        pipeline_status["latest_message"] = (
                                            file_error_msg
                                        )
                                        pipeline_status["history_messages"].append(
                                            file_error_msg
                                        )

                            # Also check and delete files from __enqueued__ directory
                            enqueued_dir = doc_manager.input_dir / "__enqueued__"
                            if enqueued_dir.exists():
                                # Look for files with the same name or similar names (with numeric suffixes)
                                base_name = Path(result.file_path).stem
                                extension = Path(result.file_path).suffix

                                # Search for exact match and files with numeric suffixes
                                for enqueued_file in enqueued_dir.glob(
                                    f"{base_name}*{extension}"
                                ):
                                    try:
                                        enqueued_file.unlink()
                                        deleted_files.append(enqueued_file.name)
                                        logger.info(
                                            f"Successfully deleted enqueued file: {enqueued_file.name}"
                                        )
                                    except Exception as enqueued_error:
                                        file_error_msg = f"Failed to delete enqueued file {enqueued_file.name}: {str(enqueued_error)}"
                                        logger.debug(file_error_msg)
                                        async with pipeline_status_lock:
                                            pipeline_status["latest_message"] = (
                                                file_error_msg
                                            )
                                            pipeline_status["history_messages"].append(
                                                file_error_msg
                                            )

                            if deleted_files == []:
                                file_error_msg = f"File deletion skipped, missing file: {result.file_path}"
                                logger.warning(file_error_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = file_error_msg
                                    pipeline_status["history_messages"].append(
                                        file_error_msg
                                    )

                        except Exception as file_error:
                            file_error_msg = f"Failed to delete file {result.file_path}: {str(file_error)}"
                            logger.error(file_error_msg)
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = file_error_msg
                                pipeline_status["history_messages"].append(
                                    file_error_msg
                                )
                    elif delete_file:
                        no_file_msg = (
                            f"File deletion skipped, missing file path: {doc_id}"
                        )
                        logger.warning(no_file_msg)
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = no_file_msg
                            pipeline_status["history_messages"].append(no_file_msg)
                else:
                    failed_deletions.append(doc_id)
                    error_msg = f"Failed to delete {i}/{total_docs}: {doc_id}[{file_path}] - {result.message}"
                    logger.error(error_msg)
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = error_msg
                        pipeline_status["history_messages"].append(error_msg)

            except Exception as e:
                failed_deletions.append(doc_id)
                error_msg = f"Error deleting document {i}/{total_docs}: {doc_id}[{file_path}] - {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = error_msg
                    pipeline_status["history_messages"].append(error_msg)

    except Exception as e:
        error_msg = f"Critical error during batch deletion: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        async with pipeline_status_lock:
            pipeline_status["history_messages"].append(error_msg)
    finally:
        # Final summary and check for pending requests
        async with pipeline_status_lock:
            pipeline_status["busy"] = False
            completion_msg = f"Deletion completed: {len(successful_deletions)} successful, {len(failed_deletions)} failed"
            pipeline_status["latest_message"] = completion_msg
            pipeline_status["history_messages"].append(completion_msg)

            # Check if there are pending document indexing requests
            has_pending_request = pipeline_status.get("request_pending", False)

        # If there are pending requests, start document processing pipeline
        if has_pending_request:
            try:
                logger.info(
                    "Processing pending document indexing requests after deletion"
                )
                await rag.apipeline_process_enqueue_documents()
            except Exception as e:
                logger.error(f"Error processing pending documents after deletion: {e}")


def create_document_routes(
    rag: LightRAG,
    rag_anything: RAGAnything,
    doc_manager: DocumentManager,
    api_key: Optional[str] = None,
):
    # Create combined auth dependency for document routes
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        "/schemes",
        response_model=SchemesResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_all_schemes():
        """Get all available processing schemes.

        Retrieves the complete list of processing schemes from the schemes.json file.
        Each scheme defines a processing framework (lightrag/raganything) and
        optional extractor configuration (mineru/docling).

        Returns:
            SchemesResponse: Response containing:
                - status (str): Operation status ("success")
                - message (str): Success message
                - data (List[Dict]): List of all available schemes with their configurations

        Raises:
            HTTPException: If file reading fails or JSON parsing errors occur (500)
        """
        SCHEMES_FILE = Path("./examples/schemes.json")

        if SCHEMES_FILE.exists():
            with open(SCHEMES_FILE, "r", encoding="utf-8") as f:
                try:
                    current_data = json.load(f)
                except json.JSONDecodeError:
                    current_data = []
        else:
            current_data = []
            SCHEMES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SCHEMES_FILE, "w") as f:
                json.dump(current_data, f)

        return SchemesResponse(
            status="success",
            message="Schemes retrieved successfully",
            data=current_data,
        )

    @router.post(
        "/schemes",
        response_model=SchemesResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def save_schemes(schemes: list[Scheme_include_id]):
        """Save/update processing schemes in batch.

        Updates existing schemes with new configuration data. This endpoint performs
        a partial update by modifying existing schemes based on their IDs while
        preserving other schemes in the file.

        Args:
            schemes (list[Scheme_include_id]): List of schemes to update, each containing:
                - id (int): Unique identifier of the scheme to update
                - name (str): Display name for the scheme
                - config (SchemeConfig): Configuration object with framework and extractor settings

        Returns:
            SchemesResponse: Response containing:
                - status (str): Operation status ("success")
                - message (str): Success message with count of saved schemes
                - data (List[Dict]): Updated list of all schemes after modification

        Raises:
            HTTPException: If file operations fail or JSON processing errors occur (500)
        """
        try:
            SCHEMES_FILE = Path("./examples/schemes.json")

            if SCHEMES_FILE.exists():
                with open(SCHEMES_FILE, "r", encoding="utf-8") as f:
                    try:
                        current_data = json.load(f)
                    except json.JSONDecodeError:
                        current_data = []
            else:
                current_data = []

            updated_item = {
                "id": schemes[0].id,
                "name": schemes[0].name,
                "config": {
                    "framework": schemes[0].config.framework,
                    "extractor": schemes[0].config.extractor,
                    "modelSource": schemes[0].config.modelSource,
                },
            }
            # 保存新方案
            for item in current_data:
                if item["id"] == updated_item["id"]:
                    item["name"] = updated_item["name"]
                    item["config"]["framework"] = updated_item["config"]["framework"]
                    item["config"]["extractor"] = updated_item["config"]["extractor"]
                    item["config"]["modelSource"] = updated_item["config"][
                        "modelSource"
                    ]
                    break

            # 写回文件
            with open(SCHEMES_FILE, "w", encoding="utf-8") as f:
                json.dump(current_data, f, indent=4)

            # 返回响应（从文件重新读取确保一致性）
            with open(SCHEMES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            return SchemesResponse(
                status="success",
                message=f"Successfully saved {len(schemes)} schemes",
                data=data,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/schemes/add",
        response_model=Scheme_include_id,
        dependencies=[Depends(combined_auth)],
    )
    async def add_scheme(scheme: Scheme):
        """Add a new processing scheme.

        Creates a new processing scheme with auto-generated ID and saves it to the
        schemes configuration file. The new scheme will be available for document
        processing operations.

        Args:
            scheme (Scheme): New scheme to add, containing:
                - name (str): Display name for the scheme
                - config (SchemeConfig): Configuration with framework and extractor settings

        Returns:
            Scheme_include_id: The created scheme with auto-generated ID, containing:
                - id (int): Auto-generated unique identifier
                - name (str): Display name of the scheme
                - config (SchemeConfig): Processing configuration

        Raises:
            HTTPException: If file operations fail or ID generation conflicts occur (500)
        """
        try:
            SCHEMES_FILE = Path("./examples/schemes.json")

            if SCHEMES_FILE.exists():
                with open(SCHEMES_FILE, "r", encoding="utf-8") as f:
                    try:
                        current_data = json.load(f)
                    except json.JSONDecodeError:
                        current_data = []
            else:
                current_data = []

            # 生成新ID（简单实现，实际项目应该用数据库自增ID）
            new_id = uuid.uuid4().int >> 96  # 生成一个较小的整数ID
            while new_id in current_data:
                new_id = uuid.uuid4().int >> 96

            new_scheme = {
                "id": new_id,
                "name": scheme.name,
                "config": {
                    "framework": scheme.config.framework,
                    "extractor": scheme.config.extractor,
                    "modelSource": scheme.config.modelSource,
                },
            }

            current_data.append(new_scheme)

            with open(SCHEMES_FILE, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=2)

            return new_scheme
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete(
        "/schemes/{scheme_id}",
        response_model=Dict[str, str],
        dependencies=[Depends(combined_auth)],
    )
    async def delete_scheme(scheme_id: int):
        """Delete a specific processing scheme by ID.

        Removes a processing scheme from the configuration file. Once deleted,
        the scheme will no longer be available for document processing operations.

        Args:
            scheme_id (int): Unique identifier of the scheme to delete

        Returns:
            Dict[str, str]: Success message containing:
                - message (str): Confirmation message with the deleted scheme ID

        Raises:
            HTTPException:
                - 404: If the scheme with the specified ID is not found
                - 500: If file operations fail or other errors occur
        """
        try:
            SCHEMES_FILE = Path("./examples/schemes.json")

            if SCHEMES_FILE.exists():
                with open(SCHEMES_FILE, "r", encoding="utf-8") as f:
                    try:
                        current_data = json.load(f)
                    except json.JSONDecodeError:
                        current_data = []
            else:
                current_data = []

            current_data_dict = {scheme["id"]: scheme for scheme in current_data}

            if scheme_id not in current_data_dict:  # 直接检查 id 是否存在
                raise HTTPException(status_code=404, detail="Scheme not found")

            for i, scheme in enumerate(current_data):
                if scheme["id"] == scheme_id:
                    del current_data[i]  # 直接删除列表中的元素
                    break

            with open(SCHEMES_FILE, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=2)

            return {"message": f"Scheme {scheme_id} deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/scan", response_model=ScanResponse, dependencies=[Depends(combined_auth)]
    )
    async def scan_for_new_documents(
        request: ScanRequest, background_tasks: BackgroundTasks
    ):
        """
        Trigger the scanning process for new documents.

        This endpoint initiates a background task that scans the input directory for new documents
        and processes them. If a scanning process is already running, it returns a status indicating
        that fact.

        Returns:
            ScanResponse: A response object containing the scanning status and track_id
        """
        # Generate track_id with "scan" prefix for scanning operation
        track_id = generate_track_id("scan")

        # Start the scanning process in the background with track_id
        background_tasks.add_task(
            run_scanning_process,
            rag,
            rag_anything,
            doc_manager,
            track_id,
            schemeConfig=request.schemeConfig,
        )
        return ScanResponse(
            status="scanning_started",
            message="Scanning process has been initiated in the background",
            track_id=track_id,
        )

    @router.post(
        "/upload", response_model=InsertResponse, dependencies=[Depends(combined_auth)]
    )
    async def upload_to_input_dir(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        schemeId: str = Form(...),
    ):
        """
        Upload a file to the input directory and index it.

        This API endpoint accepts a file through an HTTP POST request, checks if the
        uploaded file is of a supported type, saves it in the specified input directory,
        indexes it for retrieval, and returns a success status with relevant details.

        Args:
            background_tasks: FastAPI BackgroundTasks for async processing
            file (UploadFile): The file to be uploaded. It must have an allowed extension
            schemeId (str): ID of the processing scheme to use for this file. The scheme
                determines whether to use LightRAG or RAGAnything framework for processing.

        Returns:
            InsertResponse: A response object containing the upload status and a message.
                status can be "success", "duplicated", or error is thrown.

        Raises:
            HTTPException: If the file type is not supported (400) or other errors occur (500).
        """
        try:
            # Sanitize filename to prevent Path Traversal attacks
            safe_filename = sanitize_filename(file.filename, doc_manager.input_dir)

            if not doc_manager.is_supported_file(safe_filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Supported types: {doc_manager.supported_extensions}",
                )

            file_path = doc_manager.input_dir / safe_filename
            # Check if file already exists
            if file_path.exists():
                return InsertResponse(
                    status="duplicated",
                    message=f"File '{safe_filename}' already exists in the input directory.",
                    track_id="",
                )

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            track_id = generate_track_id("upload")

            def load_config():
                try:
                    SCHEMES_FILE = Path("./examples/schemes.json")
                    with open(SCHEMES_FILE, "r") as f:
                        schemes = json.load(f)
                    for scheme in schemes:
                        if str(scheme.get("id")) == schemeId:
                            return scheme.get("config", {})
                    return {}
                except Exception as e:
                    logger.error(
                        f"Failed to load config for scheme {schemeId}: {str(e)}"
                    )
                    return {}

            config = load_config()
            current_framework = config.get("framework")
            current_extractor = config.get("extractor")
            current_modelSource = config.get("modelSource")
            doc_pre_id = f"doc-pre-{safe_filename}"

            if current_framework and current_framework == "lightrag":
                # Add to background tasks and get track_id
                background_tasks.add_task(
                    pipeline_index_file,
                    rag,
                    file_path,
                    track_id,
                    scheme_name=current_framework,
                )
            else:
                background_tasks.add_task(
                    rag_anything.process_document_complete_lightrag_api,
                    file_path=str(file_path),
                    output_dir="./output",
                    parse_method="auto",
                    scheme_name=current_framework,
                    parser=current_extractor,
                    source=current_modelSource,
                )

            await rag.doc_status.upsert(
                {
                    doc_pre_id: {
                        "status": DocStatus.READY,
                        "content": "",
                        "content_summary": "",
                        "multimodal_content": [],
                        "scheme_name": current_framework,
                        "content_length": 0,
                        "created_at": "",
                        "updated_at": "",
                        "file_path": safe_filename,
                    }
                }
            )

            return InsertResponse(
                status="success",
                message=f"File '{safe_filename}' uploaded successfully. Processing will continue in background.",
                track_id=track_id,
            )

        except Exception as e:
            logger.error(f"Error /documents/upload: {file.filename}: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/text", response_model=InsertResponse, dependencies=[Depends(combined_auth)]
    )
    async def insert_text(
        request: InsertTextRequest, background_tasks: BackgroundTasks
    ):
        """
        Insert text into the RAG system.

        This endpoint allows you to insert text data into the RAG system for later retrieval
        and use in generating responses.

        Args:
            request (InsertTextRequest): The request body containing the text to be inserted.
            background_tasks: FastAPI BackgroundTasks for async processing

        Returns:
            InsertResponse: A response object containing the status of the operation.

        Raises:
            HTTPException: If an error occurs during text processing (500).
        """
        try:
            # Generate track_id for text insertion
            track_id = generate_track_id("insert")

            background_tasks.add_task(
                pipeline_index_texts,
                rag,
                [request.text],
                file_sources=[request.file_source],
                track_id=track_id,
            )

            return InsertResponse(
                status="success",
                message="Text successfully received. Processing will continue in background.",
                track_id=track_id,
            )
        except Exception as e:
            logger.error(f"Error /documents/text: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/texts",
        response_model=InsertResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def insert_texts(
        request: InsertTextsRequest, background_tasks: BackgroundTasks
    ):
        """
        Insert multiple texts into the RAG system.

        This endpoint allows you to insert multiple text entries into the RAG system
        in a single request.

        Args:
            request (InsertTextsRequest): The request body containing the list of texts.
            background_tasks: FastAPI BackgroundTasks for async processing

        Returns:
            InsertResponse: A response object containing the status of the operation.

        Raises:
            HTTPException: If an error occurs during text processing (500).
        """
        try:
            # Generate track_id for texts insertion
            track_id = generate_track_id("insert")

            background_tasks.add_task(
                pipeline_index_texts,
                rag,
                request.texts,
                file_sources=request.file_sources,
                track_id=track_id,
            )

            return InsertResponse(
                status="success",
                message="Texts successfully received. Processing will continue in background.",
                track_id=track_id,
            )
        except Exception as e:
            logger.error(f"Error /documents/texts: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete(
        "", response_model=ClearDocumentsResponse, dependencies=[Depends(combined_auth)]
    )
    async def clear_documents():
        """
        Clear all documents from the RAG system.

        This endpoint deletes all documents, entities, relationships, and files from the system.
        It uses the storage drop methods to properly clean up all data and removes all files
        from the input directory.

        Returns:
            ClearDocumentsResponse: A response object containing the status and message.
                - status="success":           All documents and files were successfully cleared.
                - status="partial_success":   Document clear job exit with some errors.
                - status="busy":              Operation could not be completed because the pipeline is busy.
                - status="fail":              All storage drop operations failed, with message
                - message: Detailed information about the operation results, including counts
                  of deleted files and any errors encountered.

        Raises:
            HTTPException: Raised when a serious error occurs during the clearing process,
                          with status code 500 and error details in the detail field.
        """
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )

        # Get pipeline status and lock
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Check and set status with lock
        async with pipeline_status_lock:
            if pipeline_status.get("busy", False):
                return ClearDocumentsResponse(
                    status="busy",
                    message="Cannot clear documents while pipeline is busy",
                )
            # Set busy to true
            pipeline_status.update(
                {
                    "busy": True,
                    "job_name": "Clearing Documents",
                    "job_start": datetime.now().isoformat(),
                    "docs": 0,
                    "batchs": 0,
                    "cur_batch": 0,
                    "request_pending": False,  # Clear any previous request
                    "latest_message": "Starting document clearing process",
                }
            )
            # Cleaning history_messages without breaking it as a shared list object
            del pipeline_status["history_messages"][:]
            pipeline_status["history_messages"].append(
                "Starting document clearing process"
            )

        try:
            # Use drop method to clear all data
            drop_tasks = []
            storages = [
                rag.text_chunks,
                rag.full_docs,
                rag.full_entities,
                rag.full_relations,
                rag.entities_vdb,
                rag.relationships_vdb,
                rag.chunks_vdb,
                rag.chunk_entity_relation_graph,
                rag.doc_status,
            ]

            # Log storage drop start
            if "history_messages" in pipeline_status:
                pipeline_status["history_messages"].append(
                    "Starting to drop storage components"
                )

            for storage in storages:
                if storage is not None:
                    drop_tasks.append(storage.drop())

            # Wait for all drop tasks to complete
            drop_results = await asyncio.gather(*drop_tasks, return_exceptions=True)

            # Check for errors and log results
            errors = []
            storage_success_count = 0
            storage_error_count = 0

            for i, result in enumerate(drop_results):
                storage_name = storages[i].__class__.__name__
                if isinstance(result, Exception):
                    error_msg = f"Error dropping {storage_name}: {str(result)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    storage_error_count += 1
                else:
                    namespace = storages[i].namespace
                    workspace = storages[i].workspace
                    logger.info(
                        f"Successfully dropped {storage_name}: {workspace}/{namespace}"
                    )
                    storage_success_count += 1

            # Log storage drop results
            if "history_messages" in pipeline_status:
                if storage_error_count > 0:
                    pipeline_status["history_messages"].append(
                        f"Dropped {storage_success_count} storage components with {storage_error_count} errors"
                    )
                else:
                    pipeline_status["history_messages"].append(
                        f"Successfully dropped all {storage_success_count} storage components"
                    )

            # Clean all parse_cache entries after successful storage drops
            if storage_success_count > 0:
                try:
                    if "history_messages" in pipeline_status:
                        pipeline_status["history_messages"].append(
                            "Cleaning parse_cache entries"
                        )

                    parse_cache_result = await rag.aclean_all_parse_cache()
                    if parse_cache_result.get("error"):
                        cache_error_msg = f"Warning: Failed to clean parse_cache: {parse_cache_result['error']}"
                        logger.warning(cache_error_msg)
                        if "history_messages" in pipeline_status:
                            pipeline_status["history_messages"].append(cache_error_msg)
                    else:
                        deleted_count = parse_cache_result.get("deleted_count", 0)
                        if deleted_count > 0:
                            cache_success_msg = f"Successfully cleaned {deleted_count} parse_cache entries"
                            logger.info(cache_success_msg)
                            if "history_messages" in pipeline_status:
                                pipeline_status["history_messages"].append(
                                    cache_success_msg
                                )
                        else:
                            cache_empty_msg = "No parse_cache entries to clean"
                            logger.info(cache_empty_msg)
                            if "history_messages" in pipeline_status:
                                pipeline_status["history_messages"].append(
                                    cache_empty_msg
                                )
                except Exception as cache_error:
                    cache_error_msg = f"Warning: Exception while cleaning parse_cache: {str(cache_error)}"
                    logger.warning(cache_error_msg)
                    if "history_messages" in pipeline_status:
                        pipeline_status["history_messages"].append(cache_error_msg)

            # If all storage operations failed, return error status and don't proceed with file deletion
            if storage_success_count == 0 and storage_error_count > 0:
                error_message = "All storage drop operations failed. Aborting document clearing process."
                logger.error(error_message)
                if "history_messages" in pipeline_status:
                    pipeline_status["history_messages"].append(error_message)
                return ClearDocumentsResponse(status="fail", message=error_message)

            # Log file deletion start
            if "history_messages" in pipeline_status:
                pipeline_status["history_messages"].append(
                    "Starting to delete files in input directory"
                )

            # Delete only files in the current directory, preserve files in subdirectories
            deleted_files_count = 0
            file_errors_count = 0

            for file_path in doc_manager.input_dir.glob("*"):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_files_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {str(e)}")
                        file_errors_count += 1

            # Log file deletion results
            if "history_messages" in pipeline_status:
                if file_errors_count > 0:
                    pipeline_status["history_messages"].append(
                        f"Deleted {deleted_files_count} files with {file_errors_count} errors"
                    )
                    errors.append(f"Failed to delete {file_errors_count} files")
                else:
                    pipeline_status["history_messages"].append(
                        f"Successfully deleted {deleted_files_count} files"
                    )

            # Prepare final result message
            final_message = ""
            if errors:
                final_message = f"Cleared documents with some errors. Deleted {deleted_files_count} files."
                status = "partial_success"
            else:
                final_message = f"All documents cleared successfully. Deleted {deleted_files_count} files."
                status = "success"

            # Log final result
            if "history_messages" in pipeline_status:
                pipeline_status["history_messages"].append(final_message)

            # Return response based on results
            return ClearDocumentsResponse(status=status, message=final_message)
        except Exception as e:
            error_msg = f"Error clearing documents: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            if "history_messages" in pipeline_status:
                pipeline_status["history_messages"].append(error_msg)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Reset busy status after completion
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                completion_msg = "Document clearing process completed"
                pipeline_status["latest_message"] = completion_msg
                if "history_messages" in pipeline_status:
                    pipeline_status["history_messages"].append(completion_msg)

    @router.get(
        "/pipeline_status",
        dependencies=[Depends(combined_auth)],
        response_model=PipelineStatusResponse,
    )
    async def get_pipeline_status() -> PipelineStatusResponse:
        """
        Get the current status of the document indexing pipeline.

        This endpoint returns information about the current state of the document processing pipeline,
        including the processing status, progress information, and history messages.

        Returns:
            PipelineStatusResponse: A response object containing:
                - autoscanned (bool): Whether auto-scan has started
                - busy (bool): Whether the pipeline is currently busy
                - job_name (str): Current job name (e.g., indexing files/indexing texts)
                - job_start (str, optional): Job start time as ISO format string
                - docs (int): Total number of documents to be indexed
                - batchs (int): Number of batches for processing documents
                - cur_batch (int): Current processing batch
                - request_pending (bool): Flag for pending request for processing
                - latest_message (str): Latest message from pipeline processing
                - history_messages (List[str], optional): List of history messages (limited to latest 1000 entries,
                  with truncation message if more than 1000 messages exist)

        Raises:
            HTTPException: If an error occurs while retrieving pipeline status (500)
        """
        try:
            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_all_update_flags_status,
            )

            pipeline_status = await get_namespace_data("pipeline_status")

            # Get update flags status for all namespaces
            update_status = await get_all_update_flags_status()

            # Convert MutableBoolean objects to regular boolean values
            processed_update_status = {}
            for namespace, flags in update_status.items():
                processed_flags = []
                for flag in flags:
                    # Handle both multiprocess and single process cases
                    if hasattr(flag, "value"):
                        processed_flags.append(bool(flag.value))
                    else:
                        processed_flags.append(bool(flag))
                processed_update_status[namespace] = processed_flags

            # Convert to regular dict if it's a Manager.dict
            status_dict = dict(pipeline_status)

            # Add processed update_status to the status dictionary
            status_dict["update_status"] = processed_update_status

            # Convert history_messages to a regular list if it's a Manager.list
            # and limit to latest 1000 entries with truncation message if needed
            if "history_messages" in status_dict:
                history_list = list(status_dict["history_messages"])
                total_count = len(history_list)

                if total_count > 1000:
                    # Calculate truncated message count
                    truncated_count = total_count - 1000

                    # Take only the latest 1000 messages
                    latest_messages = history_list[-1000:]

                    # Add truncation message at the beginning
                    truncation_message = (
                        f"[Truncated history messages: {truncated_count}/{total_count}]"
                    )
                    status_dict["history_messages"] = [
                        truncation_message
                    ] + latest_messages
                else:
                    # No truncation needed, return all messages
                    status_dict["history_messages"] = history_list

            # Ensure job_start is properly formatted as a string with timezone information
            if "job_start" in status_dict and status_dict["job_start"]:
                # Use format_datetime to ensure consistent formatting
                status_dict["job_start"] = format_datetime(status_dict["job_start"])

            return PipelineStatusResponse(**status_dict)
        except Exception as e:
            logger.error(f"Error getting pipeline status: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "", response_model=DocsStatusesResponse, dependencies=[Depends(combined_auth)]
    )
    async def documents() -> DocsStatusesResponse:
        """
        Get the status of all documents in the system.

        This endpoint retrieves the current status of all documents, grouped by their
        processing status (READY, HANDLING, PENDING, PROCESSING, PROCESSED, FAILED).

        Returns:
            DocsStatusesResponse: A response object containing a dictionary where keys are
                                DocStatus values and values are lists of DocStatusResponse
                                objects representing documents in each status category.

        Raises:
            HTTPException: If an error occurs while retrieving document statuses (500).
        """
        try:
            statuses = (
                DocStatus.READY,
                DocStatus.HANDLING,
                DocStatus.PENDING,
                DocStatus.PROCESSING,
                DocStatus.PROCESSED,
                DocStatus.FAILED,
            )

            tasks = [rag.get_docs_by_status(status) for status in statuses]
            results: List[Dict[str, DocProcessingStatus]] = await asyncio.gather(*tasks)

            response = DocsStatusesResponse()

            for idx, result in enumerate(results):
                status = statuses[idx]
                for doc_id, doc_status in result.items():
                    if status not in response.statuses:
                        response.statuses[status] = []
                    response.statuses[status].append(
                        DocStatusResponse(
                            id=doc_id,
                            content_summary=doc_status.content_summary,
                            multimodal_content=doc_status.multimodal_content,
                            content_length=doc_status.content_length,
                            status=doc_status.status,
                            created_at=format_datetime(doc_status.created_at),
                            updated_at=format_datetime(doc_status.updated_at),
                            track_id=doc_status.track_id,
                            chunks_count=doc_status.chunks_count,
                            error_msg=doc_status.error_msg,
                            metadata=doc_status.metadata,
                            file_path=doc_status.file_path,
                            scheme_name=doc_status.scheme_name,
                        )
                    )
            return response
        except Exception as e:
            logger.error(f"Error GET /documents: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    class DeleteDocByIdResponse(BaseModel):
        """Response model for single document deletion operation."""

        status: Literal["deletion_started", "busy", "not_allowed"] = Field(
            description="Status of the deletion operation"
        )
        message: str = Field(description="Message describing the operation result")
        doc_id: str = Field(description="The ID of the document to delete")

    @router.delete(
        "/delete_document",
        response_model=DeleteDocByIdResponse,
        dependencies=[Depends(combined_auth)],
        summary="Delete a document and all its associated data by its ID.",
    )
    async def delete_document(
        delete_request: DeleteDocRequest,
        background_tasks: BackgroundTasks,
    ) -> DeleteDocByIdResponse:
        """
        Delete documents and all their associated data by their IDs using background processing.

        Deletes specific documents and all their associated data, including their status,
        text chunks, vector embeddings, and any related graph data.
        The deletion process runs in the background to avoid blocking the client connection.
        It is disabled when llm cache for entity extraction is disabled.

        This operation is irreversible and will interact with the pipeline status.

        Args:
            delete_request (DeleteDocRequest): The request containing the document IDs and delete_file options.
            background_tasks: FastAPI BackgroundTasks for async processing

        Returns:
            DeleteDocByIdResponse: The result of the deletion operation.
                - status="deletion_started": The document deletion has been initiated in the background.
                - status="busy": The pipeline is busy with another operation.
                - status="not_allowed": Operation not allowed when LLM cache for entity extraction is disabled.

        Raises:
            HTTPException:
              - 500: If an unexpected internal error occurs during initialization.
        """
        doc_ids = delete_request.doc_ids

        # The rag object is initialized from the server startup args,
        # so we can access its properties here.
        if not rag.enable_llm_cache_for_entity_extract:
            return DeleteDocByIdResponse(
                status="not_allowed",
                message="Operation not allowed when LLM cache for entity extraction is disabled.",
                doc_id=", ".join(delete_request.doc_ids),
            )

        try:
            from lightrag.kg.shared_storage import get_namespace_data

            pipeline_status = await get_namespace_data("pipeline_status")

            # Check if pipeline is busy
            if pipeline_status.get("busy", False):
                return DeleteDocByIdResponse(
                    status="busy",
                    message="Cannot delete documents while pipeline is busy",
                    doc_id=", ".join(doc_ids),
                )

            # Add deletion task to background tasks
            background_tasks.add_task(
                background_delete_documents,
                rag,
                doc_manager,
                doc_ids,
                delete_request.delete_file,
            )

            return DeleteDocByIdResponse(
                status="deletion_started",
                message=f"Document deletion for {len(doc_ids)} documents has been initiated. Processing will continue in background.",
                doc_id=", ".join(doc_ids),
            )

        except Exception as e:
            error_msg = f"Error initiating document deletion for {delete_request.doc_ids}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)

    @router.post(
        "/clear_cache",
        response_model=ClearCacheResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def clear_cache(request: ClearCacheRequest):
        """
        Clear all cache data from the LLM response cache storage.

        This endpoint clears all cached LLM responses regardless of mode.
        The request body is accepted for API compatibility but is ignored.

        Args:
            request (ClearCacheRequest): The request body (ignored for compatibility).

        Returns:
            ClearCacheResponse: A response object containing the status and message.

        Raises:
            HTTPException: If an error occurs during cache clearing (500).
        """
        try:
            # Call the aclear_cache method (no modes parameter)
            await rag.aclear_cache()

            # Prepare success message
            message = "Successfully cleared all cache"

            return ClearCacheResponse(status="success", message=message)
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete(
        "/delete_entity",
        response_model=DeletionResult,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_entity(request: DeleteEntityRequest):
        """
        Delete an entity and all its relationships from the knowledge graph.

        Args:
            request (DeleteEntityRequest): The request body containing the entity name.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.

        Raises:
            HTTPException: If the entity is not found (404) or an error occurs (500).
        """
        try:
            result = await rag.adelete_by_entity(entity_name=request.entity_name)
            if result.status == "not_found":
                raise HTTPException(status_code=404, detail=result.message)
            if result.status == "fail":
                raise HTTPException(status_code=500, detail=result.message)
            # Set doc_id to empty string since this is an entity operation, not document
            result.doc_id = ""
            return result
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Error deleting entity '{request.entity_name}': {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)

    @router.delete(
        "/delete_relation",
        response_model=DeletionResult,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_relation(request: DeleteRelationRequest):
        """
        Delete a relationship between two entities from the knowledge graph.

        Args:
            request (DeleteRelationRequest): The request body containing the source and target entity names.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.

        Raises:
            HTTPException: If the relation is not found (404) or an error occurs (500).
        """
        try:
            result = await rag.adelete_by_relation(
                source_entity=request.source_entity,
                target_entity=request.target_entity,
            )
            if result.status == "not_found":
                raise HTTPException(status_code=404, detail=result.message)
            if result.status == "fail":
                raise HTTPException(status_code=500, detail=result.message)
            # Set doc_id to empty string since this is a relation operation, not document
            result.doc_id = ""
            return result
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Error deleting relation from '{request.source_entity}' to '{request.target_entity}': {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)

    @router.get(
        "/track_status/{track_id}",
        response_model=TrackStatusResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_track_status(track_id: str) -> TrackStatusResponse:
        """
        Get the processing status of documents by tracking ID.

        This endpoint retrieves all documents associated with a specific tracking ID,
        allowing users to monitor the processing progress of their uploaded files or inserted texts.

        Args:
            track_id (str): The tracking ID returned from upload, text, or texts endpoints

        Returns:
            TrackStatusResponse: A response object containing:
                - track_id: The tracking ID
                - documents: List of documents associated with this track_id
                - total_count: Total number of documents for this track_id

        Raises:
            HTTPException: If track_id is invalid (400) or an error occurs (500).
        """
        try:
            # Validate track_id
            if not track_id or not track_id.strip():
                raise HTTPException(status_code=400, detail="Track ID cannot be empty")

            track_id = track_id.strip()

            # Get documents by track_id
            docs_by_track_id = await rag.aget_docs_by_track_id(track_id)

            # Convert to response format
            documents = []
            status_summary = {}

            for doc_id, doc_status in docs_by_track_id.items():
                documents.append(
                    DocStatusResponse(
                        id=doc_id,
                        content_summary=doc_status.content_summary,
                        content_length=doc_status.content_length,
                        status=doc_status.status,
                        created_at=format_datetime(doc_status.created_at),
                        updated_at=format_datetime(doc_status.updated_at),
                        track_id=doc_status.track_id,
                        chunks_count=doc_status.chunks_count,
                        error_msg=doc_status.error_msg,
                        metadata=doc_status.metadata,
                        file_path=doc_status.file_path,
                    )
                )

                # Build status summary
                # Handle both DocStatus enum and string cases for robust deserialization
                status_key = str(doc_status.status)
                status_summary[status_key] = status_summary.get(status_key, 0) + 1

            return TrackStatusResponse(
                track_id=track_id,
                documents=documents,
                total_count=len(documents),
                status_summary=status_summary,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting track status for {track_id}: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/paginated",
        response_model=PaginatedDocsResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_documents_paginated(
        request: DocumentsRequest,
    ) -> PaginatedDocsResponse:
        """
        Get documents with pagination support.

        This endpoint retrieves documents with pagination, filtering, and sorting capabilities.
        It provides better performance for large document collections by loading only the
        requested page of data.

        Args:
            request (DocumentsRequest): The request body containing pagination parameters

        Returns:
            PaginatedDocsResponse: A response object containing:
                - documents: List of documents for the current page
                - pagination: Pagination information (page, total_count, etc.)
                - status_counts: Count of documents by status for all documents

        Raises:
            HTTPException: If an error occurs while retrieving documents (500).
        """
        try:
            # Get paginated documents and status counts in parallel
            docs_task = rag.doc_status.get_docs_paginated(
                status_filter=request.status_filter,
                page=request.page,
                page_size=request.page_size,
                sort_field=request.sort_field,
                sort_direction=request.sort_direction,
            )
            status_counts_task = rag.doc_status.get_all_status_counts()

            # Execute both queries in parallel
            (documents_with_ids, total_count), status_counts = await asyncio.gather(
                docs_task, status_counts_task
            )

            # Convert documents to response format
            doc_responses = []
            for doc_id, doc in documents_with_ids:
                doc_responses.append(
                    DocStatusResponse(
                        id=doc_id,
                        content_summary=doc.content_summary,
                        content_length=doc.content_length,
                        status=doc.status,
                        created_at=format_datetime(doc.created_at),
                        updated_at=format_datetime(doc.updated_at),
                        track_id=doc.track_id,
                        chunks_count=doc.chunks_count,
                        error_msg=doc.error_msg,
                        metadata=doc.metadata,
                        file_path=doc.file_path,
                        scheme_name=doc.scheme_name,
                        multimodal_content=doc.multimodal_content,
                    )
                )

            # Calculate pagination info
            total_pages = (total_count + request.page_size - 1) // request.page_size
            has_next = request.page < total_pages
            has_prev = request.page > 1

            pagination = PaginationInfo(
                page=request.page,
                page_size=request.page_size,
                total_count=total_count,
                total_pages=total_pages,
                has_next=has_next,
                has_prev=has_prev,
            )

            return PaginatedDocsResponse(
                documents=doc_responses,
                pagination=pagination,
                status_counts=status_counts,
            )

        except Exception as e:
            logger.error(f"Error getting paginated documents: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/status_counts",
        response_model=StatusCountsResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_document_status_counts() -> StatusCountsResponse:
        """
        Get counts of documents by status.

        This endpoint retrieves the count of documents in each processing status
        (PENDING, PROCESSING, PROCESSED, FAILED) for all documents in the system.

        Returns:
            StatusCountsResponse: A response object containing status counts

        Raises:
            HTTPException: If an error occurs while retrieving status counts (500).
        """
        try:
            status_counts = await rag.doc_status.get_all_status_counts()
            return StatusCountsResponse(status_counts=status_counts)

        except Exception as e:
            logger.error(f"Error getting document status counts: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    return router
