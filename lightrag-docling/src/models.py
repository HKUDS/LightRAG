"""
Pydantic models for the Docling service API.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
import base64


class ProcessingStatus(str, Enum):
    """Document processing status."""

    SUCCESS = "success"
    FAILED = "failed"
    PROCESSING = "processing"
    QUEUED = "queued"


class ExportFormat(str, Enum):
    """Supported export formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"
    HTML = "html"


class DoclingConfig(BaseModel):
    """Docling processing configuration."""

    # Export format
    export_format: ExportFormat = ExportFormat.MARKDOWN

    # Processing options
    enable_ocr: bool = True
    enable_table_structure: bool = True
    enable_figures: bool = True
    process_images: bool = True

    # Model selection
    layout_model: str = "auto"
    ocr_model: str = "auto"
    table_model: str = "auto"

    # Content processing
    include_page_numbers: bool = True
    include_headings: bool = True
    extract_metadata: bool = True

    # Quality settings
    image_dpi: int = Field(default=300, ge=72, le=600)
    ocr_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    table_confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    # Performance
    max_workers: int = Field(default=2, ge=1, le=8)

    # Caching
    enable_cache: bool = True
    cache_ttl_hours: int = Field(default=168, ge=1)  # 7 days default

    @validator("export_format", pre=True)
    def validate_export_format(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v


class DocumentProcessRequest(BaseModel):
    """Request model for document processing."""

    # File content (base64 encoded)
    file_content: str = Field(..., description="Base64 encoded file content")
    filename: str = Field(..., description="Original filename with extension")

    # Processing configuration
    config: DoclingConfig = Field(default_factory=DoclingConfig)

    # Optional cache key for custom caching
    cache_key: Optional[str] = Field(None, description="Custom cache key")

    # Request metadata
    request_id: Optional[str] = Field(None, description="Request tracking ID")

    @validator("file_content")
    def validate_file_content(cls, v):
        """Validate base64 encoded content."""
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 encoded file content")

    @validator("filename")
    def validate_filename(cls, v):
        """Validate filename has supported extension."""
        supported_extensions = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"}
        if not any(v.lower().endswith(ext) for ext in supported_extensions):
            raise ValueError(
                f"Unsupported file type. Supported: {supported_extensions}"
            )
        return v


class ProcessingMetadata(BaseModel):
    """Metadata about the processing operation."""

    processing_time_seconds: float
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None

    # Model information
    models_used: Dict[str, str] = Field(default_factory=dict)

    # Processing details
    ocr_applied: bool = False
    tables_extracted: int = 0
    figures_extracted: int = 0

    # Cache information
    cache_hit: bool = False
    cache_key: Optional[str] = None

    # Configuration used
    config_hash: Optional[str] = None


class ProcessingResult(BaseModel):
    """Result of document processing."""

    # Core results
    content: str = Field(..., description="Processed document content")
    status: ProcessingStatus = ProcessingStatus.SUCCESS

    # Metadata
    metadata: ProcessingMetadata

    # Request tracking
    request_id: Optional[str] = None
    processed_at: str = Field(..., description="ISO timestamp of processing")

    # Error information (if status is FAILED)
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class BatchProcessRequest(BaseModel):
    """Request model for batch processing."""

    documents: List[DocumentProcessRequest] = Field(..., max_items=10)

    # Batch configuration
    parallel_processing: bool = True
    fail_fast: bool = False  # Stop on first error

    # Request metadata
    batch_id: Optional[str] = None


class BatchProcessResult(BaseModel):
    """Result of batch processing."""

    results: List[ProcessingResult]

    # Batch metadata
    batch_id: Optional[str] = None
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_processing_time_seconds: float
    processed_at: str


class HealthStatus(BaseModel):
    """Service health status."""

    status: str = "healthy"
    timestamp: str
    version: str

    # Service details
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float

    # Dependencies
    docling_available: bool
    cache_available: bool

    # Performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_processing_time_seconds: float = 0.0

    # Configuration
    max_workers: int
    cache_enabled: bool
    supported_formats: List[str]


class ServiceConfiguration(BaseModel):
    """Service configuration information."""

    version: str
    supported_formats: List[str]
    default_config: DoclingConfig
    limits: Dict[str, Any]

    # Feature flags
    features: Dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: str
