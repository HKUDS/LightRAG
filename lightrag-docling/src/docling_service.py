"""
FastAPI service for LightRAG Docling document processing.
"""

import time
from datetime import datetime, timezone
from typing import List, Dict, Any
import psutil

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from .models import (
    DocumentProcessRequest,
    ProcessingResult,
    BatchProcessRequest,
    BatchProcessResult,
    HealthStatus,
    ServiceConfiguration,
    ErrorResponse,
    ProcessingStatus,
)
from .processors import processor
from config.docling_config import (
    service_settings,
    get_supported_formats,
    get_feature_flags,
    get_service_limits,
    get_default_docling_config,
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global metrics
service_metrics = {
    "start_time": time.time(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_processing_time": 0.0,
}

# Create FastAPI app
app = FastAPI(
    title="LightRAG Docling Service",
    description="Document processing microservice using Docling",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=service_settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request, call_next):
    """Track request metrics."""
    start_time = time.time()
    service_metrics["total_requests"] += 1

    try:
        response = await call_next(request)

        # Track success/failure
        if response.status_code < 400:
            service_metrics["successful_requests"] += 1
        else:
            service_metrics["failed_requests"] += 1

        return response

    except Exception as e:
        service_metrics["failed_requests"] += 1
        logger.error("Request processing failed", error=str(e))
        raise

    finally:
        # Track processing time
        processing_time = time.time() - start_time
        service_metrics["total_processing_time"] += processing_time


# Dependency for API key authentication (if enabled)
async def verify_api_key(request) -> bool:
    """Verify API key if authentication is enabled."""
    if not service_settings.api_key:
        return True  # No authentication required

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
        )

    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
        )

    token = auth_header[7:]  # Remove "Bearer " prefix
    if token != service_settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    return True


@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """Service health check endpoint."""
    try:
        # Get system metrics
        memory_usage = psutil.virtual_memory().used / 1024 / 1024  # MB
        cpu_usage = psutil.cpu_percent(interval=1)
        uptime = time.time() - service_metrics["start_time"]

        # Calculate average processing time
        avg_processing_time = 0.0
        if service_metrics["total_requests"] > 0:
            avg_processing_time = (
                service_metrics["total_processing_time"]
                / service_metrics["total_requests"]
            )

        # Check dependencies
        docling_available = processor._check_docling_availability()
        cache_available = processor.cache is not None

        # Determine overall health status
        health_status = "healthy"
        if not docling_available:
            health_status = "degraded"

        return HealthStatus(
            status=health_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="1.0.0",
            uptime_seconds=uptime,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            docling_available=docling_available,
            cache_available=cache_available,
            total_requests=service_metrics["total_requests"],
            successful_requests=service_metrics["successful_requests"],
            failed_requests=service_metrics["failed_requests"],
            average_processing_time_seconds=avg_processing_time,
            max_workers=service_settings.default_max_workers,
            cache_enabled=service_settings.cache_enabled,
            supported_formats=[
                ".pdf",
                ".docx",
                ".pptx",
                ".xlsx",
                ".txt",
                ".md",
                ".html",
            ],
        )

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}",
        )


@app.get("/config", response_model=ServiceConfiguration, tags=["Configuration"])
async def get_configuration():
    """Get service configuration information."""
    return ServiceConfiguration(
        version="1.0.0",
        supported_formats=[".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".html"],
        default_config=get_default_docling_config(),
        limits=get_service_limits(),
        features=get_feature_flags(),
    )


@app.post("/process", response_model=ProcessingResult, tags=["Processing"])
async def process_document(request: DocumentProcessRequest):
    """Process a single document."""
    try:
        logger.info(
            "Processing document",
            filename=request.filename,
            request_id=request.request_id,
        )

        # Validate file size
        file_size_mb = (
            len(request.file_content) * 3 / 4 / 1024 / 1024
        )  # Rough base64 to bytes
        if file_size_mb > service_settings.max_file_size_mb:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {service_settings.max_file_size_mb}MB",
            )

        # Process document
        result = await processor.process_document(request)

        if result.status == ProcessingStatus.FAILED:
            logger.warning(
                "Document processing failed",
                filename=request.filename,
                error=result.error_message,
            )
        else:
            logger.info(
                "Document processed successfully",
                filename=request.filename,
                processing_time=result.metadata.processing_time_seconds,
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Unexpected error processing document",
            filename=request.filename,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}",
        )


@app.post("/process/batch", response_model=BatchProcessResult, tags=["Processing"])
async def process_batch(request: BatchProcessRequest):
    """Process multiple documents in batch."""
    try:
        # Validate batch size
        if len(request.documents) > service_settings.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch too large. Maximum size: {service_settings.max_batch_size}",
            )

        logger.info(
            "Processing batch",
            batch_id=request.batch_id,
            document_count=len(request.documents),
        )

        start_time = time.time()

        # Process documents
        results = await processor.process_batch(
            request.documents,
            parallel=request.parallel_processing,
            fail_fast=request.fail_fast,
        )

        total_processing_time = time.time() - start_time

        # Calculate statistics
        successful_count = sum(
            1 for r in results if r.status == ProcessingStatus.SUCCESS
        )
        failed_count = len(results) - successful_count

        batch_result = BatchProcessResult(
            results=results,
            batch_id=request.batch_id,
            total_documents=len(request.documents),
            successful_documents=successful_count,
            failed_documents=failed_count,
            total_processing_time_seconds=total_processing_time,
            processed_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "Batch processing completed",
            batch_id=request.batch_id,
            successful=successful_count,
            failed=failed_count,
            total_time=total_processing_time,
        )

        return batch_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Unexpected error processing batch", batch_id=request.batch_id, error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}",
        )


@app.get("/formats", response_model=List[str], tags=["Configuration"])
async def get_formats():
    """Get list of supported file formats."""
    return get_supported_formats()


@app.get("/cache/stats", response_model=Dict[str, Any], tags=["Cache"])
async def get_cache_stats():
    """Get cache statistics."""
    return processor.get_cache_stats()


@app.delete("/cache", response_model=Dict[str, bool], tags=["Cache"])
async def clear_cache(authenticated: bool = Depends(verify_api_key)):
    """Clear all cached results."""
    try:
        success = processor.clear_cache()
        return {"cache_cleared": success}
    except Exception as e:
        logger.error("Failed to clear cache", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}",
        )


@app.get("/metrics", response_model=Dict[str, Any], tags=["Monitoring"])
async def get_metrics():
    """Get service metrics."""
    uptime = time.time() - service_metrics["start_time"]

    # Calculate rates
    request_rate = service_metrics["total_requests"] / max(uptime, 1)
    error_rate = service_metrics["failed_requests"] / max(
        service_metrics["total_requests"], 1
    )

    return {
        "uptime_seconds": uptime,
        "total_requests": service_metrics["total_requests"],
        "successful_requests": service_metrics["successful_requests"],
        "failed_requests": service_metrics["failed_requests"],
        "request_rate_per_second": request_rate,
        "error_rate": error_rate,
        "total_processing_time_seconds": service_metrics["total_processing_time"],
        "average_processing_time_seconds": (
            service_metrics["total_processing_time"]
            / max(service_metrics["total_requests"], 1)
        ),
        "cache_stats": processor.get_cache_stats(),
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).dict(),
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Service startup initialization."""
    logger.info(
        "Starting LightRAG Docling Service",
        version="1.0.0",
        port=service_settings.port,
        cache_enabled=service_settings.cache_enabled,
    )

    # Initialize processor and check dependencies
    docling_available = processor._check_docling_availability()
    if not docling_available:
        logger.warning(
            "Docling is not available - service will operate in degraded mode"
        )
    else:
        logger.info("Docling is available and ready")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Service shutdown cleanup."""
    logger.info("Shutting down LightRAG Docling Service")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "docling_service:app",
        host=service_settings.host,
        port=service_settings.port,
        reload=service_settings.reload,
        log_level=service_settings.log_level.lower(),
        access_log=service_settings.access_log,
    )
