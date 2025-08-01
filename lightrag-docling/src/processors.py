"""
Core document processing logic for the Docling service.
"""

import asyncio
import base64
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List
import tempfile
import os

import structlog
from diskcache import Cache

from .models import (
    DoclingConfig, 
    DocumentProcessRequest, 
    ProcessingResult, 
    ProcessingStatus,
    ProcessingMetadata,
    ExportFormat
)
from ..config.docling_config import service_settings

logger = structlog.get_logger(__name__)


class DoclingProcessor:
    """Core document processor using Docling."""
    
    def __init__(self):
        self.cache: Optional[Cache] = None
        self._initialize_cache()
        self._docling_available = None
        self._models_loaded = False
        
    def _initialize_cache(self) -> None:
        """Initialize disk cache if enabled."""
        if service_settings.cache_enabled:
            try:
                cache_dir = Path(service_settings.cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Convert GB to bytes for cache size
                cache_size_bytes = service_settings.cache_max_size_gb * 1024 * 1024 * 1024
                
                self.cache = Cache(
                    directory=str(cache_dir),
                    size_limit=cache_size_bytes,
                    eviction_policy='least-recently-used'
                )
                logger.info("Cache initialized", cache_dir=str(cache_dir))
            except Exception as e:
                logger.error("Failed to initialize cache", error=str(e))
                self.cache = None
    
    def _check_docling_availability(self) -> bool:
        """Check if docling is available and working."""
        if self._docling_available is not None:
            return self._docling_available
            
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import ConversionStatus
            from docling.datamodel.document import ConversionResult
            
            # Try to create a converter to verify everything works
            converter = DocumentConverter()
            self._docling_available = True
            logger.info("Docling is available and ready")
            
        except ImportError as e:
            logger.error("Docling not available - import error", error=str(e))
            self._docling_available = False
        except Exception as e:
            logger.error("Docling not available - initialization error", error=str(e))
            self._docling_available = False
            
        return self._docling_available
    
    def _generate_cache_key(self, request: DocumentProcessRequest) -> str:
        """Generate cache key for request."""
        # Include file content hash, filename, and config in cache key
        content_hash = hashlib.md5(request.file_content.encode()).hexdigest()
        config_hash = hashlib.md5(
            json.dumps(request.config.dict(), sort_keys=True).encode()
        ).hexdigest()
        
        cache_key = f"{content_hash}_{request.filename}_{config_hash}"
        return cache_key
    
    async def _get_cached_result(self, cache_key: str) -> Optional[ProcessingResult]:
        """Get cached processing result if available and valid."""
        if not self.cache:
            return None
            
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data is None:
                return None
                
            # Check if cache entry is still valid
            cache_timestamp = cached_data.get('timestamp', 0)
            cache_age_hours = (time.time() - cache_timestamp) / 3600
            
            if cache_age_hours > service_settings.cache_ttl_hours:
                # Cache expired, remove it
                self.cache.delete(cache_key)
                return None
            
            # Reconstruct ProcessingResult from cached data
            result_data = cached_data.get('result')
            if result_data:
                result = ProcessingResult(**result_data)
                # Update metadata to indicate cache hit
                result.metadata.cache_hit = True
                result.metadata.cache_key = cache_key
                return result
                
        except Exception as e:
            logger.warning("Error retrieving cached result", error=str(e), cache_key=cache_key)
            
        return None
    
    async def _cache_result(self, cache_key: str, result: ProcessingResult) -> None:
        """Cache processing result."""
        if not self.cache:
            return
            
        try:
            cache_data = {
                'timestamp': time.time(),
                'result': result.dict(),
            }
            self.cache.set(cache_key, cache_data)
            logger.debug("Result cached", cache_key=cache_key)
            
        except Exception as e:
            logger.warning("Error caching result", error=str(e), cache_key=cache_key)
    
    async def _process_with_docling(
        self, 
        file_path: Path, 
        config: DoclingConfig
    ) -> tuple[str, ProcessingMetadata]:
        """Process document using docling."""
        start_time = time.time()
        
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import ConversionStatus
            
            # Initialize converter with configuration
            converter = DocumentConverter()
            
            logger.info("Processing document with docling", 
                       filename=file_path.name, 
                       config=config.dict())
            
            # Convert document
            result = converter.convert(file_path)
            
            if result.status != ConversionStatus.SUCCESS:
                raise Exception(f"Docling conversion failed with status: {result.status}")
            
            # Export in requested format
            content = ""
            if config.export_format == ExportFormat.MARKDOWN:
                content = result.document.export_to_markdown()
            elif config.export_format == ExportFormat.JSON:
                content = result.document.export_to_json()
            elif config.export_format == ExportFormat.TEXT:
                content = result.document.export_to_text()
            elif config.export_format == ExportFormat.HTML:
                content = result.document.export_to_html()
            else:
                content = result.document.export_to_markdown()  # Default fallback
            
            # Extract metadata
            processing_time = time.time() - start_time
            page_count = len(result.pages) if result.pages else None
            
            # Count words and characters
            word_count = len(content.split()) if content else 0
            character_count = len(content) if content else 0
            
            # Build metadata
            metadata = ProcessingMetadata(
                processing_time_seconds=processing_time,
                page_count=page_count,
                word_count=word_count,
                character_count=character_count,
                models_used={
                    "layout": config.layout_model,
                    "ocr": config.ocr_model if config.enable_ocr else "disabled",
                    "table": config.table_model if config.enable_table_structure else "disabled",
                },
                ocr_applied=config.enable_ocr,
                tables_extracted=0,  # TODO: Count tables if available in result
                figures_extracted=0,  # TODO: Count figures if available in result
                cache_hit=False,
                config_hash=hashlib.md5(
                    json.dumps(config.dict(), sort_keys=True).encode()
                ).hexdigest()[:8]
            )
            
            # Add metadata section to content if requested
            if config.extract_metadata:
                metadata_section = self._generate_metadata_section(metadata, config)
                content = metadata_section + "\n\n" + content
            
            logger.info("Document processed successfully", 
                       filename=file_path.name,
                       processing_time=processing_time,
                       content_length=len(content))
            
            return content, metadata
            
        except Exception as e:
            logger.error("Docling processing failed", 
                        filename=file_path.name, 
                        error=str(e))
            raise
    
    def _generate_metadata_section(
        self, 
        metadata: ProcessingMetadata, 
        config: DoclingConfig
    ) -> str:
        """Generate metadata section for document."""
        lines = ["# Document Processing Metadata", ""]
        
        lines.append(f"- **Processed At**: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"- **Processing Time**: {metadata.processing_time_seconds:.2f} seconds")
        
        if metadata.page_count:
            lines.append(f"- **Page Count**: {metadata.page_count}")
        if metadata.word_count:
            lines.append(f"- **Word Count**: {metadata.word_count:,}")
        if metadata.character_count:
            lines.append(f"- **Character Count**: {metadata.character_count:,}")
            
        lines.append(f"- **Export Format**: {config.export_format.value}")
        
        if config.enable_ocr:
            lines.append(f"- **OCR Enabled**: Yes (confidence: {config.ocr_confidence})")
        if config.enable_table_structure:
            lines.append(f"- **Table Recognition**: Yes (confidence: {config.table_confidence})")
        if config.enable_figures:
            lines.append("- **Figure Extraction**: Yes")
            
        if metadata.models_used:
            lines.append("- **Models Used**:")
            for model_type, model_name in metadata.models_used.items():
                lines.append(f"  - {model_type.title()}: {model_name}")
        
        return "\n".join(lines)
    
    async def process_document(self, request: DocumentProcessRequest) -> ProcessingResult:
        """Process a single document."""
        start_time = time.time()
        request_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Check docling availability
        if not self._check_docling_availability():
            return ProcessingResult(
                content="",
                status=ProcessingStatus.FAILED,
                metadata=ProcessingMetadata(
                    processing_time_seconds=time.time() - start_time,
                    cache_hit=False
                ),
                request_id=request.request_id,
                processed_at=request_timestamp,
                error_message="Docling is not available",
                error_details={"reason": "docling_unavailable"}
            )
        
        # Generate cache key
        cache_key = request.cache_key or self._generate_cache_key(request)
        
        # Check cache first
        if request.config.enable_cache:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                cached_result.request_id = request.request_id
                cached_result.processed_at = request_timestamp
                logger.info("Returning cached result", cache_key=cache_key)
                return cached_result
        
        try:
            # Decode file content
            try:
                file_data = base64.b64decode(request.file_content)
            except Exception as e:
                raise ValueError(f"Invalid base64 file content: {e}")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix=Path(request.filename).suffix,
                delete=False
            ) as temp_file:
                temp_file.write(file_data)
                temp_file_path = Path(temp_file.name)
            
            try:
                # Process document
                content, metadata = await self._process_with_docling(
                    temp_file_path, 
                    request.config
                )
                
                # Create successful result
                result = ProcessingResult(
                    content=content,
                    status=ProcessingStatus.SUCCESS,
                    metadata=metadata,
                    request_id=request.request_id,
                    processed_at=request_timestamp
                )
                
                # Cache result if enabled
                if request.config.enable_cache:
                    await self._cache_result(cache_key, result)
                
                return result
                
            finally:
                # Clean up temporary file
                if temp_file_path.exists():
                    temp_file_path.unlink()
                    
        except Exception as e:
            logger.error("Document processing failed", 
                        filename=request.filename,
                        error=str(e))
            
            return ProcessingResult(
                content="",
                status=ProcessingStatus.FAILED,
                metadata=ProcessingMetadata(
                    processing_time_seconds=time.time() - start_time,
                    cache_hit=False
                ),
                request_id=request.request_id,
                processed_at=request_timestamp,
                error_message=str(e),
                error_details={"error_type": type(e).__name__}
            )
    
    async def process_batch(
        self, 
        requests: List[DocumentProcessRequest],
        parallel: bool = True,
        fail_fast: bool = False
    ) -> List[ProcessingResult]:
        """Process multiple documents."""
        if parallel and len(requests) > 1:
            # Process documents in parallel
            tasks = [self.process_document(req) for req in requests]
            
            if fail_fast:
                # Stop on first error
                results = []
                for task in asyncio.as_completed(tasks):
                    result = await task
                    results.append(result)
                    if result.status == ProcessingStatus.FAILED:
                        # Cancel remaining tasks
                        for remaining_task in tasks:
                            if not remaining_task.done():
                                remaining_task.cancel()
                        break
                return results
            else:
                # Wait for all to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Convert exceptions to failed results
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        processed_results.append(ProcessingResult(
                            content="",
                            status=ProcessingStatus.FAILED,
                            metadata=ProcessingMetadata(
                                processing_time_seconds=0,
                                cache_hit=False
                            ),
                            request_id=requests[i].request_id,
                            processed_at=datetime.now(timezone.utc).isoformat(),
                            error_message=str(result),
                            error_details={"error_type": type(result).__name__}
                        ))
                    else:
                        processed_results.append(result)
                
                return processed_results
        else:
            # Process sequentially
            results = []
            for request in requests:
                result = await self.process_document(request)
                results.append(result)
                
                if fail_fast and result.status == ProcessingStatus.FAILED:
                    break
                    
            return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {"cache_enabled": False}
            
        try:
            stats = self.cache.stats(enable=True)
            return {
                "cache_enabled": True,
                "cache_size": self.cache.volume(),
                "cache_hits": stats.get('cache_hits', 0),
                "cache_misses": stats.get('cache_misses', 0),
                "cache_hit_ratio": stats.get('cache_hits', 0) / max(1, stats.get('cache_hits', 0) + stats.get('cache_misses', 0)),
            }
        except Exception:
            return {"cache_enabled": True, "cache_stats_error": True}
    
    def clear_cache(self) -> bool:
        """Clear all cached results."""
        if not self.cache:
            return False
            
        try:
            self.cache.clear()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.error("Failed to clear cache", error=str(e))
            return False


# Global processor instance
processor = DoclingProcessor()