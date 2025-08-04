"""
Enhanced document processing with Docling service integration.
"""

from pathlib import Path
from typing import Dict, Any, Tuple

from lightrag.utils import logger
from lightrag.docling_client import (
    DoclingClient,
    DoclingServiceUnavailable,
    DoclingServiceTimeout,
    DoclingProcessingError,
)
from lightrag.docling_client.fallback import fallback_processor
from lightrag.docling_client.service_discovery import service_discovery
from ..config import global_args


class DoclingServiceProcessor:
    """Enhanced document processor with Docling service integration."""

    def __init__(self):
        self.docling_client = DoclingClient()
        self.service_mode = self._get_service_mode()

    def _get_service_mode(self) -> str:
        """Get current service mode from configuration."""
        from lightrag.utils import get_env_value

        # Check service mode configuration
        mode = get_env_value("DOCLING_SERVICE_MODE", "auto").lower()

        # Validate mode
        valid_modes = ["auto", "service", "inline", "disabled"]
        if mode not in valid_modes:
            logger.warning(f"Invalid service mode '{mode}', using 'auto'")
            mode = "auto"

        return mode

    def _build_docling_config(self) -> Dict[str, Any]:
        """Build docling configuration from global args."""
        return {
            "export_format": global_args.docling_export_format,
            "enable_ocr": global_args.docling_enable_ocr,
            "enable_table_structure": global_args.docling_enable_table_structure,
            "enable_figures": global_args.docling_enable_figures,
            "process_images": global_args.docling_process_images,
            "layout_model": global_args.docling_layout_model,
            "ocr_model": global_args.docling_ocr_model,
            "table_model": global_args.docling_table_model,
            "include_page_numbers": global_args.docling_include_page_numbers,
            "include_headings": global_args.docling_include_headings,
            "extract_metadata": global_args.docling_extract_metadata,
            "image_dpi": global_args.docling_image_dpi,
            "ocr_confidence": global_args.docling_ocr_confidence,
            "table_confidence": global_args.docling_table_confidence,
            "max_workers": global_args.docling_max_workers,
            "enable_cache": global_args.docling_enable_cache,
            "cache_ttl_hours": global_args.docling_cache_ttl_hours,
        }

    async def should_use_docling_service(self, file_path: Path) -> Tuple[bool, str]:
        """
        Determine if Docling service should be used for processing.

        Returns:
            Tuple of (should_use_service, reason)
        """

        # Check service mode
        if self.service_mode == "disabled":
            return False, "service_disabled"

        if self.service_mode == "inline":
            return False, "inline_mode"

        # Check if file type is supported by docling
        extension = file_path.suffix.lower()
        docling_supported = extension in [".pdf", ".docx", ".pptx", ".xlsx"]

        if not docling_supported:
            return False, "unsupported_format"

        # In service mode, always try to use service
        if self.service_mode == "service":
            return True, "service_mode"

        # Auto mode: check service availability
        if self.service_mode == "auto":
            try:
                service_available = await service_discovery.is_service_available()
                if service_available:
                    return True, "service_available"
                else:
                    return False, "service_unavailable"
            except Exception as e:
                logger.debug(f"Error checking service availability: {e}")
                return False, "service_check_failed"

        return False, "unknown"

    async def process_with_docling_service(self, file_path: Path) -> Dict[str, Any]:
        """Process document using Docling service."""
        try:
            logger.info(f"Processing {file_path.name} with Docling service")

            # Build configuration
            config = self._build_docling_config()

            # Process document
            result = await self.docling_client.process_document(file_path, **config)

            if result.success:
                return {
                    "content": result.content,
                    "metadata": {
                        "processing_time_seconds": result.processing_time_seconds,
                        "page_count": result.page_count,
                        "word_count": result.word_count,
                        "processor": "docling_service",
                        "docling_service_used": True,
                        "cache_hit": result.cache_hit,
                        "service_url": service_discovery.get_service_url(),
                    },
                    "success": True,
                }
            else:
                return {
                    "content": "",
                    "metadata": {
                        "processor": "docling_service",
                        "docling_service_used": True,
                        "processing_error": result.error_message,
                    },
                    "success": False,
                    "error": result.error_message
                    or "Docling service processing failed",
                }

        except DoclingServiceUnavailable as e:
            logger.warning(f"Docling service unavailable for {file_path.name}: {e}")
            return {
                "content": "",
                "metadata": {
                    "processor": "docling_service",
                    "docling_service_used": False,
                    "service_error": "unavailable",
                },
                "success": False,
                "error": f"Docling service unavailable: {e}",
            }

        except DoclingServiceTimeout as e:
            logger.warning(f"Docling service timeout for {file_path.name}: {e}")
            return {
                "content": "",
                "metadata": {
                    "processor": "docling_service",
                    "docling_service_used": False,
                    "service_error": "timeout",
                },
                "success": False,
                "error": f"Docling service timeout: {e}",
            }

        except DoclingProcessingError as e:
            logger.error(f"Docling processing error for {file_path.name}: {e}")
            return {
                "content": "",
                "metadata": {
                    "processor": "docling_service",
                    "docling_service_used": False,
                    "service_error": "processing_failed",
                },
                "success": False,
                "error": f"Docling processing failed: {e}",
            }

        except Exception as e:
            logger.error(
                f"Unexpected error processing {file_path.name} with Docling service: {e}"
            )
            return {
                "content": "",
                "metadata": {
                    "processor": "docling_service",
                    "docling_service_used": False,
                    "service_error": "unexpected_error",
                },
                "success": False,
                "error": f"Unexpected Docling service error: {e}",
            }

    async def process_with_fallback(self, file_path: Path) -> Dict[str, Any]:
        """Process document using fallback processors."""
        logger.info(f"Processing {file_path.name} with fallback processor")

        try:
            result = await fallback_processor.process_document_with_fallback(file_path)
            return result
        except Exception as e:
            logger.error(f"Fallback processing failed for {file_path.name}: {e}")
            return {
                "content": "",
                "metadata": {
                    "processor": "fallback",
                    "docling_service_used": False,
                    "fallback_used": True,
                    "fallback_error": str(e),
                },
                "success": False,
                "error": f"Fallback processing failed: {e}",
            }

    async def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Process document with intelligent routing.

        Tries Docling service first (if appropriate), falls back to basic processors.
        """

        # Determine processing strategy
        use_service, reason = await self.should_use_docling_service(file_path)

        logger.debug(
            f"Processing strategy for {file_path.name}: "
            f"use_service={use_service}, reason={reason}"
        )

        if use_service:
            # Try Docling service first
            service_result = await self.process_with_docling_service(file_path)

            if service_result["success"]:
                logger.info(
                    f"Successfully processed {file_path.name} with Docling service"
                )
                return service_result
            else:
                # Service failed, try fallback if enabled
                logger.warning(
                    f"Docling service failed for {file_path.name}, trying fallback"
                )

                # Check if fallback is enabled
                from lightrag.utils import get_env_value

                fallback_enabled = get_env_value("DOCLING_FALLBACK_ENABLED", True, bool)

                if fallback_enabled:
                    fallback_result = await self.process_with_fallback(file_path)

                    # Add service attempt info to metadata
                    fallback_result["metadata"]["docling_service_attempted"] = True
                    fallback_result["metadata"]["docling_service_error"] = (
                        service_result.get("error")
                    )

                    return fallback_result
                else:
                    logger.error(
                        f"Docling service failed and fallback disabled for {file_path.name}"
                    )
                    return service_result  # Return the service error
        else:
            # Use fallback processor directly
            logger.info(
                f"Using fallback processor for {file_path.name} (reason: {reason})"
            )
            fallback_result = await self.process_with_fallback(file_path)

            # Add routing info to metadata
            fallback_result["metadata"]["docling_service_attempted"] = False
            fallback_result["metadata"]["routing_reason"] = reason

            return fallback_result

    async def close(self):
        """Close resources."""
        await self.docling_client.close()


# Global processor instance
docling_processor = DoclingServiceProcessor()
