"""
HTTP client for Docling service integration.
"""

import asyncio
import base64
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import httpx
from lightrag.utils import get_env_value

from .exceptions import (
    DoclingServiceError,
    DoclingServiceUnavailable,
    DoclingServiceTimeout,
    DoclingProcessingError,
    DoclingConfigurationError,
)
from .service_discovery import service_discovery

logger = logging.getLogger(__name__)


class DoclingProcessingResult:
    """Result of document processing from Docling service."""

    def __init__(self, data: Dict[str, Any]):
        self.content: str = data.get("content", "")
        self.status: str = data.get("status", "failed")
        self.metadata: Dict[str, Any] = data.get("metadata", {})
        self.request_id: Optional[str] = data.get("request_id")
        self.processed_at: str = data.get("processed_at", "")
        self.error_message: Optional[str] = data.get("error_message")
        self.error_details: Optional[Dict[str, Any]] = data.get("error_details")

        # Extract common metadata fields
        self.processing_time_seconds: float = self.metadata.get(
            "processing_time_seconds", 0.0
        )
        self.page_count: Optional[int] = self.metadata.get("page_count")
        self.word_count: Optional[int] = self.metadata.get("word_count")
        self.cache_hit: bool = self.metadata.get("cache_hit", False)

    @property
    def success(self) -> bool:
        """Check if processing was successful."""
        return self.status == "success"

    @property
    def failed(self) -> bool:
        """Check if processing failed."""
        return self.status == "failed"


class DoclingClient:
    """HTTP client for Docling service."""

    def __init__(self):
        self.timeout = int(get_env_value("DOCLING_SERVICE_TIMEOUT", 300))
        self.retries = int(get_env_value("DOCLING_SERVICE_RETRIES", 3))
        self.retry_delay = float(get_env_value("DOCLING_SERVICE_RETRY_DELAY", 1.0))
        self.api_key = get_env_value("DOCLING_SERVICE_API_KEY", None)

        # Client session
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    timeout=self.timeout, connect=10.0, read=self.timeout
                ),
                headers=headers,
            )

        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def is_service_available(self) -> bool:
        """Check if Docling service is available."""
        return await service_discovery.is_service_available()

    async def get_service_health(self) -> Dict[str, Any]:
        """Get detailed service health information."""
        service_url = service_discovery.get_service_url()
        if not service_url:
            raise DoclingServiceUnavailable("Docling service URL not configured")

        try:
            client = await self._get_client()
            health_url = urljoin(service_url, "/health")
            response = await client.get(health_url)

            if response.status_code == 200:
                return response.json()
            else:
                raise DoclingServiceError(
                    f"Health check failed with status {response.status_code}",
                    {"status_code": response.status_code, "response": response.text},
                )

        except httpx.TimeoutException:
            raise DoclingServiceTimeout("Service health check timed out")
        except httpx.ConnectError:
            raise DoclingServiceUnavailable("Cannot connect to Docling service")

    async def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        service_url = service_discovery.get_service_url()
        if not service_url or not await self.is_service_available():
            raise DoclingServiceUnavailable("Docling service not available")

        try:
            client = await self._get_client()
            formats_url = urljoin(service_url, "/formats")
            response = await client.get(formats_url)

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning("Failed to get supported formats, using defaults")
                return [".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"]

        except Exception as e:
            logger.warning(f"Error getting supported formats: {e}")
            return [".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"]

    def _build_docling_config(self, **config_kwargs) -> Dict[str, Any]:
        """Build docling configuration from keyword arguments."""
        # Map LightRAG config to Docling service config
        docling_config = {}

        # Direct mappings
        direct_mappings = {
            "export_format": "export_format",
            "enable_ocr": "enable_ocr",
            "enable_table_structure": "enable_table_structure",
            "enable_figures": "enable_figures",
            "process_images": "process_images",
            "layout_model": "layout_model",
            "ocr_model": "ocr_model",
            "table_model": "table_model",
            "include_page_numbers": "include_page_numbers",
            "include_headings": "include_headings",
            "extract_metadata": "extract_metadata",
            "image_dpi": "image_dpi",
            "ocr_confidence": "ocr_confidence",
            "table_confidence": "table_confidence",
            "max_workers": "max_workers",
            "enable_cache": "enable_cache",
            "cache_ttl_hours": "cache_ttl_hours",
        }

        # Map provided config
        for lightrag_key, docling_key in direct_mappings.items():
            if lightrag_key in config_kwargs:
                docling_config[docling_key] = config_kwargs[lightrag_key]

        # Handle special cases
        if "docling_export_format" in config_kwargs:
            docling_config["export_format"] = config_kwargs["docling_export_format"]

        # Set defaults for common cases
        if not docling_config:
            docling_config = {
                "export_format": "markdown",
                "enable_ocr": True,
                "enable_table_structure": True,
                "enable_figures": True,
                "enable_cache": True,
            }

        return docling_config

    async def _make_request_with_retry(
        self, method: str, url: str, **kwargs
    ) -> httpx.Response:
        """Make HTTP request with retry logic."""
        client = await self._get_client()
        last_exception = None

        for attempt in range(self.retries + 1):
            try:
                if method.upper() == "GET":
                    response = await client.get(url, **kwargs)
                elif method.upper() == "POST":
                    response = await client.post(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Check for success or client error (don't retry client errors)
                if response.status_code < 500:
                    return response

                # Server error - retry
                last_exception = DoclingServiceError(
                    f"Server error: {response.status_code}",
                    {"status_code": response.status_code, "response": response.text},
                )

            except httpx.TimeoutException as e:
                last_exception = DoclingServiceTimeout(f"Request timed out: {e}")
            except httpx.ConnectError as e:
                last_exception = DoclingServiceUnavailable(f"Connection failed: {e}")
            except Exception as e:
                last_exception = DoclingServiceError(f"Request failed: {e}")

            # Don't retry on the last attempt
            if attempt < self.retries:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.retries + 1}), retrying...",
                    error=str(last_exception),
                )
                await asyncio.sleep(
                    self.retry_delay * (2**attempt)
                )  # Exponential backoff

        # All retries failed
        if last_exception:
            raise last_exception
        else:
            raise DoclingServiceError("All retry attempts failed")

    async def process_document(
        self, file_path: Union[str, Path], **config_kwargs
    ) -> DoclingProcessingResult:
        """Process a single document using Docling service."""
        file_path = Path(file_path)

        # Check service availability
        if not await self.is_service_available():
            raise DoclingServiceUnavailable("Docling service is not available")

        service_url = service_discovery.get_service_url()
        if not service_url:
            raise DoclingConfigurationError("Docling service URL not configured")

        # Read and encode file
        try:
            with open(file_path, "rb") as f:
                file_data = f.read()
            file_content_b64 = base64.b64encode(file_data).decode()
        except Exception as e:
            raise DoclingProcessingError(f"Failed to read file {file_path}: {e}")

        # Build request
        request_data = {
            "file_content": file_content_b64,
            "filename": file_path.name,
            "config": self._build_docling_config(**config_kwargs),
            "request_id": f"lightrag_{int(time.time())}_{file_path.name}",
        }

        # Make request
        process_url = urljoin(service_url, "/process")

        try:
            logger.info(
                f"Processing document {file_path.name} via Docling service at {service_url}"
            )

            response = await self._make_request_with_retry(
                "POST", process_url, json=request_data
            )

            if response.status_code == 200:
                result_data = response.json()
                result = DoclingProcessingResult(result_data)

                logger.info(
                    f"Document {file_path.name} processed successfully "
                    f"(time: {result.processing_time_seconds:.2f}s, "
                    f"cache_hit: {result.cache_hit}, success: {result.success})"
                )

                return result

            elif response.status_code == 413:
                raise DoclingProcessingError(
                    f"File too large: {file_path.name}",
                    {"status_code": response.status_code},
                )
            elif response.status_code == 422:
                error_data = (
                    response.json()
                    if response.headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    else {}
                )
                raise DoclingProcessingError(
                    f"Invalid request for {file_path.name}: {error_data.get('detail', 'Validation error')}",
                    {"status_code": response.status_code, "details": error_data},
                )
            else:
                raise DoclingServiceError(
                    f"Processing failed with status {response.status_code}",
                    {"status_code": response.status_code, "response": response.text},
                )

        except DoclingServiceError:
            raise
        except Exception as e:
            raise DoclingProcessingError(
                f"Unexpected error processing {file_path.name}: {e}"
            )

    async def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        parallel_processing: bool = True,
        fail_fast: bool = False,
        **config_kwargs,
    ) -> List[DoclingProcessingResult]:
        """Process multiple documents in batch."""
        if not await self.is_service_available():
            raise DoclingServiceUnavailable("Docling service is not available")

        service_url = service_discovery.get_service_url()
        if not service_url:
            raise DoclingConfigurationError("Docling service URL not configured")

        # Prepare batch request
        documents = []
        config = self._build_docling_config(**config_kwargs)

        for i, file_path in enumerate(file_paths):
            file_path = Path(file_path)
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()
                file_content_b64 = base64.b64encode(file_data).decode()

                documents.append(
                    {
                        "file_content": file_content_b64,
                        "filename": file_path.name,
                        "config": config,
                        "request_id": f"lightrag_batch_{int(time.time())}_{i}_{file_path.name}",
                    }
                )
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                # Add a failed result
                documents.append(
                    {
                        "file_content": "",
                        "filename": file_path.name,
                        "config": config,
                        "request_id": f"lightrag_batch_{int(time.time())}_{i}_{file_path.name}_failed",
                    }
                )

        batch_request = {
            "documents": documents,
            "parallel_processing": parallel_processing,
            "fail_fast": fail_fast,
            "batch_id": f"lightrag_batch_{int(time.time())}",
        }

        # Make batch request
        batch_url = urljoin(service_url, "/process/batch")

        try:
            logger.info(
                "Processing document batch via Docling service",
                document_count=len(file_paths),
                parallel=parallel_processing,
            )

            response = await self._make_request_with_retry(
                "POST", batch_url, json=batch_request
            )

            if response.status_code == 200:
                batch_data = response.json()
                results = [
                    DoclingProcessingResult(result) for result in batch_data["results"]
                ]

                successful = sum(1 for r in results if r.success)
                failed = len(results) - successful

                logger.info(
                    f"Batch processing completed: {successful}/{len(results)} successful, {failed} failed "
                    f"(total_time: {batch_data.get('total_processing_time_seconds', 0):.2f}s)"
                )

                return results
            else:
                raise DoclingServiceError(
                    f"Batch processing failed with status {response.status_code}",
                    {"status_code": response.status_code, "response": response.text},
                )

        except DoclingServiceError:
            raise
        except Exception as e:
            raise DoclingProcessingError(f"Unexpected error in batch processing: {e}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from service."""
        if not await self.is_service_available():
            return {"cache_enabled": False, "service_available": False}

        service_url = service_discovery.get_service_url()
        try:
            client = await self._get_client()
            stats_url = urljoin(service_url, "/cache/stats")
            response = await client.get(stats_url)

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "cache_enabled": False,
                    "error": f"Status {response.status_code}",
                }

        except Exception as e:
            return {"cache_enabled": False, "error": str(e)}

    async def clear_cache(self) -> bool:
        """Clear service cache."""
        if not await self.is_service_available():
            return False

        service_url = service_discovery.get_service_url()
        try:
            client = await self._get_client()
            clear_url = urljoin(service_url, "/cache")
            response = await client.delete(clear_url)

            if response.status_code == 200:
                result = response.json()
                return result.get("cache_cleared", False)
            else:
                logger.warning(
                    f"Cache clear failed with status code: {response.status_code}"
                )
                return False

        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return False


# Global client instance
docling_client = DoclingClient()
