"""
REST API client for LightRAG MCP integration.

Provides async HTTP client interface for communicating with
LightRAG API server with connection pooling, error handling,
and retry logic.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, AsyncIterator
from pathlib import Path

import httpx
from httpx import HTTPError, TimeoutException, ConnectError

from ..config import LightRAGMCPConfig
from ..utils import generate_correlation_id, MCPError

logger = logging.getLogger("lightrag-mcp.api_client")


class LightRAGAPIClient:
    """Async HTTP client for LightRAG API with connection pooling and error handling."""

    def __init__(self, config: LightRAGMCPConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._session_id = generate_correlation_id()

    async def __aenter__(self) -> "LightRAGAPIClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None or self._client.is_closed:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"{self.config.mcp_server_name}/{self.config.mcp_server_version}",
                "X-Session-ID": self._session_id,
            }

            if self.config.lightrag_api_key:
                headers["Authorization"] = f"Bearer {self.config.lightrag_api_key}"

            # Configure connection limits and timeouts
            limits = httpx.Limits(
                max_connections=self.config.http_max_connections,
                max_keepalive_connections=self.config.http_max_keepalive,
            )

            timeout = httpx.Timeout(
                connect=10.0, read=self.config.http_timeout, write=10.0, pool=5.0
            )

            self._client = httpx.AsyncClient(
                base_url=self.config.lightrag_api_url,
                headers=headers,
                limits=limits,
                timeout=timeout,
                follow_redirects=True,
            )

            logger.debug(f"Initialized HTTP client for {self.config.lightrag_api_url}")

        return self._client

    async def _handle_response(
        self, response: httpx.Response, operation: str
    ) -> Dict[str, Any]:
        """Handle HTTP response with error checking."""
        correlation_id = generate_correlation_id()

        try:
            response.raise_for_status()

            # Handle empty responses
            if not response.content:
                return {}

            # Parse JSON response
            data = response.json()

            logger.debug(
                f"Operation {operation} successful",
                extra={"correlation_id": correlation_id},
            )

            return data

        except httpx.HTTPStatusError as e:
            error_detail = None
            try:
                error_detail = response.json()
            except (json.JSONDecodeError, ValueError):
                error_detail = {"message": response.text}

            logger.error(
                f"HTTP {response.status_code} error in {operation}: {error_detail}",
                extra={"correlation_id": correlation_id},
            )

            # Map HTTP status codes to MCP errors
            if response.status_code == 401:
                raise MCPError(
                    "UNAUTHORIZED",
                    "Authentication required or invalid",
                    {"status_code": response.status_code, "detail": error_detail},
                )
            elif response.status_code == 403:
                raise MCPError(
                    "FORBIDDEN",
                    "Operation not permitted",
                    {"status_code": response.status_code, "detail": error_detail},
                )
            elif response.status_code == 404:
                raise MCPError(
                    "NOT_FOUND",
                    "Resource not found",
                    {"status_code": response.status_code, "detail": error_detail},
                )
            elif response.status_code == 429:
                raise MCPError(
                    "RATE_LIMITED",
                    "Rate limit exceeded",
                    {"status_code": response.status_code, "detail": error_detail},
                )
            elif response.status_code >= 500:
                raise MCPError(
                    "SERVICE_UNAVAILABLE",
                    "LightRAG service unavailable",
                    {"status_code": response.status_code, "detail": error_detail},
                )
            else:
                raise MCPError(
                    "API_ERROR",
                    f"API request failed: {e}",
                    {"status_code": response.status_code, "detail": error_detail},
                )

        except (ConnectError, TimeoutException) as e:
            logger.error(
                f"Connection error in {operation}: {e}",
                extra={"correlation_id": correlation_id},
            )
            raise MCPError(
                "LIGHTRAG_UNAVAILABLE",
                f"Cannot connect to LightRAG API: {e}",
                {"operation": operation},
            )

        except Exception as e:
            logger.error(
                f"Unexpected error in {operation}: {e}",
                extra={"correlation_id": correlation_id},
            )
            raise MCPError(
                "INTERNAL_ERROR",
                f"Internal error during {operation}: {e}",
                {"operation": operation},
            )

    # Query operations
    async def query(self, query: str, mode: str = "hybrid", **params) -> Dict[str, Any]:
        """Execute a RAG query."""
        client = await self._ensure_client()

        payload = {"query": query, "mode": mode, "param": params}

        logger.info(f"Executing query: {query[:100]}... (mode: {mode})")

        response = await client.post("/query", json=payload)
        return await self._handle_response(response, "query")

    async def stream_query(
        self, query: str, mode: str = "hybrid", **params
    ) -> AsyncIterator[str]:
        """Execute a streaming RAG query."""
        client = await self._ensure_client()

        payload = {"query": query, "mode": mode, "param": params}

        logger.info(f"Executing streaming query: {query[:100]}... (mode: {mode})")

        try:
            async with client.stream("POST", "/query/stream", json=payload) as response:
                await self._handle_response(response, "stream_query")

                async for chunk in response.aiter_text():
                    if chunk.strip():
                        yield chunk

        except HTTPError as e:
            logger.error(f"Streaming query failed: {e}")
            raise MCPError("QUERY_FAILED", f"Streaming query failed: {e}")

    # Document operations
    async def insert_text(
        self, text: str, title: str = "", metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Insert text document."""
        client = await self._ensure_client()

        payload = {"text": text}
        if title:
            payload["title"] = title
        if metadata:
            payload["metadata"] = metadata

        logger.info(f"Inserting text document: {title or 'Untitled'}")

        response = await client.post("/documents/text", json=payload)
        return await self._handle_response(response, "insert_text")

    async def insert_file(self, file_path: str, **options) -> Dict[str, Any]:
        """Insert file document."""
        client = await self._ensure_client()

        # Check if file exists
        path = Path(file_path)
        if not path.exists():
            raise MCPError("FILE_NOT_FOUND", f"File not found: {file_path}")

        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise MCPError(
                "FILE_TOO_LARGE",
                f"File size {file_size_mb:.1f}MB exceeds limit {self.config.max_file_size_mb}MB",
            )

        # Check file type
        file_ext = path.suffix.lower()
        if file_ext not in self.config.allowed_file_types:
            raise MCPError(
                "UNSUPPORTED_FORMAT",
                f"File type {file_ext} not supported. Allowed: {self.config.allowed_file_types}",
            )

        logger.info(f"Inserting file: {file_path}")

        # Upload file
        with open(file_path, "rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            data = options if options else {}

            response = await client.post("/documents/upload", files=files, data=data)
            return await self._handle_response(response, "insert_file")

    async def list_documents(
        self, status_filter: str = "", limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """List documents with filtering."""
        client = await self._ensure_client()

        params = {"limit": limit, "offset": offset}
        if status_filter:
            params["status"] = status_filter

        logger.debug(
            f"Listing documents: limit={limit}, offset={offset}, status={status_filter}"
        )

        response = await client.get("/documents", params=params)
        return await self._handle_response(response, "list_documents")

    async def get_document(self, document_id: str) -> Dict[str, Any]:
        """Get specific document."""
        client = await self._ensure_client()

        logger.debug(f"Getting document: {document_id}")

        response = await client.get(f"/documents/{document_id}")
        return await self._handle_response(response, "get_document")

    async def delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple documents."""
        client = await self._ensure_client()

        payload = {"document_ids": document_ids}

        logger.info(f"Deleting {len(document_ids)} documents")

        response = await client.post("/documents/delete", json=payload)
        return await self._handle_response(response, "delete_documents")

    # Graph operations
    async def get_graph(self, **params) -> Dict[str, Any]:
        """Get knowledge graph data."""
        client = await self._ensure_client()

        logger.debug(f"Getting graph data with params: {params}")

        response = await client.get("/graphs", params=params)
        return await self._handle_response(response, "get_graph")

    async def search_entities(
        self, query: str, limit: int = 20, **params
    ) -> Dict[str, Any]:
        """Search entities."""
        client = await self._ensure_client()

        search_params = {"query": query, "limit": limit, **params}

        logger.debug(f"Searching entities: {query}")

        response = await client.get("/graph/entities/search", params=search_params)
        return await self._handle_response(response, "search_entities")

    async def update_entity(
        self, entity_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update entity properties."""
        client = await self._ensure_client()

        logger.info(f"Updating entity: {entity_id}")

        response = await client.post(f"/graph/entity/{entity_id}/edit", json=updates)
        return await self._handle_response(response, "update_entity")

    async def get_entity_relationships(
        self, entity_id: str, **params
    ) -> Dict[str, Any]:
        """Get entity relationships."""
        client = await self._ensure_client()

        logger.debug(f"Getting relationships for entity: {entity_id}")

        response = await client.get(
            f"/graph/entities/{entity_id}/relationships", params=params
        )
        return await self._handle_response(response, "get_entity_relationships")

    # System operations
    async def health_check(self) -> Dict[str, Any]:
        """Check system health."""
        client = await self._ensure_client()

        logger.debug("Performing health check")

        try:
            response = await client.get("/health")
            return await self._handle_response(response, "health_check")
        except MCPError as e:
            # Return structured error for health checks
            if e.error_code == "LIGHTRAG_UNAVAILABLE":
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "message": "Failed to connect to LightRAG API",
                    "api_url": self.config.lightrag_api_url,
                }
            raise

    async def get_system_stats(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get system statistics."""
        client = await self._ensure_client()

        params = {"time_range": time_range}

        logger.debug(f"Getting system stats for {time_range}")

        response = await client.get("/stats", params=params)
        return await self._handle_response(response, "get_system_stats")

    async def clear_cache(self, cache_types: List[str]) -> Dict[str, Any]:
        """Clear system caches."""
        client = await self._ensure_client()

        payload = {"cache_types": cache_types}

        logger.info(f"Clearing caches: {cache_types}")

        response = await client.post("/cache/clear", json=payload)
        return await self._handle_response(response, "clear_cache")


@asynccontextmanager
async def get_api_client(config: LightRAGMCPConfig) -> AsyncIterator[LightRAGAPIClient]:
    """Context manager for LightRAG API client."""
    async with LightRAGAPIClient(config) as client:
        yield client
