"""
Direct library client for LightRAG MCP integration.

Provides direct access to LightRAG library functions without going
through the REST API. Used when MCP server runs in the same environment
as LightRAG core library.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, AsyncIterator
from pathlib import Path

from ..config import LightRAGMCPConfig
from ..utils import MCPError

logger = logging.getLogger("lightrag-mcp.direct_client")


class DirectClientError(Exception):
    """Exception raised by direct client operations."""

    pass


class LightRAGDirectClient:
    """Direct library interface for LightRAG."""

    def __init__(self, config: LightRAGMCPConfig):
        self.config = config
        self._lightrag = None
        self._initialized = False

    async def __aenter__(self) -> "LightRAGDirectClient":
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._lightrag:
            try:
                if hasattr(self._lightrag, "finalize_storages"):
                    await self._lightrag.finalize_storages()
            except Exception as e:
                logger.warning(f"Error finalizing LightRAG storages: {e}")
            self._lightrag = None
            self._initialized = False

    async def _ensure_initialized(self):
        """Ensure LightRAG instance is initialized."""
        if self._initialized and self._lightrag:
            return

        try:
            # Import LightRAG here to avoid dependency issues
            from lightrag import LightRAG

            # Determine working directory
            working_dir = self.config.lightrag_working_dir or "./rag_storage"

            logger.info(f"Initializing LightRAG with working directory: {working_dir}")

            # Create LightRAG instance
            self._lightrag = LightRAG(
                working_dir=working_dir,
                # Add other configuration parameters as needed
            )

            # Initialize storages (pipeline status initialization might not be needed)
            if hasattr(self._lightrag, "initialize_storages"):
                await self._lightrag.initialize_storages()

            self._initialized = True
            logger.info("LightRAG direct client initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import LightRAG: {e}")
            raise DirectClientError(f"LightRAG library not available: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {e}")
            raise DirectClientError(f"LightRAG initialization failed: {e}")

    async def _run_in_executor(self, func, *args, **kwargs):
        """Run synchronous function in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    # Query operations
    async def query(self, query: str, mode: str = "hybrid", **params) -> Dict[str, Any]:
        """Execute a RAG query using direct library access."""
        await self._ensure_initialized()

        try:
            logger.info(f"Executing direct query: {query[:100]}... (mode: {mode})")

            # Execute query using LightRAG
            result = await self._lightrag.aquery(query, param=params, mode=mode)

            # Format response to match API format
            return {
                "response": result,
                "mode": mode,
                "metadata": {
                    "processing_time": 0.0,  # Would need to measure this
                    "entities_used": 0,  # Would need to extract from result
                    "relations_used": 0,  # Would need to extract from result
                    "chunks_used": 0,  # Would need to extract from result
                    "token_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
                "sources": [],  # Would need to extract source attribution
            }

        except Exception as e:
            logger.error(f"Direct query failed: {e}")
            raise MCPError("QUERY_FAILED", f"Query execution failed: {e}")

    async def stream_query(
        self, query: str, mode: str = "hybrid", **params
    ) -> AsyncIterator[str]:
        """Execute a streaming RAG query (not implemented for direct access)."""
        # Direct streaming is more complex to implement
        # For now, fall back to regular query and yield the result
        result = await self.query(query, mode, **params)
        yield result["response"]

    # Document operations
    async def insert_text(
        self, text: str, title: str = "", metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Insert text document using direct library access."""
        await self._ensure_initialized()

        try:
            logger.info(f"Inserting text document: {title or 'Untitled'}")

            # Insert text using LightRAG
            await self._lightrag.ainsert(text)

            # Return formatted response
            return {
                "document_id": f"direct_{hash(text)[:8]}",  # Simple ID generation
                "status": "processed",
                "message": "Document inserted successfully",
                "processing_info": {
                    "chunks_created": 0,  # Would need to track this
                    "entities_extracted": 0,  # Would need to track this
                    "relationships_created": 0,  # Would need to track this
                },
            }

        except Exception as e:
            logger.error(f"Direct text insertion failed: {e}")
            raise MCPError("PROCESSING_FAILED", f"Text insertion failed: {e}")

    async def insert_file(self, file_path: str, **options) -> Dict[str, Any]:
        """Insert file document using direct library access."""
        await self._ensure_initialized()

        # Validate file
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

        try:
            logger.info(f"Inserting file: {file_path}")

            # Read file content
            content = path.read_text(encoding="utf-8")

            # Insert using text insertion
            return await self.insert_text(content, title=path.name)

        except UnicodeDecodeError:
            logger.error(f"Failed to decode file: {file_path}")
            raise MCPError(
                "UNSUPPORTED_FORMAT", f"Cannot decode file as text: {file_path}"
            )

        except Exception as e:
            logger.error(f"Direct file insertion failed: {e}")
            raise MCPError("PROCESSING_FAILED", f"File insertion failed: {e}")

    async def list_documents(
        self, status_filter: str = "", limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """List documents (limited functionality in direct mode)."""
        # Direct mode doesn't have built-in document tracking
        # Would need to implement custom tracking or return empty results
        logger.warning("Document listing not fully supported in direct mode")

        return {
            "documents": [],
            "total": 0,
            "has_more": False,
            "message": "Document listing not available in direct mode",
        }

    async def get_document(self, document_id: str) -> Dict[str, Any]:
        """Get specific document (not implemented in direct mode)."""
        raise MCPError(
            "NOT_IMPLEMENTED", "Document retrieval not available in direct mode"
        )

    async def delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """Delete documents (not implemented in direct mode)."""
        raise MCPError(
            "NOT_IMPLEMENTED", "Document deletion not available in direct mode"
        )

    # Graph operations
    async def get_graph(self, **params) -> Dict[str, Any]:
        """Get knowledge graph data (limited in direct mode)."""
        logger.warning("Graph extraction not fully implemented in direct mode")

        return {
            "nodes": [],
            "edges": [],
            "statistics": {
                "total_nodes": 0,
                "total_edges": 0,
                "node_types": {},
                "edge_types": {},
            },
            "message": "Graph extraction not fully available in direct mode",
        }

    async def search_entities(
        self, query: str, limit: int = 20, **params
    ) -> Dict[str, Any]:
        """Search entities (not implemented in direct mode)."""
        raise MCPError("NOT_IMPLEMENTED", "Entity search not available in direct mode")

    async def update_entity(
        self, entity_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update entity (not implemented in direct mode)."""
        raise MCPError("NOT_IMPLEMENTED", "Entity updates not available in direct mode")

    async def get_entity_relationships(
        self, entity_id: str, **params
    ) -> Dict[str, Any]:
        """Get entity relationships (not implemented in direct mode)."""
        raise MCPError(
            "NOT_IMPLEMENTED", "Entity relationships not available in direct mode"
        )

    # System operations
    async def health_check(self) -> Dict[str, Any]:
        """Check system health."""
        try:
            await self._ensure_initialized()

            return {
                "status": "healthy",
                "version": "direct-mode",
                "uptime": "unknown",
                "configuration": {
                    "mode": "direct",
                    "working_dir": self.config.lightrag_working_dir or "./rag_storage",
                },
                "statistics": {
                    "total_documents": 0,  # Would need to implement counting
                    "total_entities": 0,  # Would need to implement counting
                    "total_relationships": 0,  # Would need to implement counting
                },
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "LightRAG direct client not available",
            }

    async def get_system_stats(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get system statistics (limited in direct mode)."""
        return {
            "time_range": time_range,
            "message": "Detailed statistics not available in direct mode",
            "query_statistics": {
                "total_queries": 0,
                "queries_by_mode": {},
                "average_response_time": 0.0,
                "cache_hit_rate": 0.0,
            },
        }

    async def clear_cache(self, cache_types: List[str]) -> Dict[str, Any]:
        """Clear caches (not implemented in direct mode)."""
        return {
            "cleared_caches": [],
            "message": "Cache clearing not available in direct mode",
        }
