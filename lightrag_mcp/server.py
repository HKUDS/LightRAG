#!/usr/bin/env python3
"""
LightRAG MCP Server - Main server implementation

This module implements the Model Context Protocol server for LightRAG,
providing comprehensive RAG and knowledge graph capabilities to Claude CLI
and other MCP-compatible clients.

Usage:
    python -m lightrag_mcp.server
    python lightrag_mcp/server.py

Environment Variables:
    LIGHTRAG_API_URL: LightRAG API endpoint (default: http://localhost:9621)
    LIGHTRAG_API_KEY: Optional API key for authentication
    MCP_SERVER_NAME: MCP server name (default: lightrag-mcp)

    See config.py for full list of configuration options.
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    print(
        "Error: MCP library not installed. Please install with: pip install mcp",
        file=sys.stderr,
    )
    print(f"Original error: {e}", file=sys.stderr)
    sys.exit(1)

from .config import get_config, LightRAGMCPConfig
from .tools.query_tools import lightrag_query, lightrag_stream_query
from .tools.document_tools import (
    lightrag_insert_text,
    lightrag_insert_file,
    lightrag_list_documents,
    lightrag_delete_documents,
    lightrag_batch_process,
)
from .tools.graph_tools import (
    lightrag_get_graph,
    lightrag_search_entities,
    lightrag_update_entity,
    lightrag_get_entity_relationships,
)
from .tools.system_tools import (
    lightrag_health_check,
    lightrag_clear_cache,
    lightrag_get_system_stats,
)

# Configure logging
logger = logging.getLogger("lightrag-mcp.server")


class LightRAGMCPServer:
    """LightRAG Model Context Protocol Server implementation."""

    def __init__(self, config: LightRAGMCPConfig):
        self.config = config
        self.mcp = FastMCP(name=config.mcp_server_name)
        self._setup_tools()
        self._setup_resources()
        self._setup_error_handling()

        logger.info(
            f"Initialized {config.mcp_server_name} v{config.mcp_server_version}"
        )

    def _setup_tools(self):
        """Register all MCP tools."""

        # Query tools
        @self.mcp.tool()
        async def lightrag_query_tool(
            query: str,
            mode: str = "hybrid",
            top_k: int = 40,
            chunk_top_k: int = 10,
            cosine_threshold: float = 0.2,
            max_tokens: int = 30000,
            enable_rerank: bool = False,
            history_turns: int = 0,
        ) -> Dict[str, Any]:
            """Execute a RAG query using LightRAG's multiple retrieval modes."""
            return await lightrag_query(
                query=query,
                mode=mode,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                cosine_threshold=cosine_threshold,
                max_tokens=max_tokens,
                enable_rerank=enable_rerank,
                history_turns=history_turns,
            )

        # Only register streaming tool if enabled
        if self.config.enable_streaming:

            @self.mcp.tool()
            async def lightrag_stream_query_tool(
                query: str,
                mode: str = "hybrid",
                top_k: int = 40,
                chunk_top_k: int = 10,
                cosine_threshold: float = 0.2,
                max_tokens: int = 30000,
            ):
                """Execute a streaming RAG query with real-time response generation."""
                async for chunk in lightrag_stream_query(
                    query=query,
                    mode=mode,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    cosine_threshold=cosine_threshold,
                    max_tokens=max_tokens,
                ):
                    yield chunk

        # Document tools (if document upload is enabled)
        if self.config.enable_document_upload:

            @self.mcp.tool()
            async def lightrag_insert_text_tool(
                text: str, title: str = "", metadata: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
                """Insert text document directly into the knowledge base."""
                return await lightrag_insert_text(
                    text=text, title=title, metadata=metadata
                )

            @self.mcp.tool()
            async def lightrag_insert_file_tool(
                file_path: str, processing_options: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
                """Process and index files from filesystem."""
                return await lightrag_insert_file(
                    file_path=file_path, processing_options=processing_options
                )

            @self.mcp.tool()
            async def lightrag_batch_process_tool(
                items: List[Dict[str, Any]],
                max_concurrent: int = 5,
                stop_on_error: bool = False,
            ) -> Dict[str, Any]:
                """Process multiple documents in batch with progress tracking."""
                return await lightrag_batch_process(
                    items=items,
                    max_concurrent=max_concurrent,
                    stop_on_error=stop_on_error,
                )

        # Document management tools
        @self.mcp.tool()
        async def lightrag_list_documents_tool(
            status_filter: str = "",
            limit: int = 50,
            offset: int = 0,
            sort_by: str = "created_date",
            sort_order: str = "desc",
        ) -> Dict[str, Any]:
            """List documents with filtering, sorting, and pagination."""
            return await lightrag_list_documents(
                status_filter=status_filter,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_order=sort_order,
            )

        @self.mcp.tool()
        async def lightrag_delete_documents_tool(
            document_ids: List[str],
            cascade_delete: bool = True,
            create_backup: bool = False,
        ) -> Dict[str, Any]:
            """Remove documents and associated data from knowledge base."""
            return await lightrag_delete_documents(
                document_ids=document_ids,
                cascade_delete=cascade_delete,
                create_backup=create_backup,
            )

        # Graph tools
        @self.mcp.tool()
        async def lightrag_get_graph_tool(
            label_filter: Optional[str] = None,
            max_nodes: int = 100,
            max_edges: int = 200,
            output_format: str = "json",
            include_properties: bool = True,
        ) -> Dict[str, Any]:
            """Extract knowledge graph data with filtering and formatting options."""
            return await lightrag_get_graph(
                label_filter=label_filter,
                max_nodes=max_nodes,
                max_edges=max_edges,
                output_format=output_format,
                include_properties=include_properties,
            )

        @self.mcp.tool()
        async def lightrag_search_entities_tool(
            query: str,
            search_type: str = "fuzzy",
            limit: int = 20,
            offset: int = 0,
            entity_types: Optional[List[str]] = None,
            min_confidence: float = 0.0,
        ) -> Dict[str, Any]:
            """Search entities by name, properties, or relationships."""
            return await lightrag_search_entities(
                query=query,
                search_type=search_type,
                limit=limit,
                offset=offset,
                entity_types=entity_types,
                min_confidence=min_confidence,
            )

        @self.mcp.tool()
        async def lightrag_get_entity_relationships_tool(
            entity_id: str,
            relationship_types: Optional[List[str]] = None,
            direction: str = "both",
            limit: int = 50,
            offset: int = 0,
            min_confidence: float = 0.0,
        ) -> Dict[str, Any]:
            """Get relationships for specific entities with filtering."""
            return await lightrag_get_entity_relationships(
                entity_id=entity_id,
                relationship_types=relationship_types,
                direction=direction,
                limit=limit,
                offset=offset,
                min_confidence=min_confidence,
            )

        # Graph modification tool (if enabled)
        if self.config.enable_graph_modification:

            @self.mcp.tool()
            async def lightrag_update_entity_tool(
                entity_id: str,
                properties: Optional[Dict[str, Any]] = None,
                add_labels: Optional[List[str]] = None,
                remove_labels: Optional[List[str]] = None,
                merge_mode: str = "merge",
            ) -> Dict[str, Any]:
                """Modify entity properties and labels."""
                return await lightrag_update_entity(
                    entity_id=entity_id,
                    properties=properties,
                    add_labels=add_labels,
                    remove_labels=remove_labels,
                    merge_mode=merge_mode,
                )

        # System tools
        @self.mcp.tool()
        async def lightrag_health_check_tool(
            include_detailed: bool = False, check_dependencies: bool = True
        ) -> Dict[str, Any]:
            """Comprehensive system health and status monitoring."""
            return await lightrag_health_check(
                include_detailed=include_detailed, check_dependencies=check_dependencies
            )

        @self.mcp.tool()
        async def lightrag_clear_cache_tool(cache_types: List[str]) -> Dict[str, Any]:
            """Clear various system caches with granular control."""
            return await lightrag_clear_cache(cache_types=cache_types)

        @self.mcp.tool()
        async def lightrag_get_system_stats_tool(
            time_range: str = "24h",
            include_breakdown: bool = True,
            stat_categories: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """Detailed system usage statistics and analytics."""
            return await lightrag_get_system_stats(
                time_range=time_range,
                include_breakdown=include_breakdown,
                stat_categories=stat_categories,
            )

        logger.info("MCP tools registered successfully")

    def _setup_resources(self):
        """Register MCP resources."""

        @self.mcp.resource("lightrag://system/config")
        async def get_system_config() -> str:
            """Get LightRAG system configuration."""
            try:
                health_info = await lightrag_health_check(include_detailed=True)
                config_info = health_info.get("configuration", {})

                import json

                return json.dumps(config_info, indent=2)
            except Exception as e:
                return json.dumps(
                    {
                        "error": str(e),
                        "message": "Failed to retrieve system configuration",
                    },
                    indent=2,
                )

        @self.mcp.resource("lightrag://system/health")
        async def get_system_health() -> str:
            """Get system health status."""
            try:
                health_info = await lightrag_health_check()

                import json

                return json.dumps(health_info, indent=2)
            except Exception as e:
                return json.dumps(
                    {
                        "status": "unhealthy",
                        "error": str(e),
                        "message": "Health check failed",
                    },
                    indent=2,
                )

        @self.mcp.resource("lightrag://documents/status")
        async def get_document_status() -> str:
            """Get document processing pipeline status."""
            try:
                # Get recent documents with status info
                docs_info = await lightrag_list_documents(limit=10)

                # Create status summary
                status_summary = {
                    "recent_documents": docs_info.get("documents", []),
                    "total_documents": docs_info.get("total", 0),
                    "document_statistics": docs_info.get("statistics", {}),
                    "pipeline_status": {
                        "active": True,
                        "mode": "api"
                        if not self.config.enable_direct_mode
                        else "direct",
                    },
                }

                import json

                return json.dumps(status_summary, indent=2)
            except Exception as e:
                return json.dumps(
                    {"error": str(e), "message": "Failed to retrieve document status"},
                    indent=2,
                )

        logger.info("MCP resources registered successfully")

    def _setup_error_handling(self):
        """Setup global error handling for MCP server."""

        # This would be implemented if FastMCP supports global error handlers
        # For now, error handling is done in individual tools

        logger.debug("Error handling configured")

    async def run(self):
        """Run the MCP server."""
        try:
            logger.info(f"Starting {self.config.mcp_server_name} server...")
            logger.info(
                f"Configuration: API URL={self.config.lightrag_api_url}, "
                f"Mode={'direct' if self.config.enable_direct_mode else 'api'}"
            )
            logger.info(
                f"Features: streaming={self.config.enable_streaming}, "
                f"graph_mod={self.config.enable_graph_modification}, "
                f"doc_upload={self.config.enable_document_upload}"
            )

            # Test connectivity on startup
            try:
                health_result = await lightrag_health_check()
                logger.info(
                    f"Startup health check: {health_result.get('status', 'unknown')}"
                )
            except Exception as e:
                logger.warning(f"Startup health check failed: {e}")

            # Run the MCP server
            await self.mcp.run()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            logger.info("MCP server stopped")


def main():
    """Main entry point for the MCP server."""
    try:
        # Load configuration
        config = get_config()

        # Create and run server
        server = LightRAGMCPServer(config)
        asyncio.run(server.run())

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
