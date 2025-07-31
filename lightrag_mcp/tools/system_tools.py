"""
System management tools for LightRAG MCP integration.

Implements system health monitoring, cache management, and
statistics collection tools.
"""

import logging
from typing import Dict, Any, List, Literal

from ..client.api_client import get_api_client
from ..client.direct_client import LightRAGDirectClient
from ..config import get_config
from ..utils import Validator, MCPError, format_timestamp
from ..tools.query_tools import query_cache

logger = logging.getLogger("lightrag-mcp.system_tools")


async def lightrag_health_check(
    include_detailed: bool = False, check_dependencies: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive system health and status monitoring.

    Args:
        include_detailed: Include detailed diagnostic information
        check_dependencies: Check external dependencies

    Returns:
        System health information and configuration
    """
    config = get_config()

    logger.debug("Performing health check")

    try:
        # Execute health check based on mode
        if config.enable_direct_mode:
            async with LightRAGDirectClient(config) as client:
                result = await client.health_check()
        else:
            async with get_api_client(config) as client:
                result = await client.health_check()

        # Enhance result with MCP-specific information
        mcp_info = {
            "mcp_server": {
                "name": config.mcp_server_name,
                "version": config.mcp_server_version,
                "mode": "direct" if config.enable_direct_mode else "api",
                "features": {
                    "streaming": config.enable_streaming,
                    "graph_modification": config.enable_graph_modification,
                    "document_upload": config.enable_document_upload,
                    "cache": config.cache_enabled,
                },
            },
            "configuration": {
                "api_url": config.lightrag_api_url,
                "authentication": bool(config.lightrag_api_key),
                "default_query_mode": config.default_query_mode,
                "max_file_size_mb": config.max_file_size_mb,
                "allowed_file_types": config.allowed_file_types,
            },
        }

        if include_detailed:
            mcp_info["detailed_config"] = {
                "performance": {
                    "http_timeout": config.http_timeout,
                    "max_concurrent_queries": config.max_concurrent_queries,
                    "cache_ttl_seconds": config.cache_ttl_seconds,
                },
                "limits": {
                    "max_documents_per_batch": config.max_documents_per_batch,
                    "default_top_k": config.default_top_k,
                    "default_chunk_top_k": config.default_chunk_top_k,
                },
            }

        # Add cache statistics if enabled
        if config.cache_enabled:
            mcp_info["cache_stats"] = {
                "query_cache_size": query_cache.size(),
                "cache_enabled": True,
            }

        # Merge with LightRAG health information
        if isinstance(result, dict):
            result.update(mcp_info)
        else:
            result = mcp_info

        # Determine overall status
        if "status" not in result:
            result["status"] = "healthy"

        result["timestamp"] = format_timestamp()
        result["health_check_version"] = "1.0"

        logger.info(f"Health check completed: {result.get('status', 'unknown')}")
        return result

    except MCPError as e:
        logger.warning(f"Health check failed with MCP error: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "error_code": e.error_code,
            "timestamp": format_timestamp(),
            "mcp_server": config.mcp_server_name,
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Health check encountered unexpected error",
            "timestamp": format_timestamp(),
            "mcp_server": config.mcp_server_name,
        }


async def lightrag_clear_cache(
    cache_types: List[Literal["llm", "embedding", "query", "document", "graph", "all"]],
) -> Dict[str, Any]:
    """
    Clear various system caches with granular control.

    Args:
        cache_types: Types of cache to clear
            - llm: LLM response cache
            - embedding: Embedding cache
            - query: Query result cache
            - document: Document processing cache
            - graph: Graph query cache
            - all: All cache types

    Returns:
        Cache clearing results with sizes before/after
    """
    config = get_config()

    # Validate inputs
    if not cache_types:
        raise MCPError("INVALID_PARAMETER", "Cache types list cannot be empty")

    valid_cache_types = ["llm", "embedding", "query", "document", "graph", "all"]
    invalid_types = [ct for ct in cache_types if ct not in valid_cache_types]
    if invalid_types:
        raise MCPError(
            "INVALID_PARAMETER",
            f"Invalid cache types: {invalid_types}. Valid: {valid_cache_types}",
        )

    logger.info(f"Clearing caches: {cache_types}")

    try:
        cleared_caches = []
        cache_sizes_before = {}
        cache_sizes_after = {}

        # Handle MCP-level caches
        if "query" in cache_types or "all" in cache_types:
            if config.cache_enabled:
                size_before = query_cache.size()
                cache_sizes_before["query"] = {
                    "entries": size_before,
                    "size_bytes": 0,  # We don't track byte size
                    "oldest_entry": None,
                    "newest_entry": None,
                }

                cleared_count = query_cache.clear()
                cleared_caches.append("query")

                cache_sizes_after["query"] = {
                    "entries": 0,
                    "size_bytes": 0,
                    "oldest_entry": None,
                    "newest_entry": None,
                }

                logger.info(f"Cleared query cache: {cleared_count} entries")
            else:
                logger.debug("Query cache not enabled, skipping")

        # Handle LightRAG-level caches
        lightrag_cache_types = [
            ct
            for ct in cache_types
            if ct in ["llm", "embedding", "document", "graph", "all"]
        ]

        if lightrag_cache_types:
            if config.enable_direct_mode:
                # Direct mode doesn't support cache clearing
                logger.warning("Cache clearing not fully supported in direct mode")
                result = {
                    "cleared_caches": [],
                    "message": "Cache clearing not available in direct mode",
                }
            else:
                async with get_api_client(config) as client:
                    result = await client.clear_cache(lightrag_cache_types)

                if "cleared_caches" in result:
                    cleared_caches.extend(result["cleared_caches"])

                if "cache_sizes_before" in result:
                    cache_sizes_before.update(result["cache_sizes_before"])

                if "cache_sizes_after" in result:
                    cache_sizes_after.update(result["cache_sizes_after"])

        # Compile final result
        final_result = {
            "cleared_caches": cleared_caches,
            "cache_sizes_before": cache_sizes_before,
            "cache_sizes_after": cache_sizes_after,
            "processing_time": 0.0,  # Would need to measure
            "requested_types": cache_types,
            "mcp_server": config.mcp_server_name,
            "timestamp": format_timestamp(),
        }

        # Add warnings if some caches couldn't be cleared
        warnings = []
        requested_set = set(cache_types)
        if "all" in requested_set:
            requested_set = set(valid_cache_types) - {"all"}

        cleared_set = set(cleared_caches)
        not_cleared = requested_set - cleared_set

        if not_cleared:
            warnings.append(f"Could not clear caches: {list(not_cleared)}")

        if warnings:
            final_result["warnings"] = warnings

        logger.info(f"Cache clearing completed: {len(cleared_caches)} caches cleared")
        return final_result

    except MCPError:
        raise
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        raise MCPError("INTERNAL_ERROR", f"Cache clearing failed: {e}")


async def lightrag_get_system_stats(
    time_range: str = "24h",
    include_breakdown: bool = True,
    stat_categories: List[str] = None,
) -> Dict[str, Any]:
    """
    Detailed system usage statistics and analytics.

    Args:
        time_range: Statistics time range ("1h", "24h", "7d", "30d")
        include_breakdown: Include detailed breakdowns
        stat_categories: Specific categories to include
            - queries: Query statistics
            - documents: Document processing statistics
            - resources: Resource usage statistics
            - performance: Performance metrics
            - errors: Error statistics

    Returns:
        Comprehensive system statistics
    """
    config = get_config()

    # Validate inputs
    Validator.validate_time_range(time_range)

    if stat_categories:
        valid_categories = [
            "queries",
            "documents",
            "resources",
            "performance",
            "errors",
        ]
        invalid_categories = [
            cat for cat in stat_categories if cat not in valid_categories
        ]
        if invalid_categories:
            raise MCPError(
                "INVALID_PARAMETER", f"Invalid stat categories: {invalid_categories}"
            )
    else:
        stat_categories = ["queries", "documents", "resources", "performance"]

    logger.debug(f"Getting system stats for {time_range}")

    try:
        # Get base statistics from LightRAG
        if config.enable_direct_mode:
            async with LightRAGDirectClient(config) as client:
                result = await client.get_system_stats(time_range)
        else:
            async with get_api_client(config) as client:
                result = await client.get_system_stats(time_range)

        # Enhance with MCP-specific statistics
        mcp_stats = {
            "time_range": time_range,
            "timestamp": format_timestamp(),
            "mcp_server_info": {
                "name": config.mcp_server_name,
                "version": config.mcp_server_version,
                "mode": "direct" if config.enable_direct_mode else "api",
                "uptime": "unknown",  # Would need to track server start time
            },
        }

        # Add cache statistics
        if "resources" in stat_categories and config.cache_enabled:
            mcp_stats["mcp_cache_stats"] = {
                "query_cache": {
                    "enabled": True,
                    "size": query_cache.size(),
                    "ttl_seconds": config.cache_ttl_seconds,
                }
            }

        # Add configuration statistics
        if "performance" in stat_categories:
            mcp_stats["mcp_performance_config"] = {
                "max_concurrent_queries": config.max_concurrent_queries,
                "http_timeout": config.http_timeout,
                "default_query_timeout": config.default_query_timeout,
            }

        # Add feature usage statistics
        if include_breakdown:
            mcp_stats["feature_usage"] = {
                "streaming_enabled": config.enable_streaming,
                "graph_modification_enabled": config.enable_graph_modification,
                "document_upload_enabled": config.enable_document_upload,
                "direct_mode": config.enable_direct_mode,
            }

        # Merge results
        if isinstance(result, dict):
            result.update(mcp_stats)
        else:
            result = mcp_stats

        # Filter categories if requested
        if stat_categories and len(stat_categories) < 5:  # Not all categories
            filtered_result = {
                key: value
                for key, value in result.items()
                if any(cat in key for cat in stat_categories)
                or key in ["time_range", "timestamp", "mcp_server_info"]
            }
            result = filtered_result

        logger.info(f"System stats retrieved for {time_range}")
        return result

    except MCPError:
        raise
    except Exception as e:
        logger.error(f"System stats retrieval failed: {e}")
        raise MCPError("INTERNAL_ERROR", f"System stats retrieval failed: {e}")


# Tool registration helpers
def get_system_tools() -> Dict[str, Any]:
    """Get system tools for MCP server registration."""
    return {
        "lightrag_health_check": lightrag_health_check,
        "lightrag_clear_cache": lightrag_clear_cache,
        "lightrag_get_system_stats": lightrag_get_system_stats,
    }
