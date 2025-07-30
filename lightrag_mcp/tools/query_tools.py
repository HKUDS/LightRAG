"""
Query tools for LightRAG MCP integration.

Implements RAG query tools with support for multiple retrieval modes,
streaming responses, and comprehensive result formatting.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, AsyncIterator, Literal

from ..client.api_client import get_api_client
from ..client.direct_client import LightRAGDirectClient
from ..config import get_config
from ..utils import Validator, MCPError, Cache, measure_time

logger = logging.getLogger("lightrag-mcp.query_tools")

# Query cache
query_cache = Cache(default_ttl=3600)


async def lightrag_query(
    query: str,
    mode: Literal["naive", "local", "global", "hybrid", "mix", "bypass"] = "hybrid",
    top_k: int = 40,
    chunk_top_k: int = 10,
    cosine_threshold: float = 0.2,
    max_tokens: int = 30000,
    enable_rerank: bool = False,
    history_turns: int = 0
) -> Dict[str, Any]:
    """
    Execute a RAG query using LightRAG's multiple retrieval modes.
    
    Args:
        query: The question or search query
        mode: Query mode determining retrieval strategy
            - naive: Basic vector search without graph enhancement
            - local: Context-dependent entity-focused retrieval
            - global: Global knowledge graph relationship queries  
            - hybrid: Combines local and global approaches
            - mix: Integrates knowledge graph traversal with vector similarity
            - bypass: Direct LLM query without retrieval augmentation
        top_k: Number of entities/relations to retrieve (default: 40)
        chunk_top_k: Number of document chunks to include (default: 10)
        cosine_threshold: Similarity threshold for retrieval (default: 0.2)
        max_tokens: Maximum tokens in final context (default: 30000)
        enable_rerank: Enable reranking of results (default: False)
        history_turns: Number of conversation history turns to include (default: 0)
    
    Returns:
        Query response with answer, metadata, and source attribution
    """
    config = get_config()
    
    # Validate inputs
    Validator.validate_query(query)
    Validator.validate_query_mode(mode)
    
    if top_k <= 0 or top_k > 200:
        raise MCPError("INVALID_PARAMETER", "top_k must be between 1 and 200")
    
    if chunk_top_k <= 0 or chunk_top_k > 100:
        raise MCPError("INVALID_PARAMETER", "chunk_top_k must be between 1 and 100")
    
    if not 0 <= cosine_threshold <= 1:
        raise MCPError("INVALID_PARAMETER", "cosine_threshold must be between 0 and 1")
    
    if max_tokens <= 0 or max_tokens > 100000:
        raise MCPError("INVALID_PARAMETER", "max_tokens must be between 1 and 100000")
    
    # Check cache if enabled
    cache_key = None
    if config.cache_enabled:
        cache_data = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "cosine_threshold": cosine_threshold,
            "max_tokens": max_tokens,
            "enable_rerank": enable_rerank,
            "history_turns": history_turns
        }
        cache_key = f"query:{hash(str(cache_data))}"
        
        cached_result = query_cache.get(cache_key)
        if cached_result:
            logger.debug(f"Query cache hit for: {query[:50]}...")
            cached_result["metadata"]["cache_hit"] = True
            return cached_result
    
    logger.info(f"Executing query: {query[:100]}... (mode: {mode})")
    
    try:
        # Prepare query parameters
        params = {
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "cosine_threshold": cosine_threshold,
            "max_tokens": max_tokens
        }
        
        if enable_rerank:
            params["enable_rerank"] = enable_rerank
        
        if history_turns > 0:
            params["history_turns"] = history_turns
        
        # Execute query based on mode (API or direct)
        if config.enable_direct_mode:
            async with LightRAGDirectClient(config) as client:
                result = await client.query(query, mode, **params)
        else:
            async with get_api_client(config) as client:
                result = await client.query(query, mode, **params)
        
        # Enhance result with additional metadata
        if "metadata" not in result:
            result["metadata"] = {}
        
        result["metadata"].update({
            "cache_hit": False,
            "query_mode": mode,
            "parameters_used": params,
            "mcp_server": config.mcp_server_name
        })
        
        # Cache result if caching is enabled
        if config.cache_enabled and cache_key:
            query_cache.set(cache_key, result, config.cache_ttl_seconds)
            logger.debug(f"Cached query result: {cache_key}")
        
        logger.info(f"Query completed successfully (mode: {mode})")
        return result
        
    except MCPError:
        raise
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise MCPError("QUERY_FAILED", f"Query execution failed: {e}")


async def lightrag_stream_query(
    query: str,
    mode: Literal["naive", "local", "global", "hybrid", "mix", "bypass"] = "hybrid",
    top_k: int = 40,
    chunk_top_k: int = 10,
    cosine_threshold: float = 0.2,
    max_tokens: int = 30000
) -> AsyncIterator[Dict[str, Any]]:
    """
    Execute a streaming RAG query with real-time response generation.
    
    Args:
        query: The question or search query
        mode: Query mode determining retrieval strategy
        top_k: Number of entities/relations to retrieve
        chunk_top_k: Number of document chunks to include
        cosine_threshold: Similarity threshold for retrieval
        max_tokens: Maximum tokens in final context
    
    Yields:
        Stream chunks with content, metadata, or completion signals
    """
    config = get_config()
    
    # Validate inputs (same as regular query)
    Validator.validate_query(query)
    Validator.validate_query_mode(mode)
    
    if top_k <= 0 or top_k > 200:
        raise MCPError("INVALID_PARAMETER", "top_k must be between 1 and 200")
    
    if chunk_top_k <= 0 or chunk_top_k > 100:
        raise MCPError("INVALID_PARAMETER", "chunk_top_k must be between 1 and 100")
    
    if not 0 <= cosine_threshold <= 1:
        raise MCPError("INVALID_PARAMETER", "cosine_threshold must be between 0 and 1")
    
    if max_tokens <= 0 or max_tokens > 100000:
        raise MCPError("INVALID_PARAMETER", "max_tokens must be between 1 and 100000")
    
    # Check if streaming is enabled
    if not config.enable_streaming:
        raise MCPError("NOT_IMPLEMENTED", "Streaming queries are disabled")
    
    logger.info(f"Executing streaming query: {query[:100]}... (mode: {mode})")
    
    try:
        # Prepare query parameters
        params = {
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "cosine_threshold": cosine_threshold,
            "max_tokens": max_tokens
        }
        
        chunk_count = 0
        start_time = asyncio.get_event_loop().time()
        
        # Execute streaming query based on mode
        if config.enable_direct_mode:
            # Direct mode doesn't support true streaming, fall back to regular query
            async with LightRAGDirectClient(config) as client:
                result = await client.query(query, mode, **params)
                
                # Simulate streaming by yielding the complete response
                yield {
                    "chunk_type": "content",
                    "content": result["response"],
                    "metadata": None,
                    "error": None
                }
                chunk_count = 1
        else:
            # Use API streaming
            async with get_api_client(config) as client:
                async for chunk in client.stream_query(query, mode, **params):
                    if chunk.strip():
                        yield {
                            "chunk_type": "content",
                            "content": chunk,
                            "metadata": None,
                            "error": None
                        }
                        chunk_count += 1
        
        # Send final metadata chunk
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        yield {
            "chunk_type": "complete",
            "content": None,
            "metadata": {
                "total_chunks_sent": chunk_count,
                "processing_time": processing_time,
                "query_mode": mode,
                "parameters_used": params,
                "mcp_server": config.mcp_server_name
            },
            "error": None
        }
        
        logger.info(f"Streaming query completed: {chunk_count} chunks sent")
        
    except MCPError:
        # Send error chunk
        yield {
            "chunk_type": "error",
            "content": None,
            "metadata": None,
            "error": str(e)
        }
        raise
    except Exception as e:
        logger.error(f"Streaming query failed: {e}")
        
        # Send error chunk
        yield {
            "chunk_type": "error",
            "content": None,
            "metadata": None,
            "error": f"Streaming query failed: {e}"
        }
        
        raise MCPError("QUERY_FAILED", f"Streaming query failed: {e}")


# Tool registration helpers
def get_query_tools() -> Dict[str, Any]:
    """Get query tools for MCP server registration."""
    return {
        "lightrag_query": lightrag_query,
        "lightrag_stream_query": lightrag_stream_query
    }