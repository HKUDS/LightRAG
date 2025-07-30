"""
MCP tools for LightRAG integration.

This module contains all MCP tool implementations for accessing
LightRAG functionality through the Model Context Protocol.
"""

from .query_tools import lightrag_query, lightrag_stream_query
from .document_tools import (
    lightrag_insert_text,
    lightrag_insert_file,
    lightrag_list_documents,
    lightrag_delete_documents,
    lightrag_batch_process,
)
from .graph_tools import (
    lightrag_get_graph,
    lightrag_search_entities,
    lightrag_update_entity,
    lightrag_get_entity_relationships,
)
from .system_tools import (
    lightrag_health_check,
    lightrag_clear_cache,
    lightrag_get_system_stats,
)

__all__ = [
    # Query tools
    "lightrag_query",
    "lightrag_stream_query",
    # Document tools
    "lightrag_insert_text",
    "lightrag_insert_file",
    "lightrag_list_documents",
    "lightrag_delete_documents",
    "lightrag_batch_process",
    # Graph tools
    "lightrag_get_graph",
    "lightrag_search_entities",
    "lightrag_update_entity",
    "lightrag_get_entity_relationships",
    # System tools
    "lightrag_health_check",
    "lightrag_clear_cache",
    "lightrag_get_system_stats",
]
