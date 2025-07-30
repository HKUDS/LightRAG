"""
Knowledge graph tools for LightRAG MCP integration.

Implements graph exploration, entity search, relationship traversal,
and graph manipulation tools.
"""

import logging
from typing import Dict, Any, Optional, List, Literal

from ..client.api_client import get_api_client
from ..client.direct_client import LightRAGDirectClient
from ..config import get_config
from ..utils import Validator, MCPError

logger = logging.getLogger("lightrag-mcp.graph_tools")


async def lightrag_get_graph(
    label_filter: Optional[str] = None,
    max_nodes: int = 100,
    max_edges: int = 200,
    output_format: Literal["json", "cypher", "graphml", "gexf"] = "json",
    include_properties: bool = True,
) -> Dict[str, Any]:
    """
    Extract knowledge graph data with filtering and formatting options.

    Args:
        label_filter: Filter nodes/edges by label (optional)
        max_nodes: Maximum nodes to return (default: 100, max: 1000)
        max_edges: Maximum edges to return (default: 200, max: 2000)
        output_format: Output format for graph data
        include_properties: Include node/edge properties (default: True)

    Returns:
        Knowledge graph data with nodes, edges, and statistics
    """
    config = get_config()

    # Validate inputs
    if max_nodes <= 0 or max_nodes > 1000:
        raise MCPError("INVALID_PARAMETER", "max_nodes must be between 1 and 1000")

    if max_edges <= 0 or max_edges > 2000:
        raise MCPError("INVALID_PARAMETER", "max_edges must be between 1 and 2000")

    valid_formats = ["json", "cypher", "graphml", "gexf"]
    if output_format not in valid_formats:
        raise MCPError(
            "INVALID_PARAMETER",
            f"Invalid output format. Valid options: {valid_formats}",
        )

    if label_filter and len(label_filter.strip()) == 0:
        label_filter = None

    logger.debug(
        f"Getting graph data: max_nodes={max_nodes}, max_edges={max_edges}, "
        f"format={output_format}, label_filter={label_filter}"
    )

    try:
        # Prepare parameters
        params = {
            "max_nodes": max_nodes,
            "max_edges": max_edges,
            "format": output_format,
            "include_properties": include_properties,
        }

        if label_filter:
            params["label_filter"] = label_filter

        # Execute graph extraction based on mode
        if config.enable_direct_mode:
            async with LightRAGDirectClient(config) as client:
                result = await client.get_graph(**params)
        else:
            async with get_api_client(config) as client:
                result = await client.get_graph(**params)

        # Enhance result with metadata
        if "metadata" not in result:
            result["metadata"] = {}

        result["metadata"].update(
            {
                "extraction_parameters": params,
                "format": output_format,
                "mcp_server": config.mcp_server_name,
            }
        )

        # Add truncation info if limits were reached
        nodes_count = len(result.get("nodes", []))
        edges_count = len(result.get("edges", []))

        result["metadata"]["truncated"] = (
            nodes_count >= max_nodes or edges_count >= max_edges
        )

        logger.info(f"Graph data extracted: {nodes_count} nodes, {edges_count} edges")
        return result

    except MCPError:
        raise
    except Exception as e:
        logger.error(f"Graph extraction failed: {e}")
        raise MCPError("GRAPH_ACCESS_ERROR", f"Graph extraction failed: {e}")


async def lightrag_search_entities(
    query: str,
    search_type: Literal["fuzzy", "exact", "semantic", "regex"] = "fuzzy",
    limit: int = 20,
    offset: int = 0,
    entity_types: Optional[List[str]] = None,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    """
    Search entities by name, properties, or relationships.

    Args:
        query: Search query
        search_type: Type of search to perform
        limit: Maximum results (default: 20, max: 100)
        offset: Pagination offset (default: 0)
        entity_types: Filter by entity types (optional)
        min_confidence: Minimum confidence score (default: 0.0)

    Returns:
        Entity search results with relevance scores
    """
    config = get_config()

    # Validate inputs
    Validator.validate_query(query)
    Validator.validate_limit_offset(limit, offset, max_limit=100)

    valid_search_types = ["fuzzy", "exact", "semantic", "regex"]
    if search_type not in valid_search_types:
        raise MCPError(
            "INVALID_PARAMETER",
            f"Invalid search type. Valid options: {valid_search_types}",
        )

    if not 0 <= min_confidence <= 1:
        raise MCPError("INVALID_PARAMETER", "min_confidence must be between 0 and 1")

    if entity_types:
        if not isinstance(entity_types, list):
            raise MCPError("INVALID_PARAMETER", "entity_types must be a list")

        if not all(isinstance(et, str) for et in entity_types):
            raise MCPError("INVALID_PARAMETER", "All entity types must be strings")

    logger.debug(f"Searching entities: '{query}' (type: {search_type}, limit: {limit})")

    try:
        # Prepare search parameters
        params = {
            "query": query,
            "search_type": search_type,
            "limit": limit,
            "offset": offset,
            "min_confidence": min_confidence,
        }

        if entity_types:
            params["entity_types"] = entity_types

        # Execute search based on mode
        if config.enable_direct_mode:
            async with LightRAGDirectClient(config) as client:
                result = await client.search_entities(query, limit, **params)
        else:
            async with get_api_client(config) as client:
                result = await client.search_entities(query, limit, **params)

        # Enhance result with search metadata
        if "search_metadata" not in result:
            result["search_metadata"] = {}

        result["search_metadata"].update(
            {
                "search_parameters": params,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": len(result.get("entities", [])) >= limit,
                },
                "mcp_server": config.mcp_server_name,
            }
        )

        entity_count = len(result.get("entities", []))
        logger.info(f"Entity search completed: {entity_count} entities found")
        return result

    except MCPError:
        raise
    except Exception as e:
        logger.error(f"Entity search failed: {e}")
        raise MCPError("GRAPH_ACCESS_ERROR", f"Entity search failed: {e}")


async def lightrag_update_entity(
    entity_id: str,
    properties: Optional[Dict[str, Any]] = None,
    add_labels: Optional[List[str]] = None,
    remove_labels: Optional[List[str]] = None,
    merge_mode: Literal["replace", "merge", "append"] = "merge",
) -> Dict[str, Any]:
    """
    Modify entity properties and labels.

    Args:
        entity_id: Entity identifier
        properties: Properties to update (optional)
        add_labels: Labels to add (optional)
        remove_labels: Labels to remove (optional)
        merge_mode: How to handle property updates

    Returns:
        Update results with changes summary
    """
    config = get_config()

    # Check if graph modification is enabled
    if not config.enable_graph_modification:
        raise MCPError("FORBIDDEN", "Graph modification is disabled")

    # Validate inputs
    Validator.validate_entity_id(entity_id)

    if not any([properties, add_labels, remove_labels]):
        raise MCPError(
            "INVALID_PARAMETER", "At least one update operation must be specified"
        )

    valid_merge_modes = ["replace", "merge", "append"]
    if merge_mode not in valid_merge_modes:
        raise MCPError(
            "INVALID_PARAMETER",
            f"Invalid merge mode. Valid options: {valid_merge_modes}",
        )

    if properties and not isinstance(properties, dict):
        raise MCPError("INVALID_PARAMETER", "Properties must be a dictionary")

    if add_labels:
        if not isinstance(add_labels, list):
            raise MCPError("INVALID_PARAMETER", "add_labels must be a list")
        if not all(isinstance(label, str) for label in add_labels):
            raise MCPError("INVALID_PARAMETER", "All labels must be strings")

    if remove_labels:
        if not isinstance(remove_labels, list):
            raise MCPError("INVALID_PARAMETER", "remove_labels must be a list")
        if not all(isinstance(label, str) for label in remove_labels):
            raise MCPError("INVALID_PARAMETER", "All labels must be strings")

    logger.info(f"Updating entity: {entity_id}")

    try:
        # Prepare update data
        updates = {"merge_mode": merge_mode}

        if properties:
            updates["properties"] = properties

        if add_labels:
            updates["add_labels"] = add_labels

        if remove_labels:
            updates["remove_labels"] = remove_labels

        # Execute update based on mode
        if config.enable_direct_mode:
            async with LightRAGDirectClient(config) as client:
                result = await client.update_entity(entity_id, updates)
        else:
            async with get_api_client(config) as client:
                result = await client.update_entity(entity_id, updates)

        # Enhance result
        result.update(
            {
                "entity_id": entity_id,
                "update_parameters": {
                    "merge_mode": merge_mode,
                    "properties_updated": bool(properties),
                    "labels_added": len(add_labels) if add_labels else 0,
                    "labels_removed": len(remove_labels) if remove_labels else 0,
                },
                "mcp_server": config.mcp_server_name,
            }
        )

        logger.info(f"Entity updated successfully: {entity_id}")
        return result

    except MCPError:
        raise
    except Exception as e:
        logger.error(f"Entity update failed: {e}")
        raise MCPError("GRAPH_ACCESS_ERROR", f"Entity update failed: {e}")


async def lightrag_get_entity_relationships(
    entity_id: str,
    relationship_types: Optional[List[str]] = None,
    direction: Literal["incoming", "outgoing", "both"] = "both",
    limit: int = 50,
    offset: int = 0,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    """
    Get relationships for specific entities with filtering.

    Args:
        entity_id: Entity identifier
        relationship_types: Filter by relationship types (optional)
        direction: Relationship direction to include
        limit: Maximum relationships to return (default: 50)
        offset: Pagination offset (default: 0)
        min_confidence: Minimum relationship confidence (default: 0.0)

    Returns:
        Entity relationships with connected entities
    """
    config = get_config()

    # Validate inputs
    Validator.validate_entity_id(entity_id)
    Validator.validate_limit_offset(limit, offset, max_limit=200)

    valid_directions = ["incoming", "outgoing", "both"]
    if direction not in valid_directions:
        raise MCPError(
            "INVALID_PARAMETER", f"Invalid direction. Valid options: {valid_directions}"
        )

    if not 0 <= min_confidence <= 1:
        raise MCPError("INVALID_PARAMETER", "min_confidence must be between 0 and 1")

    if relationship_types:
        if not isinstance(relationship_types, list):
            raise MCPError("INVALID_PARAMETER", "relationship_types must be a list")
        if not all(isinstance(rt, str) for rt in relationship_types):
            raise MCPError(
                "INVALID_PARAMETER", "All relationship types must be strings"
            )

    logger.debug(
        f"Getting relationships for entity: {entity_id} "
        f"(direction: {direction}, limit: {limit})"
    )

    try:
        # Prepare parameters
        params = {
            "direction": direction,
            "limit": limit,
            "offset": offset,
            "min_confidence": min_confidence,
        }

        if relationship_types:
            params["relationship_types"] = relationship_types

        # Execute query based on mode
        if config.enable_direct_mode:
            async with LightRAGDirectClient(config) as client:
                result = await client.get_entity_relationships(entity_id, **params)
        else:
            async with get_api_client(config) as client:
                result = await client.get_entity_relationships(entity_id, **params)

        # Enhance result with metadata
        if "pagination" not in result:
            relationship_count = len(result.get("relationships", []))
            result["pagination"] = {
                "limit": limit,
                "offset": offset,
                "has_more": relationship_count >= limit,
                "total_returned": relationship_count,
            }

        result.update(
            {
                "entity_id": entity_id,
                "query_parameters": params,
                "mcp_server": config.mcp_server_name,
            }
        )

        relationship_count = len(result.get("relationships", []))
        logger.info(
            f"Retrieved {relationship_count} relationships for entity: {entity_id}"
        )
        return result

    except MCPError:
        raise
    except Exception as e:
        logger.error(f"Entity relationship query failed: {e}")
        raise MCPError("GRAPH_ACCESS_ERROR", f"Entity relationship query failed: {e}")


# Tool registration helpers
def get_graph_tools() -> Dict[str, Any]:
    """Get graph tools for MCP server registration."""
    return {
        "lightrag_get_graph": lightrag_get_graph,
        "lightrag_search_entities": lightrag_search_entities,
        "lightrag_update_entity": lightrag_update_entity,
        "lightrag_get_entity_relationships": lightrag_get_entity_relationships,
    }
