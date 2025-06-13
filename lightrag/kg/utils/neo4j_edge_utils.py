"""
Neo4j edge utilities for working with typed relationships.
This module provides helper functions for edge operations in Neo4j.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np

from ...utils import logger
from .relationship_registry import (
    RelationshipTypeRegistry,
    standardize_relationship_type,
)
from .threshold_manager import ThresholdManager


def process_edge_properties(
    properties: Dict[str, Any],
    rel_type: str,
    source_id: str,
    target_id: str,
    registry: Optional[RelationshipTypeRegistry] = None,
    threshold_manager: Optional[ThresholdManager] = None,
) -> Dict[str, Any]:
    """
    Process edge properties for Neo4j storage.

    Args:
        properties: Original properties dictionary
        rel_type: Relationship type
        source_id: Source entity ID
        target_id: Target entity ID
        registry: Relationship type registry
        threshold_manager: Threshold manager

    Returns:
        Processed properties dictionary
    """
    # Create a copy to avoid modifying the original
    processed = properties.copy() if properties else {}

    # Use provided instances or create defaults
    reg = registry or RelationshipTypeRegistry()
    tm = threshold_manager or ThresholdManager()

    # Get standardized Neo4j relationship type
    neo4j_type = reg.get_neo4j_type(rel_type)

    # Store relationship type information
    processed["neo4j_type"] = neo4j_type
    processed["original_type"] = rel_type
    processed["rel_type"] = rel_type  # For backward compatibility

    # Process weight with threshold
    if "weight" not in processed or processed["weight"] is None:
        processed["weight"] = tm.get_threshold(rel_type)
    else:
        try:
            weight = float(processed["weight"])
            # Apply minimum threshold
            min_threshold = tm.get_threshold(rel_type)
            if weight < min_threshold:
                processed["weight"] = min_threshold
            else:
                processed["weight"] = weight
        except (ValueError, TypeError):
            processed["weight"] = tm.get_threshold(rel_type)

    # Ensure confidence value exists
    if "confidence" not in processed or processed["confidence"] is None:
        processed["confidence"] = processed["weight"]

    # Add timestamp if missing
    if (
        "extraction_timestamp" not in processed
        or processed["extraction_timestamp"] is None
    ):
        processed["extraction_timestamp"] = datetime.now().isoformat()

    # Add description if missing
    if "description" not in processed or not processed["description"]:
        processed["description"] = (
            f"Relationship of type {rel_type} between {source_id} and {target_id}"
        )

    # Process keywords
    if "keywords" in processed:
        # Convert string keywords to list
        if isinstance(processed["keywords"], str):
            processed["keywords"] = [
                k.strip() for k in processed["keywords"].split(",")
            ]
    else:
        # Default keywords
        processed["keywords"] = [source_id, target_id, rel_type]

    # Add extraction source if missing
    if "extraction_source" not in processed:
        processed["extraction_source"] = "system"

    return processed


def construct_relationship_queries(
    rel_types: Union[str, List[str]] = None, direction: str = "outgoing"
) -> Dict[str, str]:
    """
    Construct Cypher query patterns for relationship operations.

    Args:
        rel_types: Relationship type(s) to include in the pattern
        direction: Relationship direction ('outgoing', 'incoming', or 'both')

    Returns:
        Dictionary of query patterns for Cypher queries
    """
    # Convert single relationship type to list
    if rel_types and isinstance(rel_types, str):
        rel_types = [rel_types]

    # Direction patterns
    if direction == "outgoing":
        base_pattern = "(source)-[r]->(target)"
    elif direction == "incoming":
        base_pattern = "(source)<-[r]-(target)"
    else:  # Both directions
        base_pattern = "(source)-[r]-(target)"

    # Relationship type pattern
    if rel_types:
        # Standardize all relationship types
        std_types = [standardize_relationship_type(rt) for rt in rel_types]

        # Create a relationship type pattern that includes all types
        if len(std_types) == 1:
            # Single type
            rel_pattern = f"[r:{std_types[0]}]"
        else:
            # Multiple types
            rel_types_str = "|".join(std_types)
            rel_pattern = f"[r:{rel_types_str}]"

        # Replace [r] with the typed pattern
        typed_pattern = base_pattern.replace("[r]", rel_pattern)
    else:
        # No specific type, use the base pattern
        typed_pattern = base_pattern

    # Legacy pattern (using rel_type property)
    if rel_types:
        rel_types_str = "', '".join(rel_types)
        rel_filter = f"WHERE r.rel_type IN ['{rel_types_str}']"
        legacy_pattern = base_pattern + " " + rel_filter
    else:
        legacy_pattern = base_pattern

    # Create full pattern that supports both native types and legacy property
    if rel_types:
        # For a single relationship type, they should match the neo4j type or the property
        if len(rel_types) == 1:
            neo4j_type = standardize_relationship_type(rel_types[0])
            combined_pattern = f"""
            MATCH {base_pattern.replace("[r]", f"[r:{neo4j_type}]")}
            UNION
            MATCH {base_pattern}
            WHERE r.rel_type = '{rel_types[0]}'
            """
        else:
            # For multiple types, any of the types should match
            neo4j_types = "|".join(
                [standardize_relationship_type(rt) for rt in rel_types]
            )
            original_types = "', '".join(rel_types)
            combined_pattern = f"""
            MATCH {base_pattern.replace("[r]", f"[r:{neo4j_types}]")}
            UNION
            MATCH {base_pattern}
            WHERE r.rel_type IN ['{original_types}']
            """
    else:
        # No specific type
        combined_pattern = f"MATCH {base_pattern}"

    return {
        "base": base_pattern,
        "typed": typed_pattern,
        "legacy": legacy_pattern,
        "combined": combined_pattern,
    }


def format_cypher_where_clause(
    relationship_types: Optional[List[str]] = None,
    entity_types: Optional[List[str]] = None,
    min_weight: Optional[float] = None,
    include_legacy: bool = True,
) -> str:
    """
    Format a WHERE clause for Cypher queries based on filters.

    Args:
        relationship_types: List of relationship types to filter by
        entity_types: List of entity types to filter by
        min_weight: Minimum weight threshold
        include_legacy: Whether to include legacy relationship type filters

    Returns:
        WHERE clause string for Cypher query
    """
    conditions = []

    # Relationship type filter
    if relationship_types and len(relationship_types) > 0:
        # Convert all relationship types to Neo4j format
        neo4j_types = [standardize_relationship_type(rt) for rt in relationship_types]

        if len(relationship_types) == 1:
            # Single relationship type (optimize the query)
            type_filter = f"type(r) = '{neo4j_types[0]}'"
        else:
            # List of relationship types for MATCH clause
            type_filter = " OR ".join([f"type(r) = '{rt}'" for rt in neo4j_types])

        # Add legacy property check for backward compatibility
        if include_legacy:
            if len(relationship_types) == 1:
                legacy_filter = f"r.rel_type = '{relationship_types[0]}'"
            else:
                rel_list = "', '".join(relationship_types)
                legacy_filter = f"r.rel_type IN ['{rel_list}']"

            conditions.append(f"({type_filter} OR {legacy_filter})")
        else:
            conditions.append(f"({type_filter})")

    # Entity type filter
    if entity_types and len(entity_types) > 0:
        entity_list = "', '".join(entity_types)
        conditions.append(f"target.entity_type IN ['{entity_list}']")

    # Weight threshold
    if min_weight is not None and min_weight > 0:
        conditions.append(f"r.weight >= {min_weight}")

    # Build final WHERE clause
    if conditions:
        return "WHERE " + " AND ".join(conditions)
    else:
        return ""


def parse_relationship_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a Neo4j record into a relationship dictionary.

    Args:
        record: Neo4j record containing relationship data

    Returns:
        Dictionary with relationship data
    """
    # Ensure the record has all required fields
    if not all(k in record for k in ["source", "target", "properties"]):
        logger.warning("Invalid relationship record, missing required fields")
        return None

    source = record["source"]
    target = record["target"]
    props = dict(record["properties"])

    # Handle relationship type properties
    neo4j_type = record.get("neo4j_type", record.get("relationship_type"))
    original_type = record.get("original_type")
    rel_type = record.get("rel_type")

    # Prioritize types in this order: original_type, rel_type, neo4j_type
    if original_type:
        props["original_type"] = original_type
    elif rel_type:
        props["original_type"] = rel_type
    else:
        props["original_type"] = neo4j_type

    # Ensure backward compatibility with rel_type
    if not rel_type:
        props["rel_type"] = props.get("original_type", str(neo4j_type))

    # Store Neo4j type for reference
    props["neo4j_type"] = neo4j_type

    # Ensure weight is a float
    if "weight" in props:
        props["weight"] = float(props["weight"])

    # Add standard timestamp if missing
    if "extraction_timestamp" not in props:
        props["extraction_timestamp"] = props.get(
            "timestamp", datetime.now().isoformat()
        )

    # Ensure description exists
    if "description" not in props or not props["description"]:
        props["description"] = (
            f"Relationship of type {props.get('original_type', neo4j_type)} between {source} and {target}"
        )

    # Default confidence if not present
    if "confidence" not in props:
        props["confidence"] = props.get("weight", 0.5)  # Default to weight

    # Convert keywords to proper format if needed
    if "keywords" in props and isinstance(props["keywords"], str):
        props["keywords"] = [k.strip() for k in props["keywords"].split(";")]
    elif "keywords" not in props:
        props["keywords"] = [source, target, props.get("original_type", "related")]

    # Set extraction source if missing
    if "extraction_source" not in props:
        props["extraction_source"] = "system"

    return {
        "source": source,
        "target": target,
        "properties": props,
        "relationship_type": props.get("original_type", str(neo4j_type)),
        "neo4j_type": neo4j_type,
        "weight": props.get("weight", 0.5),
    }
