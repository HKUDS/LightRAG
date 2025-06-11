"""
Utilities for knowledge graph operations.
This package provides utilities for relationship typing, threshold management,
and other knowledge graph operations.
"""

# Import basic utilities - clean architecture without circular imports
from .threshold_manager import (
    ThresholdManager,
    get_default_threshold_manager,
    set_default_threshold_manager,
)
from .relationship_registry import (
    RelationshipTypeRegistry,
    standardize_relationship_type,
)
from .neo4j_edge_utils import (
    process_edge_properties,
    construct_relationship_queries,
    format_cypher_where_clause,
)
from .semantic_utils import process_relationship_weight, calculate_semantic_weight


# RelationshipExtractor has more complex dependencies - defer import
def get_relationship_extractor():
    """Get the RelationshipExtractor class. This is deferred to avoid circular imports."""
    from .relationship_extraction import RelationshipExtractor

    return RelationshipExtractor


# Export all utility classes and functions
__all__ = [
    "ThresholdManager",
    "get_default_threshold_manager",
    "set_default_threshold_manager",
    "RelationshipTypeRegistry",
    "standardize_relationship_type",
    "process_edge_properties",
    "construct_relationship_queries",
    "format_cypher_where_clause",
    "process_relationship_weight",
    "calculate_semantic_weight",
    "get_relationship_extractor",
]
