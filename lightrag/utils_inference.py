"""
Entity Relationship Inference Utilities

This module provides utilities for inferring missing relationships between entities
based on their co-occurrence patterns in document chunks. This helps reduce isolated
nodes in knowledge graphs and improve graph connectivity.

Key features:
- Co-occurrence-based relationship inference
- Similarity-based entity name merging
- Configurable thresholds and parameters
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Any

def infer_cooccurrence_relationships(
    all_nodes: Dict[str, List[Dict]],
    all_edges: Dict[Tuple[str, str], List[Dict]],
    min_cooccurrence: int = 2,
    enable_inference: bool = True,
) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Infer missing relationships based on entity co-occurrence in chunks.

    This helps recover relationships that were split across chunk boundaries
    and reduces isolated nodes in the knowledge graph.

    Args:
        all_nodes: Dictionary of entity names to entity node data
        all_edges: Existing relationships (edges) between entities
        min_cooccurrence: Minimum number of chunks entities must co-occur in
        enable_inference: Whether to enable inference (can be disabled via config)

    Returns:
        Updated relationships dictionary with inferred relationships
    """
    if not enable_inference:
        return all_edges

    # Build co-occurrence map: track which entities appear together in chunks
    cooccurrence_map = defaultdict(lambda: defaultdict(set))

    for entity_name, entity_list in all_nodes.items():
        # Get all chunk IDs where this entity appears
        for entity_data in entity_list:
            chunk_id = entity_data.get("source_id")
            if not chunk_id:
                continue

            # Find other entities in the same chunk
            for other_entity, other_entity_list in all_nodes.items():
                if other_entity == entity_name:
                    continue

                # Check if other entity appears in same chunk
                for other_data in other_entity_list:
                    other_chunk_id = other_data.get("source_id")
                    if chunk_id == other_chunk_id:
                        # They co-occur in this chunk
                        cooccurrence_map[entity_name][other_entity].add(chunk_id)

    # Infer relationships for frequently co-occurring entities
    inferred_count = 0
    updated_edges = dict(all_edges)

    for entity1 in sorted(all_nodes.keys()):
        if entity1 not in cooccurrence_map:
            continue

        for entity2, chunk_ids in cooccurrence_map[entity1].items():
            count = len(chunk_ids)
            if count < min_cooccurrence:
                continue

            # Create sorted edge key for undirected graph
            edge_key = tuple(sorted([entity1, entity2]))

            # Check if relationship already exists
            if edge_key in updated_edges:
                continue

            # Get entity types for context
            entity1_type = all_nodes[entity1][0].get("entity_type", "unknown")
            entity2_type = all_nodes[entity2][0].get("entity_type", "unknown")

            # Create inferred relationship
            inferred_rel = {
                "src_id": entity1,
                "tgt_id": entity2,
                "weight": count,  # Weight by co-occurrence count
                "keywords": f"associated_with, co-occurs_{count}_times",
                "description": f"{entity1} and {entity2} appear together in {count} document sections, suggesting an association between the {entity1_type} and the {entity2_type}.",
                "source_id": "inferred_cooccurrence",
                "inferred": True,
            }

            updated_edges[edge_key] = [inferred_rel]
            inferred_count += 1

    if inferred_count > 0:
        from .utils import logger
        logger.info(
            f"  ✓ Inferred {inferred_count} additional relationship(s) from entity co-occurrence"
        )

    return updated_edges

