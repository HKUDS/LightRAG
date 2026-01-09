"""
Entity Relationship Inference Utilities

This module provides utilities for inferring missing relationships between entities
based on their co-occurrence patterns in document chunks. This helps reduce isolated
nodes in knowledge graphs and improve graph connectivity.

Key features:
- High-quality co-occurrence-based relationship inference
- Entity-type-specific inference rules and thresholds
- Business-relevant relationship type assignment
- Quality filters to prevent weak "trash" relationships
- Configurable thresholds and parameters
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


# High-quality relationship inference rules based on entity type combinations
# Only infer relationships that are LIKELY TO BE TRUE and BUSINESS-RELEVANT
INFERENCE_RULES = {
    # Person-Organization relationships (employment, affiliation)
    ("Person", "Organization"): {
        "min_cooccurrence": 3,  # Require strong evidence
        "keywords": "professional_affiliation, works_for",
        "description_template": "{entity1} is professionally affiliated with {entity2} (co-occurs {count} times across document sections)",
    },

    # Organization-Organization relationships (B2B associations)
    ("Organization", "Organization"): {
        "min_cooccurrence": 4,  # Very conservative - needs strong evidence
        "keywords": "business_association, ecosystem_connection",
        "description_template": "{entity1} and {entity2} have a business association (mentioned together in {count} contexts)",
    },

    # Organization-Location relationships
    ("Organization", "Location"): {
        "min_cooccurrence": 2,
        "keywords": "geographic_presence, operates_in",
        "description_template": "{entity1} has a presence in or operates in {entity2} (co-occurs {count} times)",
    },

    # Organization-Product relationships
    ("Organization", "Product"): {
        "min_cooccurrence": 3,
        "keywords": "product_association, portfolio_connection",
        "description_template": "{entity1} is associated with product/service {entity2} (mentioned together {count} times)",
    },

    # Organization-Technology_Field relationships
    ("Organization", "Technology_Field"): {
        "min_cooccurrence": 3,
        "keywords": "technology_involvement, domain_activity",
        "description_template": "{entity1} is involved in {entity2} technology domain (co-occurs {count} times)",
    },

    # Person-Person relationships (be very conservative)
    ("Person", "Person"): {
        "min_cooccurrence": 5,  # High threshold - only strong connections
        "keywords": "professional_connection, colleagues",
        "description_template": "{entity1} and {entity2} have a professional connection (appear together {count} times)",
    },

    # Customer-Organization relationships (when Customer is an entity type)
    ("Customer", "Organization"): {
        "min_cooccurrence": 2,
        "keywords": "customer_relationship, business_connection",
        "description_template": "{entity1} has a business relationship with {entity2} (co-occurs {count} times)",
    },

    # Partner-Organization relationships (when Partner is an entity type)
    ("Partner", "Organization"): {
        "min_cooccurrence": 2,
        "keywords": "partnership, collaboration",
        "description_template": "{entity1} collaborates with {entity2} (co-occurs {count} times)",
    },
}


def get_inference_rule(type1: str, type2: str) -> Optional[Dict]:
    """
    Get inference rule for a pair of entity types.

    Returns None if no rule exists (meaning we should NOT infer this relationship).
    """
    # Normalize types
    type1 = type1.strip() if type1 else "unknown"
    type2 = type2.strip() if type2 else "unknown"

    # Try both orderings
    if (type1, type2) in INFERENCE_RULES:
        return INFERENCE_RULES[(type1, type2)]
    if (type2, type1) in INFERENCE_RULES:
        return INFERENCE_RULES[(type2, type1)]

    return None


def should_infer_relationship(
    entity1_name: str,
    entity2_name: str,
    entity1_type: str,
    entity2_type: str,
    cooccurrence_count: int,
    rule: Dict,
) -> bool:
    """
    Apply quality filters to determine if a relationship should be inferred.

    Returns False if the relationship would be low-quality "trash".
    """
    # Filter 1: Check type-specific minimum co-occurrence
    min_cooccur = rule.get("min_cooccurrence", 3)
    if cooccurrence_count < min_cooccur:
        return False

    # Filter 2: Reject generic entity names (likely extraction errors)
    generic_names = {"company", "person", "organization", "entity", "unknown", "other"}
    if (entity1_name.lower() in generic_names or
        entity2_name.lower() in generic_names):
        return False

    # Filter 3: Reject very short entity names (likely incomplete extractions)
    if len(entity1_name) < 2 or len(entity2_name) < 2:
        return False

    # Filter 4: For Organization-Organization, be extra conservative
    if (entity1_type == "Organization" and entity2_type == "Organization"):
        # Require even higher threshold for org-org relationships
        if cooccurrence_count < 4:
            return False

    # Filter 5: For Person-Person, be extremely conservative
    if (entity1_type == "Person" and entity2_type == "Person"):
        # Only infer if very strong evidence
        if cooccurrence_count < 5:
            return False

    return True

def infer_cooccurrence_relationships(
    all_nodes: Dict[str, List[Dict]],
    all_edges: Dict[Tuple[str, str], List[Dict]],
    min_cooccurrence: int = 3,  # Increased from 2 for higher quality
    enable_inference: bool = True,
) -> Dict[Tuple[str, str], List[Dict]]:
    """
    HIGH-QUALITY relationship inference based on entity co-occurrence.

    Unlike basic co-occurrence, this function:
    1. Uses entity-type-specific inference rules (INFERENCE_RULES)
    2. Requires higher co-occurrence thresholds per entity type
    3. Creates specific relationship types based on entity types
    4. Applies quality filters to prevent "trash" relationships
    5. Only infers relationships that are LIKELY TO BE TRUE
    6. Marks inferred relationships clearly for transparency

    Args:
        all_nodes: Dictionary of entity names to entity node data
        all_edges: Existing relationships (edges) between entities
        min_cooccurrence: Global minimum co-occurrence threshold (default: 3, was 2)
        enable_inference: Whether to enable inference (can be disabled via config)

    Returns:
        Updated relationships dictionary with high-quality inferred relationships
    """
    if not enable_inference:
        return all_edges

    from .utils import logger

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

    # Infer HIGH-QUALITY relationships using type-specific rules
    inferred_count = 0
    skipped_low_quality = 0
    skipped_no_rule = 0
    updated_edges = dict(all_edges)

    inference_stats = defaultdict(int)  # Track inference by type pair

    for entity1 in sorted(all_nodes.keys()):
        if entity1 not in cooccurrence_map:
            continue

        for entity2, chunk_ids in cooccurrence_map[entity1].items():
            count = len(chunk_ids)

            # Get entity types
            entity1_type = all_nodes[entity1][0].get("entity_type", "unknown")
            entity2_type = all_nodes[entity2][0].get("entity_type", "unknown")

            # Check if we have an inference rule for this entity type pair
            rule = get_inference_rule(entity1_type, entity2_type)
            if rule is None:
                # No inference rule = don't infer this relationship type
                skipped_no_rule += 1
                continue

            # Apply quality filters
            if not should_infer_relationship(
                entity1, entity2,
                entity1_type, entity2_type,
                count, rule
            ):
                skipped_low_quality += 1
                continue

            # Create sorted edge key for undirected graph
            edge_key = tuple(sorted([entity1, entity2]))

            # Don't override explicit relationships from LLM extraction
            if edge_key in updated_edges:
                continue

            # Create HIGH-QUALITY inferred relationship with specific keywords
            description = rule["description_template"].format(
                entity1=entity1,
                entity2=entity2,
                count=count,
            )

            inferred_rel = {
                "src_id": entity1,
                "tgt_id": entity2,
                "weight": count,  # Weight by co-occurrence count
                "keywords": rule["keywords"],  # Type-specific keywords (not generic)
                "description": description,
                "source_id": f"inferred_cooccurrence",
                "inferred": True,
                "inference_method": "high_quality_type_based",
                "cooccurrence_count": count,
                "entity_type_pair": f"{entity1_type}-{entity2_type}",
            }

            updated_edges[edge_key] = [inferred_rel]
            inferred_count += 1

            # Track statistics
            type_pair_key = f"{entity1_type}-{entity2_type}"
            inference_stats[type_pair_key] += 1

    # Log detailed statistics
    if inferred_count > 0:
        logger.info(
            f"  ✓ HIGH-QUALITY INFERENCE: Added {inferred_count} relationship(s)"
        )
        logger.info(f"    (Skipped {skipped_low_quality} low-quality, {skipped_no_rule} without rules)")

        # Show breakdown by entity type pairs
        if len(inference_stats) > 0:
            logger.info(f"    Inference breakdown by entity types:")
            for type_pair, cnt in sorted(inference_stats.items(), key=lambda x: -x[1]):
                logger.info(f"      • {type_pair}: {cnt} relationship(s)")
    elif skipped_low_quality > 0 or skipped_no_rule > 0:
        logger.info(
            f"  ℹ️  No high-quality relationships inferred "
            f"(skipped {skipped_low_quality} low-quality, {skipped_no_rule} without rules)"
        )

    return updated_edges

