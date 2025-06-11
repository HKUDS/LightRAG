"""
Chunk-level relationship post-processing implementation for LightRAG.

This module provides functionality to validate and refine relationships immediately
after they are extracted from each text chunk, improving both accuracy and performance
compared to document-level post-processing.
"""

from __future__ import annotations
import asyncio
import json
import re
from typing import Any, Dict, List
from collections import defaultdict
import logging

from lightrag.utils import use_llm_func_with_cache

logger = logging.getLogger(__name__)

# Chunk-level relationship validation prompt
CHUNK_RELATIONSHIP_VALIDATION_PROMPT = """You are validating extracted relationships from a specific text chunk.

CHUNK CONTENT:
{chunk_content}

EXTRACTED RELATIONSHIPS:
{relationships_json}

TASK:
For each relationship, analyze if it's supported by the chunk content:
1. KEEP with current weight if clearly supported by the text
2. ADJUST weight (0.1-1.0) based on evidence strength in the chunk
3. MODIFY description for clarity if needed based on chunk context
4. REMOVE if not supported by chunk content

Weight Guidelines:
- 0.9-1.0: Explicitly stated, direct relationship
- 0.7-0.8: Strongly implied by context
- 0.5-0.6: Moderately supported by evidence
- 0.1-0.4: Weakly supported, consider removing
- Below 0.1: Remove relationship

Return JSON with this exact structure:
{{
    "validated_relationships": [
        {{
            "src_id": "entity1",
            "tgt_id": "entity2", 
            "rel_type": "relationship_type",
            "weight": 0.85,
            "description": "Updated description based on chunk",
            "keywords": "relevant, keywords, from, chunk",
            "action": "KEEP",
            "reason": "Brief explanation of decision"
        }}
    ],
    "summary": {{
        "kept": 2,
        "adjusted": 1, 
        "modified": 1,
        "removed": 1,
        "total_processed": 5
    }}
}}

Important: Only include relationships that you are KEEPING or MODIFYING. Do not include removed relationships in the output."""


def clean_llm_response(response: str) -> str:
    """
    Clean LLM response by removing markdown code blocks and extracting JSON content.

    Args:
        response: Raw LLM response string

    Returns:
        Cleaned JSON string
    """
    # Remove markdown code blocks
    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```\s*", "", response)

    # Find JSON content between braces
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return response.strip()


def validate_relationship_schema(rel: Dict[str, Any]) -> bool:
    """
    Validate that a relationship has all required fields with correct types.

    Args:
        rel: Relationship dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["src_id", "tgt_id", "rel_type", "weight", "description"]

    # Check required fields exist
    if not all(field in rel for field in required_fields):
        return False

    # Validate types
    if not isinstance(rel.get("weight"), (int, float)):
        return False

    if not isinstance(rel.get("src_id"), str) or not isinstance(rel.get("tgt_id"), str):
        return False

    if not isinstance(rel.get("rel_type"), str) or not isinstance(
        rel.get("description"), str
    ):
        return False

    # Validate weight range
    weight = float(rel["weight"])
    if not (0.0 <= weight <= 1.0):
        return False

    return True


def normalize_relationship_data(
    rel: Dict[str, Any], original_rel: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalize and complete relationship data using original relationship as template.

    Args:
        rel: Validated relationship from LLM
        original_rel: Original relationship with all fields

    Returns:
        Complete relationship dictionary
    """
    # Start with original relationship to preserve all fields
    normalized = original_rel.copy()

    # Update with validated fields
    normalized.update(
        {
            "src_id": str(rel["src_id"]),
            "tgt_id": str(rel["tgt_id"]),
            "rel_type": str(rel["rel_type"]),
            "weight": float(rel["weight"]),
            "description": str(rel["description"]),
            "keywords": rel.get("keywords", original_rel.get("keywords", "")),
        }
    )

    return normalized


def merge_validated_relationships(
    original_edges: defaultdict,
    validated_relationships: List[Dict[str, Any]],
    chunk_key: str,
    log_changes: bool = False,
) -> defaultdict:
    """
    Merge validated relationships back into the original edges structure.

    Args:
        original_edges: Original defaultdict of relationships
        validated_relationships: List of validated relationships from LLM
        chunk_key: Chunk identifier for logging

    Returns:
        Updated defaultdict with validated relationships
    """
    result_edges = defaultdict(list)

    # Create lookup for validated relationships
    validated_lookup = {}
    for rel in validated_relationships:
        key = (rel["src_id"], rel["tgt_id"])
        validated_lookup[key] = rel

    # Process original relationships
    stats = {"kept": 0, "modified": 0, "removed": 0}

    for edge_key, edge_list in original_edges.items():
        for original_rel in edge_list:
            lookup_key = (original_rel["src_id"], original_rel["tgt_id"])

            if lookup_key in validated_lookup:
                # Use validated version
                validated_rel = validated_lookup[lookup_key]
                normalized_rel = normalize_relationship_data(
                    validated_rel, original_rel
                )
                result_edges[edge_key].append(normalized_rel)

                # Determine if modified and log changes if requested
                weight_changed = (
                    abs(float(original_rel["weight"]) - float(validated_rel["weight"]))
                    > 0.05
                )
                description_changed = (
                    original_rel["description"] != validated_rel["description"]
                )

                if weight_changed or description_changed:
                    stats["modified"] += 1
                    if log_changes:
                        changes = []
                        if weight_changed:
                            changes.append(
                                f"weight: {original_rel['weight']:.2f} → {validated_rel['weight']:.2f}"
                            )
                        if description_changed:
                            changes.append(
                                f"description: '{original_rel['description'][:50]}...' → '{validated_rel['description'][:50]}...'"
                            )
                        reason = validated_rel.get("reason", "No reason provided")
                        logger.info(
                            f"Chunk {chunk_key}: Modified {original_rel['src_id']} → {original_rel['tgt_id']}: {', '.join(changes)} (Reason: {reason})"
                        )
                else:
                    stats["kept"] += 1
            else:
                # Relationship was removed
                stats["removed"] += 1
                logger.debug(
                    f"Removed relationship: {original_rel['src_id']} -> {original_rel['tgt_id']}"
                )

    logger.info(
        f"Chunk {chunk_key}: Kept {stats['kept']}, Modified {stats['modified']}, Removed {stats['removed']}"
    )
    return result_edges


async def _post_process_chunk_relationships(
    chunk_content: str,
    maybe_edges: defaultdict,
    chunk_entities: Dict[str, Any],
    llm_func: callable,
    chunk_key: str,
    global_config: Dict[str, Any],
) -> defaultdict:
    """
    Post-process relationships for a single chunk using LLM validation.

    This function validates relationships immediately after extraction from a chunk,
    using the chunk content as context for accurate validation.

    Args:
        chunk_content: The text content of the chunk
        maybe_edges: defaultdict containing extracted relationships
        chunk_entities: Dictionary of entities extracted from this chunk
        llm_func: LLM function for validation
        chunk_key: Unique identifier for the chunk
        global_config: Configuration dictionary

    Returns:
        defaultdict with validated relationships (same structure as input)
    """
    # Check if chunk post-processing is enabled
    if not global_config.get("enable_chunk_post_processing", False):
        return maybe_edges

    # Check if there are relationships to process
    if not maybe_edges:
        logger.debug(f"Chunk {chunk_key}: No relationships to validate")
        return maybe_edges

    total_relationships = sum(len(edge_list) for edge_list in maybe_edges.values())
    max_batch_size = global_config.get("chunk_validation_batch_size", 50)

    # Skip processing if too many relationships for single batch
    if total_relationships > max_batch_size:
        logger.warning(
            f"Chunk {chunk_key}: {total_relationships} relationships exceed batch size {max_batch_size}, skipping validation"
        )
        return maybe_edges

    logger.info(f"Chunk {chunk_key}: Validating {total_relationships} relationships")

    try:
        # Convert relationships to list format for LLM processing
        relationships_list = []
        for edge_key, edge_list in maybe_edges.items():
            for rel in edge_list:
                relationships_list.append(
                    {
                        "src_id": rel["src_id"],
                        "tgt_id": rel["tgt_id"],
                        "rel_type": rel["rel_type"],
                        "weight": rel["weight"],
                        "description": rel["description"],
                        "keywords": rel.get("keywords", ""),
                    }
                )

        # Prepare prompt
        relationships_json = json.dumps(relationships_list, indent=2)
        validation_prompt = CHUNK_RELATIONSHIP_VALIDATION_PROMPT.format(
            chunk_content=chunk_content[:2000],  # Limit chunk content size
            relationships_json=relationships_json,
        )

        # Call LLM with timeout
        timeout = global_config.get("chunk_validation_timeout", 30)

        # Check if post-processing cache is enabled
        llm_response_cache = global_config.get("llm_response_cache")
        enable_cache = global_config.get("enable_llm_cache_for_post_process", True)

        # Diagnostic logging
        logger.info(
            f"Chunk {chunk_key}: Cache diagnostic - llm_response_cache exists: {llm_response_cache is not None}, enable_cache: {enable_cache}"
        )
        if llm_response_cache is None:
            logger.warning(
                f"Chunk {chunk_key}: llm_response_cache is None - cache disabled"
            )

        if llm_response_cache and enable_cache:
            # Use cached LLM call
            logger.info(
                f"Chunk {chunk_key}: Checking post-processing cache for {total_relationships} relationships"
            )
            logger.debug(f"DEBUG: cache_type=post_process, enable_cache={enable_cache}")
            llm_response = await asyncio.wait_for(
                use_llm_func_with_cache(
                    validation_prompt,
                    llm_func,
                    llm_response_cache=llm_response_cache,
                    cache_type="post_process",
                ),
                timeout=timeout,
            )
        else:
            # Direct LLM call without caching
            llm_response = await asyncio.wait_for(
                llm_func(validation_prompt), timeout=timeout
            )

        # Parse and validate LLM response
        cleaned_response = clean_llm_response(llm_response)
        validation_result = json.loads(cleaned_response)

        # Validate response structure
        if "validated_relationships" not in validation_result:
            raise ValueError("Missing 'validated_relationships' field in LLM response")

        validated_relationships = validation_result["validated_relationships"]

        # Validate each relationship
        valid_relationships = []
        # Create a set of entity names from chunk_entities for faster lookup
        chunk_entity_names = set()
        if isinstance(chunk_entities, dict):
            for entity_list in chunk_entities.values():
                if isinstance(entity_list, list):
                    for entity in entity_list:
                        if isinstance(entity, dict) and "entity_name" in entity:
                            chunk_entity_names.add(entity["entity_name"])

        for rel in validated_relationships:
            if validate_relationship_schema(rel):
                # Check that entities exist in chunk (if we have entity data)
                if not chunk_entity_names or (
                    rel["src_id"] in chunk_entity_names
                    and rel["tgt_id"] in chunk_entity_names
                ):
                    valid_relationships.append(rel)
                else:
                    logger.debug(
                        f"Chunk {chunk_key}: Skipping relationship with unknown entities: {rel['src_id']} -> {rel['tgt_id']}"
                    )
            else:
                logger.debug(f"Chunk {chunk_key}: Skipping invalid relationship: {rel}")

        # Merge validated relationships back
        log_changes = global_config.get("log_validation_changes", False)
        validated_edges = merge_validated_relationships(
            maybe_edges, valid_relationships, chunk_key, log_changes
        )

        # Log summary
        if "summary" in validation_result:
            summary = validation_result["summary"]
            logger.info(f"Chunk {chunk_key}: LLM summary - {summary}")

        return validated_edges

    except asyncio.TimeoutError:
        logger.warning(
            f"Chunk {chunk_key}: Validation timed out after {timeout}s, using original relationships"
        )
        return maybe_edges

    except json.JSONDecodeError as e:
        logger.warning(f"Chunk {chunk_key}: Failed to parse LLM response as JSON: {e}")
        if global_config.get("log_validation_changes", False):
            logger.debug(f"Chunk {chunk_key}: Raw LLM response: {llm_response}")
        return maybe_edges

    except Exception as e:
        logger.warning(f"Chunk {chunk_key}: Validation failed with error: {e}")
        if global_config.get("log_validation_changes", False):
            logger.debug(f"Chunk {chunk_key}: Full error details", exc_info=True)
        return maybe_edges


def cleanup_orphaned_entities(
    all_nodes: defaultdict, all_edges: defaultdict, log_changes: bool = False
) -> defaultdict:
    """
    Remove entities that have no relationships after chunk post-processing.

    Args:
        all_nodes: defaultdict containing all entities
        all_edges: defaultdict containing all relationships
        log_changes: Whether to log detailed changes

    Returns:
        defaultdict with orphaned entities removed
    """
    # Collect all entity IDs that are referenced in relationships
    referenced_entities = set()

    for edge_list in all_edges.values():
        for edge in edge_list:
            referenced_entities.add(edge["src_id"])
            referenced_entities.add(edge["tgt_id"])

    # Create new nodes dict with only referenced entities
    cleaned_nodes = defaultdict(list)
    removed_count = 0
    kept_count = 0

    for entity_name, entity_list in all_nodes.items():
        if entity_name in referenced_entities:
            cleaned_nodes[entity_name] = entity_list
            kept_count += 1
        else:
            removed_count += 1
            if log_changes:
                logger.debug(f"Removed orphaned entity: {entity_name}")

    if removed_count > 0:
        logger.info(
            f"Entity cleanup: Kept {kept_count}, Removed {removed_count} orphaned entities"
        )
    else:
        logger.debug(f"Entity cleanup: All {kept_count} entities have relationships")

    return cleaned_nodes
