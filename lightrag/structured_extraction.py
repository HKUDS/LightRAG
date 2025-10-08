"""
Enhanced extraction functions using Pydantic structured outputs.
This module provides extraction functions that leverage Pydantic schemas
for reliable and validated entity and relationship extraction.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Dict, Tuple, Optional
from pydantic import ValidationError

from lightrag.pydantic_schemas import Entity, Relationship, ExtractionResult
from lightrag.prompt import PROMPTS
from lightrag.utils import sanitize_and_normalize_extracted_text

logger = logging.getLogger("lightrag")


async def extract_entities_relationships_structured(
    text: str,
    chunk_key: str,
    timestamp: int,
    file_path: str,
    entity_types: list,
    language: str,
    llm_func,
    tuple_delimiter: str = "<|#|>",
    completion_delimiter: str = "<|COMPLETE|>",
    max_gleaning: int = 1,
    **llm_kwargs
) -> Tuple[Dict, Dict]:
    """
    Extract entities and relationships using structured Pydantic output.
    
    Args:
        text: Input text to extract from
        chunk_key: Unique identifier for this text chunk
        timestamp: Extraction timestamp
        file_path: Source file path
        entity_types: List of allowed entity types
        language: Output language
        llm_func: LLM function to use
        tuple_delimiter: Delimiter for backward compatibility
        completion_delimiter: Completion signal
        max_gleaning: Number of additional extraction passes
        **llm_kwargs: Additional arguments for LLM
        
    Returns:
        Tuple of (nodes_dict, edges_dict) with extracted entities and relationships
    """
    # Format entity types
    entity_types_str = ", ".join(entity_types)
    
    # Build extraction prompt
    system_prompt = PROMPTS["entity_extraction_system_prompt"].format(
        entity_types=entity_types_str,
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
        language=language,
        examples=""  # You can add examples if needed
    )
    
    user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
        completion_delimiter=completion_delimiter,
        language=language
    )
    
    # Add the actual text to extract from
    full_prompt = f"{user_prompt}\n\nText to analyze:\n```\n{text}\n```"
    
    # Initialize storage
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    
    try:
        # Call LLM with structured output
        logger.info(f"{chunk_key}: Extracting entities and relationships with structured output")
        
        result = await llm_func(
            full_prompt,
            system_prompt=system_prompt,
            entity_extraction=True,  # Signal to use ExtractionResult schema
            **llm_kwargs
        )
        
        # If result is ExtractionResult (Pydantic model), process it
        if isinstance(result, ExtractionResult):
            logger.info(
                f"{chunk_key}: Successfully extracted {len(result.entities)} entities "
                f"and {len(result.relationships)} relationships"
            )
            
            # Process entities
            for entity in result.entities:
                entity_data = {
                    "entity_name": entity.entity_name,
                    "entity_type": entity.entity_type,
                    "description": entity.entity_description,
                    "source_id": chunk_key,
                    "file_path": file_path,
                    "timestamp": timestamp,
                }
                maybe_nodes[entity.entity_name].append(entity_data)
            
            # Process relationships
            for relationship in result.relationships:
                # Ensure both entities exist
                if (relationship.source_entity in maybe_nodes or 
                    relationship.target_entity in maybe_nodes):
                    
                    relationship_data = {
                        "src_id": relationship.source_entity,
                        "tgt_id": relationship.target_entity,
                        "keywords": relationship.relationship_keywords,
                        "description": relationship.relationship_description,
                        "source_id": chunk_key,
                        "file_path": file_path,
                        "timestamp": timestamp,
                        "weight": 1.0,
                    }
                    edge_key = (relationship.source_entity, relationship.target_entity)
                    maybe_edges[edge_key].append(relationship_data)
                else:
                    logger.warning(
                        f"{chunk_key}: Skipping relationship {relationship.source_entity} -> "
                        f"{relationship.target_entity} (entities not found)"
                    )
        
        else:
            # Fallback: result is string, parse with legacy method
            logger.warning(
                f"{chunk_key}: Received string instead of structured output, "
                "falling back to legacy parsing"
            )
            maybe_nodes, maybe_edges = await _parse_legacy_format(
                result, chunk_key, timestamp, file_path, tuple_delimiter, completion_delimiter
            )
        
        # Perform gleaning (additional extraction passes) if configured
        for gleaning_round in range(max_gleaning):
            logger.info(f"{chunk_key}: Gleaning round {gleaning_round + 1}/{max_gleaning}")
            
            gleaning_prompt = PROMPTS.get(
                "entity_continue_extraction_user_prompt",
                "Review the text and extract any missed entities or relationships."
            ).format(
                tuple_delimiter=tuple_delimiter,
                completion_delimiter=completion_delimiter,
                language=language
            )
            
            gleaning_result = await llm_func(
                f"{gleaning_prompt}\n\nText:\n```\n{text}\n```",
                system_prompt=system_prompt,
                entity_extraction=True,
                **llm_kwargs
            )
            
            if isinstance(gleaning_result, ExtractionResult):
                # Add newly found entities/relationships
                for entity in gleaning_result.entities:
                    if entity.entity_name not in maybe_nodes:
                        entity_data = {
                            "entity_name": entity.entity_name,
                            "entity_type": entity.entity_type,
                            "description": entity.entity_description,
                            "source_id": chunk_key,
                            "file_path": file_path,
                            "timestamp": timestamp,
                        }
                        maybe_nodes[entity.entity_name].append(entity_data)
                
                for relationship in gleaning_result.relationships:
                    edge_key = (relationship.source_entity, relationship.target_entity)
                    if edge_key not in maybe_edges:
                        relationship_data = {
                            "src_id": relationship.source_entity,
                            "tgt_id": relationship.target_entity,
                            "keywords": relationship.relationship_keywords,
                            "description": relationship.relationship_description,
                            "source_id": chunk_key,
                            "file_path": file_path,
                            "timestamp": timestamp,
                            "weight": 1.0,
                        }
                        maybe_edges[edge_key].append(relationship_data)
        
        logger.info(
            f"{chunk_key}: Extraction complete - {len(maybe_nodes)} unique entities, "
            f"{len(maybe_edges)} unique relationships"
        )
        
        return dict(maybe_nodes), dict(maybe_edges)
        
    except Exception as e:
        logger.error(f"{chunk_key}: Extraction failed with error: {e}")
        # Return empty results on failure
        return {}, {}


async def _parse_legacy_format(
    result: str,
    chunk_key: str,
    timestamp: int,
    file_path: str,
    tuple_delimiter: str,
    completion_delimiter: str
) -> Tuple[Dict, Dict]:
    """
    Fallback parser for legacy string-based extraction format.
    
    This maintains backward compatibility with the original delimiter-based format.
    """
    from lightrag.operate import split_string_by_multi_markers, fix_tuple_delimiter_corruption
    
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    
    if completion_delimiter not in result:
        logger.warning(f"{chunk_key}: Completion delimiter not found in result")
    
    # Split by newlines
    records = split_string_by_multi_markers(
        result,
        ["\n", completion_delimiter, completion_delimiter.lower()],
    )
    
    for record in records:
        record = record.strip()
        if not record:
            continue
        
        # Fix delimiter corruption
        delimiter_core = tuple_delimiter[2:-2]
        record = fix_tuple_delimiter_corruption(record, delimiter_core, tuple_delimiter)
        
        # Split by tuple delimiter
        parts = record.split(tuple_delimiter)
        
        # Parse entity
        if len(parts) == 4 and parts[0].strip().lower() == "entity":
            entity_name = sanitize_and_normalize_extracted_text(parts[1], remove_inner_quotes=True)
            entity_type = sanitize_and_normalize_extracted_text(parts[2], remove_inner_quotes=True)
            entity_type = entity_type.replace(" ", "").lower()
            entity_desc = sanitize_and_normalize_extracted_text(parts[3])
            
            if entity_name and entity_type and entity_desc:
                maybe_nodes[entity_name].append({
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "description": entity_desc,
                    "source_id": chunk_key,
                    "file_path": file_path,
                    "timestamp": timestamp,
                })
        
        # Parse relationship
        elif len(parts) == 5 and parts[0].strip().lower() in ["relation", "relationship"]:
            src = sanitize_and_normalize_extracted_text(parts[1], remove_inner_quotes=True)
            tgt = sanitize_and_normalize_extracted_text(parts[2], remove_inner_quotes=True)
            keywords = sanitize_and_normalize_extracted_text(parts[3])
            desc = sanitize_and_normalize_extracted_text(parts[4])
            
            if src and tgt and keywords and desc:
                maybe_edges[(src, tgt)].append({
                    "src_id": src,
                    "tgt_id": tgt,
                    "keywords": keywords,
                    "description": desc,
                    "source_id": chunk_key,
                    "file_path": file_path,
                    "timestamp": timestamp,
                    "weight": 1.0,
                })
    
    return dict(maybe_nodes), dict(maybe_edges)


async def extract_keywords_structured(
    query: str,
    llm_func,
    **llm_kwargs
) -> Tuple[list, list]:
    """
    Extract high-level and low-level keywords using structured output.
    
    Args:
        query: User query to extract keywords from
        llm_func: LLM function to use
        **llm_kwargs: Additional arguments for LLM
        
    Returns:
        Tuple of (high_level_keywords, low_level_keywords)
    """
    prompt = PROMPTS.get("keywords_extraction", """
Extract keywords from this query for a RAG system.
Return high-level concepts/themes and low-level specific entities/details.

Query: {query}
""").format(query=query)
    
    try:
        result = await llm_func(
            prompt,
            system_prompt="You are an expert at extracting keywords for information retrieval.",
            keyword_extraction=True,  # Signal to use KeywordExtraction schema
            **llm_kwargs
        )
        
        if hasattr(result, 'high_level_keywords') and hasattr(result, 'low_level_keywords'):
            logger.debug(
                f"Extracted keywords: {len(result.high_level_keywords)} high-level, "
                f"{len(result.low_level_keywords)} low-level"
            )
            return result.high_level_keywords, result.low_level_keywords
        else:
            logger.warning("Structured keyword extraction failed, using fallback")
            return [], []
            
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        return [], []