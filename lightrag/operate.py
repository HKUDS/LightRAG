from __future__ import annotations
from functools import partial

import asyncio
import json
import re
import os
from typing import Any, AsyncIterator
from collections import Counter, defaultdict

from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    Tokenizer,
    normalize_extracted_info,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    get_conversation_turns,
    use_llm_func_with_cache,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .validation import (
    DocumentValidator,
    EntityValidator,
    RelationshipValidator,
    DatabaseValidator,
    validate_extraction_results,
    log_validation_errors,
)
from .monitoring import (
    get_performance_monitor,
    get_processing_monitor,
    get_enhanced_logger,
)
import time
from dotenv import load_dotenv
from lightrag.kg.utils.relationship_registry import standardize_relationship_type
from .chunk_post_processor import _post_process_chunk_relationships
from .constants import (
    DEFAULT_ENABLE_CHUNK_POST_PROCESSING,
    DEFAULT_ENABLE_ENTITY_CLEANUP,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> list[dict[str, Any]]:
    # Validate input content
    validation_result = DocumentValidator.validate_content(content)
    if not validation_result.is_valid:
        logger.error("Content validation failed during chunking")
        log_validation_errors(validation_result.errors, "chunking")
        # Use original content but log the issues

    # Use sanitized content if available
    if validation_result.sanitized_data:
        content = validation_result.sanitized_data["content"]
        logger.debug(
            f"Using sanitized content: {validation_result.sanitized_data['original_length']} -> {validation_result.sanitized_data['sanitized_length']} chars"
        )

    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) <= max_token_size:
                    new_chunks.append((len(_tokens), chunk))
                else:
                    # If chunk is still too large, split it recursively
                    sub_chunks = chunking_by_token_size(
                        tokenizer,
                        chunk,
                        None,
                        False,
                        overlap_token_size,
                        max_token_size,
                    )
                    for sub_chunk in sub_chunks:
                        new_chunks.append(
                            (
                                len(tokenizer.encode(sub_chunk["content"])),
                                sub_chunk["content"],
                            )
                        )
    else:
        new_chunks = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            end = min(start + max_token_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_content = tokenizer.decode(chunk_tokens)
            if chunk_content.strip():
                new_chunks.append((len(chunk_tokens), chunk_content))

    for token_count, chunk_content in new_chunks:
        # Validate each chunk
        chunk_data = {
            "content": chunk_content,
            "tokens": token_count,
        }

        chunk_validation = DocumentValidator.validate_chunk(chunk_data)
        if chunk_validation.sanitized_data:
            results.append(chunk_validation.sanitized_data)
        else:
            # Fall back to original data if validation fails
            results.append(chunk_data)
            if chunk_validation.has_errors():
                log_validation_errors(chunk_validation.errors, f"chunk_{len(results)}")

    if not results:
        # Final fallback
        chunk_data = {
            "content": content,
            "tokens": len(tokens),
        }
        chunk_validation = DocumentValidator.validate_chunk(chunk_data)
        if chunk_validation.sanitized_data:
            results.append(chunk_validation.sanitized_data)
        else:
            results.append(chunk_data)

    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
) -> str:
    """Handle entity relation summary
    For each entity or relation, input is the combined description of already existing description and new description.
    If too long, use LLM to summarize.
    """
    use_llm_func: callable = global_config["llm_model_func"]
    # Apply higher priority (8) to entity/relation summary tasks
    use_llm_func = partial(use_llm_func, _priority=8)

    tokenizer: Tokenizer = global_config["tokenizer"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    summary_max_tokens = global_config["summary_to_max_tokens"]

    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = tokenizer.encode(description)

    ### summarize is not determined here anymore (It's determined by num_fragment now)
    # if len(tokens) < summary_max_tokens:  # No need for summary
    #     return description

    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = tokenizer.decode(tokens[:llm_max_tokens])
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")

    # Use LLM function with cache (higher priority for summary generation)
    summary = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        max_tokens=summary_max_tokens,
        cache_type="extract",
    )
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
):
    if len(record_attributes) < 4 or '"entity"' not in record_attributes[0]:
        return None

    # Clean and validate entity name
    entity_name = clean_str(record_attributes[1]).strip()
    if not entity_name:
        logger.warning(
            f"Entity extraction error: empty entity name in: {record_attributes}"
        )
        return None

    # Normalize entity name
    entity_name = normalize_extracted_info(entity_name, is_entity=True)

    # Clean and validate entity type
    entity_type = clean_str(record_attributes[2]).strip('"')
    if not entity_type.strip() or entity_type.startswith('("'):
        logger.warning(
            f"Entity extraction error: invalid entity type in: {record_attributes}"
        )
        return None

    # Clean and validate description
    entity_description = clean_str(record_attributes[3])
    entity_description = normalize_extracted_info(entity_description)

    if not entity_description.strip():
        logger.warning(
            f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
        )
        return None

    # Create entity data structure
    entity_data = dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=chunk_key,
        file_path=file_path,
    )

    # Validate and sanitize entity data
    validation_result = EntityValidator.validate_entity(entity_data)

    if validation_result.has_errors():
        logger.warning(
            f"Entity validation failed for '{entity_name}': {[e.message for e in validation_result.errors]}"
        )
        return None

    if validation_result.has_warnings():
        logger.debug(
            f"Entity validation warnings for '{entity_name}': {[w.message for w in validation_result.warnings]}"
        )

    # Return sanitized data if available, otherwise original
    return (
        validation_result.sanitized_data
        if validation_result.sanitized_data
        else entity_data
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
):
    logger.debug(
        f"Attempting to parse relationship record: {record_attributes} from chunk {chunk_key}"
    )

    # Check if this is a content_keywords record (expected and should be ignored)
    if len(record_attributes) >= 1 and '"content_keywords"' in record_attributes[0]:
        logger.debug(
            f"Skipping content_keywords record: {record_attributes[0] if len(record_attributes) > 0 else 'empty'}"
        )
        return None

    if (
        len(record_attributes) != 7 or '"relationship"' not in record_attributes[0]
    ):  # Strict check for 7 elements
        # Only log as error if it's not a known content_keywords record
        if (
            len(record_attributes) >= 1
            and '"content_keywords"' not in record_attributes[0]
        ):
            logger.warning(
                f"Malformed relationship record (expected 7 attributes, got {len(record_attributes)}): {record_attributes} from chunk {chunk_key}"
            )
        return None

    source = normalize_extracted_info(clean_str(record_attributes[1]), is_entity=True)
    target = normalize_extracted_info(clean_str(record_attributes[2]), is_entity=True)

    if not source or not target:
        logger.warning(
            f"Missing source or target for relationship in chunk {chunk_key}: src='{source}', tgt='{target}'"
        )
        return None
    if source == target:
        logger.debug(
            f"Self-loop relationship skipped for {source} in chunk {chunk_key}"
        )
        return None

    edge_description = normalize_extracted_info(clean_str(record_attributes[3]))
    raw_rel_type = clean_str(record_attributes[4])  # Actual relationship type from LLM
    edge_keywords = normalize_extracted_info(
        clean_str(record_attributes[5]), is_entity=False
    )  # Actual keywords
    edge_keywords = edge_keywords.replace("，", ",")  # Normalize Chinese comma

    raw_strength_str = record_attributes[6].strip('"').strip("'")
    weight = 0.5  # Default weight
    try:
        extracted_weight = float(raw_strength_str)
        if 0.0 <= extracted_weight <= 1.0:  # LLM asked for 0-1
            weight = extracted_weight
        elif 0.0 <= extracted_weight <= 10.0:  # LLM might give 0-10
            weight = extracted_weight / 10.0
            logger.debug(
                f"Normalized relationship strength {raw_strength_str} to {weight} for {source}-{target}"
            )
        else:
            logger.warning(
                f"Relationship strength '{raw_strength_str}' for {source}-{target} is out of 0.0-10.0 range. Defaulting to {weight}."
            )
    except ValueError:
        logger.warning(
            f"Invalid relationship strength '{raw_strength_str}' for {source}-{target}. Defaulting to {weight}."
        )

    logger.info(
        f"Parsed relationship from chunk {chunk_key}: {source} -[{raw_rel_type}({weight})]-> {target}, Keywords: '{edge_keywords}'"
    )

    relationship_data = dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        relationship_type=raw_rel_type,  # This is the raw type from LLM
        rel_type=raw_rel_type,  # CRITICAL: This field was missing - needed for LLM post-processing
        keywords=edge_keywords,
        source_id=chunk_key,
        file_path=file_path,
    )

    # Validate and sanitize relationship data
    validation_result = RelationshipValidator.validate_relationship(relationship_data)

    if validation_result.has_errors():
        logger.warning(
            f"Relationship validation failed for '{source}' -> '{target}': {[e.message for e in validation_result.errors]}"
        )
        return None

    if validation_result.has_warnings():
        logger.debug(
            f"Relationship validation warnings for '{source}' -> '{target}': {[w.message for w in validation_result.warnings]}"
        )

    # Return sanitized data if available, otherwise original
    return (
        validation_result.sanitized_data
        if validation_result.sanitized_data
        else relationship_data
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
):
    # Initialize monitoring
    perf_monitor = get_performance_monitor()
    proc_monitor = get_processing_monitor()
    enhanced_logger = get_enhanced_logger("lightrag.node_merge")

    with perf_monitor.measure(
        "merge_nodes_upsert", entity_name=entity_name, nodes_count=len(nodes_data)
    ):
        enhanced_logger.debug(
            f"Merging {len(nodes_data)} nodes for entity: {entity_name}"
        )

        try:
            logger.debug(
                f"Starting node merge for entity: {entity_name} with {len(nodes_data)} data entries"
            )

            # Data validation for input parameters
            if not entity_name or not entity_name.strip():
                logger.error(f"Invalid entity_name provided: '{entity_name}'")
                raise ValueError("Entity name cannot be empty or None")

            if not nodes_data or not isinstance(nodes_data, list):
                logger.error(
                    f"Invalid nodes_data provided for entity '{entity_name}': {type(nodes_data)}"
                )
                raise ValueError("nodes_data must be a non-empty list")

            # Validate that all nodes_data entries have required fields
            for i, node_data in enumerate(nodes_data):
                required_fields = [
                    "entity_type",
                    "description",
                    "source_id",
                    "file_path",
                ]
                for field in required_fields:
                    if field not in node_data:
                        logger.warning(
                            f"Missing required field '{field}' in nodes_data[{i}] for entity '{entity_name}', using default"
                        )
                        # Set default values for missing fields
                        if field == "entity_type":
                            node_data[field] = "UNKNOWN"
                        elif field == "description":
                            node_data[field] = f"Entity: {entity_name}"
                        elif field == "source_id":
                            node_data[field] = "unknown_source"
                        elif field == "file_path":
                            node_data[field] = "unknown_file"

            already_entity_types = []
            already_source_ids = []
            already_description = []
            already_file_paths = []

            already_node = await knowledge_graph_inst.get_node(entity_name)
            if already_node is not None:
                logger.debug(f"Found existing node for entity: {entity_name}")

                # Validate existing node has required fields
                if "entity_type" not in already_node:
                    logger.warning(
                        f"Existing node for '{entity_name}' missing entity_type, using 'UNKNOWN'"
                    )
                    already_node["entity_type"] = "UNKNOWN"

                already_entity_types.append(already_node["entity_type"])

                # Add data validation to prevent KeyError - get source_id with empty string default if missing
                if already_node.get("source_id") is not None:
                    already_source_ids.extend(
                        split_string_by_multi_markers(
                            already_node["source_id"], [GRAPH_FIELD_SEP]
                        )
                    )
                else:
                    logger.debug(
                        f"No source_id found in existing node for entity '{entity_name}'"
                    )

                # Add data validation to prevent KeyError - get file_path with empty string default if missing
                if already_node.get("file_path") is not None:
                    already_file_paths.extend(
                        split_string_by_multi_markers(
                            already_node["file_path"], [GRAPH_FIELD_SEP]
                        )
                    )
                else:
                    logger.debug(
                        f"No file_path found in existing node for entity '{entity_name}'"
                    )

                # Add data validation to prevent KeyError - get description with empty string default if missing
                if already_node.get("description") is not None:
                    already_description.append(already_node["description"])
                else:
                    logger.debug(
                        f"No description found in existing node for entity '{entity_name}'"
                    )
            else:
                logger.debug(f"No existing node found for entity: {entity_name}")

            entity_type = sorted(
                Counter(
                    [dp["entity_type"] for dp in nodes_data] + already_entity_types
                ).items(),
                key=lambda x: x[1],
                reverse=True,
            )[0][0]

            description = GRAPH_FIELD_SEP.join(
                sorted(
                    set([dp["description"] for dp in nodes_data] + already_description)
                )
            )
            source_id = GRAPH_FIELD_SEP.join(
                set([dp["source_id"] for dp in nodes_data] + already_source_ids)
            )
            file_path = GRAPH_FIELD_SEP.join(
                set([dp["file_path"] for dp in nodes_data] + already_file_paths)
            )

            force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

            num_fragment = description.count(GRAPH_FIELD_SEP) + 1
            num_new_fragment = len(set([dp["description"] for dp in nodes_data]))

            if num_fragment > 1:
                if num_fragment >= force_llm_summary_on_merge:
                    status_message = f"LLM merge N: {entity_name} | {num_new_fragment}+{num_fragment-num_new_fragment}"
                    logger.info(status_message)
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)
                    description = await _handle_entity_relation_summary(
                        entity_name,
                        description,
                        global_config,
                        pipeline_status,
                        pipeline_status_lock,
                        llm_response_cache,
                    )
                else:
                    status_message = f"Merge N: {entity_name} | {num_new_fragment}+{num_fragment-num_new_fragment}"
                    logger.info(status_message)
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

            node_data = dict(
                entity_id=entity_name,
                entity_type=entity_type,
                description=description,
                source_id=source_id,
                file_path=file_path,
                created_at=int(time.time()),
            )

            # Validate node data before database upsert
            db_validation_result = DatabaseValidator.validate_node_data(node_data)
            if db_validation_result.has_errors():
                logger.error(
                    f"Database validation failed for node '{entity_name}': {[e.message for e in db_validation_result.errors]}\""
                )
                raise ValueError(f"Node data validation failed for '{entity_name}'")

            if db_validation_result.has_warnings():
                logger.warning(
                    f"Database validation warnings for node '{entity_name}': {[w.message for w in db_validation_result.warnings]}"
                )

            # Monitor database upsert operation
            with perf_monitor.measure("database_upsert_node", entity_name=entity_name):
                try:
                    await knowledge_graph_inst.upsert_node(entity_name, node_data)
                    proc_monitor.record_database_operation(success=True)
                    enhanced_logger.debug(f"Successfully upserted node: {entity_name}")
                except Exception as e:
                    proc_monitor.record_database_operation(success=False)
                    enhanced_logger.error(
                        f"Failed to upsert node {entity_name}: {str(e)}"
                    )
                    raise

            logger.debug(
                f"Successfully upserted node for entity: {entity_name} with type: {entity_type}"
            )

            node_data["entity_name"] = entity_name
            return node_data

        except Exception as e:
            logger.error(
                f"Error in _merge_nodes_then_upsert for entity '{entity_name}': {str(e)}"
            )
            logger.error(f"nodes_data: {nodes_data}")
            # Return a basic node structure to prevent pipeline failure
            basic_node_data = {
                "entity_id": entity_name,
                "entity_name": entity_name,
                "entity_type": "UNKNOWN",
                "description": f"Entity: {entity_name}",
                "source_id": "error_recovery",
                "file_path": "unknown_file",
                "created_at": int(time.time()),
            }

            # Try to upsert the basic node structure
            try:
                await knowledge_graph_inst.upsert_node(
                    entity_name, node_data=basic_node_data
                )
                logger.info(
                    f"Created fallback node for entity '{entity_name}' after error recovery"
                )
                return basic_node_data
            except Exception as fallback_error:
                logger.error(
                    f"Failed to create fallback node for entity '{entity_name}': {str(fallback_error)}"
                )
                raise e  # Re-raise the original exception if fallback fails


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges: list[dict],  # List of edge dicts from extract_entities_with_types
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict[str, Any],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
) -> dict | None:
    """
    Merge and upsert a single edge relationship between two entities.

    Args:
        src_id: Source entity ID
        tgt_id: Target entity ID
        edges: List of edge data dictionaries to merge
        knowledge_graph_inst: Knowledge graph storage instance
        global_config: Global configuration dictionary
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        llm_response_cache: LLM response cache

    Returns:
        Merged edge data dictionary or None if merge fails
    """
    if not edges:
        logger.warning(f"No edges provided to merge for {src_id} -> {tgt_id}")
        return None

    logger.debug(f"Merging {len(edges)} edge instances for {src_id} -> {tgt_id}")

    # Initialize merged_edge with defaults that clearly indicate no specific type yet.
    # We will iterate through all edges to find the best type.
    merged_edge = {
        "src_id": src_id,
        "tgt_id": tgt_id,
        "weight": 0.0,
        "description": "",
        "keywords": [],
        "source_id": [],
        "file_path": [],
        "relationship_type": "related",  # Default human-readable std
        "original_type": "related",  # Default LLM raw
        "neo4j_type": "RELATED",  # Default Neo4j label
        "created_at": int(time.time()),
    }

    # Iterate through all edge instances to merge their properties
    all_original_types = []
    all_neo4j_types = []

    for i, edge_instance in enumerate(edges):
        logger.debug(
            f"Processing instance {i+1}/{len(edges)} for merge {src_id}->{tgt_id}: "
            f"original_type='{edge_instance.get('original_type')}', "
            f"rel_type='{edge_instance.get('relationship_type')}', "  # This is human-readable-std from advanced_operate
            f"neo4j_type='{edge_instance.get('neo4j_type')}'"
        )

        merged_edge["weight"] += float(edge_instance.get("weight", 0.5))

        if edge_instance.get("description"):
            if merged_edge["description"]:
                merged_edge[
                    "description"
                ] += f"{GRAPH_FIELD_SEP}{edge_instance['description']}"
            else:
                merged_edge["description"] = edge_instance["description"]

        # Keywords merging (ensure this is robust)
        new_keywords = edge_instance.get("keywords", [])
        current_keywords_list = merged_edge.get("keywords", [])
        if not isinstance(current_keywords_list, list):
            current_keywords_list = (
                [str(current_keywords_list)] if current_keywords_list else []
            )

        if isinstance(new_keywords, str):
            new_keywords = [kw.strip() for kw in new_keywords.split(",") if kw.strip()]
        elif not isinstance(new_keywords, list):
            new_keywords = [str(new_keywords).strip()] if new_keywords else []

        temp_new_keywords_list = []
        for item_kw in new_keywords:
            if isinstance(item_kw, str):
                temp_new_keywords_list.append(item_kw.strip())
            elif isinstance(
                item_kw, list
            ):  # Should not happen if advanced_operate is correct
                for sub_item_kw in item_kw:
                    temp_new_keywords_list.append(str(sub_item_kw).strip())
            else:
                temp_new_keywords_list.append(str(item_kw).strip())
        current_keywords_list.extend(filter(None, temp_new_keywords_list))
        merged_edge["keywords"] = current_keywords_list

        # source_id merging (robust source_id merging logic)
        new_source_ids = edge_instance.get("source_id", [])
        current_source_ids_list = merged_edge.get("source_id", [])
        if not isinstance(current_source_ids_list, list):
            current_source_ids_list = (
                [str(current_source_ids_list)] if current_source_ids_list else []
            )
        if isinstance(new_source_ids, str):
            new_source_ids = [
                sid.strip()
                for sid in new_source_ids.split(GRAPH_FIELD_SEP)
                if sid.strip()
            ]
        elif not isinstance(new_source_ids, list):
            new_source_ids = [str(new_source_ids).strip()] if new_source_ids else []
        current_source_ids_list.extend(filter(None, new_source_ids))
        merged_edge["source_id"] = current_source_ids_list

        # file_path merging (robust file_path merging logic)
        new_file_paths = edge_instance.get("file_path", [])
        current_file_paths_list = merged_edge.get("file_path", [])
        if not isinstance(current_file_paths_list, list):
            current_file_paths_list = (
                [str(current_file_paths_list)] if current_file_paths_list else []
            )
        if isinstance(new_file_paths, str):
            new_file_paths = [
                fp.strip() for fp in new_file_paths.split(GRAPH_FIELD_SEP) if fp.strip()
            ]
        elif not isinstance(new_file_paths, list):
            new_file_paths = [str(new_file_paths).strip()] if new_file_paths else []
        current_file_paths_list.extend(filter(None, new_file_paths))
        merged_edge["file_path"] = current_file_paths_list

        # Collect all types encountered for this merge group
        if edge_instance.get("original_type"):
            all_original_types.append(edge_instance["original_type"])
        if edge_instance.get("neo4j_type"):
            all_neo4j_types.append(edge_instance["neo4j_type"])

    # Finalize list fields
    merged_edge["keywords"] = list(set(merged_edge["keywords"]))  # Deduplicate
    merged_edge["source_id"] = GRAPH_FIELD_SEP.join(list(set(merged_edge["source_id"])))
    merged_edge["file_path"] = GRAPH_FIELD_SEP.join(list(set(merged_edge["file_path"])))

    # Determine the best type information after iterating all instances
    # Prioritize non-generic, non-None types.
    # If multiple specific types exist, this logic might need further refinement (e.g., most frequent, or manual resolution flag)
    final_original_type = "related"
    final_neo4j_type = "RELATED"

    # Find a specific original_type if one exists
    specific_original_types = [
        ot for ot in all_original_types if ot and ot.lower() != "related"
    ]
    if specific_original_types:
        final_original_type = specific_original_types[
            0
        ]  # Take the first specific one encountered
        logger.debug(
            f"For {src_id}->{tgt_id}, selected specific original_type: '{final_original_type}' from {specific_original_types}"
        )

    # Find a specific neo4j_type if one exists
    specific_neo4j_types = [nt for nt in all_neo4j_types if nt and nt != "RELATED"]
    if specific_neo4j_types:
        final_neo4j_type = specific_neo4j_types[0]  # Take the first specific one
        logger.debug(
            f"For {src_id}->{tgt_id}, selected specific neo4j_type: '{final_neo4j_type}' from {specific_neo4j_types}"
        )
    elif (
        final_original_type != "related"
    ):  # If no specific neo4j_type, but specific original_type, use enhanced standardization
        # Use the enhanced standardize_relationship_type function from the registry
        final_neo4j_type = standardize_relationship_type(final_original_type)
        logger.debug(
            f"For {src_id}->{tgt_id}, enhanced standardization: '{final_original_type}' -> '{final_neo4j_type}'"
        )

    merged_edge["original_type"] = final_original_type
    merged_edge["neo4j_type"] = final_neo4j_type
    merged_edge["relationship_type"] = final_neo4j_type.lower().replace(
        "_", " "
    )  # Human-readable from final Neo4j type
    merged_edge["rel_type"] = merged_edge["relationship_type"]  # Ensure consistency

    # Description summarization logic
    force_llm_summary_on_merge = global_config.get("force_llm_summary_on_merge", 6)
    num_fragment = merged_edge["description"].count(GRAPH_FIELD_SEP) + 1
    if num_fragment > 1 and num_fragment >= force_llm_summary_on_merge:
        merged_edge["description"] = await _handle_entity_relation_summary(
            f"({src_id}, {tgt_id})",
            merged_edge["description"],
            global_config,
            pipeline_status,
            pipeline_status_lock,
            llm_response_cache,
        )

    # Final log before passing to upsert_edge
    logger.info(
        f"Final merged_edge for {src_id}->{tgt_id}: "
        f"neo4j_type='{merged_edge['neo4j_type']}', "
        f"rel_type='{merged_edge['rel_type']}', "
        f"original_type='{merged_edge['original_type']}', "
        f"weight={merged_edge['weight']:.2f}"
    )

    try:
        # Pass the fully populated merged_edge dictionary to upsert_edge
        await knowledge_graph_inst.upsert_edge(src_id, tgt_id, merged_edge)
        logger.debug(f"Successfully upserted edge: {src_id} -> {tgt_id}")

        # Return the merged edge data for vector database updates
        return merged_edge

    except Exception as e:
        logger.error(f"Failed to upsert edge {src_id} -> {tgt_id}: {str(e)}")
        return None


def _calculate_string_similarity(str1: str, str2: str) -> float:
    """
    Calculate string similarity using simple character-based approach.
    Returns similarity ratio between 0 and 1.
    """
    if not str1 or not str2:
        return 0.0

    str1, str2 = str1.lower().strip(), str2.lower().strip()
    if str1 == str2:
        return 1.0

    # Simple approach: count common characters
    common_chars = sum(
        (str1.count(c) == str2.count(c)) and str1.count(c) > 0 for c in set(str1 + str2)
    )
    total_chars = len(set(str1 + str2))

    return common_chars / total_chars if total_chars > 0 else 0.0


def _is_abstract_entity(entity_name: str) -> bool:
    """
    Check if an entity name represents an abstract concept rather than a concrete entity.
    """
    name = entity_name.lower().strip()

    # Only filter the most obviously abstract patterns - be conservative
    ABSTRACT_ENTITY_PATTERNS = [
        r"^(users|user)$",  # Generic user references only
        r"^parallel technical tasks$",  # Specific abstract concept
        r"^business research$",  # Specific abstract concept
        r"^data transformations$",  # Specific abstract concept
    ]

    return any(re.match(pattern, name) for pattern in ABSTRACT_ENTITY_PATTERNS)


# Legacy confidence scoring removed - LLM provides more accurate quality assessment


def _validate_relationship_context(
    src_id: str, tgt_id: str, rel_type: str, description: str
) -> bool:
    """
    Validate relationship context to filter out low-quality or abstract relationships.
    """
    # Filter relationships where entities are too similar
    similarity = _calculate_string_similarity(src_id, tgt_id)
    if similarity > 0.8:
        return False

    # Check if either entity is abstract
    if _is_abstract_entity(src_id) or _is_abstract_entity(tgt_id):
        # Require higher standards for abstract entities
        concrete_indicators = [
            "uses",
            "calls",
            "accesses",
            "creates",
            "debugs",
            "runs_on",
        ]
        return rel_type.lower() in concrete_indicators and len(description) > 50

    # Filter very generic relationships with poor descriptions
    if rel_type.lower() in ["related", "associated_with"] and len(description) < 30:
        return False

    return True


def _apply_relationship_quality_filter(
    all_edges: dict, global_config: dict = None
) -> dict:
    """
    Enhanced post-processing filter with type-specific intelligence and comprehensive metrics.

    Uses data-driven relationship categorization based on actual Neo4j patterns to apply
    type-specific confidence thresholds and validation rules.

    Args:
        all_edges: Dictionary of edge lists keyed by sorted edge tuples

    Returns:
        Filtered dictionary with type-aware relationship filtering
    """
    # Check configuration and import the enhanced classifier
    global_config = global_config or {}

    # Debug: Log what's actually in the config
    logger.debug(
        f"Enhanced filter config check: enable_enhanced_relationship_filter = {global_config.get('enable_enhanced_relationship_filter', 'NOT_FOUND')}"
    )

    enable_enhanced_filter = global_config.get(
        "enable_enhanced_relationship_filter", False
    )  # Default to False
    log_classification = global_config.get("log_relationship_classification", False)
    track_performance = global_config.get(
        "relationship_filter_performance_tracking", True
    )
    monitoring_mode = global_config.get("enhanced_filter_monitoring_mode", False)

    use_enhanced_classification = False
    classifier = None
    enhanced_logger = None

    # Initialize enhanced logging ONLY if enhanced filter is enabled
    if enable_enhanced_filter:
        try:
            from .kg.utils.enhanced_filter_logger import get_enhanced_filter_logger

            console_logging = global_config.get(
                "enhanced_filter_console_logging", False
            )
            enhanced_logger = get_enhanced_filter_logger(
                enable_console_logging=console_logging
            )
        except ImportError:
            enhanced_logger = None

    if enable_enhanced_filter:
        try:
            from .kg.utils.enhanced_relationship_classifier import (
                EnhancedRelationshipClassifier,
            )

            classifier = EnhancedRelationshipClassifier()
            use_enhanced_classification = True
            logger.debug("Using enhanced type-specific relationship filtering")
        except ImportError as e:
            logger.warning(
                f"Enhanced classifier not available, falling back to basic filtering: {e}"
            )
            if enhanced_logger:
                enhanced_logger.log_error(
                    "EnhancedClassifier", e, "Failed to import enhanced classifier"
                )
    else:
        logger.debug("Enhanced relationship filtering disabled by configuration")
        return all_edges  # Return original edges without any filtering

    # Fallback abstract/generic relationship types for basic filtering
    ABSTRACT_RELATIONSHIPS = {
        "implements",
        "supports",
        "enables",
        "involves",
        "includes",
        "contains",
        "related",
        "part_of",
        "applies_to",
        "affects",
        "investigates",
        "analyzes",
        "optimizes",
        "facilitates",
    }

    # Define synonym pairs to detect redundant relationships
    SYNONYM_CONCEPTS = [
        {"web scraping", "data extraction"},
        {"email communication", "gmail"},
        {"client data management", "sail pos"},
        {"information retrieval", "data processing"},
        {"workflow automation", "automation"},
        {"ai assistance", "google gemini chat model"},
        {"screen sharing", "remote collaboration"},
    ]

    filtered_edges = {}
    filter_stats = {
        "abstract_relationships": 0,
        "synonym_relationships": 0,
        "low_quality_relationships": 0,
        "abstract_entities": 0,
        "low_confidence": 0,
        "context_validation": 0,
        "type_specific_filtered": 0,
        "total_before": 0,
        "total_after": 0,
    }

    # Enhanced statistics for type-specific filtering
    if use_enhanced_classification:
        filter_stats.update(
            {
                "technical_core_filtered": 0,
                "development_operations_filtered": 0,
                "system_interactions_filtered": 0,
                "troubleshooting_support_filtered": 0,
                "abstract_conceptual_filtered": 0,
                "data_flow_filtered": 0,
                "category_stats": defaultdict(
                    lambda: {"total": 0, "kept": 0, "filtered": 0}
                ),
            }
        )

    # Log filter session start
    total_relationships = sum(len(edges) for edges in all_edges.values())
    if enhanced_logger:
        enhanced_logger.log_filter_session_start(
            total_relationships, "enhanced" if use_enhanced_classification else "basic"
        )

    for edge_key, edges in all_edges.items():
        filter_stats["total_before"] += len(edges)
        filtered_edge_list = []

        for edge in edges:
            src_id = edge.get("src_id", "").lower()
            tgt_id = edge.get("tgt_id", "").lower()
            rel_type = edge.get("rel_type", "").lower()
            weight = edge.get("weight", 0)
            description = edge.get("description", "")

            # NEW: Enhanced type-specific filtering
            if use_enhanced_classification:
                classification = classifier.classify_relationship(
                    rel_type, src_id, tgt_id, description
                )

                category = classification["category"]
                confidence = classification["confidence"]
                should_keep = classification["should_keep"]
                threshold = classification["threshold"]

                # Track category statistics
                filter_stats["category_stats"][category]["total"] += 1

                # Enhanced logging for classification results
                if enhanced_logger and (log_classification or not should_keep):
                    enhanced_logger.log_classification_result(
                        rel_type, src_id, tgt_id, classification
                    )

                # CRITICAL FIX: Check monitoring mode
                if monitoring_mode:
                    # In monitoring mode, log the decision but don't actually filter
                    if not should_keep:
                        logger.info(
                            f"MONITORING: Would filter {src_id} -[{rel_type}]-> {tgt_id} "
                            f"(category: {category}, confidence: {confidence:.2f}, threshold: {threshold:.2f})"
                        )
                        filter_stats["category_stats"][category]["filtered"] += 1
                    else:
                        filter_stats["category_stats"][category]["kept"] += 1
                        if log_classification:
                            logger.debug(
                                f"MONITORING: Would keep {src_id} -[{rel_type}]-> {tgt_id} "
                                f"(category: {category}, confidence: {confidence:.2f}, threshold: {threshold:.2f})"
                            )
                    # Always keep the relationship in monitoring mode
                elif not should_keep:
                    # Filter based on type-specific confidence thresholds
                    filter_stats["type_specific_filtered"] += 1
                    filter_stats[f"{category}_filtered"] += 1
                    filter_stats["category_stats"][category]["filtered"] += 1

                    if log_classification:
                        logger.info(
                            f"Type-specific filter: {src_id} -[{rel_type}]-> {tgt_id} "
                            f"(category: {category}, confidence: {confidence:.2f}, threshold: {threshold:.2f})"
                        )
                    else:
                        logger.debug(
                            f"Type-specific filter: {src_id} -[{rel_type}]-> {tgt_id} "
                            f"(category: {category}, confidence: {confidence:.2f}, threshold: {threshold:.2f})"
                        )
                    continue
                else:
                    filter_stats["category_stats"][category]["kept"] += 1
                    if log_classification:
                        logger.debug(
                            f"Type-specific keep: {src_id} -[{rel_type}]-> {tgt_id} "
                            f"(category: {category}, confidence: {confidence:.2f}, threshold: {threshold:.2f})"
                        )

            # FALLBACK: Basic abstract relationship filtering (if enhanced classification not available)
            elif rel_type in ABSTRACT_RELATIONSHIPS and weight < 0.8:
                filter_stats["abstract_relationships"] += 1
                logger.debug(
                    f"Filtered abstract relationship: {src_id} -[{rel_type}]-> {tgt_id} (weight: {weight})"
                )
                continue

            # Filter 2: Remove relationships between synonymous concepts
            is_synonym_relationship = False
            for synonym_group in SYNONYM_CONCEPTS:
                if src_id in synonym_group and tgt_id in synonym_group:
                    filter_stats["synonym_relationships"] += 1
                    logger.debug(
                        f"Filtered synonym relationship: {src_id} -[{rel_type}]-> {tgt_id}"
                    )
                    is_synonym_relationship = True
                    break

            if is_synonym_relationship:
                continue

            # Filter 3: Remove relationships involving abstract entities
            if _is_abstract_entity(src_id) or _is_abstract_entity(tgt_id):
                if not _validate_relationship_context(
                    src_id, tgt_id, rel_type, description
                ):
                    filter_stats["abstract_entities"] += 1
                    logger.debug(
                        f"Filtered abstract entity relationship: {src_id} -[{rel_type}]-> {tgt_id}"
                    )
                    continue

            # Filter 4: Apply basic weight filtering (very lenient - only filter obvious noise)
            weight = edge.get("weight", 1.0)
            if (
                weight < 0.1
            ):  # Very lenient threshold - only filter extremely low weight relationships
                filter_stats["low_confidence"] += 1
                logger.debug(
                    f"Filtered low-weight relationship: {src_id} -[{rel_type}]-> {tgt_id} (weight: {weight:.2f})"
                )
                continue

            # Filter 5: Skip context validation - LLM will handle this more intelligently
            # if not _validate_relationship_context(src_id, tgt_id, rel_type, description):
            #     filter_stats['context_validation'] += 1
            #     logger.debug(f"Filtered context validation: {src_id} -[{rel_type}]-> {tgt_id}")
            #     continue

            # Filter 6: Remove low-quality generic relationships (enhanced)
            if (
                rel_type in ["related", "associated_with"]
                and weight < 0.7
                and len(description) < 30
            ):
                filter_stats["low_quality_relationships"] += 1
                logger.debug(
                    f"Filtered low-quality relationship: {src_id} -[{rel_type}]-> {tgt_id}"
                )
                continue

            # Keep this relationship
            filtered_edge_list.append(edge)

        if filtered_edge_list:
            filtered_edges[edge_key] = filtered_edge_list

        filter_stats["total_after"] += len(filtered_edge_list)

    # Log enhanced filter statistics
    removed = filter_stats["total_before"] - filter_stats["total_after"]
    if removed > 0:
        logger.info(
            f"Enhanced relationship quality filter removed {removed}/{filter_stats['total_before']} relationships:"
        )

        if use_enhanced_classification:
            # Log type-specific filtering results
            logger.info(
                f"  - Type-specific filtered: {filter_stats['type_specific_filtered']}"
            )

            # Log category-specific statistics
            for category, stats in filter_stats["category_stats"].items():
                if stats["total"] > 0:
                    retention_rate = stats["kept"] / stats["total"]
                    logger.info(
                        f"    • {category}: {stats['kept']}/{stats['total']} kept ({retention_rate:.1%})"
                    )
        else:
            # Log basic filtering results
            logger.info(
                f"  - Abstract relationships: {filter_stats['abstract_relationships']}"
            )

        # Log remaining filter categories
        logger.info(
            f"  - Synonym relationships: {filter_stats['synonym_relationships']}"
        )
        logger.info(f"  - Abstract entities: {filter_stats['abstract_entities']}")
        logger.info(f"  - Low confidence: {filter_stats['low_confidence']}")
        logger.info(f"  - Context validation: {filter_stats['context_validation']}")
        logger.info(
            f"  - Low-quality relationships: {filter_stats['low_quality_relationships']}"
        )

        # Calculate quality metrics
        if filter_stats["total_after"] > 0:
            quality_ratio = filter_stats["total_after"] / filter_stats["total_before"]
            logger.info(f"  - Relationship retention rate: {quality_ratio:.1%}")

            # Log enhanced quality assessment and record metrics if available
            if use_enhanced_classification and filter_stats["category_stats"]:
                # Convert to format expected by classifier
                relationships_for_assessment = []
                for edge_key, edges in all_edges.items():
                    relationships_for_assessment.extend(edges)

                try:
                    recommendations = classifier.get_validation_recommendations(
                        relationships_for_assessment[:100]
                    )  # Sample for performance
                    overall_quality = recommendations.get("overall_quality", "N/A")
                    logger.info(f"  - Overall Quality Assessment: {overall_quality}")

                    # Enhanced logging for quality assessment
                    if enhanced_logger:
                        enhanced_logger.log_quality_assessment(
                            overall_quality,
                            recommendations.get("category_insights", {}).get(
                                "insights", []
                            ),
                        )
                except Exception as e:
                    logger.debug(f"Could not generate quality assessment: {e}")
                    if enhanced_logger:
                        enhanced_logger.log_error(
                            "QualityAssessment",
                            e,
                            "Failed to generate quality recommendations",
                        )

                # Record metrics if performance tracking is enabled
                if track_performance:
                    try:
                        from .kg.utils.relationship_filter_metrics import (
                            get_filter_metrics,
                        )

                        metrics = get_filter_metrics()
                        metrics.record_filter_session(
                            filter_stats, filter_stats["category_stats"]
                        )

                        # Enhanced logging for metrics collection
                        if enhanced_logger:
                            summary = metrics.get_session_summary()
                            enhanced_logger.log_metrics_collection(summary)

                        logger.debug("Recorded filter performance metrics")
                    except Exception as e:
                        logger.debug(f"Could not record filter metrics: {e}")
                        if enhanced_logger:
                            enhanced_logger.log_error(
                                "MetricsCollection",
                                e,
                                "Failed to record filter metrics",
                            )

    # SANITY CHECK: Detect abnormal retention rates
    total_before = filter_stats.get("total_before", 0)
    total_after = filter_stats.get("total_after", 0)
    retention_rate = total_after / max(total_before, 1)

    if retention_rate > 0.95 and total_before > 10:
        logger.warning(
            f"🚨 ABNORMALLY HIGH RETENTION RATE: {retention_rate:.1%} ({total_after}/{total_before})"
        )
        logger.warning(
            "   This suggests the enhanced filter may not be working properly"
        )
        logger.warning(
            "   Consider adjusting confidence thresholds or checking classification logic"
        )
    elif retention_rate < 0.5 and total_before > 10:
        logger.warning(
            f"🚨 ABNORMALLY LOW RETENTION RATE: {retention_rate:.1%} ({total_after}/{total_before})"
        )
        logger.warning("   This suggests the enhanced filter may be too aggressive")
        logger.warning("   Consider lowering confidence thresholds")
    elif 0.75 <= retention_rate <= 0.9:
        logger.info(f"✅ HEALTHY RETENTION RATE: {retention_rate:.1%} (target: 75-90%)")

    # Log filter session end with enhanced details
    if enhanced_logger:
        enhanced_stats = (
            filter_stats.get("category_stats", {})
            if use_enhanced_classification
            else None
        )
        enhanced_logger.log_filter_session_end(filter_stats, enhanced_stats)

    return filtered_edges


# Legacy extraction quality logging removed - deprecated metrics no longer needed


async def _llm_post_process_relationships(
    document_text: str,
    all_entities: list,
    all_relationships: list,
    llm_response_cache: BaseKVStorage,
    global_config: dict,
) -> tuple[list, dict]:
    """
    Use LLM to post-process and validate extracted relationships for improved accuracy.

    Args:
        document_text: Original document content for context
        all_entities: List of extracted entities
        all_relationships: List of extracted relationships to validate
        llm_response_cache: Cache for LLM responses
        global_config: Global configuration

    Returns:
        Tuple of (validated_relationships, processing_stats)
    """
    if not all_relationships:
        return [], {"total_input": 0, "validated": 0, "removed": 0}

    # Prepare entities summary (limit for token efficiency)
    entities_summary = "\n".join(
        [
            f"- {e.get('entity_name', 'Unknown')}: {e.get('entity_type', 'Unknown')}"
            for e in all_entities[:50]  # Limit to first 50 entities
        ]
    )

    # Create temporary file with all relationships in JSON format for LLM to manipulate
    import tempfile
    import json
    import os

    temp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False)
    temp_file_path = temp_file.name

    try:
        # Write all relationships to temporary file with original types preserved
        relationships_data = {
            "relationships": [
                {
                    "id": f"rel_{i}",
                    "src_id": r.get("src_id", ""),
                    "tgt_id": r.get("tgt_id", ""),
                    "rel_type": r.get("rel_type", "related"),
                    "description": r.get("description", ""),
                    "weight": r.get("weight", 0.8),
                    "source_id": r.get("source_id", ""),
                    "keywords": r.get("keywords", []),
                }
                for i, r in enumerate(all_relationships)
            ]
        }

        json.dump(relationships_data, temp_file, indent=2)
        temp_file.close()

        logger.info(f"📁 Created temporary relationships file: {temp_file_path}")
        logger.info(
            f"📁 File contains {len(relationships_data['relationships'])} relationships with preserved types"
        )

        # Debug: Log first 5 relationships that went into the temp file
        logger.info("🔍 First 5 relationships stored in temp file:")
        for i, rel in enumerate(relationships_data["relationships"][:5]):
            logger.info(
                f"  {i+1}. {rel['src_id']} -[{rel['rel_type']}]-> {rel['tgt_id']}"
            )

        # Debug: Also log 5 random original relationships for comparison
        logger.info("🔍 Original relationships for comparison:")
        for i, rel in enumerate(all_relationships[:5]):
            logger.info(
                f"  {i+1}. {rel.get('src_id', '')} -[{rel.get('rel_type', '')}]-> {rel.get('tgt_id', '')}"
            )

        # Read file content for LLM prompt
        with open(temp_file_path, "r") as f:
            relationships_file_content = f.read()

        # Prepare entities summary (limit for token efficiency)
        entities_summary = "\n".join(
            [
                f"- {e.get('entity_name', 'Unknown')}: {e.get('entity_type', 'Unknown')}"
                for e in all_entities[:50]  # Limit to first 50 entities
            ]
        )

        # Simplified prompt focused ONLY on removal, NO modification
        prompt_text = f"""You are filtering extracted relationships based on document evidence. 

DOCUMENT:
{document_text}

RELATIONSHIPS TO FILTER:
{relationships_file_content}

TASK: Remove relationships that are NOT clearly supported by the document. 
- DO NOT modify any rel_type values
- DO NOT change field values  
- ONLY remove entire relationship entries if unsupported

Return the filtered JSON with the same exact structure. Only keep relationships with clear document evidence.

Example output (keep exact same format and field values):
```json
{{
  "relationships": [
    {{
      "id": "rel_0",
      "src_id": "same_as_input",
      "tgt_id": "same_as_input", 
      "rel_type": "same_as_input",
      "description": "same_as_input",
      "weight": 0.9,
      "source_id": "same_as_input",
      "keywords": ["same_as_input"]
    }}
  ]
}}
```

CRITICAL: Preserve ALL field values exactly. Only remove unsupported relationships."""

        # Call LLM for post-processing
        logger.info(
            f"Starting LLM post-processing of {len(all_relationships)} relationships..."
        )

        # Get the LLM function from global config
        llm_func = global_config.get("llm_model_func")
        if not llm_func:
            raise ValueError("No LLM function available in global_config")

        # Don't use cache for this processing since it's document-specific
        response = await llm_func(prompt_text)

        # Clean up JSON response (remove markdown formatting if present)
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        # Parse LLM response (file-based format)
        result = json.loads(cleaned_response)
        validated_relationships = result.get("relationships", [])

        # Calculate processing stats
        input_count = len(all_relationships)
        validated_count = len(validated_relationships)
        removed_count = input_count - validated_count
        processing_stats = {
            "total_input": input_count,
            "validated": validated_count,
            "removed": removed_count,
            "accuracy_improvement": f"File-based processing preserved {validated_count}/{input_count} relationships with original types",
            "average_quality_score": (
                sum(r.get("weight", 0.8) for r in validated_relationships)
                / len(validated_relationships)
                if validated_relationships
                else 0
            ),
        }

        # Log results
        logger.info("🎯 LLM post-processing completed:")
        logger.info(f"  - Input relationships: {input_count}")
        logger.info(f"  - Validated relationships: {validated_count}")
        logger.info(f"  - Removed relationships: {removed_count}")
        logger.info(f"  - Retention rate: {validated_count/input_count*100:.1f}%")
        logger.info(
            f"  - Average quality score: {processing_stats['average_quality_score']:.1f}"
        )
        logger.info(f"  - Improvement: {processing_stats['accuracy_improvement']}")

        # Log examples of validated relationships with preserved types
        logger.info("✅ File-based relationships with preserved types:")
        for i, rel in enumerate(validated_relationships[:3]):
            logger.info(
                f"  {i+1}. {rel.get('src_id', '')} -[{rel.get('rel_type', '')}]-> {rel.get('tgt_id', '')}"
            )

        # Save validated relationships to a new temp file for debugging
        validated_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix="_validated.json", delete=False
        )
        validated_data = {"relationships": validated_relationships}
        json.dump(validated_data, validated_file, indent=2)
        validated_file.close()
        logger.info(f"📁 Saved validated relationships to: {validated_file.name}")

        # Convert file-based relationships directly to expected format (NO BROKEN PRESERVATION LOGIC)
        formatted_relationships = []
        from lightrag.kg.utils.relationship_registry import (
            standardize_relationship_type,
        )

        for rel in validated_relationships:
            rel_type = rel.get("rel_type", "related")  # Should be preserved from file

            formatted_rel = {
                "src_id": rel.get("src_id", ""),
                "tgt_id": rel.get("tgt_id", ""),
                "rel_type": rel_type,  # Direct from file - SHOULD be preserved!
                "relationship_type": rel_type,  # Human-readable type for merge process
                "original_type": rel_type,  # Original extracted type
                "neo4j_type": standardize_relationship_type(
                    rel_type
                ),  # Standardized Neo4j type
                "description": rel.get("description", ""),
                "weight": rel.get("weight", 0.8),
                "quality_score": rel.get("quality_score", 6),
                "evidence": rel.get("evidence", ""),
                "source_id": rel.get("source_id", "file_based_processing"),
            }
            formatted_relationships.append(formatted_rel)

        # Log final formatting examples
        logger.info("📝 Final formatted relationships (should preserve types):")
        for i, rel in enumerate(formatted_relationships[:3]):
            logger.info(
                f"  {i+1}. {rel['src_id']} -[{rel['rel_type']}|{rel['neo4j_type']}]-> {rel['tgt_id']}"
            )

        # Cleanup temp files
        try:
            os.unlink(temp_file_path)
            logger.info(f"🗑️ Cleaned up temp file: {temp_file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {temp_file_path}: {e}")

        return formatted_relationships, processing_stats

    except json.JSONDecodeError as e:
        logger.warning(f"LLM post-processing failed to parse JSON: {e}")
        logger.warning(f"Raw response: {response[:500]}...")
        logger.warning("Falling back to original relationships")
        return all_relationships, {"error": "JSON parsing failed"}

    except Exception as e:
        logger.error(f"LLM post-processing failed with error: {e}")
        logger.warning("Falling back to original relationships")
        return all_relationships, {"error": str(e)}

    finally:
        # Ensure temp files are cleaned up
        try:
            if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except Exception:
            pass


def _preserve_original_relationship_metadata(
    original_relationships: list, validated_relationships: list
) -> list:
    """
    Ensure LLM-validated relationships preserve their original rel_type and source_id.

    Args:
        original_relationships: List of original extracted relationships
        validated_relationships: List of LLM-validated relationships

    Returns:
        List of validated relationships with preserved metadata
    """
    # Create mapping of original relationships by (src_id, tgt_id) for fast lookup
    # Using normalized names for better matching
    original_map = {}
    for rel in original_relationships:
        src_id = rel.get("src_id", "").strip()
        tgt_id = rel.get("tgt_id", "").strip()
        # Create both directions for undirected relationships
        for key in [(src_id, tgt_id), (tgt_id, src_id)]:
            if key not in original_map:
                original_map[key] = rel

    logger.info(
        f"🔍 Relationship preservation: Created lookup map with {len(original_map)} entries"
    )

    # Update validated relationships to preserve original metadata
    preserved_relationships = []
    preservation_stats = {"preserved": 0, "not_found": 0, "total": 0}

    for validated in validated_relationships:
        preservation_stats["total"] += 1
        src_id = validated.get("src_id", "").strip()
        tgt_id = validated.get("tgt_id", "").strip()

        # Try to find original relationship
        original = None
        for key in [(src_id, tgt_id), (tgt_id, src_id)]:
            if key in original_map:
                original = original_map[key]
                break

        if original:
            # ALWAYS preserve original rel_type - LLM is converting everything to "related"
            original_rel_type = original.get("rel_type", "related")
            validated["rel_type"] = original_rel_type

            # Preserve original source_id for traceability
            original_source_id = original.get("source_id", "llm_post_processed")
            validated["source_id"] = original_source_id

            preservation_stats["preserved"] += 1
            logger.debug(f"✅ Preserved: {src_id} -[{original_rel_type}]-> {tgt_id}")
        else:
            preservation_stats["not_found"] += 1
            logger.debug(
                f"❌ Not found: {src_id} -[{validated.get('rel_type', 'unknown')}]-> {tgt_id}"
            )

        preserved_relationships.append(validated)

    # Log preservation statistics
    preserved_pct = (
        (preservation_stats["preserved"] / preservation_stats["total"]) * 100
        if preservation_stats["total"] > 0
        else 0
    )
    logger.info("🔧 Relationship preservation completed:")
    logger.info(f"  - Total relationships: {preservation_stats['total']}")
    logger.info(
        f"  - Preserved original types: {preservation_stats['preserved']} ({preserved_pct:.1f}%)"
    )
    logger.info(f"  - Not found in original: {preservation_stats['not_found']}")

    return preserved_relationships


def _rebuild_edges_from_validated(validated_relationships: list) -> dict:
    """
    Rebuild the edges dictionary from validated relationships.

    Args:
        validated_relationships: List of validated relationships from LLM

    Returns:
        Dictionary of edge lists keyed by sorted edge tuples
    """
    edges_dict = defaultdict(list)

    for rel in validated_relationships:
        src_id = rel.get("src_id", "")
        tgt_id = rel.get("tgt_id", "")

        if src_id and tgt_id:
            # Create sorted edge key for undirected graph
            edge_key = tuple(sorted([src_id, tgt_id]))
            edges_dict[edge_key].append(rel)

    return dict(edges_dict)


async def merge_nodes_and_edges(
    chunk_results: list,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    current_file_number: int = 0,
    total_files: int = 0,
    file_path: str = "unknown_source",
    document_text: str = None,  # Add document text parameter
) -> None:
    """Merge nodes and edges from extraction results

    Args:
        chunk_results: List of tuples (maybe_nodes, maybe_edges) containing extracted entities and relationships
        knowledge_graph_inst: Knowledge graph storage
        entity_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        llm_response_cache: LLM response cache
    """
    # Get lock manager from shared storage
    from .kg.shared_storage import get_graph_db_lock

    # Collect all nodes and edges from all chunks
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)

    for maybe_nodes, maybe_edges in chunk_results:
        # Collect nodes
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)

        # Collect edges with sorted keys for undirected graph
        for edge_key, edges in maybe_edges.items():
            sorted_edge_key = tuple(sorted(edge_key))
            # Deduplicate edges based on relationship type and properties to prevent excessive processing
            edge_signatures_seen = set()
            for edge in edges:
                # Create a more comprehensive signature for the edge to identify duplicates
                # Include source_id (chunk) to prevent same relationship from same chunk being added twice
                edge_signature = (
                    edge.get("src_id"),
                    edge.get("tgt_id"),
                    edge.get("rel_type"),
                    edge.get(
                        "source_id"
                    ),  # Include chunk ID to prevent intra-chunk duplicates
                    edge.get("description", "")[
                        :100
                    ],  # More chars for better distinction
                )

                # Check if we've already seen this exact signature in this batch
                if edge_signature not in edge_signatures_seen:
                    # Also check if this relationship already exists in accumulated edges
                    existing_sigs = {
                        (
                            e.get("src_id"),
                            e.get("tgt_id"),
                            e.get("rel_type"),
                            e.get("source_id"),
                            e.get("description", "")[:100],
                        )
                        for e in all_edges[sorted_edge_key]
                    }

                    if edge_signature not in existing_sigs:
                        all_edges[sorted_edge_key].append(edge)
                        edge_signatures_seen.add(edge_signature)
                    else:
                        logger.debug(
                            f"Skipping cross-chunk duplicate relationship: {edge_signature[0]} -> {edge_signature[1]} ({edge_signature[2]})"
                        )
                else:
                    logger.debug(
                        f"Skipping intra-batch duplicate relationship: {edge_signature[0]} -> {edge_signature[1]} ({edge_signature[2]})"
                    )

    # Apply basic post-processing filters to remove redundant relationships
    # Note: Made more lenient since LLM post-processing will do the heavy lifting
    all_edges = _apply_relationship_quality_filter(all_edges, global_config)

    # Clean up orphaned entities after chunk post-processing
    # Note: Entity cleanup can be controlled independently from chunk post-processing
    if global_config.get("enable_chunk_post_processing", False) and global_config.get(
        "enable_entity_cleanup", DEFAULT_ENABLE_ENTITY_CLEANUP
    ):
        from .chunk_post_processor import cleanup_orphaned_entities

        log_changes = global_config.get("log_validation_changes", False)
        original_entity_count = len(all_nodes)
        all_nodes = cleanup_orphaned_entities(all_nodes, all_edges, log_changes)
        logger.info(
            f"Post-processing entity cleanup: {original_entity_count} → {len(all_nodes)} entities"
        )
    elif global_config.get("enable_chunk_post_processing", False):
        logger.info(
            f"Entity cleanup disabled: Keeping all {len(all_nodes)} entities (some may be orphaned)"
        )

    # NEW: LLM-based post-processing for enhanced accuracy
    all_entities_list = [
        entity for entities in all_nodes.values() for entity in entities
    ]
    all_relationships_list = [edge for edges in all_edges.values() for edge in edges]

    # Simplified post-processing diagnostics (removed verbose debug logging)
    logger.info("=== LLM Post-Processing Status ===")
    logger.info(f"  - Entities: {len(all_entities_list)}")
    logger.info(f"  - Relationships: {len(all_relationships_list)}")
    logger.info(
        f"  - LLM processing enabled: {global_config.get('enable_llm_post_processing', True)}"
    )
    if document_text:
        logger.info(f"  - Document available: {len(document_text)} chars")

    if (
        llm_response_cache
        and global_config.get("enable_llm_post_processing", True)
        and len(all_relationships_list) > 0
        and document_text
    ):

        logger.info("✅ Starting LLM-based relationship post-processing...")

        try:
            validated_relationships, processing_stats = (
                await _llm_post_process_relationships(
                    document_text,
                    all_entities_list,
                    all_relationships_list,
                    llm_response_cache,
                    global_config,
                )
            )

            # Rebuild edges from validated relationships
            if validated_relationships:
                all_edges = _rebuild_edges_from_validated(validated_relationships)
                all_relationships_list = validated_relationships
                logger.info("LLM post-processing completed successfully")
            else:
                logger.warning(
                    "LLM post-processing returned no relationships, keeping originals"
                )

        except Exception as e:
            logger.error(f"LLM post-processing failed: {e}")
            logger.warning("Continuing with basic filtering results")
    else:
        logger.warning("❌ Skipping LLM post-processing due to:")
        if not document_text:
            logger.warning("  - No document text available")
        if not llm_response_cache:
            logger.warning("  - No LLM cache available")
        if len(all_relationships_list) == 0:
            logger.warning("  - No relationships to process")
        if not global_config.get("enable_llm_post_processing", True):
            logger.warning("  - LLM post-processing disabled in config")

    # Legacy extraction quality logging removed - using more accurate LLM-based quality metrics instead

    # Centralized processing of all nodes and edges
    entities_data = []
    relationships_data = []

    # Merge nodes and edges
    # Use graph database lock to ensure atomic merges and updates
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    async with graph_db_lock:
        async with pipeline_status_lock:
            log_message = (
                f"Merging stage {current_file_number}/{total_files}: {file_path}"
            )
            logger.info(log_message)
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)

        # Process and update all entities at once
        for entity_name, entities in all_nodes.items():
            entity_data = await _merge_nodes_then_upsert(
                entity_name,
                entities,
                knowledge_graph_inst,
                global_config,
                pipeline_status,
                pipeline_status_lock,
                llm_response_cache,
            )
            entities_data.append(entity_data)

        # Process and update all relationships at once
        for edge_key, edges in all_edges.items():
            edge_data = await _merge_edges_then_upsert(
                edge_key[0],
                edge_key[1],
                edges,
                knowledge_graph_inst,
                global_config,
                pipeline_status,
                pipeline_status_lock,
                llm_response_cache,
            )
            if edge_data is not None:
                relationships_data.append(edge_data)

        # Update total counts
        total_entities_count = len(entities_data)
        total_relations_count = len(relationships_data)

        log_message = f"Updating {total_entities_count} entities  {current_file_number}/{total_files}: {file_path}"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # Update vector databases with all collected data
        if entity_vdb is not None and entities_data:
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "entity_name": dp["entity_name"],
                    "entity_type": dp["entity_type"],
                    "content": f"{dp['entity_name']}\n{dp['description']}",
                    "source_id": dp["source_id"],
                    "file_path": dp.get("file_path", "unknown_source"),
                }
                for dp in entities_data
            }
            await entity_vdb.upsert(data_for_vdb)

        log_message = f"Updating {total_relations_count} relations {current_file_number}/{total_files}: {file_path}"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        if relationships_vdb is not None and relationships_data:
            data_for_vdb = {
                compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                    "src_id": dp["src_id"],
                    "tgt_id": dp["tgt_id"],
                    "keywords": dp["keywords"],
                    "content": f"{dp['src_id']}\t{dp['tgt_id']}\n{dp['keywords']}\n{dp['description']}",
                    "source_id": dp["source_id"],
                    "file_path": dp.get("file_path", "unknown_source"),
                }
                for dp in relationships_data
            }
            await relationships_vdb.upsert(data_for_vdb)


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
) -> list:
    # Initialize monitoring
    perf_monitor = get_performance_monitor()
    proc_monitor = get_processing_monitor()
    enhanced_logger = get_enhanced_logger("lightrag.extraction")

    # Start processing session
    proc_monitor.start_session()
    proc_monitor.update_status("entity_extraction", total_files=len(chunks))

    # Define total_chunks early for use in nested functions
    total_chunks = len(chunks)

    # Start performance monitoring for the overall extraction process
    with perf_monitor.measure("extract_entities_total", chunks_count=len(chunks)):
        enhanced_logger.info(f"Starting entity extraction for {len(chunks)} chunks")

        # Add llm_response_cache to global_config for post-processing
        if llm_response_cache is not None:
            global_config["llm_response_cache"] = llm_response_cache

        use_llm_func: callable = global_config["llm_model_func"]
        entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

        # Track processed chunks count
        processed_chunks = 0
        completed_chunks = 0  # Add this counter for progress tracking

        ordered_chunks = list(chunks.items())
        # add language and example number params to prompt
        language = global_config["addon_params"].get(
            "language", PROMPTS["DEFAULT_LANGUAGE"]
        )
        entity_types = global_config["addon_params"].get(
            "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
        )
        example_number = global_config["addon_params"].get("example_number", None)
        if example_number and example_number < len(
            PROMPTS["entity_extraction_examples"]
        ):
            examples = "\n".join(
                PROMPTS["entity_extraction_examples"][: int(example_number)]
            )
        else:
            examples = "\n".join(PROMPTS["entity_extraction_examples"])

        example_context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=", ".join(entity_types),
            language=language,
        )
        # add example's format
        examples = examples.format(**example_context_base)

        entity_extract_prompt = PROMPTS["entity_extraction"]
        context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=",".join(entity_types),
            examples=examples,
            language=language,
            relationship_types=global_config.get(
                "relationship_types",
                "related, uses, creates, implements, integrates_with, configures, troubleshoots, optimizes",
            ),
            relationship_examples=global_config.get(
                "relationship_examples",
                "uses: for tool usage, creates: for artifact creation, implements: for feature implementation, troubleshoots: for debugging activities",
            ),
        )

        continue_prompt = PROMPTS["entity_continue_extraction"].format(**context_base)
        if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

        # Define total_chunks for logging
        total_chunks = len(chunks)

        async def _process_extraction_result(
            result: str, chunk_key: str, file_path: str = "unknown_source"
        ):
            with perf_monitor.measure("process_extraction_result", chunk_key=chunk_key):
                """Process a single extraction result (either initial or gleaning)
                Args:
                    result (str): The extraction result to process
                    chunk_key (str): The chunk key for source tracking
                    file_path (str): The file path for citation
                Returns:
                    tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
                """
                maybe_nodes = defaultdict(list)
                maybe_edges = defaultdict(list)

                records = split_string_by_multi_markers(
                    result,
                    [
                        context_base["record_delimiter"],
                        context_base["completion_delimiter"],
                    ],
                )

                for record in records:
                    record = re.search(r"\((.*)\)", record)
                    if record is None:
                        continue
                    record = record.group(1)
                    record_attributes = split_string_by_multi_markers(
                        record, [context_base["tuple_delimiter"]]
                    )

                    # Skip content_keywords records entirely
                    if (
                        len(record_attributes) >= 1
                        and '"content_keywords"' in record_attributes[0]
                    ):
                        logger.debug(
                            f"Skipping content_keywords record: {record_attributes[0] if len(record_attributes) > 0 else 'empty'}"
                        )
                        continue

                    if_entities = await _handle_single_entity_extraction(
                        record_attributes, chunk_key, file_path
                    )
                    if if_entities is not None:
                        maybe_nodes[if_entities["entity_name"]].append(if_entities)
                        continue

                    if_relation = await _handle_single_relationship_extraction(
                        record_attributes, chunk_key, file_path
                    )
                    if if_relation is not None:
                        maybe_edges[
                            (if_relation["src_id"], if_relation["tgt_id"])
                        ].append(if_relation)

                # Track extraction statistics
                entities_extracted = len(maybe_nodes) if maybe_nodes else 0
                relationships_extracted = len(maybe_edges) if maybe_edges else 0

                # Validate and track results
                if maybe_nodes or maybe_edges:
                    # Flatten the defaultdict structures into lists for validation
                    entities_list = []
                    for entity_name, entity_instances in maybe_nodes.items():
                        entities_list.extend(entity_instances)

                    relationships_list = []
                    for edge_key, edge_instances in maybe_edges.items():
                        relationships_list.extend(edge_instances)

                    valid_entities, valid_relationships, validation_errors = (
                        validate_extraction_results(entities_list, relationships_list)
                    )

                    # Record processing statistics
                    proc_monitor.record_extraction_results(
                        entities_extracted=entities_extracted,
                        entities_validated=len(valid_entities),
                        entities_failed=entities_extracted - len(valid_entities),
                        relationships_extracted=relationships_extracted,
                        relationships_validated=len(valid_relationships),
                        relationships_failed=relationships_extracted
                        - len(valid_relationships),
                    )

                    # Record validation statistics
                    proc_monitor.record_validation_results(
                        errors=len(
                            [e for e in validation_errors if e.severity == "error"]
                        ),
                        warnings=len(
                            [e for e in validation_errors if e.severity == "warning"]
                        ),
                    )

                    enhanced_logger.debug(
                        f"Extraction results for {chunk_key}",
                        entities_extracted=entities_extracted,
                        entities_validated=len(valid_entities),
                        relationships_extracted=relationships_extracted,
                        relationships_validated=len(valid_relationships),
                        validation_errors=len(validation_errors),
                    )

                    # Rebuild defaultdict structures with validated data for consistent return format
                    validated_nodes = defaultdict(list)
                    for entity in valid_entities:
                        if "entity_name" in entity:
                            validated_nodes[entity["entity_name"]].append(entity)

                    validated_edges = defaultdict(list)
                    for relationship in valid_relationships:
                        if "src_id" in relationship and "tgt_id" in relationship:
                            edge_key = (relationship["src_id"], relationship["tgt_id"])
                            validated_edges[edge_key].append(relationship)

                    return validated_nodes, validated_edges

                return maybe_nodes, maybe_edges

        async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
            chunk_key, chunk_dp = chunk_key_dp

            # Monitor chunk processing
            with perf_monitor.measure("process_single_chunk", chunk_key=chunk_key):
                enhanced_logger.debug(f"Processing chunk: {chunk_key}")

                nonlocal processed_chunks
                content = chunk_dp["content"]
                # Get file path from chunk data or use default
                file_path = chunk_dp.get("file_path", "unknown_source")

                # Get initial extraction
                hint_prompt = entity_extract_prompt.format(
                    **{**context_base, "input_text": content}
                )

                final_result = await use_llm_func_with_cache(
                    hint_prompt,
                    use_llm_func,
                    llm_response_cache=llm_response_cache,
                    cache_type="extract",
                )
                history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

                # Process initial extraction with file path
                maybe_nodes, maybe_edges = await _process_extraction_result(
                    final_result, chunk_key, file_path
                )

                # Process additional gleaning results
                for now_glean_index in range(entity_extract_max_gleaning):
                    glean_result = await use_llm_func_with_cache(
                        continue_prompt,
                        use_llm_func,
                        llm_response_cache=llm_response_cache,
                        history_messages=history,
                        cache_type="extract",
                    )

                    history += pack_user_ass_to_openai_messages(
                        continue_prompt, glean_result
                    )

                    # Process gleaning result separately with file path
                    glean_nodes, glean_edges = await _process_extraction_result(
                        glean_result, chunk_key, file_path
                    )

                    # Merge results - only add entities and edges with new names
                    for entity_name, entities in glean_nodes.items():
                        if (
                            entity_name not in maybe_nodes
                        ):  # Only accetp entities with new name in gleaning stage
                            maybe_nodes[entity_name].extend(entities)
                    for edge_key, edges in glean_edges.items():
                        if (
                            edge_key not in maybe_edges
                        ):  # Only accetp edges with new name in gleaning stage
                            maybe_edges[edge_key].extend(edges)

                    if now_glean_index == entity_extract_max_gleaning - 1:
                        break

                    if_loop_result: str = await use_llm_func_with_cache(
                        if_loop_prompt,
                        use_llm_func,
                        llm_response_cache=llm_response_cache,
                        history_messages=history,
                        cache_type="extract",
                    )
                    if_loop_result = (
                        if_loop_result.strip().strip('"').strip("'").lower()
                    )
                    if if_loop_result != "yes":
                        break

                processed_chunks += 1
                entities_count = len(maybe_nodes)
                relations_count = len(maybe_edges)
                log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel"
                logger.info(log_message)
                if pipeline_status is not None:
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                # Apply chunk-level relationship post-processing if enabled
                if global_config.get(
                    "enable_chunk_post_processing", DEFAULT_ENABLE_CHUNK_POST_PROCESSING
                ):
                    try:
                        maybe_edges = await _post_process_chunk_relationships(
                            content,
                            maybe_edges,
                            maybe_nodes,
                            use_llm_func,
                            chunk_key,
                            global_config,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Chunk post-processing failed for {chunk_key}: {e}"
                        )
                        logger.info("Continuing with original relationships")

                # Return the extracted nodes and edges for centralized processing
                return maybe_nodes, maybe_edges

        # Get max async tasks limit from global_config
        llm_model_max_async = global_config.get("llm_model_max_async", 4)
        semaphore = asyncio.Semaphore(llm_model_max_async)

        async def _process_with_semaphore(chunk):
            async with semaphore:
                nonlocal completed_chunks, processed_chunks

                # Monitor individual chunk processing with semaphore
                with perf_monitor.measure("chunk_with_semaphore", chunk_key=chunk[0]):
                    try:
                        result = await _process_single_content(chunk)

                        # Update progress
                        completed_chunks += 1
                        progress = completed_chunks / len(chunks)
                        proc_monitor.update_status(
                            "entity_extraction",
                            current_file=chunk[0],
                            progress=progress,
                            completed_files=completed_chunks,
                            total_files=len(chunks),
                        )

                        # Record successful chunk processing
                        proc_monitor.record_chunk_processing(processed=1, failed=0)
                        enhanced_logger.debug(
                            f"Completed chunk {chunk[0]} ({completed_chunks}/{len(chunks)})"
                        )

                        return result
                    except Exception as e:
                        # Record failed chunk processing
                        proc_monitor.record_chunk_processing(processed=0, failed=1)
                        enhanced_logger.error(
                            f"Failed to process chunk {chunk[0]}: {str(e)}"
                        )
                        raise

        # Enhanced error handling for task processing
        tasks = []
        failed_chunks = []
        successful_results = []
        # Change default to fail-fast for all-or-nothing processing
        enable_graceful_degradation = global_config.get(
            "enable_graceful_degradation", False
        )

        for c in ordered_chunks:
            task = asyncio.create_task(_process_with_semaphore(c))
            tasks.append(task)

        # Process tasks with enhanced error handling and recovery
        if enable_graceful_degradation:
            # Graceful degradation mode: fail on ANY chunk failure (0% tolerance)
            logger.info(
                f"Processing {len(tasks)} chunks with graceful degradation enabled (0% failure tolerance)"
            )

            # Wait for all tasks to complete
            done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

            for i, task in enumerate(done):
                try:
                    if task.exception():
                        failed_chunks.append(
                            (i, ordered_chunks[i], str(task.exception()))
                        )
                        logger.error(f"Chunk {i} failed: {str(task.exception())}")

                    else:
                        result = task.result()
                        successful_results.append(result)

                except Exception as e:
                    logger.error(f"Error processing task {i}: {str(e)}")
                    failed_chunks.append((i, ordered_chunks[i], str(e)))

            # Fail immediately if ANY chunks failed (0% tolerance)
            if failed_chunks:
                failure_rate = len(failed_chunks) / len(tasks) * 100
                logger.error(
                    f"Pipeline failed: {len(failed_chunks)}/{len(tasks)} chunks failed ({failure_rate:.1f}% failure rate)"
                )

                # Log detailed failure information for debugging
                failure_types = {}
                for chunk_idx, chunk_data, error_msg in failed_chunks:
                    error_type = "general"
                    if "rate" in error_msg.lower() and "limit" in error_msg.lower():
                        error_type = "rate_limit"
                    elif "timeout" in error_msg.lower():
                        error_type = "timeout"
                    elif (
                        "connection" in error_msg.lower()
                        or "network" in error_msg.lower()
                    ):
                        error_type = "network"
                    elif "total_chunks" in error_msg.lower():
                        error_type = "variable_error"

                    failure_types[error_type] = failure_types.get(error_type, 0) + 1

                logger.error(f"Failure breakdown: {failure_types}")
                raise RuntimeError(
                    f"Entity extraction failed: {len(failed_chunks)}/{len(tasks)} chunks failed. All chunks must succeed for processing to continue."
                )
            else:
                logger.info("All chunks processed successfully")

            chunk_results = successful_results

        else:
            # Fail-fast mode: default behavior - stop on first failure
            logger.info(
                f"Processing {len(tasks)} chunks with fail-fast mode (recommended)"
            )

            # Wait for tasks to complete or for the first exception to occur
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )

            # Check if any task raised an exception
            for task in done:
                if task.exception():
                    # If a task failed, cancel all pending tasks
                    for pending_task in pending:
                        pending_task.cancel()

                    # Wait for cancellation to complete
                    if pending:
                        await asyncio.wait(pending)

                    # Re-raise the exception to notify the caller
                    logger.error(
                        f"Chunk processing failed in fail-fast mode: {str(task.exception())}"
                    )
                    raise task.exception()

            # If all tasks completed successfully, collect results
            chunk_results = [task.result() for task in tasks]

        # Return the chunk_results for later processing in merge_nodes_and_edges
        return chunk_results


async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    chunks_vdb: BaseVectorStorage = None,
) -> str | AsyncIterator[str]:
    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    # Handle cache
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
        logger.warning(
            "low_level_keywords is empty, switching from %s mode to global mode",
            query_param.mode,
        )
        query_param.mode = "global"
    if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
        logger.warning(
            "high_level_keywords is empty, switching from %s mode to local mode",
            query_param.mode,
        )
        query_param.mode = "local"

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build context
    context = await _build_query_context(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # Build system prompt
    user_prompt = (
        query_param.user_prompt
        if query_param.user_prompt
        else PROMPTS["DEFAULT_USER_PROMPT"]
    )
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
        user_prompt=user_prompt,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    if hashing_kv.global_config.get("enable_llm_cache"):
        # Save to cache
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=query_param.mode,
                cache_type="query",
            ),
        )

    return response


async def get_keywords_from_query(
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Retrieves high-level and low-level keywords for RAG operations.

    This function checks if keywords are already provided in query parameters,
    and if not, extracts them from the query text using LLM.

    Args:
        query: The user's query text
        query_param: Query parameters that may contain pre-defined keywords
        global_config: Global configuration dictionary
        hashing_kv: Optional key-value storage for caching results

    Returns:
        A tuple containing (high_level_keywords, low_level_keywords)
    """
    # Check if pre-defined keywords are already provided
    if query_param.hl_keywords or query_param.ll_keywords:
        return query_param.hl_keywords, query_param.ll_keywords

    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )
    return hl_keywords, ll_keywords


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # 1. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(param.mode, text, cache_type="keywords")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_response is not None:
        try:
            keywords_data = json.loads(cached_response)
            return (
                keywords_data["high_level_keywords"],
                keywords_data["low_level_keywords"],
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 2. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # 3. Process conversation history
    history_context = ""
    if param.conversation_history:
        history_context = get_conversation_turns(
            param.conversation_history, param.history_turns
        )

    # 4. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text, examples=examples, language=language, history=history_context
    )

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(kw_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    # 5. Call the LLM for keyword extraction
    if param.model_func:
        use_model_func = param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    result = await use_model_func(kw_prompt, keyword_extraction=True)

    # 6. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        keywords_data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    # 7. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        if hashing_kv.global_config.get("enable_llm_cache"):
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=json.dumps(cache_data),
                    prompt=text,
                    quantized=quantized,
                    min_val=min_val,
                    max_val=max_val,
                    mode=param.mode,
                    cache_type="keywords",
                ),
            )

    return hl_keywords, ll_keywords


async def _get_vector_context(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    tokenizer: Tokenizer,
) -> tuple[list, list, list] | None:
    """
    Retrieve vector context from the vector database.

    This function performs vector search to find relevant text chunks for a query,
    formats them with file path and creation time information.

    Args:
        query: The query string to search for
        chunks_vdb: Vector database containing document chunks
        query_param: Query parameters including top_k and ids
        tokenizer: Tokenizer for counting tokens

    Returns:
        Tuple (empty_entities, empty_relations, text_units) for combine_contexts,
        compatible with _get_edge_data and _get_node_data format
    """
    try:
        results = await chunks_vdb.query(
            query, top_k=query_param.top_k, ids=query_param.ids
        )
        if not results:
            return [], [], []

        valid_chunks = []
        for result in results:
            if "content" in result:
                # Directly use content from chunks_vdb.query result
                chunk_with_time = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                }
                valid_chunks.append(chunk_with_time)

        if not valid_chunks:
            return [], [], []

        maybe_trun_chunks = truncate_list_by_token_size(
            valid_chunks,
            key=lambda x: x["content"],
            max_token_size=query_param.max_token_for_text_unit,
            tokenizer=tokenizer,
        )

        logger.debug(
            f"Truncate chunks from {len(valid_chunks)} to {len(maybe_trun_chunks)} (max tokens:{query_param.max_token_for_text_unit})"
        )
        logger.info(
            f"Vector query: {len(maybe_trun_chunks)} chunks, top_k: {query_param.top_k}"
        )

        if not maybe_trun_chunks:
            return [], [], []

        # Create empty entities and relations contexts
        entities_context = []
        relations_context = []

        # Create text_units_context directly as a list of dictionaries
        text_units_context = []
        for i, chunk in enumerate(maybe_trun_chunks):
            text_units_context.append(
                {
                    "id": i + 1,
                    "content": chunk["content"],
                    "file_path": chunk["file_path"],
                }
            )

        return entities_context, relations_context, text_units_context
    except Exception as e:
        logger.error(f"Error in _get_vector_context: {e}")
        return [], [], []


async def _build_query_context(
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,  # Add chunks_vdb parameter for mix mode
):
    logger.info(f"Process {os.getpid()} building query context...")

    # Handle local and global modes as before
    if query_param.mode == "local":
        entities_context, relations_context, text_units_context = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    else:  # hybrid or mix mode
        ll_data = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
        hl_data = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

        (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        ) = ll_data

        (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        ) = hl_data

        # Initialize vector data with empty lists
        vector_entities_context, vector_relations_context, vector_text_units_context = (
            [],
            [],
            [],
        )

        # Only get vector data if in mix mode
        if query_param.mode == "mix" and hasattr(query_param, "original_query"):
            # Get tokenizer from text_chunks_db
            tokenizer = text_chunks_db.global_config.get("tokenizer")

            # Get vector context in triple format
            vector_data = await _get_vector_context(
                query_param.original_query,  # We need to pass the original query
                chunks_vdb,
                query_param,
                tokenizer,
            )

            # If vector_data is not None, unpack it
            if vector_data is not None:
                (
                    vector_entities_context,
                    vector_relations_context,
                    vector_text_units_context,
                ) = vector_data

        # Combine and deduplicate the entities, relationships, and sources
        entities_context = process_combine_contexts(
            hl_entities_context, ll_entities_context, vector_entities_context
        )
        relations_context = process_combine_contexts(
            hl_relations_context, ll_relations_context, vector_relations_context
        )
        text_units_context = process_combine_contexts(
            hl_text_units_context, ll_text_units_context, vector_text_units_context
        )
    # not necessary to use LLM to generate a response
    if not entities_context and not relations_context:
        return None

    # 转换为 JSON 字符串
    entities_str = json.dumps(entities_context, ensure_ascii=False)
    relations_str = json.dumps(relations_context, ensure_ascii=False)
    text_units_str = json.dumps(text_units_context, ensure_ascii=False)

    result = f"""-----Entities(KG)-----

```json
{entities_str}
```

-----Relationships(KG)-----

```json
{relations_str}
```

-----Document Chunks(DC)-----

```json
{text_units_str}
```

"""
    return result


async def _get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    # get similar entities
    logger.info(
        f"Query nodes: {query}, top_k: {query_param.top_k}, cosine: {entities_vdb.cosine_better_than_threshold}"
    )

    results = await entities_vdb.query(
        query, top_k=query_param.top_k, ids=query_param.ids
    )

    if not len(results):
        return "", "", ""

    # Extract all entity IDs from your results list
    node_ids = [r["entity_name"] for r in results]

    # Call the batch node retrieval and degree functions concurrently.
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(node_ids),
        knowledge_graph_inst.node_degrees_batch(node_ids),
    )

    # Now, if you need the node data and degree in order:
    node_datas = [nodes_dict.get(nid) for nid in node_ids]
    node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {
            **n,
            "entity_name": k["entity_name"],
            "rank": d,
            "created_at": k.get("created_at"),
        }
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entitytext chunk
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas,
        query_param,
        text_chunks_db,
        knowledge_graph_inst,
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas,
        query_param,
        knowledge_graph_inst,
    )

    tokenizer: Tokenizer = text_chunks_db.global_config.get("tokenizer")
    len_node_datas = len(node_datas)
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"] if x["description"] is not None else "",
        max_token_size=query_param.max_token_for_local_context,
        tokenizer=tokenizer,
    )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} chunks"
    )

    # build prompt
    entities_context = []
    for i, n in enumerate(node_datas):
        created_at = n.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Get file path from node data
        file_path = n.get("file_path", "unknown_source")

        entities_context.append(
            {
                "id": i + 1,
                "entity": n["entity_name"],
                "type": n.get("entity_type", "UNKNOWN"),
                "description": n.get("description", "UNKNOWN"),
                "rank": n["rank"],
                "created_at": created_at,
                "file_path": file_path,
            }
        )

    relations_context = []
    for i, e in enumerate(use_relations):
        created_at = e.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Get file path from edge data
        file_path = e.get("file_path", "unknown_source")

        relations_context.append(
            {
                "id": i + 1,
                "entity1": e["src_tgt"][0],
                "entity2": e["src_tgt"][1],
                "description": e["description"],
                "keywords": e["keywords"],
                "weight": e["weight"],
                "rank": e["rank"],
                "created_at": created_at,
                "file_path": file_path,
            }
        )

    text_units_context = []
    for i, t in enumerate(use_text_units):
        text_units_context.append(
            {
                "id": i + 1,
                "content": t["content"],
                "file_path": t.get("file_path", "unknown_source"),
            }
        )
    return entities_context, relations_context, text_units_context


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
        if dp["source_id"] is not None
    ]

    node_names = [dp["entity_name"] for dp in node_datas]
    batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)
    # Build the edges list in the same order as node_datas.
    edges = [batch_edges_dict.get(name, []) for name in node_names]

    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)

    # Batch retrieve one-hop node data using get_nodes_batch
    all_one_hop_nodes_data_dict = await knowledge_graph_inst.get_nodes_batch(
        all_one_hop_nodes
    )
    all_one_hop_nodes_data = [
        all_one_hop_nodes_data_dict.get(e) for e in all_one_hop_nodes
    ]

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    tasks = []

    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = index
                tasks.append((c_id, index, this_edges))

    # Process in batches tasks at a time to avoid overwhelming resources
    batch_size = 5
    results = []

    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i : i + batch_size]
        batch_results = await asyncio.gather(
            *[text_chunks_db.get_by_id(c_id) for c_id, _, _ in batch_tasks]
        )
        results.extend(batch_results)

    for (c_id, index, this_edges), data in zip(tasks, results):
        all_text_units_lookup[c_id] = {
            "data": data,
            "order": index,
            "relation_counts": 0,
        }

        if this_edges:
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    tokenizer: Tokenizer = text_chunks_db.global_config.get("tokenizer")
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
        tokenizer=tokenizer,
    )

    logger.debug(
        f"Truncate chunks from {len(all_text_units_lookup)} to {len(all_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    node_names = [dp["entity_name"] for dp in node_datas]
    batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)

    all_edges = []
    seen = set()

    for node_name in node_names:
        this_edges = batch_edges_dict.get(node_name, [])
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]
    # For edge degrees, use tuples.
    edge_pairs_tuples = list(all_edges)  # all_edges is already a list of tuples

    # Call the batched functions concurrently.
    edge_data_dict, edge_degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
        knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
    )

    # Reconstruct edge_datas list in the same order as the deduplicated results.
    all_edges_data = []
    for pair in all_edges:
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 0.0"
                )
                edge_props["weight"] = 0.0

            combined = {
                "src_tgt": pair,
                "rank": edge_degrees_dict.get(pair, 0),
                **edge_props,
            }
            all_edges_data.append(combined)

    tokenizer: Tokenizer = knowledge_graph_inst.global_config.get("tokenizer")
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"] if x["description"] is not None else "",
        max_token_size=query_param.max_token_for_global_context,
        tokenizer=tokenizer,
    )

    logger.debug(
        f"Truncate relations from {len(all_edges)} to {len(all_edges_data)} (max tokens:{query_param.max_token_for_global_context})"
    )

    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    logger.info(
        f"Query edges: {keywords}, top_k: {query_param.top_k}, cosine: {relationships_vdb.cosine_better_than_threshold}"
    )

    results = await relationships_vdb.query(
        keywords, top_k=query_param.top_k, ids=query_param.ids
    )

    if not len(results):
        return "", "", ""

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in results]
    # For edge degrees, use tuples.
    edge_pairs_tuples = [(r["src_id"], r["tgt_id"]) for r in results]

    # Call the batched functions concurrently.
    edge_data_dict, edge_degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
        knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
    )

    # Reconstruct edge_datas list in the same order as results.
    edge_datas = []
    for k in results:
        pair = (k["src_id"], k["tgt_id"])
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 0.0"
                )
                edge_props["weight"] = 0.0

            # Use edge degree from the batch as rank.
            combined = {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "rank": edge_degrees_dict.get(pair, k.get("rank", 0)),
                "created_at": k.get("created_at", None),
                **edge_props,
            }
            edge_datas.append(combined)

    tokenizer: Tokenizer = text_chunks_db.global_config.get("tokenizer")
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"] if x["description"] is not None else "",
        max_token_size=query_param.max_token_for_global_context,
        tokenizer=tokenizer,
    )
    use_entities, use_text_units = await asyncio.gather(
        _find_most_related_entities_from_relationships(
            edge_datas,
            query_param,
            knowledge_graph_inst,
        ),
        _find_related_text_unit_from_relationships(
            edge_datas,
            query_param,
            text_chunks_db,
            knowledge_graph_inst,
        ),
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} chunks"
    )

    relations_context = []
    for i, e in enumerate(edge_datas):
        created_at = e.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Get file path from edge data
        file_path = e.get("file_path", "unknown_source")

        relations_context.append(
            {
                "id": i + 1,
                "entity1": e["src_id"],
                "entity2": e["tgt_id"],
                "description": e["description"],
                "keywords": e["keywords"],
                "weight": e["weight"],
                "rank": e["rank"],
                "created_at": created_at,
                "file_path": file_path,
            }
        )

    entities_context = []
    for i, n in enumerate(use_entities):
        created_at = n.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Get file path from node data
        file_path = n.get("file_path", "unknown_source")

        entities_context.append(
            {
                "id": i + 1,
                "entity": n["entity_name"],
                "type": n.get("entity_type", "UNKNOWN"),
                "description": n.get("description", "UNKNOWN"),
                "rank": n["rank"],
                "created_at": created_at,
                "file_path": file_path,
            }
        )

    text_units_context = []
    for i, t in enumerate(use_text_units):
        text_units_context.append(
            {
                "id": i + 1,
                "content": t["content"],
                "file_path": t.get("file_path", "unknown"),
            }
        )
    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    # Batch approach: Retrieve nodes and their degrees concurrently with one query each.
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(entity_names),
        knowledge_graph_inst.node_degrees_batch(entity_names),
    )

    # Rebuild the list in the same order as entity_names
    node_datas = []
    for entity_name in entity_names:
        node = nodes_dict.get(entity_name)
        degree = degrees_dict.get(entity_name, 0)
        if node is None:
            logger.warning(f"Node '{entity_name}' not found in batch retrieval.")
            continue
        # Combine the node data with the entity name and computed degree (as rank)
        combined = {**node, "entity_name": entity_name, "rank": degree}
        node_datas.append(combined)

    tokenizer: Tokenizer = knowledge_graph_inst.global_config.get("tokenizer")
    len_node_datas = len(node_datas)
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"] if x["description"] is not None else "",
        max_token_size=query_param.max_token_for_local_context,
        tokenizer=tokenizer,
    )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
        if dp["source_id"] is not None
    ]
    all_text_units_lookup = {}

    async def fetch_chunk_data(c_id, index):
        if c_id not in all_text_units_lookup:
            chunk_data = await text_chunks_db.get_by_id(c_id)
            # Only store valid data
            if chunk_data is not None and "content" in chunk_data:
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                }

    tasks = []
    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            tasks.append(fetch_chunk_data(c_id, index))

    await asyncio.gather(*tasks)

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    tokenizer: Tokenizer = text_chunks_db.global_config.get("tokenizer")
    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
        tokenizer=tokenizer,
    )

    logger.debug(
        f"Truncate chunks from {len(valid_text_units)} to {len(truncated_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    # Handle cache
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    tokenizer: Tokenizer = global_config["tokenizer"]

    _, _, text_units_context = await _get_vector_context(
        query, chunks_vdb, query_param, tokenizer
    )

    if text_units_context is None or len(text_units_context) == 0:
        return PROMPTS["fail_response"]

    text_units_str = json.dumps(text_units_context, ensure_ascii=False)
    if query_param.only_need_context:
        return f"""
---Document Chunks---

```json
{text_units_str}
```

"""
    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # Build system prompt
    user_prompt = (
        query_param.user_prompt
        if query_param.user_prompt
        else PROMPTS["DEFAULT_USER_PROMPT"]
    )
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=text_units_str,
        response_type=query_param.response_type,
        history=history_context,
        user_prompt=user_prompt,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(f"[naive_query]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    if hashing_kv.global_config.get("enable_llm_cache"):
        # Save to cache
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=query_param.mode,
                cache_type="query",
            ),
        )

    return response


# TODO: Deprecated, use user_prompt in QueryParam instead
async def kg_query_with_keywords(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    ll_keywords: list[str] = [],
    hl_keywords: list[str] = [],
    chunks_vdb: BaseVectorStorage | None = None,
) -> str | AsyncIterator[str]:
    """
    Refactored kg_query that does NOT extract keywords by itself.
    It expects hl_keywords and ll_keywords to be set in query_param, or defaults to empty.
    Then it uses those to build context and produce a final LLM response.
    """
    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # If neither has any keywords, you could handle that logic here.
    if not hl_keywords and not ll_keywords:
        logger.warning(
            "No keywords found in query_param. Could default to global mode or fail."
        )
        return PROMPTS["fail_response"]
    if not ll_keywords and query_param.mode in ["local", "hybrid"]:
        logger.warning("low_level_keywords is empty, switching to global mode.")
        query_param.mode = "global"
    if not hl_keywords and query_param.mode in ["global", "hybrid"]:
        logger.warning("high_level_keywords is empty, switching to local mode.")
        query_param.mode = "local"

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    context = await _build_query_context(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb=chunks_vdb,
    )
    if not context:
        return PROMPTS["fail_response"]

    if query_param.only_need_context:
        return context

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(f"[kg_query_with_keywords]Prompt Tokens: {len_of_prompts}")

    # 6. Generate response
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    # Clean up response content
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

        if hashing_kv.global_config.get("enable_llm_cache"):
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    quantized=quantized,
                    min_val=min_val,
                    max_val=max_val,
                    mode=query_param.mode,
                    cache_type="query",
                ),
            )

    return response


# TODO: Deprecated, use user_prompt in QueryParam instead
async def query_with_keywords(
    query: str,
    prompt: str,
    param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> str | AsyncIterator[str]:
    """
    Extract keywords from the query and then use them for retrieving information.

    1. Extracts high-level and low-level keywords from the query
    2. Formats the query with the extracted keywords and prompt
    3. Uses the appropriate query method based on param.mode

    Args:
        query: The user's query
        prompt: Additional prompt to prepend to the query
        param: Query parameters
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entities vector database
        relationships_vdb: Relationships vector database
        chunks_vdb: Document chunks vector database
        text_chunks_db: Text chunks storage
        global_config: Global configuration
        hashing_kv: Cache storage

    Returns:
        Query response or async iterator
    """
    # Extract keywords
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query=query,
        query_param=param,
        global_config=global_config,
        hashing_kv=hashing_kv,
    )

    # Create a new string with the prompt and the keywords
    keywords_str = ", ".join(ll_keywords + hl_keywords)
    formatted_question = (
        f"{prompt}\n\n### Keywords\n\n{keywords_str}\n\n### Query\n\n{query}"
    )

    param.original_query = query

    # Use appropriate query method based on mode
    if param.mode in ["local", "global", "hybrid", "mix"]:
        return await kg_query_with_keywords(
            formatted_question,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
            hl_keywords=hl_keywords,
            ll_keywords=ll_keywords,
            chunks_vdb=chunks_vdb,
        )
    elif param.mode == "naive":
        return await naive_query(
            formatted_question,
            chunks_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
        )
    else:
        raise ValueError(f"Unknown mode {param.mode}")
