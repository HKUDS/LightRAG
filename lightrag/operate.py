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
    is_float_regex,
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
    update_chunk_cache_list,
    remove_think_tags,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import PROMPTS
from .constants import (
    GRAPH_FIELD_SEP,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_RELATED_CHUNK_NUMBER,
)
from .kg.shared_storage import get_storage_keyed_lock
import time
from dotenv import load_dotenv

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
                if len(_tokens) > max_token_size:
                    for start in range(
                        0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + max_token_size]
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + max_token_size])
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
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
        # max_tokens=summary_max_tokens,
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

    # Check if entity name became empty after normalization
    if not entity_name or not entity_name.strip():
        logger.warning(
            f"Entity extraction error: entity name became empty after normalization. Original: '{record_attributes[1]}'"
        )
        return None

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

    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=chunk_key,
        file_path=file_path,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
):
    if len(record_attributes) < 5 or '"relationship"' not in record_attributes[0]:
        return None
    # add this record as edge
    source = clean_str(record_attributes[1])
    target = clean_str(record_attributes[2])

    # Normalize source and target entity names
    source = normalize_extracted_info(source, is_entity=True)
    target = normalize_extracted_info(target, is_entity=True)

    # Check if source or target became empty after normalization
    if not source or not source.strip():
        logger.warning(
            f"Relationship extraction error: source entity became empty after normalization. Original: '{record_attributes[1]}'"
        )
        return None

    if not target or not target.strip():
        logger.warning(
            f"Relationship extraction error: target entity became empty after normalization. Original: '{record_attributes[2]}'"
        )
        return None

    if source == target:
        logger.debug(
            f"Relationship source and target are the same in: {record_attributes}"
        )
        return None

    edge_description = clean_str(record_attributes[3])
    edge_description = normalize_extracted_info(edge_description)

    edge_keywords = normalize_extracted_info(
        clean_str(record_attributes[4]), is_entity=True
    )
    edge_keywords = edge_keywords.replace("，", ",")

    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1].strip('"').strip("'"))
        if is_float_regex(record_attributes[-1].strip('"').strip("'"))
        else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        file_path=file_path,
    )


async def _rebuild_knowledge_from_chunks(
    entities_to_rebuild: dict[str, set[str]],
    relationships_to_rebuild: dict[tuple[str, str], set[str]],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_storage: BaseKVStorage,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
) -> None:
    """Rebuild entity and relationship descriptions from cached extraction results with parallel processing

    This method uses cached LLM extraction results instead of calling LLM again,
    following the same approach as the insert process. Now with parallel processing
    controlled by llm_model_max_async and using get_storage_keyed_lock for data consistency.

    Args:
        entities_to_rebuild: Dict mapping entity_name -> set of remaining chunk_ids
        relationships_to_rebuild: Dict mapping (src, tgt) -> set of remaining chunk_ids
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_storage: Text chunks storage
        llm_response_cache: LLM response cache
        global_config: Global configuration containing llm_model_max_async
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
    """
    if not entities_to_rebuild and not relationships_to_rebuild:
        return

    # Get all referenced chunk IDs
    all_referenced_chunk_ids = set()
    for chunk_ids in entities_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)
    for chunk_ids in relationships_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)

    status_message = f"Rebuilding knowledge from {len(all_referenced_chunk_ids)} cached chunk extractions (parallel processing)"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Get cached extraction results for these chunks using storage
    #    cached_results： chunk_id -> [list of extraction result from LLM cache sorted by created_at]
    cached_results = await _get_cached_extraction_results(
        llm_response_cache,
        all_referenced_chunk_ids,
        text_chunks_storage=text_chunks_storage,
    )

    if not cached_results:
        status_message = "No cached extraction results found, cannot rebuild"
        logger.warning(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
        return

    # Process cached results to get entities and relationships for each chunk
    chunk_entities = {}  # chunk_id -> {entity_name: [entity_data]}
    chunk_relationships = {}  # chunk_id -> {(src, tgt): [relationship_data]}

    for chunk_id, extraction_results in cached_results.items():
        try:
            # Handle multiple extraction results per chunk
            chunk_entities[chunk_id] = defaultdict(list)
            chunk_relationships[chunk_id] = defaultdict(list)

            # process multiple LLM extraction results for a single chunk_id
            for extraction_result in extraction_results:
                entities, relationships = await _parse_extraction_result(
                    text_chunks_storage=text_chunks_storage,
                    extraction_result=extraction_result,
                    chunk_id=chunk_id,
                )

                # Merge entities and relationships from this extraction result
                # Only keep the first occurrence of each entity_name in the same chunk_id
                for entity_name, entity_list in entities.items():
                    if (
                        entity_name not in chunk_entities[chunk_id]
                        or len(chunk_entities[chunk_id][entity_name]) == 0
                    ):
                        chunk_entities[chunk_id][entity_name].extend(entity_list)

                # Only keep the first occurrence of each rel_key in the same chunk_id
                for rel_key, rel_list in relationships.items():
                    if (
                        rel_key not in chunk_relationships[chunk_id]
                        or len(chunk_relationships[chunk_id][rel_key]) == 0
                    ):
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)

        except Exception as e:
            status_message = (
                f"Failed to parse cached extraction result for chunk {chunk_id}: {e}"
            )
            logger.info(status_message)  # Per requirement, change to info
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            continue

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # Counters for tracking progress
    rebuilt_entities_count = 0
    rebuilt_relationships_count = 0
    failed_entities_count = 0
    failed_relationships_count = 0

    async def _locked_rebuild_entity(entity_name, chunk_ids):
        nonlocal rebuilt_entities_count, failed_entities_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    await _rebuild_single_entity(
                        knowledge_graph_inst=knowledge_graph_inst,
                        entities_vdb=entities_vdb,
                        entity_name=entity_name,
                        chunk_ids=chunk_ids,
                        chunk_entities=chunk_entities,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                    )
                    rebuilt_entities_count += 1
                    status_message = (
                        f"Rebuilt entity: {entity_name} from {len(chunk_ids)} chunks"
                    )
                    logger.info(status_message)
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)
                except Exception as e:
                    failed_entities_count += 1
                    status_message = f"Failed to rebuild entity {entity_name}: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    async def _locked_rebuild_relationship(src, tgt, chunk_ids):
        nonlocal rebuilt_relationships_count, failed_relationships_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                f"{src}-{tgt}", namespace=namespace, enable_logging=False
            ):
                try:
                    await _rebuild_single_relationship(
                        knowledge_graph_inst=knowledge_graph_inst,
                        relationships_vdb=relationships_vdb,
                        src=src,
                        tgt=tgt,
                        chunk_ids=chunk_ids,
                        chunk_relationships=chunk_relationships,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                    )
                    rebuilt_relationships_count += 1
                    status_message = f"Rebuilt relationship: {src}->{tgt} from {len(chunk_ids)} chunks"
                    logger.info(status_message)
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)
                except Exception as e:
                    failed_relationships_count += 1
                    status_message = f"Failed to rebuild relationship {src}->{tgt}: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    # Create tasks for parallel processing
    tasks = []

    # Add entity rebuilding tasks
    for entity_name, chunk_ids in entities_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_entity(entity_name, chunk_ids))
        tasks.append(task)

    # Add relationship rebuilding tasks
    for (src, tgt), chunk_ids in relationships_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_relationship(src, tgt, chunk_ids))
        tasks.append(task)

    # Log parallel processing start
    status_message = f"Starting parallel rebuild of {len(entities_to_rebuild)} entities and {len(relationships_to_rebuild)} relationships (async: {graph_max_async})"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Execute all tasks in parallel with semaphore control
    await asyncio.gather(*tasks)

    # Final status report
    status_message = f"KG rebuild completed: {rebuilt_entities_count} entities and {rebuilt_relationships_count} relationships rebuilt successfully."
    if failed_entities_count > 0 or failed_relationships_count > 0:
        status_message += f" Failed: {failed_entities_count} entities, {failed_relationships_count} relationships."

    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _get_cached_extraction_results(
    llm_response_cache: BaseKVStorage,
    chunk_ids: set[str],
    text_chunks_storage: BaseKVStorage,
) -> dict[str, list[str]]:
    """Get cached extraction results for specific chunk IDs

    Args:
        llm_response_cache: LLM response cache storage
        chunk_ids: Set of chunk IDs to get cached results for
        text_chunks_data: Pre-loaded chunk data (optional, for performance)
        text_chunks_storage: Text chunks storage (fallback if text_chunks_data is None)

    Returns:
        Dict mapping chunk_id -> list of extraction_result_text
    """
    cached_results = {}

    # Collect all LLM cache IDs from chunks
    all_cache_ids = set()

    # Read from storage
    chunk_data_list = await text_chunks_storage.get_by_ids(list(chunk_ids))
    for chunk_id, chunk_data in zip(chunk_ids, chunk_data_list):
        if chunk_data and isinstance(chunk_data, dict):
            llm_cache_list = chunk_data.get("llm_cache_list", [])
            if llm_cache_list:
                all_cache_ids.update(llm_cache_list)
        else:
            logger.warning(
                f"Chunk {chunk_id} data is invalid or None: {type(chunk_data)}"
            )

    if not all_cache_ids:
        logger.warning(f"No LLM cache IDs found for {len(chunk_ids)} chunk IDs")
        return cached_results

    # Batch get LLM cache entries
    cache_data_list = await llm_response_cache.get_by_ids(list(all_cache_ids))

    # Process cache entries and group by chunk_id
    valid_entries = 0
    for cache_id, cache_entry in zip(all_cache_ids, cache_data_list):
        if (
            cache_entry is not None
            and isinstance(cache_entry, dict)
            and cache_entry.get("cache_type") == "extract"
            and cache_entry.get("chunk_id") in chunk_ids
        ):
            chunk_id = cache_entry["chunk_id"]
            extraction_result = cache_entry["return"]
            create_time = cache_entry.get(
                "create_time", 0
            )  # Get creation time, default to 0
            valid_entries += 1

            # Support multiple LLM caches per chunk
            if chunk_id not in cached_results:
                cached_results[chunk_id] = []
            # Store tuple with extraction result and creation time for sorting
            cached_results[chunk_id].append((extraction_result, create_time))

    # Sort extraction results by create_time for each chunk
    for chunk_id in cached_results:
        # Sort by create_time (x[1]), then extract only extraction_result (x[0])
        cached_results[chunk_id].sort(key=lambda x: x[1])
        cached_results[chunk_id] = [item[0] for item in cached_results[chunk_id]]

    logger.info(
        f"Found {valid_entries} valid cache entries, {len(cached_results)} chunks with results"
    )
    return cached_results


async def _parse_extraction_result(
    text_chunks_storage: BaseKVStorage, extraction_result: str, chunk_id: str
) -> tuple[dict, dict]:
    """Parse cached extraction result using the same logic as extract_entities

    Args:
        text_chunks_storage: Text chunks storage to get chunk data
        extraction_result: The cached LLM extraction result
        chunk_id: The chunk ID for source tracking

    Returns:
        Tuple of (entities_dict, relationships_dict)
    """

    # Get chunk data for file_path from storage
    chunk_data = await text_chunks_storage.get_by_id(chunk_id)
    file_path = (
        chunk_data.get("file_path", "unknown_source")
        if chunk_data
        else "unknown_source"
    )
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    )
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    # Parse the extraction result using the same logic as in extract_entities
    records = split_string_by_multi_markers(
        extraction_result,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )
    for record in records:
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [context_base["tuple_delimiter"]]
        )

        # Try to parse as entity
        entity_data = await _handle_single_entity_extraction(
            record_attributes, chunk_id, file_path
        )
        if entity_data is not None:
            maybe_nodes[entity_data["entity_name"]].append(entity_data)
            continue

        # Try to parse as relationship
        relationship_data = await _handle_single_relationship_extraction(
            record_attributes, chunk_id, file_path
        )
        if relationship_data is not None:
            maybe_edges[
                (relationship_data["src_id"], relationship_data["tgt_id"])
            ].append(relationship_data)

    return dict(maybe_nodes), dict(maybe_edges)


async def _rebuild_single_entity(
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name: str,
    chunk_ids: set[str],
    chunk_entities: dict,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
) -> None:
    """Rebuild a single entity from cached extraction results"""

    # Get current entity data
    current_entity = await knowledge_graph_inst.get_node(entity_name)
    if not current_entity:
        return

    # Helper function to update entity in both graph and vector storage
    async def _update_entity_storage(
        final_description: str, entity_type: str, file_paths: set[str]
    ):
        # Update entity in graph storage
        updated_entity_data = {
            **current_entity,
            "description": final_description,
            "entity_type": entity_type,
            "source_id": GRAPH_FIELD_SEP.join(chunk_ids),
            "file_path": GRAPH_FIELD_SEP.join(file_paths)
            if file_paths
            else current_entity.get("file_path", "unknown_source"),
        }
        await knowledge_graph_inst.upsert_node(entity_name, updated_entity_data)

        # Update entity in vector database
        entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")

        # Delete old vector record first
        try:
            await entities_vdb.delete([entity_vdb_id])
        except Exception as e:
            logger.debug(
                f"Could not delete old entity vector record {entity_vdb_id}: {e}"
            )

        # Insert new vector record
        entity_content = f"{entity_name}\n{final_description}"
        await entities_vdb.upsert(
            {
                entity_vdb_id: {
                    "content": entity_content,
                    "entity_name": entity_name,
                    "source_id": updated_entity_data["source_id"],
                    "description": final_description,
                    "entity_type": entity_type,
                    "file_path": updated_entity_data["file_path"],
                }
            }
        )

    # Helper function to generate final description with optional LLM summary
    async def _generate_final_description(combined_description: str) -> str:
        force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]
        num_fragment = combined_description.count(GRAPH_FIELD_SEP) + 1

        if num_fragment >= force_llm_summary_on_merge:
            return await _handle_entity_relation_summary(
                entity_name,
                combined_description,
                global_config,
                llm_response_cache=llm_response_cache,
            )
        else:
            return combined_description

    # Collect all entity data from relevant chunks
    all_entity_data = []
    for chunk_id in chunk_ids:
        if chunk_id in chunk_entities and entity_name in chunk_entities[chunk_id]:
            all_entity_data.extend(chunk_entities[chunk_id][entity_name])

    if not all_entity_data:
        logger.warning(
            f"No cached entity data found for {entity_name}, trying to rebuild from relationships"
        )

        # Get all edges connected to this entity
        edges = await knowledge_graph_inst.get_node_edges(entity_name)
        if not edges:
            logger.warning(f"No relationships found for entity {entity_name}")
            return

        # Collect relationship data to extract entity information
        relationship_descriptions = []
        file_paths = set()

        # Get edge data for all connected relationships
        for src_id, tgt_id in edges:
            edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)
            if edge_data:
                if edge_data.get("description"):
                    relationship_descriptions.append(edge_data["description"])

                if edge_data.get("file_path"):
                    edge_file_paths = edge_data["file_path"].split(GRAPH_FIELD_SEP)
                    file_paths.update(edge_file_paths)

        # Generate description from relationships or fallback to current
        if relationship_descriptions:
            combined_description = GRAPH_FIELD_SEP.join(relationship_descriptions)
            final_description = await _generate_final_description(combined_description)
        else:
            final_description = current_entity.get("description", "")

        entity_type = current_entity.get("entity_type", "UNKNOWN")
        await _update_entity_storage(final_description, entity_type, file_paths)
        return

    # Process cached entity data
    descriptions = []
    entity_types = []
    file_paths = set()

    for entity_data in all_entity_data:
        if entity_data.get("description"):
            descriptions.append(entity_data["description"])
        if entity_data.get("entity_type"):
            entity_types.append(entity_data["entity_type"])
        if entity_data.get("file_path"):
            file_paths.add(entity_data["file_path"])

    # Combine all descriptions
    combined_description = (
        GRAPH_FIELD_SEP.join(descriptions)
        if descriptions
        else current_entity.get("description", "")
    )

    # Get most common entity type
    entity_type = (
        max(set(entity_types), key=entity_types.count)
        if entity_types
        else current_entity.get("entity_type", "UNKNOWN")
    )

    # Generate final description and update storage
    final_description = await _generate_final_description(combined_description)
    await _update_entity_storage(final_description, entity_type, file_paths)


async def _rebuild_single_relationship(
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    src: str,
    tgt: str,
    chunk_ids: set[str],
    chunk_relationships: dict,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
) -> None:
    """Rebuild a single relationship from cached extraction results

    Note: This function assumes the caller has already acquired the appropriate
    keyed lock for the relationship pair to ensure thread safety.
    """

    # Get current relationship data
    current_relationship = await knowledge_graph_inst.get_edge(src, tgt)
    if not current_relationship:
        return

    # Collect all relationship data from relevant chunks
    all_relationship_data = []
    for chunk_id in chunk_ids:
        if chunk_id in chunk_relationships:
            # Check both (src, tgt) and (tgt, src) since relationships can be bidirectional
            for edge_key in [(src, tgt), (tgt, src)]:
                if edge_key in chunk_relationships[chunk_id]:
                    all_relationship_data.extend(
                        chunk_relationships[chunk_id][edge_key]
                    )

    if not all_relationship_data:
        logger.warning(f"No cached relationship data found for {src}-{tgt}")
        return

    # Merge descriptions and keywords
    descriptions = []
    keywords = []
    weights = []
    file_paths = set()

    for rel_data in all_relationship_data:
        if rel_data.get("description"):
            descriptions.append(rel_data["description"])
        if rel_data.get("keywords"):
            keywords.append(rel_data["keywords"])
        if rel_data.get("weight"):
            weights.append(rel_data["weight"])
        if rel_data.get("file_path"):
            file_paths.add(rel_data["file_path"])

    # Combine descriptions and keywords
    combined_description = (
        GRAPH_FIELD_SEP.join(descriptions)
        if descriptions
        else current_relationship.get("description", "")
    )
    combined_keywords = (
        ", ".join(set(keywords))
        if keywords
        else current_relationship.get("keywords", "")
    )
    # weight = (
    #     sum(weights) / len(weights)
    #     if weights
    #     else current_relationship.get("weight", 1.0)
    # )
    weight = sum(weights) if weights else current_relationship.get("weight", 1.0)

    # Use summary if description has too many fragments
    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]
    num_fragment = combined_description.count(GRAPH_FIELD_SEP) + 1

    if num_fragment >= force_llm_summary_on_merge:
        final_description = await _handle_entity_relation_summary(
            f"{src}-{tgt}",
            combined_description,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        final_description = combined_description

    # Update relationship in graph storage
    updated_relationship_data = {
        **current_relationship,
        "description": final_description,
        "keywords": combined_keywords,
        "weight": weight,
        "source_id": GRAPH_FIELD_SEP.join(chunk_ids),
        "file_path": GRAPH_FIELD_SEP.join(file_paths)
        if file_paths
        else current_relationship.get("file_path", "unknown_source"),
    }
    await knowledge_graph_inst.upsert_edge(src, tgt, updated_relationship_data)

    # Update relationship in vector database
    rel_vdb_id = compute_mdhash_id(src + tgt, prefix="rel-")
    rel_vdb_id_reverse = compute_mdhash_id(tgt + src, prefix="rel-")

    # Delete old vector records first (both directions to be safe)
    try:
        await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
    except Exception as e:
        logger.debug(
            f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
        )

    # Insert new vector record
    rel_content = f"{combined_keywords}\t{src}\n{tgt}\n{final_description}"
    await relationships_vdb.upsert(
        {
            rel_vdb_id: {
                "src_id": src,
                "tgt_id": tgt,
                "source_id": updated_relationship_data["source_id"],
                "content": rel_content,
                "keywords": combined_keywords,
                "description": final_description,
                "weight": weight,
                "file_path": updated_relationship_data["file_path"],
            }
        }
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
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_file_paths.extend(
            split_string_by_multi_markers(already_node["file_path"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
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
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
):
    if src_id == tgt_id:
        return None

    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_file_paths = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            # Get weight with default 0.0 if missing
            already_weights.append(already_edge.get("weight", 0.0))

            # Get source_id with empty string default if missing or None
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    split_string_by_multi_markers(
                        already_edge["source_id"], [GRAPH_FIELD_SEP]
                    )
                )

            # Get file_path with empty string default if missing or None
            if already_edge.get("file_path") is not None:
                already_file_paths.extend(
                    split_string_by_multi_markers(
                        already_edge["file_path"], [GRAPH_FIELD_SEP]
                    )
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.append(already_edge["description"])

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )

    # Process edges_data with None checks
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(
            set(
                [dp["description"] for dp in edges_data if dp.get("description")]
                + already_description
            )
        )
    )

    # Split all existing and new keywords into individual terms, then combine and deduplicate
    all_keywords = set()
    # Process already_keywords (which are comma-separated)
    for keyword_str in already_keywords:
        if keyword_str:  # Skip empty strings
            all_keywords.update(k.strip() for k in keyword_str.split(",") if k.strip())
    # Process new keywords from edges_data
    for edge in edges_data:
        if edge.get("keywords"):
            all_keywords.update(
                k.strip() for k in edge["keywords"].split(",") if k.strip()
            )
    # Join all unique keywords with commas
    keywords = ",".join(sorted(all_keywords))

    source_id = GRAPH_FIELD_SEP.join(
        set(
            [dp["source_id"] for dp in edges_data if dp.get("source_id")]
            + already_source_ids
        )
    )
    file_path = GRAPH_FIELD_SEP.join(
        set(
            [dp["file_path"] for dp in edges_data if dp.get("file_path")]
            + already_file_paths
        )
    )

    for need_insert_id in [src_id, tgt_id]:
        workspace = global_config.get("workspace", "")
        namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
        async with get_storage_keyed_lock(
            [need_insert_id], namespace=namespace, enable_logging=False
        ):
            if not (await knowledge_graph_inst.has_node(need_insert_id)):
                await knowledge_graph_inst.upsert_node(
                    need_insert_id,
                    node_data={
                        "entity_id": need_insert_id,
                        "source_id": source_id,
                        "description": description,
                        "entity_type": "UNKNOWN",
                        "file_path": file_path,
                        "created_at": int(time.time()),
                    },
                )

    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    num_fragment = description.count(GRAPH_FIELD_SEP) + 1
    num_new_fragment = len(
        set([dp["description"] for dp in edges_data if dp.get("description")])
    )

    if num_fragment > 1:
        if num_fragment >= force_llm_summary_on_merge:
            status_message = f"LLM merge E: {src_id} - {tgt_id} | {num_new_fragment}+{num_fragment-num_new_fragment}"
            logger.info(status_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            description = await _handle_entity_relation_summary(
                f"({src_id}, {tgt_id})",
                description,
                global_config,
                llm_response_cache,
            )
        else:
            status_message = f"Merge E: {src_id} - {tgt_id} | {num_new_fragment}+{num_fragment-num_new_fragment}"
            logger.info(status_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)

    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            file_path=file_path,
            created_at=int(time.time()),
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
        created_at=int(time.time()),
    )

    return edge_data


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
            all_edges[sorted_edge_key].extend(edges)

    # Centralized processing of all nodes and edges
    total_entities_count = len(all_nodes)
    total_relations_count = len(all_edges)

    # Merge nodes and edges
    log_message = f"Merging stage {current_file_number}/{total_files}: {file_path}"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # Process and update all entities and relationships in parallel
    log_message = f"Processing: {total_entities_count} entities and {total_relations_count} relations (async: {graph_max_async})"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    async def _locked_process_entity_name(entity_name, entities):
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                entity_data = await _merge_nodes_then_upsert(
                    entity_name,
                    entities,
                    knowledge_graph_inst,
                    global_config,
                    pipeline_status,
                    pipeline_status_lock,
                    llm_response_cache,
                )
                if entity_vdb is not None:
                    data_for_vdb = {
                        compute_mdhash_id(entity_data["entity_name"], prefix="ent-"): {
                            "entity_name": entity_data["entity_name"],
                            "entity_type": entity_data["entity_type"],
                            "content": f"{entity_data['entity_name']}\n{entity_data['description']}",
                            "source_id": entity_data["source_id"],
                            "file_path": entity_data.get("file_path", "unknown_source"),
                        }
                    }
                    await entity_vdb.upsert(data_for_vdb)
                return entity_data

    async def _locked_process_edges(edge_key, edges):
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                f"{edge_key[0]}-{edge_key[1]}",
                namespace=namespace,
                enable_logging=False,
            ):
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
                if edge_data is None:
                    return None

                if relationships_vdb is not None:
                    data_for_vdb = {
                        compute_mdhash_id(
                            edge_data["src_id"] + edge_data["tgt_id"], prefix="rel-"
                        ): {
                            "src_id": edge_data["src_id"],
                            "tgt_id": edge_data["tgt_id"],
                            "keywords": edge_data["keywords"],
                            "content": f"{edge_data['src_id']}\t{edge_data['tgt_id']}\n{edge_data['keywords']}\n{edge_data['description']}",
                            "source_id": edge_data["source_id"],
                            "file_path": edge_data.get("file_path", "unknown_source"),
                            "weight": 0
                        }
                    }
                    await relationships_vdb.upsert(data_for_vdb)
                return edge_data

    # Create a single task queue for both entities and edges
    tasks = []

    # Add entity processing tasks
    for entity_name, entities in all_nodes.items():
        tasks.append(
            asyncio.create_task(_locked_process_entity_name(entity_name, entities))
        )

    # Add edge processing tasks
    for edge_key, edges in all_edges.items():
        tasks.append(asyncio.create_task(_locked_process_edges(edge_key, edges)))

    # Execute all tasks in parallel with semaphore control
    await asyncio.gather(*tasks)


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    text_chunks_storage: BaseKVStorage | None = None,
) -> list:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
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
    )

    continue_prompt = PROMPTS["entity_continue_extraction"].format(**context_base)
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    async def _process_extraction_result(
        result: str, chunk_key: str, file_path: str = "unknown_source"
    ):
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
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

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
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )

        return maybe_nodes, maybe_edges

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        Returns:
            tuple: (maybe_nodes, maybe_edges) containing extracted entities and relationships
        """
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # Get file path from chunk data or use default
        file_path = chunk_dp.get("file_path", "unknown_source")

        # Create cache keys collector for batch processing
        cache_keys_collector = []

        # Get initial extraction
        hint_prompt = entity_extract_prompt.format(
            **{**context_base, "input_text": content}
        )

        final_result = await use_llm_func_with_cache(
            hint_prompt,
            use_llm_func,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            cache_keys_collector=cache_keys_collector,
        )

        # Store LLM cache reference in chunk (will be handled by use_llm_func_with_cache)
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
                chunk_id=chunk_key,
                cache_keys_collector=cache_keys_collector,
            )

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)

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
                cache_keys_collector=cache_keys_collector,
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        # Batch update chunk's llm_cache_list with all collected cache keys
        if cache_keys_collector and text_chunks_storage:
            await update_chunk_cache_list(
                chunk_key,
                text_chunks_storage,
                cache_keys_collector,
                "entity_extraction",
            )

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # Return the extracted nodes and edges for centralized processing
        return maybe_nodes, maybe_edges

    # Get max async tasks limit from global_config
    chunk_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = asyncio.Semaphore(chunk_max_async)

    async def _process_with_semaphore(chunk):
        async with semaphore:
            return await _process_single_content(chunk)

    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    # Wait for tasks to complete or for the first exception to occur
    # This allows us to cancel remaining tasks if any task fails
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception
    for task in done:
        if task.exception():
            # If a task failed, cancel all pending tasks
            # This prevents unnecessary processing since the parent function will abort anyway
            for pending_task in pending:
                pending_task.cancel()

            # Wait for cancellation to complete
            if pending:
                await asyncio.wait(pending)

            # Re-raise the exception to notify the caller
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
    args_hash = compute_args_hash(query_param.mode, query)
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
        query,
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
        return context if context is not None else PROMPTS["fail_response"]
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
    logger.debug(
        f"[kg_query] Sending to LLM: {len_of_prompts:,} tokens (Query: {len(tokenizer.encode(query))}, System: {len(tokenizer.encode(sys_prompt))})"
    )

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
    args_hash = compute_args_hash(param.mode, text)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_response is not None:
        try:
            keywords_data = json.loads(cached_response)
            return keywords_data["high_level_keywords"], keywords_data[
                "low_level_keywords"
            ]
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
    logger.debug(
        f"[extract_keywords] Sending to LLM: {len_of_prompts:,} tokens (Prompt: {len_of_prompts})"
    )

    # 5. Call the LLM for keyword extraction
    if param.model_func:
        use_model_func = param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    result = await use_model_func(kw_prompt, keyword_extraction=True)

    # 6. Parse out JSON from the LLM response
    result = remove_think_tags(result)
    match = re.search(r"\{.*?\}", result, re.DOTALL)
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
) -> list[dict]:
    """
    Retrieve text chunks from the vector database without reranking or truncation.

    This function performs vector search to find relevant text chunks for a query.
    Reranking and truncation will be handled later in the unified processing.

    Args:
        query: The query string to search for
        chunks_vdb: Vector database containing document chunks
        query_param: Query parameters including chunk_top_k and ids

    Returns:
        List of text chunks with metadata
    """
    try:
        # Use chunk_top_k if specified, otherwise fall back to top_k
        search_top_k = query_param.chunk_top_k or query_param.top_k

        results = await chunks_vdb.query(query, top_k=search_top_k, ids=query_param.ids)
        if not results:
            return []

        valid_chunks = []
        for result in results:
            if "content" in result:
                chunk_with_metadata = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                    "source_type": "vector",  # Mark the source type
                }
                valid_chunks.append(chunk_with_metadata)

        logger.info(
            f"Naive query: {len(valid_chunks)} chunks (chunk_top_k: {search_top_k})"
        )
        return valid_chunks

    except Exception as e:
        logger.error(f"Error in _get_vector_context: {e}")
        return []


async def _build_query_context(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
):
    logger.info(f"Process {os.getpid()} building query context...")

    # Collect all chunks from different sources
    all_chunks = []
    entities_context = []
    relations_context = []

    # Store original data for later text chunk retrieval
    original_node_datas = []
    original_edge_datas = []

    # Handle local and global modes
    if query_param.mode == "local":
        (
            entities_context,
            relations_context,
            node_datas,
            use_relations,
        ) = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
        )
        original_node_datas = node_datas
        original_edge_datas = use_relations

    elif query_param.mode == "global":
        (
            entities_context,
            relations_context,
            edge_datas,
            use_entities,
        ) = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
        )
        original_edge_datas = edge_datas
        original_node_datas = use_entities

    else:  # hybrid or mix mode
        ll_data = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
        )
        hl_data = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
        )

        (ll_entities_context, ll_relations_context, ll_node_datas, ll_edge_datas) = (
            ll_data
        )
        (hl_entities_context, hl_relations_context, hl_edge_datas, hl_node_datas) = (
            hl_data
        )

        # Get vector chunks first if in mix mode
        if query_param.mode == "mix" and chunks_vdb:
            vector_chunks = await _get_vector_context(
                query,
                chunks_vdb,
                query_param,
            )
            all_chunks.extend(vector_chunks)

        # Store original data from both sources
        original_node_datas = ll_node_datas + hl_node_datas
        original_edge_datas = ll_edge_datas + hl_edge_datas

        # Combine entities and relations contexts
        entities_context = process_combine_contexts(
            ll_entities_context, hl_entities_context
        )
        relations_context = process_combine_contexts(
            hl_relations_context, ll_relations_context
        )

    logger.info(
        f"Initial context: {len(entities_context)} entities, {len(relations_context)} relations, {len(all_chunks)} chunks"
    )

    # Unified token control system - Apply precise token limits to entities and relations
    tokenizer = text_chunks_db.global_config.get("tokenizer")
    if tokenizer:
        # Get new token limits from query_param (with fallback to global_config)
        max_entity_tokens = getattr(
            query_param,
            "max_entity_tokens",
            text_chunks_db.global_config.get(
                "max_entity_tokens", DEFAULT_MAX_ENTITY_TOKENS
            ),
        )
        max_relation_tokens = getattr(
            query_param,
            "max_relation_tokens",
            text_chunks_db.global_config.get(
                "max_relation_tokens", DEFAULT_MAX_RELATION_TOKENS
            ),
        )
        max_total_tokens = getattr(
            query_param,
            "max_total_tokens",
            text_chunks_db.global_config.get(
                "max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS
            ),
        )

        # Truncate entities based on complete JSON serialization
        if entities_context:
            original_entity_count = len(entities_context)

            # Process entities context to replace GRAPH_FIELD_SEP with : in file_path fields
            for entity in entities_context:
                if "file_path" in entity and entity["file_path"]:
                    entity["file_path"] = entity["file_path"].replace(
                        GRAPH_FIELD_SEP, ";"
                    )

            entities_context = truncate_list_by_token_size(
                entities_context,
                key=lambda x: json.dumps(x, ensure_ascii=False),
                max_token_size=max_entity_tokens,
                tokenizer=tokenizer,
            )
            if len(entities_context) < original_entity_count:
                logger.debug(
                    f"Truncated entities: {original_entity_count} -> {len(entities_context)} (entity max tokens: {max_entity_tokens})"
                )

        # Truncate relations based on complete JSON serialization
        if relations_context:
            original_relation_count = len(relations_context)

            # Process relations context to replace GRAPH_FIELD_SEP with : in file_path fields
            for relation in relations_context:
                if "file_path" in relation and relation["file_path"]:
                    relation["file_path"] = relation["file_path"].replace(
                        GRAPH_FIELD_SEP, ";"
                    )

            relations_context = truncate_list_by_token_size(
                relations_context,
                key=lambda x: json.dumps(x, ensure_ascii=False),
                max_token_size=max_relation_tokens,
                tokenizer=tokenizer,
            )
            if len(relations_context) < original_relation_count:
                logger.debug(
                    f"Truncated relations: {original_relation_count} -> {len(relations_context)} (relation max tokens: {max_relation_tokens})"
                )

    # After truncation, get text chunks based on final entities and relations
    logger.info("Getting text chunks based on truncated entities and relations...")

    # Create filtered data based on truncated context
    final_node_datas = []
    if entities_context and original_node_datas:
        final_entity_names = {e["entity"] for e in entities_context}
        seen_nodes = set()
        for node in original_node_datas:
            name = node.get("entity_name")
            if name in final_entity_names and name not in seen_nodes:
                final_node_datas.append(node)
                seen_nodes.add(name)

    final_edge_datas = []
    if relations_context and original_edge_datas:
        final_relation_pairs = {(r["entity1"], r["entity2"]) for r in relations_context}
        seen_edges = set()
        for edge in original_edge_datas:
            src, tgt = edge.get("src_id"), edge.get("tgt_id")
            if src is None or tgt is None:
                src, tgt = edge.get("src_tgt", (None, None))

            pair = (src, tgt)
            if pair in final_relation_pairs and pair not in seen_edges:
                final_edge_datas.append(edge)
                seen_edges.add(pair)

    # Get text chunks based on final filtered data
    text_chunk_tasks = []

    if final_node_datas:
        text_chunk_tasks.append(
            _find_most_related_text_unit_from_entities(
                final_node_datas,
                query_param,
                text_chunks_db,
                knowledge_graph_inst,
            )
        )

    if final_edge_datas:
        text_chunk_tasks.append(
            _find_related_text_unit_from_relationships(
                final_edge_datas,
                query_param,
                text_chunks_db,
            )
        )

    # Execute text chunk retrieval in parallel
    if text_chunk_tasks:
        text_chunk_results = await asyncio.gather(*text_chunk_tasks)
        for chunks in text_chunk_results:
            if chunks:
                all_chunks.extend(chunks)

    # Apply token processing to chunks if tokenizer is available
    text_units_context = []
    if tokenizer and all_chunks:
        # Calculate dynamic token limit for text chunks
        entities_str = json.dumps(entities_context, ensure_ascii=False)
        relations_str = json.dumps(relations_context, ensure_ascii=False)

        # Calculate base context tokens (entities + relations + template)
        kg_context_template = """-----Entities(KG)-----

```json
{entities_str}
```

-----Relationships(KG)-----

```json
{relations_str}
```

-----Document Chunks(DC)-----

```json
[]
```

"""
        kg_context = kg_context_template.format(
            entities_str=entities_str, relations_str=relations_str
        )
        kg_context_tokens = len(tokenizer.encode(kg_context))

        # Calculate actual system prompt overhead dynamically
        # 1. Calculate conversation history tokens
        history_context = ""
        if query_param.conversation_history:
            history_context = get_conversation_turns(
                query_param.conversation_history, query_param.history_turns
            )
        history_tokens = (
            len(tokenizer.encode(history_context)) if history_context else 0
        )

        # 2. Calculate system prompt template tokens (excluding context_data)
        user_prompt = query_param.user_prompt if query_param.user_prompt else ""
        response_type = (
            query_param.response_type
            if query_param.response_type
            else "Multiple Paragraphs"
        )

        # Get the system prompt template from PROMPTS
        sys_prompt_template = text_chunks_db.global_config.get(
            "system_prompt_template", PROMPTS["rag_response"]
        )

        # Create a sample system prompt with placeholders filled (excluding context_data)
        sample_sys_prompt = sys_prompt_template.format(
            history=history_context,
            context_data="",  # Empty for overhead calculation
            response_type=response_type,
            user_prompt=user_prompt,
        )
        sys_prompt_template_tokens = len(tokenizer.encode(sample_sys_prompt))

        # Total system prompt overhead = template + query tokens
        query_tokens = len(tokenizer.encode(query))
        sys_prompt_overhead = sys_prompt_template_tokens + query_tokens

        buffer_tokens = 100  # Safety buffer as requested

        # Calculate available tokens for text chunks
        used_tokens = kg_context_tokens + sys_prompt_overhead + buffer_tokens
        available_chunk_tokens = max_total_tokens - used_tokens

        logger.debug(
            f"Token allocation - Total: {max_total_tokens}, History: {history_tokens}, SysPrompt: {sys_prompt_overhead}, KG: {kg_context_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
        )

        # Re-process chunks with dynamic token limit
        if all_chunks:
            # Create a temporary query_param copy with adjusted chunk token limit
            temp_chunks = [
                {"content": chunk["content"], "file_path": chunk["file_path"]}
                for chunk in all_chunks
            ]

            # Apply token truncation to chunks using the dynamic limit
            truncated_chunks = await process_chunks_unified(
                query=query,
                chunks=temp_chunks,
                query_param=query_param,
                global_config=text_chunks_db.global_config,
                source_type="mixed",
                chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
            )

            # Rebuild text_units_context with truncated chunks
            for i, chunk in enumerate(truncated_chunks):
                text_units_context.append(
                    {
                        "id": i + 1,
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                    }
                )

            logger.debug(
                f"Re-truncated chunks for dynamic token limit: {len(temp_chunks)} -> {len(text_units_context)} (chunk available tokens: {available_chunk_tokens})"
            )

    logger.info(
        f"Final context: {len(entities_context)} entities, {len(relations_context)} relations, {len(text_units_context)} chunks"
    )

    # not necessary to use LLM to generate a response
    if not entities_context and not relations_context:
        return None

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
        return "", "", [], []

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
    ]

    use_relations = await _find_most_related_edges_from_entities(
        node_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Local query: {len(node_datas)} entites, {len(use_relations)} relations"
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
                "created_at": created_at,
                "file_path": file_path,
            }
        )

    return entities_context, relations_context, node_datas, use_relations


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
):
    logger.debug(f"Searching text chunks for {len(node_datas)} entities")

    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])[
            : text_chunks_db.global_config.get(
                "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
            )
        ]
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

    # Sort by relation counts and order, but don't truncate
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    logger.debug(f"Found {len(all_text_units)} entity-related chunks")

    # Add source type marking and return chunk data
    result_chunks = []
    for t in all_text_units:
        chunk_data = t["data"].copy()
        chunk_data["source_type"] = "entity"
        result_chunks.append(chunk_data)

    return result_chunks


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

    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )

    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    query_param: QueryParam,
):
    logger.info(
        f"Query edges: {keywords}, top_k: {query_param.top_k}, cosine: {relationships_vdb.cosine_better_than_threshold}"
    )

    results = await relationships_vdb.query(
        keywords, top_k=query_param.top_k, ids=query_param.ids
    )

    if not len(results):
        return "", "", [], []

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

    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Global query: {len(use_entities)} entites, {len(edge_datas)} relations"
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
                "created_at": created_at,
                "file_path": file_path,
            }
        )

    # Return original data for later text chunk retrieval
    return entities_context, relations_context, edge_datas, use_entities


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

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
):
    logger.debug(f"Searching text chunks for {len(edge_datas)} relationships")

    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])[
            : text_chunks_db.global_config.get(
                "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
            )
        ]
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

    logger.debug(f"Found {len(valid_text_units)} relationship-related chunks")

    # Add source type marking and return chunk data
    result_chunks = []
    for t in valid_text_units:
        chunk_data = t["data"].copy()
        chunk_data["source_type"] = "relationship"
        result_chunks.append(chunk_data)

    return result_chunks


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
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    tokenizer: Tokenizer = global_config["tokenizer"]

    chunks = await _get_vector_context(query, chunks_vdb, query_param)

    if chunks is None or len(chunks) == 0:
        return PROMPTS["fail_response"]

    # Calculate dynamic token limit for chunks
    # Get token limits from query_param (with fallback to global_config)
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Calculate conversation history tokens
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )
    history_tokens = len(tokenizer.encode(history_context)) if history_context else 0

    # Calculate system prompt template tokens (excluding content_data)
    user_prompt = query_param.user_prompt if query_param.user_prompt else ""
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Use the provided system prompt or default
    sys_prompt_template = (
        system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    )

    # Create a sample system prompt with empty content_data to calculate overhead
    sample_sys_prompt = sys_prompt_template.format(
        content_data="",  # Empty for overhead calculation
        response_type=response_type,
        history=history_context,
        user_prompt=user_prompt,
    )
    sys_prompt_template_tokens = len(tokenizer.encode(sample_sys_prompt))

    # Total system prompt overhead = template + query tokens
    query_tokens = len(tokenizer.encode(query))
    sys_prompt_overhead = sys_prompt_template_tokens + query_tokens

    buffer_tokens = 100  # Safety buffer

    # Calculate available tokens for chunks
    used_tokens = sys_prompt_overhead + buffer_tokens
    available_chunk_tokens = max_total_tokens - used_tokens

    logger.debug(
        f"Naive query token allocation - Total: {max_total_tokens}, History: {history_tokens}, SysPrompt: {sys_prompt_overhead}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # Process chunks using unified processing with dynamic token limit
    processed_chunks = await process_chunks_unified(
        query=query,
        chunks=chunks,
        query_param=query_param,
        global_config=global_config,
        source_type="vector",
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
    )

    logger.info(f"Final context: {len(processed_chunks)} chunks")

    # Build text_units_context from processed chunks
    text_units_context = []
    for i, chunk in enumerate(processed_chunks):
        text_units_context.append(
            {
                "id": i + 1,
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
            }
        )

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
    logger.debug(
        f"[naive_query] Sending to LLM: {len_of_prompts:,} tokens (Query: {len(tokenizer.encode(query))}, System: {len(tokenizer.encode(sys_prompt))})"
    )

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

    args_hash = compute_args_hash(query_param.mode, query)
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
        query,
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
    logger.debug(
        f"[kg_query_with_keywords] Sending to LLM: {len_of_prompts:,} tokens (Query: {len(tokenizer.encode(query))}, System: {len(tokenizer.encode(sys_prompt))})"
    )

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


async def apply_rerank_if_enabled(
    query: str,
    retrieved_docs: list[dict],
    global_config: dict,
    enable_rerank: bool = True,
    top_k: int = None,
) -> list[dict]:
    """
    Apply reranking to retrieved documents if rerank is enabled.

    Args:
        query: The search query
        retrieved_docs: List of retrieved documents
        global_config: Global configuration containing rerank settings
        enable_rerank: Whether to enable reranking from query parameter
        top_k: Number of top documents to return after reranking

    Returns:
        Reranked documents if rerank is enabled, otherwise original documents
    """
    if not enable_rerank or not retrieved_docs:
        return retrieved_docs

    rerank_func = global_config.get("rerank_model_func")
    if not rerank_func:
        logger.warning(
            "Rerank is enabled but no rerank model is configured. Please set up a rerank model or set enable_rerank=False in query parameters."
        )
        return retrieved_docs

    try:
        logger.debug(
            f"Applying rerank to {len(retrieved_docs)} documents, returning top {top_k}"
        )

        # Apply reranking - let rerank_model_func handle top_k internally
        reranked_docs = await rerank_func(
            query=query,
            documents=retrieved_docs,
            top_k=top_k,
        )
        if reranked_docs and len(reranked_docs) > 0:
            if len(reranked_docs) > top_k:
                reranked_docs = reranked_docs[:top_k]
            logger.info(
                f"Successfully reranked {len(retrieved_docs)} documents to {len(reranked_docs)}"
            )
            return reranked_docs
        else:
            logger.warning("Rerank returned empty results, using original documents")
            return retrieved_docs

    except Exception as e:
        logger.error(f"Error during reranking: {e}, using original documents")
        return retrieved_docs


async def process_chunks_unified(
    query: str,
    chunks: list[dict],
    query_param: QueryParam,
    global_config: dict,
    source_type: str = "mixed",
    chunk_token_limit: int = None,  # Add parameter for dynamic token limit
) -> list[dict]:
    """
    Unified processing for text chunks: deduplication, chunk_top_k limiting, reranking, and token truncation.

    Args:
        query: Search query for reranking
        chunks: List of text chunks to process
        query_param: Query parameters containing configuration
        global_config: Global configuration dictionary
        source_type: Source type for logging ("vector", "entity", "relationship", "mixed")
        chunk_token_limit: Dynamic token limit for chunks (if None, uses default)

    Returns:
        Processed and filtered list of text chunks
    """
    if not chunks:
        return []

    # 1. Deduplication based on content
    seen_content = set()
    unique_chunks = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if content and content not in seen_content:
            seen_content.add(content)
            unique_chunks.append(chunk)

    logger.debug(
        f"Deduplication: {len(unique_chunks)} chunks (original: {len(chunks)})"
    )

    # 2. Apply reranking if enabled and query is provided
    if query_param.enable_rerank and query and unique_chunks:
        rerank_top_k = query_param.chunk_top_k or len(unique_chunks)
        unique_chunks = await apply_rerank_if_enabled(
            query=query,
            retrieved_docs=unique_chunks,
            global_config=global_config,
            enable_rerank=query_param.enable_rerank,
            top_k=rerank_top_k,
        )
        logger.debug(f"Rerank: {len(unique_chunks)} chunks (source: {source_type})")

    # 3. Apply chunk_top_k limiting if specified
    if query_param.chunk_top_k is not None and query_param.chunk_top_k > 0:
        if len(unique_chunks) > query_param.chunk_top_k:
            unique_chunks = unique_chunks[: query_param.chunk_top_k]
            logger.debug(
                f"Chunk top-k limiting: kept {len(unique_chunks)} chunks (chunk_top_k={query_param.chunk_top_k})"
            )

    # 4. Token-based final truncation
    tokenizer = global_config.get("tokenizer")
    if tokenizer and unique_chunks:
        # Set default chunk_token_limit if not provided
        if chunk_token_limit is None:
            # Get default from query_param or global_config
            chunk_token_limit = getattr(
                query_param,
                "max_total_tokens",
                global_config.get("MAX_TOTAL_TOKENS", 32000),
            )

        original_count = len(unique_chunks)
        unique_chunks = truncate_list_by_token_size(
            unique_chunks,
            key=lambda x: x.get("content", ""),
            max_token_size=chunk_token_limit,
            tokenizer=tokenizer,
        )
        logger.debug(
            f"Token truncation: {len(unique_chunks)} chunks from {original_count} "
            f"(chunk available tokens: {chunk_token_limit}, source: {source_type})"
        )

    return unique_chunks
