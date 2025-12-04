from __future__ import annotations

import time
import asyncio
from typing import Any, cast

from .base import DeletionResult
from .kg.shared_storage import get_graph_db_lock
from .constants import GRAPH_FIELD_SEP
from .utils import compute_mdhash_id, logger
from .base import StorageNameSpace


async def _persist_graph_updates(
    entities_vdb=None,
    relationships_vdb=None,
    chunk_entity_relation_graph=None,
    entity_chunks_storage=None,
    relation_chunks_storage=None,
) -> None:
    """Unified callback to persist updates after graph operations.

    Ensures all relevant storage instances are properly persisted after
    operations like delete, edit, create, or merge.

    Args:
        entities_vdb: Entity vector database storage (optional)
        relationships_vdb: Relationship vector database storage (optional)
        chunk_entity_relation_graph: Graph storage instance (optional)
        entity_chunks_storage: Entity-chunk tracking storage (optional)
        relation_chunks_storage: Relation-chunk tracking storage (optional)
    """
    storages = []

    # Collect all non-None storage instances
    if entities_vdb is not None:
        storages.append(entities_vdb)
    if relationships_vdb is not None:
        storages.append(relationships_vdb)
    if chunk_entity_relation_graph is not None:
        storages.append(chunk_entity_relation_graph)
    if entity_chunks_storage is not None:
        storages.append(entity_chunks_storage)
    if relation_chunks_storage is not None:
        storages.append(relation_chunks_storage)

    # Persist all storage instances in parallel
    if storages:
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in storages  # type: ignore
            ]
        )


async def adelete_by_entity(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    entity_name: str,
    entity_chunks_storage=None,
    relation_chunks_storage=None,
) -> DeletionResult:
    """Asynchronously delete an entity and all its relationships.

    Also cleans up entity_chunks_storage and relation_chunks_storage to remove chunk tracking.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        entity_name: Name of the entity to delete
        entity_chunks_storage: Optional KV storage for tracking chunks that reference this entity
        relation_chunks_storage: Optional KV storage for tracking chunks that reference relations
    """
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            # Check if the entity exists
            if not await chunk_entity_relation_graph.has_node(entity_name):
                logger.warning(f"Entity '{entity_name}' not found.")
                return DeletionResult(
                    status="not_found",
                    doc_id=entity_name,
                    message=f"Entity '{entity_name}' not found.",
                    status_code=404,
                )
            # Retrieve related relationships before deleting the node
            edges = await chunk_entity_relation_graph.get_node_edges(entity_name)
            related_relations_count = len(edges) if edges else 0

            # Clean up chunk tracking storages before deletion
            if entity_chunks_storage is not None:
                # Delete entity's entry from entity_chunks_storage
                await entity_chunks_storage.delete([entity_name])
                logger.info(
                    f"Entity Delete: removed chunk tracking for `{entity_name}`"
                )

            if relation_chunks_storage is not None and edges:
                # Delete all related relationships from relation_chunks_storage
                from .utils import make_relation_chunk_key

                relation_keys_to_delete = []
                for src, tgt in edges:
                    # Normalize entity order for consistent key generation
                    normalized_src, normalized_tgt = sorted([src, tgt])
                    storage_key = make_relation_chunk_key(
                        normalized_src, normalized_tgt
                    )
                    relation_keys_to_delete.append(storage_key)

                if relation_keys_to_delete:
                    await relation_chunks_storage.delete(relation_keys_to_delete)
                    logger.info(
                        f"Entity Delete: removed chunk tracking for {len(relation_keys_to_delete)} relations"
                    )

            await entities_vdb.delete_entity(entity_name)
            await relationships_vdb.delete_entity_relation(entity_name)
            await chunk_entity_relation_graph.delete_node(entity_name)

            message = f"Entity Delete: remove '{entity_name}' and its {related_relations_count} relations"
            logger.info(message)
            await _persist_graph_updates(
                entities_vdb=entities_vdb,
                relationships_vdb=relationships_vdb,
                chunk_entity_relation_graph=chunk_entity_relation_graph,
                entity_chunks_storage=entity_chunks_storage,
                relation_chunks_storage=relation_chunks_storage,
            )
            return DeletionResult(
                status="success",
                doc_id=entity_name,
                message=message,
                status_code=200,
            )
        except Exception as e:
            error_message = f"Error while deleting entity '{entity_name}': {e}"
            logger.error(error_message)
            return DeletionResult(
                status="fail",
                doc_id=entity_name,
                message=error_message,
                status_code=500,
            )


async def adelete_by_relation(
    chunk_entity_relation_graph,
    relationships_vdb,
    source_entity: str,
    target_entity: str,
    relation_chunks_storage=None,
) -> DeletionResult:
    """Asynchronously delete a relation between two entities.

    Also cleans up relation_chunks_storage to remove chunk tracking.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        relationships_vdb: Vector database storage for relationships
        source_entity: Name of the source entity
        target_entity: Name of the target entity
        relation_chunks_storage: Optional KV storage for tracking chunks that reference this relation
    """
    relation_str = f"{source_entity} -> {target_entity}"
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            # Normalize entity order for undirected graph (ensures consistent key generation)
            if source_entity > target_entity:
                source_entity, target_entity = target_entity, source_entity

            # Check if the relation exists
            edge_exists = await chunk_entity_relation_graph.has_edge(
                source_entity, target_entity
            )
            if not edge_exists:
                message = f"Relation from '{source_entity}' to '{target_entity}' does not exist"
                logger.warning(message)
                return DeletionResult(
                    status="not_found",
                    doc_id=relation_str,
                    message=message,
                    status_code=404,
                )

            # Clean up chunk tracking storage before deletion
            if relation_chunks_storage is not None:
                from .utils import make_relation_chunk_key

                # Normalize entity order for consistent key generation
                normalized_src, normalized_tgt = sorted([source_entity, target_entity])
                storage_key = make_relation_chunk_key(normalized_src, normalized_tgt)

                await relation_chunks_storage.delete([storage_key])
                logger.info(
                    f"Relation Delete: removed chunk tracking for `{source_entity}`~`{target_entity}`"
                )

            # Delete relation from vector database
            rel_ids_to_delete = [
                compute_mdhash_id(source_entity + target_entity, prefix="rel-"),
                compute_mdhash_id(target_entity + source_entity, prefix="rel-"),
            ]

            await relationships_vdb.delete(rel_ids_to_delete)

            # Delete relation from knowledge graph
            await chunk_entity_relation_graph.remove_edges(
                [(source_entity, target_entity)]
            )

            message = f"Relation Delete: `{source_entity}`~`{target_entity}` deleted successfully"
            logger.info(message)
            await _persist_graph_updates(
                relationships_vdb=relationships_vdb,
                chunk_entity_relation_graph=chunk_entity_relation_graph,
                relation_chunks_storage=relation_chunks_storage,
            )
            return DeletionResult(
                status="success",
                doc_id=relation_str,
                message=message,
                status_code=200,
            )
        except Exception as e:
            error_message = f"Error while deleting relation from '{source_entity}' to '{target_entity}': {e}"
            logger.error(error_message)
            return DeletionResult(
                status="fail",
                doc_id=relation_str,
                message=error_message,
                status_code=500,
            )


async def aedit_entity(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    entity_name: str,
    updated_data: dict[str, str],
    allow_rename: bool = True,
    entity_chunks_storage=None,
    relation_chunks_storage=None,
) -> dict[str, Any]:
    """Asynchronously edit entity information.

    Updates entity information in the knowledge graph and re-embeds the entity in the vector database.
    Also synchronizes entity_chunks_storage and relation_chunks_storage to track chunk references.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        entity_name: Name of the entity to edit
        updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "entity_type": "new type"}
        allow_rename: Whether to allow entity renaming, defaults to True
        entity_chunks_storage: Optional KV storage for tracking chunks that reference this entity
        relation_chunks_storage: Optional KV storage for tracking chunks that reference relations

    Returns:
        Dictionary containing updated entity information
    """
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            # Save original entity name for chunk tracking updates
            original_entity_name = entity_name

            # 1. Get current entity information
            node_exists = await chunk_entity_relation_graph.has_node(entity_name)
            if not node_exists:
                raise ValueError(f"Entity '{entity_name}' does not exist")
            node_data = await chunk_entity_relation_graph.get_node(entity_name)

            # Check if entity is being renamed
            new_entity_name = updated_data.get("entity_name", entity_name)
            is_renaming = new_entity_name != entity_name

            # If renaming, check if new name already exists
            if is_renaming:
                if not allow_rename:
                    raise ValueError(
                        "Entity renaming is not allowed. Set allow_rename=True to enable this feature"
                    )

                existing_node = await chunk_entity_relation_graph.has_node(
                    new_entity_name
                )
                if existing_node:
                    raise ValueError(
                        f"Entity name '{new_entity_name}' already exists, cannot rename"
                    )

            # 2. Update entity information in the graph
            new_node_data = {**node_data, **updated_data}
            new_node_data["entity_id"] = new_entity_name

            if "entity_name" in new_node_data:
                del new_node_data[
                    "entity_name"
                ]  # Node data should not contain entity_name field

            # If renaming entity
            if is_renaming:
                logger.info(
                    f"Entity Edit: renaming `{entity_name}` to `{new_entity_name}`"
                )

                # Create new entity
                await chunk_entity_relation_graph.upsert_node(
                    new_entity_name, new_node_data
                )

                # Store relationships that need to be updated
                relations_to_update = []
                relations_to_delete = []
                # Get all edges related to the original entity
                edges = await chunk_entity_relation_graph.get_node_edges(entity_name)
                if edges:
                    # Recreate edges for the new entity
                    for source, target in edges:
                        edge_data = await chunk_entity_relation_graph.get_edge(
                            source, target
                        )
                        if edge_data:
                            relations_to_delete.append(
                                compute_mdhash_id(source + target, prefix="rel-")
                            )
                            relations_to_delete.append(
                                compute_mdhash_id(target + source, prefix="rel-")
                            )
                            if source == entity_name:
                                await chunk_entity_relation_graph.upsert_edge(
                                    new_entity_name, target, edge_data
                                )
                                relations_to_update.append(
                                    (new_entity_name, target, edge_data)
                                )
                            else:  # target == entity_name
                                await chunk_entity_relation_graph.upsert_edge(
                                    source, new_entity_name, edge_data
                                )
                                relations_to_update.append(
                                    (source, new_entity_name, edge_data)
                                )

                # Delete old entity
                await chunk_entity_relation_graph.delete_node(entity_name)

                # Delete old entity record from vector database
                old_entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                await entities_vdb.delete([old_entity_id])

                # Delete old relation records from vector database
                await relationships_vdb.delete(relations_to_delete)

                # Update relationship vector representations
                for src, tgt, edge_data in relations_to_update:
                    # Normalize entity order for consistent vector ID generation
                    normalized_src, normalized_tgt = sorted([src, tgt])

                    description = edge_data.get("description", "")
                    keywords = edge_data.get("keywords", "")
                    source_id = edge_data.get("source_id", "")
                    weight = float(edge_data.get("weight", 1.0))

                    # Create content using normalized order
                    content = (
                        f"{normalized_src}\t{normalized_tgt}\n{keywords}\n{description}"
                    )

                    # Calculate relationship ID using normalized order
                    relation_id = compute_mdhash_id(
                        normalized_src + normalized_tgt, prefix="rel-"
                    )

                    # Prepare data for vector database update
                    relation_data = {
                        relation_id: {
                            "content": content,
                            "src_id": normalized_src,
                            "tgt_id": normalized_tgt,
                            "source_id": source_id,
                            "description": description,
                            "keywords": keywords,
                            "weight": weight,
                        }
                    }

                    # Update vector database
                    await relationships_vdb.upsert(relation_data)

                # Update working entity name to new name
                entity_name = new_entity_name

            else:
                # If not renaming, directly update node data
                await chunk_entity_relation_graph.upsert_node(
                    entity_name, new_node_data
                )

            # 3. Recalculate entity's vector representation and update vector database
            description = new_node_data.get("description", "")
            source_id = new_node_data.get("source_id", "")
            entity_type = new_node_data.get("entity_type", "")
            content = entity_name + "\n" + description

            # Calculate entity ID
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")

            # Prepare data for vector database update
            entity_data = {
                entity_id: {
                    "content": content,
                    "entity_name": entity_name,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": entity_type,
                }
            }

            # Update vector database
            await entities_vdb.upsert(entity_data)

            # 4. Update chunk tracking storages
            if entity_chunks_storage is not None or relation_chunks_storage is not None:
                from .utils import (
                    make_relation_chunk_key,
                    compute_incremental_chunk_ids,
                )

                # 4.1 Handle entity chunk tracking
                if entity_chunks_storage is not None:
                    # Get storage key (use original name for renaming scenario)
                    storage_key = original_entity_name if is_renaming else entity_name
                    stored_data = await entity_chunks_storage.get_by_id(storage_key)
                    has_stored_data = (
                        stored_data
                        and isinstance(stored_data, dict)
                        and stored_data.get("chunk_ids")
                    )

                    # Get old and new source_id
                    old_source_id = node_data.get("source_id", "")
                    old_chunk_ids = [
                        cid for cid in old_source_id.split(GRAPH_FIELD_SEP) if cid
                    ]

                    new_source_id = new_node_data.get("source_id", "")
                    new_chunk_ids = [
                        cid for cid in new_source_id.split(GRAPH_FIELD_SEP) if cid
                    ]

                    source_id_changed = set(new_chunk_ids) != set(old_chunk_ids)

                    # Update if: source_id changed OR storage has no data
                    if source_id_changed or not has_stored_data:
                        # Get existing full chunk_ids from storage
                        existing_full_chunk_ids = []
                        if has_stored_data:
                            existing_full_chunk_ids = [
                                cid for cid in stored_data.get("chunk_ids", []) if cid
                            ]

                        # If no stored data exists, use old source_id as baseline
                        if not existing_full_chunk_ids:
                            existing_full_chunk_ids = old_chunk_ids.copy()

                        # Use utility function to compute incremental updates
                        updated_chunk_ids = compute_incremental_chunk_ids(
                            existing_full_chunk_ids, old_chunk_ids, new_chunk_ids
                        )

                        # Update storage (even if updated_chunk_ids is empty)
                        if is_renaming:
                            # Renaming: delete old + create new
                            await entity_chunks_storage.delete([original_entity_name])
                            await entity_chunks_storage.upsert(
                                {
                                    entity_name: {
                                        "chunk_ids": updated_chunk_ids,
                                        "count": len(updated_chunk_ids),
                                    }
                                }
                            )
                        else:
                            # Non-renaming: direct update
                            await entity_chunks_storage.upsert(
                                {
                                    entity_name: {
                                        "chunk_ids": updated_chunk_ids,
                                        "count": len(updated_chunk_ids),
                                    }
                                }
                            )

                        logger.info(
                            f"Entity Edit: find {len(updated_chunk_ids)} chunks related to `{entity_name}`"
                        )

                # 4.2 Handle relation chunk tracking if entity was renamed
                if (
                    is_renaming
                    and relation_chunks_storage is not None
                    and relations_to_update
                ):
                    for src, tgt, edge_data in relations_to_update:
                        # Determine old entity pair (before rename)
                        old_src = original_entity_name if src == entity_name else src
                        old_tgt = original_entity_name if tgt == entity_name else tgt

                        # Normalize entity order for both old and new keys
                        old_normalized_src, old_normalized_tgt = sorted(
                            [old_src, old_tgt]
                        )
                        new_normalized_src, new_normalized_tgt = sorted([src, tgt])

                        # Generate storage keys
                        old_storage_key = make_relation_chunk_key(
                            old_normalized_src, old_normalized_tgt
                        )
                        new_storage_key = make_relation_chunk_key(
                            new_normalized_src, new_normalized_tgt
                        )

                        # If keys are different, we need to move the chunk tracking
                        if old_storage_key != new_storage_key:
                            # Get complete chunk IDs from storage first (preserves all existing references)
                            old_stored_data = await relation_chunks_storage.get_by_id(
                                old_storage_key
                            )
                            relation_chunk_ids = []

                            if old_stored_data and isinstance(old_stored_data, dict):
                                # Use complete chunk_ids from storage
                                relation_chunk_ids = [
                                    cid
                                    for cid in old_stored_data.get("chunk_ids", [])
                                    if cid
                                ]
                            else:
                                # Fallback: if storage has no data, use graph's source_id
                                relation_source_id = edge_data.get("source_id", "")
                                relation_chunk_ids = [
                                    cid
                                    for cid in relation_source_id.split(GRAPH_FIELD_SEP)
                                    if cid
                                ]

                            # Delete old relation chunk tracking
                            await relation_chunks_storage.delete([old_storage_key])

                            # Create new relation chunk tracking (migrate complete data)
                            if relation_chunk_ids:
                                await relation_chunks_storage.upsert(
                                    {
                                        new_storage_key: {
                                            "chunk_ids": relation_chunk_ids,
                                            "count": len(relation_chunk_ids),
                                        }
                                    }
                                )
                    logger.info(
                        f"Entity Edit: migrate {len(relations_to_update)} relations after rename"
                    )

            # 5. Save changes
            await _persist_graph_updates(
                entities_vdb=entities_vdb,
                relationships_vdb=relationships_vdb,
                chunk_entity_relation_graph=chunk_entity_relation_graph,
                entity_chunks_storage=entity_chunks_storage,
                relation_chunks_storage=relation_chunks_storage,
            )

            logger.info(f"Entity Edit: `{entity_name}` successfully updated")
            return await get_entity_info(
                chunk_entity_relation_graph,
                entities_vdb,
                entity_name,
                include_vector_data=True,
            )
        except Exception as e:
            logger.error(f"Error while editing entity '{entity_name}': {e}")
            raise


async def aedit_relation(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    source_entity: str,
    target_entity: str,
    updated_data: dict[str, Any],
    relation_chunks_storage=None,
) -> dict[str, Any]:
    """Asynchronously edit relation information.

    Updates relation (edge) information in the knowledge graph and re-embeds the relation in the vector database.
    Also synchronizes the relation_chunks_storage to track which chunks reference this relation.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        source_entity: Name of the source entity
        target_entity: Name of the target entity
        updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "keywords": "new keywords"}
        relation_chunks_storage: Optional KV storage for tracking chunks that reference this relation

    Returns:
        Dictionary containing updated relation information
    """
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            # Normalize entity order for undirected graph (ensures consistent key generation)
            if source_entity > target_entity:
                source_entity, target_entity = target_entity, source_entity

            # 1. Get current relation information
            edge_exists = await chunk_entity_relation_graph.has_edge(
                source_entity, target_entity
            )
            if not edge_exists:
                raise ValueError(
                    f"Relation from '{source_entity}' to '{target_entity}' does not exist"
                )
            edge_data = await chunk_entity_relation_graph.get_edge(
                source_entity, target_entity
            )
            # Important: First delete the old relation record from the vector database
            # Delete both permutations to handle relationships created before normalization
            rel_ids_to_delete = [
                compute_mdhash_id(source_entity + target_entity, prefix="rel-"),
                compute_mdhash_id(target_entity + source_entity, prefix="rel-"),
            ]
            await relationships_vdb.delete(rel_ids_to_delete)
            logger.debug(
                f"Relation Delete: delete vdb for `{source_entity}`~`{target_entity}`"
            )

            # 2. Update relation information in the graph
            new_edge_data = {**edge_data, **updated_data}
            await chunk_entity_relation_graph.upsert_edge(
                source_entity, target_entity, new_edge_data
            )

            # 3. Recalculate relation's vector representation and update vector database
            description = new_edge_data.get("description", "")
            keywords = new_edge_data.get("keywords", "")
            source_id = new_edge_data.get("source_id", "")
            weight = float(new_edge_data.get("weight", 1.0))

            # Create content for embedding
            content = f"{source_entity}\t{target_entity}\n{keywords}\n{description}"

            # Calculate relation ID
            relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-"
            )

            # Prepare data for vector database update
            relation_data = {
                relation_id: {
                    "content": content,
                    "src_id": source_entity,
                    "tgt_id": target_entity,
                    "source_id": source_id,
                    "description": description,
                    "keywords": keywords,
                    "weight": weight,
                }
            }

            # Update vector database
            await relationships_vdb.upsert(relation_data)

            # 4. Update relation_chunks_storage in two scenarios:
            #    - source_id has changed (edit scenario)
            #    - relation_chunks_storage has no existing data (migration/initialization scenario)
            if relation_chunks_storage is not None:
                from .utils import (
                    make_relation_chunk_key,
                    compute_incremental_chunk_ids,
                )

                storage_key = make_relation_chunk_key(source_entity, target_entity)

                # Check if storage has existing data
                stored_data = await relation_chunks_storage.get_by_id(storage_key)
                has_stored_data = (
                    stored_data
                    and isinstance(stored_data, dict)
                    and stored_data.get("chunk_ids")
                )

                # Get old and new source_id
                old_source_id = edge_data.get("source_id", "")
                old_chunk_ids = [
                    cid for cid in old_source_id.split(GRAPH_FIELD_SEP) if cid
                ]

                new_source_id = new_edge_data.get("source_id", "")
                new_chunk_ids = [
                    cid for cid in new_source_id.split(GRAPH_FIELD_SEP) if cid
                ]

                source_id_changed = set(new_chunk_ids) != set(old_chunk_ids)

                # Update if: source_id changed OR storage has no data
                if source_id_changed or not has_stored_data:
                    # Get existing full chunk_ids from storage
                    existing_full_chunk_ids = []
                    if has_stored_data:
                        existing_full_chunk_ids = [
                            cid for cid in stored_data.get("chunk_ids", []) if cid
                        ]

                    # If no stored data exists, use old source_id as baseline
                    if not existing_full_chunk_ids:
                        existing_full_chunk_ids = old_chunk_ids.copy()

                    # Use utility function to compute incremental updates
                    updated_chunk_ids = compute_incremental_chunk_ids(
                        existing_full_chunk_ids, old_chunk_ids, new_chunk_ids
                    )

                    # Update storage (Update even if updated_chunk_ids is empty)
                    await relation_chunks_storage.upsert(
                        {
                            storage_key: {
                                "chunk_ids": updated_chunk_ids,
                                "count": len(updated_chunk_ids),
                            }
                        }
                    )

                    logger.info(
                        f"Relation Delete: update chunk tracking for `{source_entity}`~`{target_entity}`"
                    )

            # 5. Save changes
            await _persist_graph_updates(
                relationships_vdb=relationships_vdb,
                chunk_entity_relation_graph=chunk_entity_relation_graph,
                relation_chunks_storage=relation_chunks_storage,
            )

            logger.info(
                f"Relation Delete: `{source_entity}`~`{target_entity}`' successfully updated"
            )
            return await get_relation_info(
                chunk_entity_relation_graph,
                relationships_vdb,
                source_entity,
                target_entity,
                include_vector_data=True,
            )
        except Exception as e:
            logger.error(
                f"Error while editing relation from '{source_entity}' to '{target_entity}': {e}"
            )
            raise


async def acreate_entity(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    entity_name: str,
    entity_data: dict[str, Any],
    entity_chunks_storage=None,
    relation_chunks_storage=None,
) -> dict[str, Any]:
    """Asynchronously create a new entity.

    Creates a new entity in the knowledge graph and adds it to the vector database.
    Also synchronizes entity_chunks_storage to track chunk references.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        entity_name: Name of the new entity
        entity_data: Dictionary containing entity attributes, e.g. {"description": "description", "entity_type": "type"}
        entity_chunks_storage: Optional KV storage for tracking chunks that reference this entity
        relation_chunks_storage: Optional KV storage for tracking chunks that reference relations

    Returns:
        Dictionary containing created entity information
    """
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            # Check if entity already exists
            existing_node = await chunk_entity_relation_graph.has_node(entity_name)
            if existing_node:
                raise ValueError(f"Entity '{entity_name}' already exists")

            # Prepare node data with defaults if missing
            node_data = {
                "entity_id": entity_name,
                "entity_type": entity_data.get("entity_type", "UNKNOWN"),
                "description": entity_data.get("description", ""),
                "source_id": entity_data.get("source_id", "manual_creation"),
                "file_path": entity_data.get("file_path", "manual_creation"),
                "created_at": int(time.time()),
            }

            # Add entity to knowledge graph
            await chunk_entity_relation_graph.upsert_node(entity_name, node_data)

            # Prepare content for entity
            description = node_data.get("description", "")
            source_id = node_data.get("source_id", "")
            entity_type = node_data.get("entity_type", "")
            content = entity_name + "\n" + description

            # Calculate entity ID
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")

            # Prepare data for vector database update
            entity_data_for_vdb = {
                entity_id: {
                    "content": content,
                    "entity_name": entity_name,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": entity_type,
                    "file_path": entity_data.get("file_path", "manual_creation"),
                }
            }

            # Update vector database
            await entities_vdb.upsert(entity_data_for_vdb)

            # Update entity_chunks_storage to track chunk references
            if entity_chunks_storage is not None:
                source_id = node_data.get("source_id", "")
                chunk_ids = [cid for cid in source_id.split(GRAPH_FIELD_SEP) if cid]

                if chunk_ids:
                    await entity_chunks_storage.upsert(
                        {
                            entity_name: {
                                "chunk_ids": chunk_ids,
                                "count": len(chunk_ids),
                            }
                        }
                    )
                    logger.info(
                        f"Entity Create: tracked {len(chunk_ids)} chunks for `{entity_name}`"
                    )

            # Save changes
            await _persist_graph_updates(
                entities_vdb=entities_vdb,
                relationships_vdb=relationships_vdb,
                chunk_entity_relation_graph=chunk_entity_relation_graph,
                entity_chunks_storage=entity_chunks_storage,
                relation_chunks_storage=relation_chunks_storage,
            )

            logger.info(f"Entity Create: '{entity_name}' successfully created")
            return await get_entity_info(
                chunk_entity_relation_graph,
                entities_vdb,
                entity_name,
                include_vector_data=True,
            )
        except Exception as e:
            logger.error(f"Error while creating entity '{entity_name}': {e}")
            raise


async def acreate_relation(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    source_entity: str,
    target_entity: str,
    relation_data: dict[str, Any],
    relation_chunks_storage=None,
) -> dict[str, Any]:
    """Asynchronously create a new relation between entities.

    Creates a new relation (edge) in the knowledge graph and adds it to the vector database.
    Also synchronizes relation_chunks_storage to track chunk references.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        source_entity: Name of the source entity
        target_entity: Name of the target entity
        relation_data: Dictionary containing relation attributes, e.g. {"description": "description", "keywords": "keywords"}
        relation_chunks_storage: Optional KV storage for tracking chunks that reference this relation

    Returns:
        Dictionary containing created relation information
    """
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            # Check if both entities exist
            source_exists = await chunk_entity_relation_graph.has_node(source_entity)
            target_exists = await chunk_entity_relation_graph.has_node(target_entity)

            if not source_exists:
                raise ValueError(f"Source entity '{source_entity}' does not exist")
            if not target_exists:
                raise ValueError(f"Target entity '{target_entity}' does not exist")

            # Check if relation already exists
            existing_edge = await chunk_entity_relation_graph.has_edge(
                source_entity, target_entity
            )
            if existing_edge:
                raise ValueError(
                    f"Relation from '{source_entity}' to '{target_entity}' already exists"
                )

            # Prepare edge data with defaults if missing
            edge_data = {
                "description": relation_data.get("description", ""),
                "keywords": relation_data.get("keywords", ""),
                "source_id": relation_data.get("source_id", "manual_creation"),
                "weight": float(relation_data.get("weight", 1.0)),
                "file_path": relation_data.get("file_path", "manual_creation"),
                "created_at": int(time.time()),
            }

            # Add relation to knowledge graph
            await chunk_entity_relation_graph.upsert_edge(
                source_entity, target_entity, edge_data
            )

            # Normalize entity order for undirected relation vector (ensures consistent key generation)
            if source_entity > target_entity:
                source_entity, target_entity = target_entity, source_entity

            # Prepare content for embedding
            description = edge_data.get("description", "")
            keywords = edge_data.get("keywords", "")
            source_id = edge_data.get("source_id", "")
            weight = edge_data.get("weight", 1.0)

            # Create content for embedding
            content = f"{keywords}\t{source_entity}\n{target_entity}\n{description}"

            # Calculate relation ID
            relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-"
            )

            # Prepare data for vector database update
            relation_data_for_vdb = {
                relation_id: {
                    "content": content,
                    "src_id": source_entity,
                    "tgt_id": target_entity,
                    "source_id": source_id,
                    "description": description,
                    "keywords": keywords,
                    "weight": weight,
                    "file_path": relation_data.get("file_path", "manual_creation"),
                }
            }

            # Update vector database
            await relationships_vdb.upsert(relation_data_for_vdb)

            # Update relation_chunks_storage to track chunk references
            if relation_chunks_storage is not None:
                from .utils import make_relation_chunk_key

                # Normalize entity order for consistent key generation
                normalized_src, normalized_tgt = sorted([source_entity, target_entity])
                storage_key = make_relation_chunk_key(normalized_src, normalized_tgt)

                source_id = edge_data.get("source_id", "")
                chunk_ids = [cid for cid in source_id.split(GRAPH_FIELD_SEP) if cid]

                if chunk_ids:
                    await relation_chunks_storage.upsert(
                        {
                            storage_key: {
                                "chunk_ids": chunk_ids,
                                "count": len(chunk_ids),
                            }
                        }
                    )
                    logger.info(
                        f"Relation Create: tracked {len(chunk_ids)} chunks for `{source_entity}`~`{target_entity}`"
                    )

            # Save changes
            await _persist_graph_updates(
                relationships_vdb=relationships_vdb,
                chunk_entity_relation_graph=chunk_entity_relation_graph,
                relation_chunks_storage=relation_chunks_storage,
            )

            logger.info(
                f"Relation Create: `{source_entity}`~`{target_entity}` successfully created"
            )
            return await get_relation_info(
                chunk_entity_relation_graph,
                relationships_vdb,
                source_entity,
                target_entity,
                include_vector_data=True,
            )
        except Exception as e:
            logger.error(
                f"Error while creating relation from '{source_entity}' to '{target_entity}': {e}"
            )
            raise


async def amerge_entities(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    source_entities: list[str],
    target_entity: str,
    merge_strategy: dict[str, str] = None,
    target_entity_data: dict[str, Any] = None,
) -> dict[str, Any]:
    """Asynchronously merge multiple entities into one entity.

    Merges multiple source entities into a target entity, handling all relationships,
    and updating both the knowledge graph and vector database.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        source_entities: List of source entity names to merge
        target_entity: Name of the target entity after merging
        merge_strategy: Deprecated (Each field uses its own default strategy). If provided,
            customizations are applied but a warning is logged.
        target_entity_data: Dictionary of specific values to set for the target entity,
            overriding any merged values, e.g. {"description": "custom description", "entity_type": "PERSON"}

    Returns:
        Dictionary containing the merged entity information
    """
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            # Default merge strategy for entities
            default_entity_merge_strategy = {
                "description": "concatenate",
                "entity_type": "keep_first",
                "source_id": "join_unique",
                "file_path": "join_unique",
            }
            effective_entity_merge_strategy = default_entity_merge_strategy
            if merge_strategy:
                logger.warning(
                    "merge_strategy parameter is deprecated and will be ignored in a future "
                    "release. Provided overrides will be applied for now."
                )
                effective_entity_merge_strategy = {
                    **default_entity_merge_strategy,
                    **merge_strategy,
                }
            target_entity_data = (
                {} if target_entity_data is None else target_entity_data
            )

            # 1. Check if all source entities exist
            source_entities_data = {}
            for entity_name in source_entities:
                node_exists = await chunk_entity_relation_graph.has_node(entity_name)
                if not node_exists:
                    raise ValueError(f"Source entity '{entity_name}' does not exist")
                node_data = await chunk_entity_relation_graph.get_node(entity_name)
                source_entities_data[entity_name] = node_data

            # 2. Check if target entity exists and get its data if it does
            target_exists = await chunk_entity_relation_graph.has_node(target_entity)
            existing_target_entity_data = {}
            if target_exists:
                existing_target_entity_data = (
                    await chunk_entity_relation_graph.get_node(target_entity)
                )
                logger.info(
                    f"Target entity '{target_entity}' already exists, will merge data"
                )

            # 3. Merge entity data
            merged_entity_data = _merge_attributes(
                list(source_entities_data.values())
                + ([existing_target_entity_data] if target_exists else []),
                effective_entity_merge_strategy,
                filter_none_only=False,  # Use entity behavior: filter falsy values
            )

            # Apply any explicitly provided target entity data (overrides merged data)
            for key, value in target_entity_data.items():
                merged_entity_data[key] = value

            # 4. Get all relationships of the source entities and target entity (if exists)
            all_relations = []
            entities_to_collect = source_entities.copy()
            
            # If target entity exists, also collect its relationships for merging
            if target_exists:
                entities_to_collect.append(target_entity)
            
            for entity_name in entities_to_collect:
                # Get all relationships of the entities
                edges = await chunk_entity_relation_graph.get_node_edges(entity_name)
                if edges:
                    for src, tgt in edges:
                        # Ensure src is the current entity
                        if src == entity_name:
                            edge_data = await chunk_entity_relation_graph.get_edge(
                                src, tgt
                            )
                            all_relations.append((src, tgt, edge_data))

            # 5. Create or update the target entity
            merged_entity_data["entity_id"] = target_entity
            if not target_exists:
                await chunk_entity_relation_graph.upsert_node(
                    target_entity, merged_entity_data
                )
                logger.info(f"Created new target entity '{target_entity}'")
            else:
                await chunk_entity_relation_graph.upsert_node(
                    target_entity, merged_entity_data
                )
                logger.info(f"Updated existing target entity '{target_entity}'")

            # 6. Recreate all relationships, pointing to the target entity
            relation_updates = {}  # Track relationships that need to be merged
            relations_to_delete = []

            for src, tgt, edge_data in all_relations:
                relations_to_delete.append(compute_mdhash_id(src + tgt, prefix="rel-"))
                relations_to_delete.append(compute_mdhash_id(tgt + src, prefix="rel-"))
                new_src = target_entity if src in source_entities else src
                new_tgt = target_entity if tgt in source_entities else tgt

                # Skip relationships between source entities to avoid self-loops
                if new_src == new_tgt:
                    logger.info(
                        f"Skipping relationship between source entities: {src} -> {tgt} to avoid self-loop"
                    )
                    continue

                # Check if the same relationship already exists
                relation_key = f"{new_src}|{new_tgt}"
                if relation_key in relation_updates:
                    # Merge relationship data
                    existing_data = relation_updates[relation_key]["data"]
                    merged_relation = _merge_attributes(
                        [existing_data, edge_data],
                        {
                            "description": "concatenate",
                            "keywords": "join_unique_comma",
                            "source_id": "join_unique",
                            "file_path": "join_unique",
                            "weight": "max",
                        },
                        filter_none_only=True,  # Use relation behavior: only filter None
                    )
                    relation_updates[relation_key]["data"] = merged_relation
                    logger.info(
                        f"Merged duplicate relationship: {new_src} -> {new_tgt}"
                    )
                else:
                    relation_updates[relation_key] = {
                        "src": new_src,
                        "tgt": new_tgt,
                        "data": edge_data.copy(),
                    }

            # Apply relationship updates
            for rel_data in relation_updates.values():
                await chunk_entity_relation_graph.upsert_edge(
                    rel_data["src"], rel_data["tgt"], rel_data["data"]
                )
                logger.info(
                    f"Created or updated relationship: {rel_data['src']} -> {rel_data['tgt']}"
                )

                # Delete relationships records from vector database
                await relationships_vdb.delete(relations_to_delete)
                logger.info(
                    f"Deleted {len(relations_to_delete)} relation records for entity from vector database"
                )

            # 7. Update entity vector representation
            description = merged_entity_data.get("description", "")
            source_id = merged_entity_data.get("source_id", "")
            entity_type = merged_entity_data.get("entity_type", "")
            content = target_entity + "\n" + description

            entity_id = compute_mdhash_id(target_entity, prefix="ent-")
            entity_data_for_vdb = {
                entity_id: {
                    "content": content,
                    "entity_name": target_entity,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": entity_type,
                }
            }

            await entities_vdb.upsert(entity_data_for_vdb)

            # 8. Update relationship vector representations
            for rel_data in relation_updates.values():
                src = rel_data["src"]
                tgt = rel_data["tgt"]
                edge_data = rel_data["data"]

                # Normalize entity order for consistent vector storage
                normalized_src, normalized_tgt = sorted([src, tgt])

                description = edge_data.get("description", "")
                keywords = edge_data.get("keywords", "")
                source_id = edge_data.get("source_id", "")
                weight = float(edge_data.get("weight", 1.0))

                # Use normalized order for content and relation ID
                content = (
                    f"{keywords}\t{normalized_src}\n{normalized_tgt}\n{description}"
                )
                relation_id = compute_mdhash_id(
                    normalized_src + normalized_tgt, prefix="rel-"
                )

                relation_data_for_vdb = {
                    relation_id: {
                        "content": content,
                        "src_id": normalized_src,
                        "tgt_id": normalized_tgt,
                        "source_id": source_id,
                        "description": description,
                        "keywords": keywords,
                        "weight": weight,
                    }
                }

                await relationships_vdb.upsert(relation_data_for_vdb)

            # 9. Delete source entities
            for entity_name in source_entities:
                if entity_name == target_entity:
                    logger.info(
                        f"Skipping deletion of '{entity_name}' as it's also the target entity"
                    )
                    continue

                # Delete entity node from knowledge graph
                await chunk_entity_relation_graph.delete_node(entity_name)

                # Delete entity record from vector database
                entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                await entities_vdb.delete([entity_id])

                logger.info(
                    f"Deleted source entity '{entity_name}' and its vector embedding from database"
                )

            # 10. Save changes
            await _persist_graph_updates(
                entities_vdb=entities_vdb,
                relationships_vdb=relationships_vdb,
                chunk_entity_relation_graph=chunk_entity_relation_graph,
            )

            logger.info(
                f"Successfully merged {len(source_entities)} entities into '{target_entity}'"
            )
            return await get_entity_info(
                chunk_entity_relation_graph,
                entities_vdb,
                target_entity,
                include_vector_data=True,
            )

        except Exception as e:
            logger.error(f"Error merging entities: {e}")
            raise


def _merge_attributes(
    data_list: list[dict[str, Any]],
    merge_strategy: dict[str, str],
    filter_none_only: bool = False,
) -> dict[str, Any]:
    """Merge attributes from multiple entities or relationships.

    This unified function handles merging of both entity and relationship attributes,
    applying different merge strategies per field.

    Args:
        data_list: List of dictionaries containing entity or relationship data
        merge_strategy: Merge strategy for each field. Supported strategies:
            - "concatenate": Join all values with GRAPH_FIELD_SEP
            - "keep_first": Keep the first non-empty value
            - "keep_last": Keep the last non-empty value
            - "join_unique": Join unique items separated by GRAPH_FIELD_SEP
            - "join_unique_comma": Join unique items separated by comma and space
            - "max": Keep the maximum numeric value (for numeric fields)
        filter_none_only: If True, only filter None values (keep empty strings, 0, etc.).
            If False, filter all falsy values. Default is False for backward compatibility.

    Returns:
        Dictionary containing merged data
    """
    merged_data = {}

    # Collect all possible keys
    all_keys = set()
    for data in data_list:
        all_keys.update(data.keys())

    # Merge values for each key
    for key in all_keys:
        # Get all values for this key based on filtering mode
        if filter_none_only:
            values = [data.get(key) for data in data_list if data.get(key) is not None]
        else:
            values = [data.get(key) for data in data_list if data.get(key)]

        if not values:
            continue

        # Merge values according to strategy
        strategy = merge_strategy.get(key, "keep_first")

        if strategy == "concatenate":
            # Convert all values to strings and join with GRAPH_FIELD_SEP
            merged_data[key] = GRAPH_FIELD_SEP.join(str(v) for v in values)
        elif strategy == "keep_first":
            merged_data[key] = values[0]
        elif strategy == "keep_last":
            merged_data[key] = values[-1]
        elif strategy == "join_unique":
            # Handle fields separated by GRAPH_FIELD_SEP
            unique_items = set()
            for value in values:
                items = str(value).split(GRAPH_FIELD_SEP)
                unique_items.update(items)
            merged_data[key] = GRAPH_FIELD_SEP.join(unique_items)
        elif strategy == "join_unique_comma":
            # Handle fields separated by comma, join unique items with comma
            unique_items = set()
            for value in values:
                items = str(value).split(",")
                unique_items.update(item.strip() for item in items if item.strip())
            merged_data[key] = ",".join(sorted(unique_items))
        elif strategy == "max":
            # For numeric fields like weight
            try:
                merged_data[key] = max(float(v) for v in values)
            except (ValueError, TypeError):
                # Fallback to first value if conversion fails
                merged_data[key] = values[0]
        else:
            # Default strategy: keep first value
            merged_data[key] = values[0]

    return merged_data


async def get_entity_info(
    chunk_entity_relation_graph,
    entities_vdb,
    entity_name: str,
    include_vector_data: bool = False,
) -> dict[str, str | None | dict[str, str]]:
    """Get detailed information of an entity"""

    # Get information from the graph
    node_data = await chunk_entity_relation_graph.get_node(entity_name)
    source_id = node_data.get("source_id") if node_data else None

    result: dict[str, str | None | dict[str, str]] = {
        "entity_name": entity_name,
        "source_id": source_id,
        "graph_data": node_data,
    }

    # Optional: Get vector database information
    if include_vector_data:
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        vector_data = await entities_vdb.get_by_id(entity_id)
        result["vector_data"] = vector_data

    return result


async def get_relation_info(
    chunk_entity_relation_graph,
    relationships_vdb,
    src_entity: str,
    tgt_entity: str,
    include_vector_data: bool = False,
) -> dict[str, str | None | dict[str, str]]:
    """
    Get detailed information of a relationship between two entities.
    Relationship is unidirectional, swap src_entity and tgt_entity does not change the relationship.

    Args:
        src_entity: Source entity name
        tgt_entity: Target entity name
        include_vector_data: Whether to include vector database information

    Returns:
        Dictionary containing relationship information
    """

    # Get information from the graph
    edge_data = await chunk_entity_relation_graph.get_edge(src_entity, tgt_entity)
    source_id = edge_data.get("source_id") if edge_data else None

    result: dict[str, str | None | dict[str, str]] = {
        "src_entity": src_entity,
        "tgt_entity": tgt_entity,
        "source_id": source_id,
        "graph_data": edge_data,
    }

    # Optional: Get vector database information
    if include_vector_data:
        rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
        vector_data = await relationships_vdb.get_by_id(rel_id)
        result["vector_data"] = vector_data

    return result
