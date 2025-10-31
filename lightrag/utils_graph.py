from __future__ import annotations

import time
import asyncio
from typing import Any, cast

from .base import DeletionResult
from .kg.shared_storage import get_storage_keyed_lock
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
    # Use keyed lock for entity to ensure atomic graph and vector db operations
    workspace = entities_vdb.global_config.get("workspace", "")
    namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
    async with get_storage_keyed_lock(
        [entity_name], namespace=namespace, enable_logging=False
    ):
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
    # Normalize entity order for undirected graph (ensures consistent key generation)
    if source_entity > target_entity:
        source_entity, target_entity = target_entity, source_entity

    # Use keyed lock for relation to ensure atomic graph and vector db operations
    workspace = relationships_vdb.global_config.get("workspace", "")
    namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
    sorted_edge_key = sorted([source_entity, target_entity])
    async with get_storage_keyed_lock(
        sorted_edge_key, namespace=namespace, enable_logging=False
    ):
        try:
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


async def _edit_entity_impl(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    entity_name: str,
    updated_data: dict[str, str],
    *,
    entity_chunks_storage=None,
    relation_chunks_storage=None,
) -> dict[str, Any]:
    """Internal helper that edits an entity without acquiring storage locks.

    This function performs the actual entity edit operations without lock management.
    It should only be called by public APIs that have already acquired necessary locks.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        entity_name: Name of the entity to edit
        updated_data: Dictionary containing updated attributes (including optional entity_name for renaming)
        entity_chunks_storage: Optional KV storage for tracking chunks
        relation_chunks_storage: Optional KV storage for tracking relation chunks

    Returns:
        Dictionary containing updated entity information

    Note:
        Caller must acquire appropriate locks before calling this function.
        If renaming (entity_name in updated_data), this function will check if the new name exists.
    """
    new_entity_name = updated_data.get("entity_name", entity_name)
    is_renaming = new_entity_name != entity_name

    original_entity_name = entity_name

    node_exists = await chunk_entity_relation_graph.has_node(entity_name)
    if not node_exists:
        raise ValueError(f"Entity '{entity_name}' does not exist")
    node_data = await chunk_entity_relation_graph.get_node(entity_name)

    if is_renaming:
        existing_node = await chunk_entity_relation_graph.has_node(new_entity_name)
        if existing_node:
            raise ValueError(
                f"Entity name '{new_entity_name}' already exists, cannot rename"
            )

    new_node_data = {**node_data, **updated_data}
    new_node_data["entity_id"] = new_entity_name

    if "entity_name" in new_node_data:
        del new_node_data[
            "entity_name"
        ]  # Node data should not contain entity_name field

    if is_renaming:
        logger.info(f"Entity Edit: renaming `{entity_name}` to `{new_entity_name}`")

        await chunk_entity_relation_graph.upsert_node(new_entity_name, new_node_data)

        relations_to_update = []
        relations_to_delete = []
        edges = await chunk_entity_relation_graph.get_node_edges(entity_name)
        if edges:
            for source, target in edges:
                edge_data = await chunk_entity_relation_graph.get_edge(source, target)
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
                        relations_to_update.append((new_entity_name, target, edge_data))
                    else:  # target == entity_name
                        await chunk_entity_relation_graph.upsert_edge(
                            source, new_entity_name, edge_data
                        )
                        relations_to_update.append((source, new_entity_name, edge_data))

        await chunk_entity_relation_graph.delete_node(entity_name)

        old_entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        await entities_vdb.delete([old_entity_id])

        await relationships_vdb.delete(relations_to_delete)

        for src, tgt, edge_data in relations_to_update:
            normalized_src, normalized_tgt = sorted([src, tgt])

            description = edge_data.get("description", "")
            keywords = edge_data.get("keywords", "")
            source_id = edge_data.get("source_id", "")
            weight = float(edge_data.get("weight", 1.0))

            content = f"{normalized_src}\t{normalized_tgt}\n{keywords}\n{description}"

            relation_id = compute_mdhash_id(
                normalized_src + normalized_tgt, prefix="rel-"
            )

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

            await relationships_vdb.upsert(relation_data)

        entity_name = new_entity_name
    else:
        await chunk_entity_relation_graph.upsert_node(entity_name, new_node_data)

    description = new_node_data.get("description", "")
    source_id = new_node_data.get("source_id", "")
    entity_type = new_node_data.get("entity_type", "")
    content = entity_name + "\n" + description

    entity_id = compute_mdhash_id(entity_name, prefix="ent-")

    entity_data = {
        entity_id: {
            "content": content,
            "entity_name": entity_name,
            "source_id": source_id,
            "description": description,
            "entity_type": entity_type,
        }
    }

    await entities_vdb.upsert(entity_data)

    if entity_chunks_storage is not None or relation_chunks_storage is not None:
        from .utils import make_relation_chunk_key, compute_incremental_chunk_ids

        if entity_chunks_storage is not None:
            storage_key = original_entity_name if is_renaming else entity_name
            stored_data = await entity_chunks_storage.get_by_id(storage_key)
            has_stored_data = (
                stored_data
                and isinstance(stored_data, dict)
                and stored_data.get("chunk_ids")
            )

            old_source_id = node_data.get("source_id", "")
            old_chunk_ids = [cid for cid in old_source_id.split(GRAPH_FIELD_SEP) if cid]

            new_source_id = new_node_data.get("source_id", "")
            new_chunk_ids = [cid for cid in new_source_id.split(GRAPH_FIELD_SEP) if cid]

            source_id_changed = set(new_chunk_ids) != set(old_chunk_ids)

            if source_id_changed or not has_stored_data or is_renaming:
                existing_full_chunk_ids = []
                if has_stored_data:
                    existing_full_chunk_ids = [
                        cid for cid in stored_data.get("chunk_ids", []) if cid
                    ]

                if not existing_full_chunk_ids:
                    existing_full_chunk_ids = old_chunk_ids.copy()

                updated_chunk_ids = compute_incremental_chunk_ids(
                    existing_full_chunk_ids, old_chunk_ids, new_chunk_ids
                )

                if is_renaming:
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

        if is_renaming and relation_chunks_storage is not None and relations_to_update:
            for src, tgt, edge_data in relations_to_update:
                old_src = original_entity_name if src == entity_name else src
                old_tgt = original_entity_name if tgt == entity_name else tgt

                old_normalized_src, old_normalized_tgt = sorted([old_src, old_tgt])
                new_normalized_src, new_normalized_tgt = sorted([src, tgt])

                old_storage_key = make_relation_chunk_key(
                    old_normalized_src, old_normalized_tgt
                )
                new_storage_key = make_relation_chunk_key(
                    new_normalized_src, new_normalized_tgt
                )

                if old_storage_key != new_storage_key:
                    old_stored_data = await relation_chunks_storage.get_by_id(
                        old_storage_key
                    )
                    relation_chunk_ids = []

                    if old_stored_data and isinstance(old_stored_data, dict):
                        relation_chunk_ids = [
                            cid for cid in old_stored_data.get("chunk_ids", []) if cid
                        ]
                    else:
                        relation_source_id = edge_data.get("source_id", "")
                        relation_chunk_ids = [
                            cid
                            for cid in relation_source_id.split(GRAPH_FIELD_SEP)
                            if cid
                        ]

                    await relation_chunks_storage.delete([old_storage_key])

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


async def aedit_entity(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    entity_name: str,
    updated_data: dict[str, str],
    allow_rename: bool = True,
    allow_merge: bool = False,
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
        allow_merge: Whether to merge into an existing entity when renaming to an existing name, defaults to False
        entity_chunks_storage: Optional KV storage for tracking chunks that reference this entity
        relation_chunks_storage: Optional KV storage for tracking chunks that reference relations

    Returns:
        Dictionary containing updated entity information and operation summary with the following structure:
        {
            "entity_name": str,           # Name of the entity
            "description": str,           # Entity description
            "entity_type": str,           # Entity type
            "source_id": str,            # Source chunk IDs
            ...                          # Other entity properties
            "operation_summary": {
                "merged": bool,          # Whether entity was merged
                "merge_status": str,     # "success" | "failed" | "not_attempted"
                "merge_error": str | None,  # Error message if merge failed
                "operation_status": str, # "success" | "partial_success" | "failure"
                "target_entity": str | None,  # Target entity name if renaming/merging
                "final_entity": str,     # Final entity name after operation
                "renamed": bool          # Whether entity was renamed
            }
        }

        operation_status values:
            - "success": Operation completed successfully (update/rename/merge all succeeded)
            - "partial_success": Non-name updates succeeded but merge failed
            - "failure": Operation failed completely

        merge_status values:
            - "success": Entity successfully merged into target
            - "failed": Merge operation failed
            - "not_attempted": No merge was attempted (normal update/rename)
    """
    new_entity_name = updated_data.get("entity_name", entity_name)
    is_renaming = new_entity_name != entity_name

    lock_keys = sorted({entity_name, new_entity_name}) if is_renaming else [entity_name]

    workspace = entities_vdb.global_config.get("workspace", "")
    namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"

    operation_summary: dict[str, Any] = {
        "merged": False,
        "merge_status": "not_attempted",
        "merge_error": None,
        "operation_status": "success",
        "target_entity": None,
        "final_entity": new_entity_name if is_renaming else entity_name,
        "renamed": is_renaming,
    }
    async with get_storage_keyed_lock(
        lock_keys, namespace=namespace, enable_logging=False
    ):
        try:
            if is_renaming and not allow_rename:
                raise ValueError(
                    "Entity renaming is not allowed. Set allow_rename=True to enable this feature"
                )

            if is_renaming:
                target_exists = await chunk_entity_relation_graph.has_node(
                    new_entity_name
                )
                if target_exists:
                    if not allow_merge:
                        raise ValueError(
                            f"Entity name '{new_entity_name}' already exists, cannot rename"
                        )

                    logger.info(
                        f"Entity Edit: `{entity_name}` will be merged into `{new_entity_name}`"
                    )

                    # Track whether non-name updates were applied
                    non_name_updates_applied = False
                    non_name_updates = {
                        key: value
                        for key, value in updated_data.items()
                        if key != "entity_name"
                    }

                    # Apply non-name updates first
                    if non_name_updates:
                        try:
                            logger.info(
                                "Entity Edit: applying non-name updates before merge"
                            )
                            await _edit_entity_impl(
                                chunk_entity_relation_graph,
                                entities_vdb,
                                relationships_vdb,
                                entity_name,
                                non_name_updates,
                                entity_chunks_storage=entity_chunks_storage,
                                relation_chunks_storage=relation_chunks_storage,
                            )
                            non_name_updates_applied = True
                        except Exception as update_error:
                            # If update fails, re-raise immediately
                            logger.error(
                                f"Entity Edit: non-name updates failed: {update_error}"
                            )
                            raise

                    # Attempt to merge entities
                    try:
                        merge_result = await _merge_entities_impl(
                            chunk_entity_relation_graph,
                            entities_vdb,
                            relationships_vdb,
                            [entity_name],
                            new_entity_name,
                            merge_strategy=None,
                            target_entity_data=None,
                            entity_chunks_storage=entity_chunks_storage,
                            relation_chunks_storage=relation_chunks_storage,
                        )

                        # Merge succeeded
                        operation_summary.update(
                            {
                                "merged": True,
                                "merge_status": "success",
                                "merge_error": None,
                                "operation_status": "success",
                                "target_entity": new_entity_name,
                                "final_entity": new_entity_name,
                            }
                        )
                        return {**merge_result, "operation_summary": operation_summary}

                    except Exception as merge_error:
                        # Merge failed, but update may have succeeded
                        logger.error(f"Entity Edit: merge failed: {merge_error}")

                        # Return partial success status (update succeeded but merge failed)
                        operation_summary.update(
                            {
                                "merged": False,
                                "merge_status": "failed",
                                "merge_error": str(merge_error),
                                "operation_status": "partial_success"
                                if non_name_updates_applied
                                else "failure",
                                "target_entity": new_entity_name,
                                "final_entity": entity_name,  # Keep source entity name
                            }
                        )

                        # Get current entity info (with applied updates if any)
                        entity_info = await get_entity_info(
                            chunk_entity_relation_graph,
                            entities_vdb,
                            entity_name,
                            include_vector_data=True,
                        )
                        return {**entity_info, "operation_summary": operation_summary}

            # Normal edit flow (no merge involved)
            edit_result = await _edit_entity_impl(
                chunk_entity_relation_graph,
                entities_vdb,
                relationships_vdb,
                entity_name,
                updated_data,
                entity_chunks_storage=entity_chunks_storage,
                relation_chunks_storage=relation_chunks_storage,
            )
            operation_summary["operation_status"] = "success"
            return {**edit_result, "operation_summary": operation_summary}

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
    # Normalize entity order for undirected graph (ensures consistent key generation)
    if source_entity > target_entity:
        source_entity, target_entity = target_entity, source_entity

    # Use keyed lock for relation to ensure atomic graph and vector db operations
    workspace = relationships_vdb.global_config.get("workspace", "")
    namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
    sorted_edge_key = sorted([source_entity, target_entity])
    async with get_storage_keyed_lock(
        sorted_edge_key, namespace=namespace, enable_logging=False
    ):
        try:
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
    # Use keyed lock for entity to ensure atomic graph and vector db operations
    workspace = entities_vdb.global_config.get("workspace", "")
    namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
    async with get_storage_keyed_lock(
        [entity_name], namespace=namespace, enable_logging=False
    ):
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
    # Use keyed lock for relation to ensure atomic graph and vector db operations
    workspace = relationships_vdb.global_config.get("workspace", "")
    namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
    sorted_edge_key = sorted([source_entity, target_entity])
    async with get_storage_keyed_lock(
        sorted_edge_key, namespace=namespace, enable_logging=False
    ):
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


async def _merge_entities_impl(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    source_entities: list[str],
    target_entity: str,
    *,
    merge_strategy: dict[str, str] = None,
    target_entity_data: dict[str, Any] = None,
    entity_chunks_storage=None,
    relation_chunks_storage=None,
) -> dict[str, Any]:
    """Internal helper that merges entities without acquiring storage locks.

    This function performs the actual entity merge operations without lock management.
    It should only be called by public APIs that have already acquired necessary locks.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        source_entities: List of source entity names to merge
        target_entity: Name of the target entity after merging
        merge_strategy: Deprecated. Merge strategy for each field (optional)
        target_entity_data: Dictionary of specific values to set for target entity (optional)
        entity_chunks_storage: Optional KV storage for tracking chunks
        relation_chunks_storage: Optional KV storage for tracking relation chunks

    Returns:
        Dictionary containing the merged entity information

    Note:
        Caller must acquire appropriate locks before calling this function.
        All source entities and the target entity should be locked together.
    """
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
            "Entity Merge: merge_strategy parameter is deprecated and will be ignored in a future release."
        )
        effective_entity_merge_strategy = {
            **default_entity_merge_strategy,
            **merge_strategy,
        }
    target_entity_data = {} if target_entity_data is None else target_entity_data

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
        existing_target_entity_data = await chunk_entity_relation_graph.get_node(
            target_entity
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

    # If target entity exists and not already in source_entities, add it
    if target_exists and target_entity not in source_entities:
        entities_to_collect.append(target_entity)

    for entity_name in entities_to_collect:
        # Get all relationships of the entities
        edges = await chunk_entity_relation_graph.get_node_edges(entity_name)
        if edges:
            for src, tgt in edges:
                # Ensure src is the current entity
                if src == entity_name:
                    edge_data = await chunk_entity_relation_graph.get_edge(src, tgt)
                    all_relations.append((src, tgt, edge_data))

    # 5. Create or update the target entity
    merged_entity_data["entity_id"] = target_entity
    if not target_exists:
        await chunk_entity_relation_graph.upsert_node(target_entity, merged_entity_data)
        logger.info(f"Entity Merge: created target '{target_entity}'")
    else:
        await chunk_entity_relation_graph.upsert_node(target_entity, merged_entity_data)
        logger.info(f"Entity Merge: Updated target '{target_entity}'")

    # 6. Recreate all relations pointing to the target entity in KG
    # Also collect chunk tracking information in the same loop
    relation_updates = {}  # Track relationships that need to be merged
    relations_to_delete = []

    # Initialize chunk tracking variables
    relation_chunk_tracking = {}  # key: storage_key, value: list of chunk_ids
    old_relation_keys_to_delete = []

    for src, tgt, edge_data in all_relations:
        relations_to_delete.append(compute_mdhash_id(src + tgt, prefix="rel-"))
        relations_to_delete.append(compute_mdhash_id(tgt + src, prefix="rel-"))

        # Collect old chunk tracking key for deletion
        if relation_chunks_storage is not None:
            from .utils import make_relation_chunk_key

            old_storage_key = make_relation_chunk_key(src, tgt)
            old_relation_keys_to_delete.append(old_storage_key)

        new_src = target_entity if src in source_entities else src
        new_tgt = target_entity if tgt in source_entities else tgt

        # Skip relationships between source entities to avoid self-loops
        if new_src == new_tgt:
            logger.info(f"Entity Merge: skipping `{src}`~`{tgt}` to avoid self-loop")
            continue

        # Normalize entity order for consistent duplicate detection (undirected relationships)
        normalized_src, normalized_tgt = sorted([new_src, new_tgt])
        relation_key = f"{normalized_src}|{normalized_tgt}"

        # Process chunk tracking for this relation
        if relation_chunks_storage is not None:
            storage_key = make_relation_chunk_key(normalized_src, normalized_tgt)

            # Get chunk_ids from storage for this original relation
            stored = await relation_chunks_storage.get_by_id(old_storage_key)

            if stored is not None and isinstance(stored, dict):
                chunk_ids = [cid for cid in stored.get("chunk_ids", []) if cid]
            else:
                # Fallback to source_id from graph
                source_id = edge_data.get("source_id", "")
                chunk_ids = [cid for cid in source_id.split(GRAPH_FIELD_SEP) if cid]

            # Accumulate chunk_ids with ordered deduplication
            if storage_key not in relation_chunk_tracking:
                relation_chunk_tracking[storage_key] = []

            existing_chunks = set(relation_chunk_tracking[storage_key])
            for chunk_id in chunk_ids:
                if chunk_id not in existing_chunks:
                    existing_chunks.add(chunk_id)
                    relation_chunk_tracking[storage_key].append(chunk_id)

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
            logger.debug(
                f"Entity Merge: deduplicating relation `{normalized_src}`~`{normalized_tgt}`"
            )
        else:
            relation_updates[relation_key] = {
                "graph_src": new_src,
                "graph_tgt": new_tgt,
                "norm_src": normalized_src,
                "norm_tgt": normalized_tgt,
                "data": edge_data.copy(),
            }

    # Apply relationship updates
    logger.info(f"Entity Merge: updatign {len(relation_updates)} relations")
    for rel_data in relation_updates.values():
        await chunk_entity_relation_graph.upsert_edge(
            rel_data["graph_src"], rel_data["graph_tgt"], rel_data["data"]
        )
        logger.info(
            f"Entity Merge: updating relation `{rel_data['graph_src']}`~`{rel_data['graph_tgt']}`"
        )

    # Update relation chunk tracking storage
    if relation_chunks_storage is not None and all_relations:
        if old_relation_keys_to_delete:
            await relation_chunks_storage.delete(old_relation_keys_to_delete)

        if relation_chunk_tracking:
            updates = {}
            for storage_key, chunk_ids in relation_chunk_tracking.items():
                updates[storage_key] = {
                    "chunk_ids": chunk_ids,
                    "count": len(chunk_ids),
                }

            await relation_chunks_storage.upsert(updates)
            logger.info(
                f"Entity Merge: {len(updates)} relation chunk tracking records updated"
            )

    # 7. Update relationship vector representations
    logger.debug(
        f"Entity Merge: deleting {len(relations_to_delete)} relations from vdb"
    )
    await relationships_vdb.delete(relations_to_delete)

    for rel_data in relation_updates.values():
        edge_data = rel_data["data"]
        normalized_src = rel_data["norm_src"]
        normalized_tgt = rel_data["norm_tgt"]

        description = edge_data.get("description", "")
        keywords = edge_data.get("keywords", "")
        source_id = edge_data.get("source_id", "")
        weight = float(edge_data.get("weight", 1.0))

        # Use normalized order for content and relation ID
        content = f"{keywords}\t{normalized_src}\n{normalized_tgt}\n{description}"
        relation_id = compute_mdhash_id(normalized_src + normalized_tgt, prefix="rel-")

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
        logger.debug(
            f"Entity Merge: updating vdb `{normalized_src}`~`{normalized_tgt}`"
        )

    logger.info(f"Entity Merge: {len(relation_updates)} relations in vdb updated")

    # 8. Update entity vector representation
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
    logger.info(f"Entity Merge: updating vdb `{target_entity}`")

    # 9. Merge entity chunk tracking (source entities first, then target entity)
    if entity_chunks_storage is not None:
        all_chunk_id_lists = []

        # Build list of entities to process (source entities first, then target entity)
        entities_to_process = []

        # Add source entities first (excluding target if it's already in source list)
        for entity_name in source_entities:
            if entity_name != target_entity:
                entities_to_process.append(entity_name)

        # Add target entity last (if it exists)
        if target_exists:
            entities_to_process.append(target_entity)

        # Process all entities in order with unified logic
        for entity_name in entities_to_process:
            stored = await entity_chunks_storage.get_by_id(entity_name)
            if stored and isinstance(stored, dict):
                chunk_ids = [cid for cid in stored.get("chunk_ids", []) if cid]
                if chunk_ids:
                    all_chunk_id_lists.append(chunk_ids)

        # Merge chunk_ids with ordered deduplication (preserves order, source entities first)
        merged_chunk_ids = []
        seen = set()
        for chunk_id_list in all_chunk_id_lists:
            for chunk_id in chunk_id_list:
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    merged_chunk_ids.append(chunk_id)

        # Delete source entities' chunk tracking records
        entity_keys_to_delete = [e for e in source_entities if e != target_entity]
        if entity_keys_to_delete:
            await entity_chunks_storage.delete(entity_keys_to_delete)

        # Update target entity's chunk tracking
        if merged_chunk_ids:
            await entity_chunks_storage.upsert(
                {
                    target_entity: {
                        "chunk_ids": merged_chunk_ids,
                        "count": len(merged_chunk_ids),
                    }
                }
            )
            logger.info(
                f"Entity Merge: find {len(merged_chunk_ids)} chunks related to '{target_entity}'"
            )

    # 10. Delete source entities
    for entity_name in source_entities:
        if entity_name == target_entity:
            logger.warning(
                f"Entity Merge: source entity'{entity_name}' is same as target entity"
            )
            continue

        logger.info(f"Entity Merge: deleting '{entity_name}' from KG and vdb")

        # Delete entity node and related edges from knowledge graph
        await chunk_entity_relation_graph.delete_node(entity_name)

        # Delete entity record from vector database
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        await entities_vdb.delete([entity_id])

    # 11. Save changes
    await _persist_graph_updates(
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        chunk_entity_relation_graph=chunk_entity_relation_graph,
        entity_chunks_storage=entity_chunks_storage,
        relation_chunks_storage=relation_chunks_storage,
    )

    logger.info(
        f"Entity Merge: successfully merged {len(source_entities)} entities into '{target_entity}'"
    )
    return await get_entity_info(
        chunk_entity_relation_graph,
        entities_vdb,
        target_entity,
        include_vector_data=True,
    )


async def amerge_entities(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    source_entities: list[str],
    target_entity: str,
    merge_strategy: dict[str, str] = None,
    target_entity_data: dict[str, Any] = None,
    entity_chunks_storage=None,
    relation_chunks_storage=None,
) -> dict[str, Any]:
    """Asynchronously merge multiple entities into one entity.

    Merges multiple source entities into a target entity, handling all relationships,
    and updating both the knowledge graph and vector database.
    Also merges chunk tracking information from entity_chunks_storage and relation_chunks_storage.

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
        entity_chunks_storage: Optional KV storage for tracking chunks that reference entities
        relation_chunks_storage: Optional KV storage for tracking chunks that reference relations

    Returns:
        Dictionary containing the merged entity information
    """
    # Collect all entities involved (source + target) and lock them all in sorted order
    all_entities = set(source_entities)
    all_entities.add(target_entity)
    lock_keys = sorted(all_entities)

    workspace = entities_vdb.global_config.get("workspace", "")
    namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
    async with get_storage_keyed_lock(
        lock_keys, namespace=namespace, enable_logging=False
    ):
        try:
            return await _merge_entities_impl(
                chunk_entity_relation_graph,
                entities_vdb,
                relationships_vdb,
                source_entities,
                target_entity,
                merge_strategy=merge_strategy,
                target_entity_data=target_entity_data,
                entity_chunks_storage=entity_chunks_storage,
                relation_chunks_storage=relation_chunks_storage,
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
