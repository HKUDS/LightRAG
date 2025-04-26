from __future__ import annotations

import asyncio
from typing import Any, cast

from .kg.shared_storage import get_graph_db_lock
from .prompt import GRAPH_FIELD_SEP
from .utils import compute_mdhash_id, logger
from .base import StorageNameSpace


async def adelete_by_entity(
    chunk_entity_relation_graph, entities_vdb, relationships_vdb, entity_name: str
) -> None:
    """Asynchronously delete an entity and all its relationships.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        entity_name: Name of the entity to delete
    """
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            await entities_vdb.delete_entity(entity_name)
            await relationships_vdb.delete_entity_relation(entity_name)
            await chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await _delete_by_entity_done(
                entities_vdb, relationships_vdb, chunk_entity_relation_graph
            )
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")


async def _delete_by_entity_done(
    entities_vdb, relationships_vdb, chunk_entity_relation_graph
) -> None:
    """Callback after entity deletion is complete, ensures updates are persisted"""
    await asyncio.gather(
        *[
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                entities_vdb,
                relationships_vdb,
                chunk_entity_relation_graph,
            ]
        ]
    )


async def adelete_by_relation(
    chunk_entity_relation_graph,
    relationships_vdb,
    source_entity: str,
    target_entity: str,
) -> None:
    """Asynchronously delete a relation between two entities.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        relationships_vdb: Vector database storage for relationships
        source_entity: Name of the source entity
        target_entity: Name of the target entity
    """
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            # Check if the relation exists
            edge_exists = await chunk_entity_relation_graph.has_edge(
                source_entity, target_entity
            )
            if not edge_exists:
                logger.warning(
                    f"Relation from '{source_entity}' to '{target_entity}' does not exist"
                )
                return

            # Delete relation from vector database
            relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-"
            )
            await relationships_vdb.delete([relation_id])

            # Delete relation from knowledge graph
            await chunk_entity_relation_graph.remove_edges(
                [(source_entity, target_entity)]
            )

            logger.info(
                f"Successfully deleted relation from '{source_entity}' to '{target_entity}'"
            )
            await _delete_relation_done(relationships_vdb, chunk_entity_relation_graph)
        except Exception as e:
            logger.error(
                f"Error while deleting relation from '{source_entity}' to '{target_entity}': {e}"
            )


async def _delete_relation_done(relationships_vdb, chunk_entity_relation_graph) -> None:
    """Callback after relation deletion is complete, ensures updates are persisted"""
    await asyncio.gather(
        *[
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                relationships_vdb,
                chunk_entity_relation_graph,
            ]
        ]
    )


async def aedit_entity(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    entity_name: str,
    updated_data: dict[str, str],
    allow_rename: bool = True,
) -> dict[str, Any]:
    """Asynchronously edit entity information.

    Updates entity information in the knowledge graph and re-embeds the entity in the vector database.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        entity_name: Name of the entity to edit
        updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "entity_type": "new type"}
        allow_rename: Whether to allow entity renaming, defaults to True

    Returns:
        Dictionary containing updated entity information
    """
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
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
                logger.info(f"Renaming entity '{entity_name}' to '{new_entity_name}'")

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
                logger.info(
                    f"Deleted old entity '{entity_name}' and its vector embedding from database"
                )

                # Delete old relation records from vector database
                await relationships_vdb.delete(relations_to_delete)
                logger.info(
                    f"Deleted {len(relations_to_delete)} relation records for entity '{entity_name}' from vector database"
                )

                # Update relationship vector representations
                for src, tgt, edge_data in relations_to_update:
                    description = edge_data.get("description", "")
                    keywords = edge_data.get("keywords", "")
                    source_id = edge_data.get("source_id", "")
                    weight = float(edge_data.get("weight", 1.0))

                    # Create new content for embedding
                    content = f"{src}\t{tgt}\n{keywords}\n{description}"

                    # Calculate relationship ID
                    relation_id = compute_mdhash_id(src + tgt, prefix="rel-")

                    # Prepare data for vector database update
                    relation_data = {
                        relation_id: {
                            "content": content,
                            "src_id": src,
                            "tgt_id": tgt,
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

            # 4. Save changes
            await _edit_entity_done(
                entities_vdb, relationships_vdb, chunk_entity_relation_graph
            )

            logger.info(f"Entity '{entity_name}' successfully updated")
            return await get_entity_info(
                chunk_entity_relation_graph,
                entities_vdb,
                entity_name,
                include_vector_data=True,
            )
        except Exception as e:
            logger.error(f"Error while editing entity '{entity_name}': {e}")
            raise


async def _edit_entity_done(
    entities_vdb, relationships_vdb, chunk_entity_relation_graph
) -> None:
    """Callback after entity editing is complete, ensures updates are persisted"""
    await asyncio.gather(
        *[
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                entities_vdb,
                relationships_vdb,
                chunk_entity_relation_graph,
            ]
        ]
    )


async def aedit_relation(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    source_entity: str,
    target_entity: str,
    updated_data: dict[str, Any],
) -> dict[str, Any]:
    """Asynchronously edit relation information.

    Updates relation (edge) information in the knowledge graph and re-embeds the relation in the vector database.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        source_entity: Name of the source entity
        target_entity: Name of the target entity
        updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "keywords": "new keywords"}

    Returns:
        Dictionary containing updated relation information
    """
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
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
            old_relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-"
            )
            await relationships_vdb.delete([old_relation_id])
            logger.info(
                f"Deleted old relation record from vector database for relation {source_entity} -> {target_entity}"
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

            # 4. Save changes
            await _edit_relation_done(relationships_vdb, chunk_entity_relation_graph)

            logger.info(
                f"Relation from '{source_entity}' to '{target_entity}' successfully updated"
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


async def _edit_relation_done(relationships_vdb, chunk_entity_relation_graph) -> None:
    """Callback after relation editing is complete, ensures updates are persisted"""
    await asyncio.gather(
        *[
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                relationships_vdb,
                chunk_entity_relation_graph,
            ]
        ]
    )


async def acreate_entity(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    entity_name: str,
    entity_data: dict[str, Any],
) -> dict[str, Any]:
    """Asynchronously create a new entity.

    Creates a new entity in the knowledge graph and adds it to the vector database.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        entity_name: Name of the new entity
        entity_data: Dictionary containing entity attributes, e.g. {"description": "description", "entity_type": "type"}

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
                "source_id": entity_data.get("source_id", "manual"),
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

            # Save changes
            await _edit_entity_done(
                entities_vdb, relationships_vdb, chunk_entity_relation_graph
            )

            logger.info(f"Entity '{entity_name}' successfully created")
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
) -> dict[str, Any]:
    """Asynchronously create a new relation between entities.

    Creates a new relation (edge) in the knowledge graph and adds it to the vector database.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        source_entity: Name of the source entity
        target_entity: Name of the target entity
        relation_data: Dictionary containing relation attributes, e.g. {"description": "description", "keywords": "keywords"}

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
                "source_id": relation_data.get("source_id", "manual"),
                "weight": float(relation_data.get("weight", 1.0)),
            }

            # Add relation to knowledge graph
            await chunk_entity_relation_graph.upsert_edge(
                source_entity, target_entity, edge_data
            )

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

            # Save changes
            await _edit_relation_done(relationships_vdb, chunk_entity_relation_graph)

            logger.info(
                f"Relation from '{source_entity}' to '{target_entity}' successfully created"
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
        merge_strategy: Merge strategy configuration, e.g. {"description": "concatenate", "entity_type": "keep_first"}
            Supported strategies:
            - "concatenate": Concatenate all values (for text fields)
            - "keep_first": Keep the first non-empty value
            - "keep_last": Keep the last non-empty value
            - "join_unique": Join all unique values (for fields separated by delimiter)
        target_entity_data: Dictionary of specific values to set for the target entity,
            overriding any merged values, e.g. {"description": "custom description", "entity_type": "PERSON"}

    Returns:
        Dictionary containing the merged entity information
    """
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            # Default merge strategy
            default_strategy = {
                "description": "concatenate",
                "entity_type": "keep_first",
                "source_id": "join_unique",
            }

            merge_strategy = (
                default_strategy
                if merge_strategy is None
                else {**default_strategy, **merge_strategy}
            )
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
            merged_entity_data = _merge_entity_attributes(
                list(source_entities_data.values())
                + ([existing_target_entity_data] if target_exists else []),
                merge_strategy,
            )

            # Apply any explicitly provided target entity data (overrides merged data)
            for key, value in target_entity_data.items():
                merged_entity_data[key] = value

            # 4. Get all relationships of the source entities
            all_relations = []
            for entity_name in source_entities:
                # Get all relationships of the source entities
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
                    merged_relation = _merge_relation_attributes(
                        [existing_data, edge_data],
                        {
                            "description": "concatenate",
                            "keywords": "join_unique",
                            "source_id": "join_unique",
                            "weight": "max",
                        },
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

                description = edge_data.get("description", "")
                keywords = edge_data.get("keywords", "")
                source_id = edge_data.get("source_id", "")
                weight = float(edge_data.get("weight", 1.0))

                content = f"{keywords}\t{src}\n{tgt}\n{description}"
                relation_id = compute_mdhash_id(src + tgt, prefix="rel-")

                relation_data_for_vdb = {
                    relation_id: {
                        "content": content,
                        "src_id": src,
                        "tgt_id": tgt,
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
            await _merge_entities_done(
                entities_vdb, relationships_vdb, chunk_entity_relation_graph
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


def _merge_entity_attributes(
    entity_data_list: list[dict[str, Any]], merge_strategy: dict[str, str]
) -> dict[str, Any]:
    """Merge attributes from multiple entities.

    Args:
        entity_data_list: List of dictionaries containing entity data
        merge_strategy: Merge strategy for each field

    Returns:
        Dictionary containing merged entity data
    """
    merged_data = {}

    # Collect all possible keys
    all_keys = set()
    for data in entity_data_list:
        all_keys.update(data.keys())

    # Merge values for each key
    for key in all_keys:
        # Get all values for this key
        values = [data.get(key) for data in entity_data_list if data.get(key)]

        if not values:
            continue

        # Merge values according to strategy
        strategy = merge_strategy.get(key, "keep_first")

        if strategy == "concatenate":
            merged_data[key] = "\n\n".join(values)
        elif strategy == "keep_first":
            merged_data[key] = values[0]
        elif strategy == "keep_last":
            merged_data[key] = values[-1]
        elif strategy == "join_unique":
            # Handle fields separated by GRAPH_FIELD_SEP
            unique_items = set()
            for value in values:
                items = value.split(GRAPH_FIELD_SEP)
                unique_items.update(items)
            merged_data[key] = GRAPH_FIELD_SEP.join(unique_items)
        else:
            # Default strategy
            merged_data[key] = values[0]

    return merged_data


def _merge_relation_attributes(
    relation_data_list: list[dict[str, Any]], merge_strategy: dict[str, str]
) -> dict[str, Any]:
    """Merge attributes from multiple relationships.

    Args:
        relation_data_list: List of dictionaries containing relationship data
        merge_strategy: Merge strategy for each field

    Returns:
        Dictionary containing merged relationship data
    """
    merged_data = {}

    # Collect all possible keys
    all_keys = set()
    for data in relation_data_list:
        all_keys.update(data.keys())

    # Merge values for each key
    for key in all_keys:
        # Get all values for this key
        values = [
            data.get(key) for data in relation_data_list if data.get(key) is not None
        ]

        if not values:
            continue

        # Merge values according to strategy
        strategy = merge_strategy.get(key, "keep_first")

        if strategy == "concatenate":
            merged_data[key] = "\n\n".join(str(v) for v in values)
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
        elif strategy == "max":
            # For numeric fields like weight
            try:
                merged_data[key] = max(float(v) for v in values)
            except (ValueError, TypeError):
                merged_data[key] = values[0]
        else:
            # Default strategy
            merged_data[key] = values[0]

    return merged_data


async def _merge_entities_done(
    entities_vdb, relationships_vdb, chunk_entity_relation_graph
) -> None:
    """Callback after entity merging is complete, ensures updates are persisted"""
    await asyncio.gather(
        *[
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                entities_vdb,
                relationships_vdb,
                chunk_entity_relation_graph,
            ]
        ]
    )


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
    """Get detailed information of a relationship"""

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
