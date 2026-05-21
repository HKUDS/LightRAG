"""Storage data migration helpers for :class:`LightRAG`.

Mixed into LightRAG and runs once at startup (``initialize_storages`` →
``check_and_migrate_data``) to upgrade legacy data layouts:

- Backfill ``full_entities`` / ``full_relations`` from the graph + doc_status
  history when those KV stores are empty (entity-relation migration).
- Rebuild ``entity_chunks`` / ``relation_chunks`` indexes by walking nodes/
  edges in the graph storage when they are empty
  (chunk-tracking migration).
"""

from __future__ import annotations

from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kg.shared_storage import get_data_init_lock
from lightrag.utils import logger, make_relation_chunk_key


class _StorageMigrationMixin:
    """Mixin that owns one-shot data migrations on :class:`LightRAG`.

    Mixed into LightRAG only. Relies on attributes that the main class
    initializes in ``__post_init__`` (``doc_status``, ``full_entities``,
    ``full_relations``, ``chunk_entity_relation_graph``, ``entity_chunks``,
    ``relation_chunks``).
    """

    async def check_and_migrate_data(self):
        """Check if data migration is needed and perform migration if necessary"""
        async with get_data_init_lock():
            try:
                # Check if migration is needed:
                # 1. chunk_entity_relation_graph has entities and relations (count > 0)
                # 2. full_entities and full_relations are empty

                # Get all entity labels from graph
                all_entity_labels = (
                    await self.chunk_entity_relation_graph.get_all_labels()
                )

                if not all_entity_labels:
                    logger.debug("No entities found in graph, skipping migration check")
                    return

                try:
                    # Initialize chunk tracking storage after migration
                    await self._migrate_chunk_tracking_storage()
                except Exception as e:
                    logger.error(f"Error during chunk_tracking migration: {e}")
                    raise e

                # Check if full_entities and full_relations are empty
                # Get all processed documents to check their entity/relation data
                try:
                    processed_docs = await self.doc_status.get_docs_by_status(
                        DocStatus.PROCESSED
                    )

                    if not processed_docs:
                        logger.debug("No processed documents found, skipping migration")
                        return

                    # Check first few documents to see if they have full_entities/full_relations data
                    migration_needed = True
                    checked_count = 0
                    max_check = min(5, len(processed_docs))  # Check up to 5 documents

                    for doc_id in list(processed_docs.keys())[:max_check]:
                        checked_count += 1
                        entity_data = await self.full_entities.get_by_id(doc_id)
                        relation_data = await self.full_relations.get_by_id(doc_id)

                        if entity_data or relation_data:
                            migration_needed = False
                            break

                    if not migration_needed:
                        logger.debug(
                            "Full entities/relations data already exists, no migration needed"
                        )
                        return

                    logger.info(
                        f"Data migration needed: found {len(all_entity_labels)} entities in graph but no full_entities/full_relations data"
                    )

                    # Perform migration
                    await self._migrate_entity_relation_data(processed_docs)

                except Exception as e:
                    logger.error(f"Error during migration check: {e}")
                    raise e

            except Exception as e:
                logger.error(f"Error in data migration check: {e}")
                raise e

    async def _migrate_entity_relation_data(self, processed_docs: dict):
        """Migrate existing entity and relation data to full_entities and full_relations storage"""
        logger.info(f"Starting data migration for {len(processed_docs)} documents")

        # Create mapping from chunk_id to doc_id
        chunk_to_doc = {}
        for doc_id, doc_status in processed_docs.items():
            chunk_ids = (
                doc_status.chunks_list
                if hasattr(doc_status, "chunks_list") and doc_status.chunks_list
                else []
            )
            for chunk_id in chunk_ids:
                chunk_to_doc[chunk_id] = doc_id

        # Initialize document entity and relation mappings
        doc_entities = {}  # doc_id -> set of entity_names
        doc_relations = {}  # doc_id -> set of relation_pairs (as tuples)

        # Get all nodes and edges from graph
        all_nodes = await self.chunk_entity_relation_graph.get_all_nodes()
        all_edges = await self.chunk_entity_relation_graph.get_all_edges()

        # Process all nodes once
        for node in all_nodes:
            if "source_id" in node:
                entity_id = node.get("entity_id") or node.get("id")
                if not entity_id:
                    continue

                # Get chunk IDs from source_id
                source_ids = node["source_id"].split(GRAPH_FIELD_SEP)

                # Find which documents this entity belongs to
                for chunk_id in source_ids:
                    doc_id = chunk_to_doc.get(chunk_id)
                    if doc_id:
                        if doc_id not in doc_entities:
                            doc_entities[doc_id] = set()
                        doc_entities[doc_id].add(entity_id)

        # Process all edges once
        for edge in all_edges:
            if "source_id" in edge:
                src = edge.get("source")
                tgt = edge.get("target")
                if not src or not tgt:
                    continue

                # Get chunk IDs from source_id
                source_ids = edge["source_id"].split(GRAPH_FIELD_SEP)

                # Find which documents this relation belongs to
                for chunk_id in source_ids:
                    doc_id = chunk_to_doc.get(chunk_id)
                    if doc_id:
                        if doc_id not in doc_relations:
                            doc_relations[doc_id] = set()
                        # Use tuple for set operations, convert to list later
                        doc_relations[doc_id].add(tuple(sorted((src, tgt))))

        # Store the results in full_entities and full_relations
        migration_count = 0

        # Store entities
        if doc_entities:
            entities_data = {}
            for doc_id, entity_set in doc_entities.items():
                entities_data[doc_id] = {
                    "entity_names": list(entity_set),
                    "count": len(entity_set),
                }
            await self.full_entities.upsert(entities_data)

        # Store relations
        if doc_relations:
            relations_data = {}
            for doc_id, relation_set in doc_relations.items():
                # Convert tuples back to lists
                relations_data[doc_id] = {
                    "relation_pairs": [list(pair) for pair in relation_set],
                    "count": len(relation_set),
                }
            await self.full_relations.upsert(relations_data)

        migration_count = len(
            set(list(doc_entities.keys()) + list(doc_relations.keys()))
        )

        # Persist the migrated data
        await self.full_entities.index_done_callback()
        await self.full_relations.index_done_callback()

        logger.info(
            f"Data migration completed: migrated {migration_count} documents with entities/relations"
        )

    async def _migrate_chunk_tracking_storage(self) -> None:
        """Ensure entity/relation chunk tracking KV stores exist and are seeded."""

        if not self.entity_chunks or not self.relation_chunks:
            return

        need_entity_migration = False
        need_relation_migration = False

        try:
            need_entity_migration = await self.entity_chunks.is_empty()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to check entity chunks storage: {exc}")
            raise exc

        try:
            need_relation_migration = await self.relation_chunks.is_empty()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to check relation chunks storage: {exc}")
            raise exc

        if not need_entity_migration and not need_relation_migration:
            return

        BATCH_SIZE = 500  # Process 500 records per batch

        if need_entity_migration:
            try:
                nodes = await self.chunk_entity_relation_graph.get_all_nodes()
            except Exception as exc:
                logger.error(f"Failed to fetch nodes for chunk migration: {exc}")
                nodes = []

            logger.info(f"Starting chunk_tracking data migration: {len(nodes)} nodes")

            # Process nodes in batches
            total_nodes = len(nodes)
            total_batches = (total_nodes + BATCH_SIZE - 1) // BATCH_SIZE
            total_migrated = 0

            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, total_nodes)
                batch_nodes = nodes[start_idx:end_idx]

                upsert_payload: dict[str, dict[str, object]] = {}
                for node in batch_nodes:
                    entity_id = node.get("entity_id") or node.get("id")
                    if not entity_id:
                        continue

                    raw_source = node.get("source_id") or ""
                    chunk_ids = [
                        chunk_id
                        for chunk_id in raw_source.split(GRAPH_FIELD_SEP)
                        if chunk_id
                    ]
                    if not chunk_ids:
                        continue

                    upsert_payload[entity_id] = {
                        "chunk_ids": chunk_ids,
                        "count": len(chunk_ids),
                    }

                if upsert_payload:
                    await self.entity_chunks.upsert(upsert_payload)
                    total_migrated += len(upsert_payload)
                    logger.info(
                        f"Processed entity batch {batch_idx + 1}/{total_batches}: {len(upsert_payload)} records (total: {total_migrated}/{total_nodes})"
                    )

            if total_migrated > 0:
                # Persist entity_chunks data to disk
                await self.entity_chunks.index_done_callback()
                logger.info(
                    f"Entity chunk_tracking migration completed: {total_migrated} records persisted"
                )

        if need_relation_migration:
            try:
                edges = await self.chunk_entity_relation_graph.get_all_edges()
            except Exception as exc:
                logger.error(f"Failed to fetch edges for chunk migration: {exc}")
                edges = []

            logger.info(f"Starting chunk_tracking data migration: {len(edges)} edges")

            # Process edges in batches
            total_edges = len(edges)
            total_batches = (total_edges + BATCH_SIZE - 1) // BATCH_SIZE
            total_migrated = 0

            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, total_edges)
                batch_edges = edges[start_idx:end_idx]

                upsert_payload: dict[str, dict[str, object]] = {}
                for edge in batch_edges:
                    src = edge.get("source") or edge.get("src_id") or edge.get("src")
                    tgt = edge.get("target") or edge.get("tgt_id") or edge.get("tgt")
                    if not src or not tgt:
                        continue

                    raw_source = edge.get("source_id") or ""
                    chunk_ids = [
                        chunk_id
                        for chunk_id in raw_source.split(GRAPH_FIELD_SEP)
                        if chunk_id
                    ]
                    if not chunk_ids:
                        continue

                    storage_key = make_relation_chunk_key(src, tgt)
                    upsert_payload[storage_key] = {
                        "chunk_ids": chunk_ids,
                        "count": len(chunk_ids),
                    }

                if upsert_payload:
                    await self.relation_chunks.upsert(upsert_payload)
                    total_migrated += len(upsert_payload)
                    logger.info(
                        f"Processed relation batch {batch_idx + 1}/{total_batches}: {len(upsert_payload)} records (total: {total_migrated}/{total_edges})"
                    )

            if total_migrated > 0:
                # Persist relation_chunks data to disk
                await self.relation_chunks.index_done_callback()
                logger.info(
                    f"Relation chunk_tracking migration completed: {total_migrated} records persisted"
                )
