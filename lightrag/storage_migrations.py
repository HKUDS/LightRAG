"""Storage data migration and repair helpers for :class:`LightRAG`.

Mixed into LightRAG. Server startup calls ``check_and_migrate_data`` after
``initialize_storages`` to upgrade legacy data layouts; SDK explicit creation
also checks chunk tracking before writing its first new row:

- Backfill ``full_entities`` / ``full_relations`` from the graph + doc_status
  history when those KV stores are empty (entity-relation migration).
- Rebuild ``entity_chunks`` / ``relation_chunks`` indexes by walking nodes/
  edges in the graph storage when they are empty
  (chunk-tracking migration).

On top of those one-shot startup migrations this module owns the operator
repair entry point ``arepair_chunk_tracking`` (see its docstring). The two are
deliberately different mechanisms and must not be confused:

===================  ==========================  ==============================
                     ``_migrate_chunk_tracking``  ``arepair_chunk_tracking``
===================  ==========================  ==============================
When                 startup / first creation     operator, on demand
Gate                 only when ``is_empty()``     never gated
Seed                 graph ``source_id``          cached extraction results
Existing rows        left untouched               dropped and rebuilt
===================  ==========================  ==============================

The seed difference is the whole point of the repair. Graph ``source_id`` is a
KEEP-truncated view, so reusing it downgrades provenance across the install;
chunk tracking is supposed to OUTRANK it. The repair therefore never reads
``source_id`` — the graph is consulted only for object existence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kg.shared_storage import get_data_init_lock
from lightrag.utils import logger, make_relation_chunk_key

# Chunks scanned per cache lookup round-trip during a chunk-tracking repair.
# Bounds peak memory of the extraction-cache read; the accumulated
# object -> chunk_ids mapping is what the repair actually keeps.
_REPAIR_CHUNK_SCAN_BATCH = 200

# Rows written per ``upsert`` while rebuilding a tracking namespace.
_REPAIR_UPSERT_BATCH = 500


@dataclass
class ChunkTrackingRepairReport:
    """Outcome of one :meth:`_StorageMigrationMixin.arepair_chunk_tracking` run."""

    scanned_documents: int = 0
    """doc_status rows whose ``chunks_list`` contributed candidate chunk ids."""

    scanned_chunks: int = 0
    """Distinct chunk ids taken from those documents."""

    chunks_with_cache: int = 0
    """Chunks whose extraction results were found in the LLM cache."""

    chunks_without_cache: int = 0
    """Chunks with no usable cached extraction result (nothing is attributed
    to them; the objects they used to support may end up with no row)."""

    entity_rows_written: int = 0
    relation_rows_written: int = 0

    entities_without_evidence: int = 0
    """Graph entities left with NO row because no cached chunk named them."""

    relations_without_evidence: int = 0
    """Graph relations left with NO row for the same reason."""

    warnings: list[str] = field(default_factory=list)
    """Operator-facing notes about degradations this run accepted."""

    def as_dict(self) -> dict:
        return asdict(self)


class _StorageMigrationMixin:
    """Mixin that owns one-shot data migrations on :class:`LightRAG`.

    Mixed into LightRAG only. Relies on attributes that the main class
    initializes in ``__post_init__`` (``doc_status``, ``full_entities``,
    ``full_relations``, ``chunk_entity_relation_graph``, ``entity_chunks``,
    ``relation_chunks``). :meth:`arepair_chunk_tracking` additionally relies on
    ``text_chunks`` and ``llm_response_cache``.
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
                    # strict=True: this mapping is the ONLY input that decides
                    # which documents get recovery anchors written. A row that
                    # cannot be parsed must abort the migration, not be skipped
                    # — a partial mapping writes anchors for its siblings, after
                    # which the "anchors already exist" check above short-
                    # circuits every later startup, so the omitted document
                    # never gets anchors and every purge of it fails closed
                    # (409) for good. See the purge recovery contract.
                    processed_docs = await self.doc_status.get_docs_by_statuses(
                        [DocStatus.PROCESSED], strict=True
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

    async def _migrate_chunk_tracking_before_creation(self) -> None:
        """Seed legacy tracking before a new row can suppress empty-store migration.

        SDK callers do not run the server's startup migrations. Use the same
        migration lock and propagate failures before any creation write. This
        checks only chunk tracking, not document-anchor migrations, and leaves
        existing non-empty namespaces (including authoritative empty rows) alone.
        A successful check is cached per instance, including when a namespace
        stays empty. Failed checks remain retryable; each worker checks once.
        """
        if self._chunk_tracking_migration_checked:
            return
        async with get_data_init_lock():
            # Another creation on this instance may have completed the check
            # while this caller waited for the shared initialization lock.
            if self._chunk_tracking_migration_checked:
                return
            await self._migrate_chunk_tracking_storage()
            self._chunk_tracking_migration_checked = True

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
                # Complete-or-raise, matching _migrate_entity_relation_data:
                # degrading to an empty list would record a "completed"
                # backfill built from nothing, and chunk-tracking would then
                # silently miss every entity until manually repaired.
                logger.error(f"Failed to fetch nodes for chunk migration: {exc}")
                raise

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
                # Same contract as the nodes read above.
                logger.error(f"Failed to fetch edges for chunk migration: {exc}")
                raise

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

    # ------------------------------------------------------------------
    # Operator repair entry point (issue #3838, R4)
    # ------------------------------------------------------------------

    async def arepair_chunk_tracking(self) -> ChunkTrackingRepairReport:
        """Drop and rebuild ``entity_chunks`` / ``relation_chunks`` from the cache.

        The recovery path for a chunk-tracking namespace that has gone wrong:
        an orphan row left behind when a crash separated a durable graph commit
        from the tracking delete that belonged with it, or a row polluted with
        chunk ids that no longer exist. ``BaseKVStorage`` has no enumeration
        API, so such a row cannot be discovered — let alone pruned — by a sweep.
        The repair is therefore whole-namespace: ``drop()`` both namespaces,
        then write back what the extraction cache can prove.

        Four properties are load-bearing (issue #3838, R4):

        * **Not gated on ``is_empty()``.** ``_migrate_chunk_tracking_storage``
          is, and that is exactly why it cannot repair anything: one leftover
          row makes the gate false and silently suppresses the rebuild for the
          whole install.
        * **Per workspace, on demand.** It runs against this instance's
          storages only, when an operator asks for it — never implicitly.
        * **Never seeded from the graph ``source_id``.** ``source_id`` is
          KEEP-truncated, and chunk tracking outranks it; seeding from it would
          downgrade provenance across the install. The graph is read ONLY to
          learn which entities and edges exist, never for their chunk ids.
        * **Chunk-granular, from the cached extraction results.** The fact a
          tracking row needs — which chunk mentioned which object — exists only
          in the LLM extraction cache (``text_chunks.llm_cache_list`` ->
          ``llm_response_cache``); ``text_chunks`` carries the text, not the
          attribution, and the ``full_entities`` / ``full_relations`` anchors
          map document -> object name, one granularity too coarse. Attributing
          an object to every chunk of every document whose anchor names it is
          FORBIDDEN: those rows feed ``existing_full_source_ids`` in
          ``_merge_edges_then_upsert``, which would break the relation weight
          contract and misclassify chunk-subset purges — the same phantom
          evidence this repair exists to remove.

        Objects whose chunks are not in the cache (cache cleared or extraction
        caching disabled) get **no row at all**. Their absence restores the
        graph ``source_id`` fallback inside the purge classifier, a known and
        bounded degradation, which is strictly better than writing a row that
        claims evidence nobody can substantiate.

        Accepted residue (see AGENTS.md, *Consistency without transactions*):

        * The repair is **not atomic across the two namespaces**: it drops
          both, then writes. A crash in between leaves tracking partially
          rebuilt, which re-running the repair fully heals — the rebuilt
          content is a pure function of the cache, so a second run is
          idempotent. The mapping is computed BEFORE the first drop, so a
          failure while reading doc_status / chunks / cache leaves every
          existing row untouched.
        * If a namespace ends up **empty** (nothing in the cache), the next
          startup's ``_migrate_chunk_tracking_storage`` sees ``is_empty()`` and
          re-seeds it from the graph ``source_id`` — precisely the provenance
          downgrade this repair refuses to perform itself. The run warns about
          it in that case; the recovery is to restore the extraction cache (or
          re-ingest) before restarting.

        Returns:
            ChunkTrackingRepairReport: counts plus operator-facing warnings.

        Raises:
            ValueError: chunk tracking is not configured on this instance.
            Exception: any doc_status / storage failure is propagated. Failures
                before the drop leave the namespaces untouched.
        """
        if not self.entity_chunks or not self.relation_chunks:
            raise ValueError(
                "Chunk tracking storages are not configured; nothing to repair."
            )

        # Same lock the migrations take: a repair must not interleave with a
        # startup migration seeding the very namespaces it is about to drop.
        async with get_data_init_lock():
            report = await self._repair_chunk_tracking_impl()

        # A repair supersedes the legacy empty-store migration for this
        # instance: re-running it now would either be a no-op or (when the
        # repair legitimately wrote nothing) re-seed from graph ``source_id``.
        self._chunk_tracking_migration_checked = True
        return report

    async def _repair_chunk_tracking_impl(self) -> ChunkTrackingRepairReport:
        """Body of :meth:`arepair_chunk_tracking`, called under the init lock."""
        report = ChunkTrackingRepairReport()

        chunk_ids, report.scanned_documents = await self._collect_corpus_chunk_ids()
        report.scanned_chunks = len(chunk_ids)

        # Existence only. `get_all_labels` / `get_all_edges` also expose
        # `source_id`; it is deliberately NOT read here (R4).
        graph_entities = set(await self.chunk_entity_relation_graph.get_all_labels())
        graph_relations = await self._collect_graph_relation_keys()

        entity_rows: dict[str, list[str]] = {}
        relation_rows: dict[str, list[str]] = {}

        if chunk_ids and self.llm_response_cache is not None:
            (
                report.chunks_with_cache,
                report.chunks_without_cache,
            ) = await self._accumulate_cached_attribution(
                chunk_ids,
                graph_entities,
                graph_relations,
                entity_rows,
                relation_rows,
            )
        else:
            report.chunks_without_cache = len(chunk_ids)
            if self.llm_response_cache is None:
                report.warnings.append(
                    "LLM response cache is not configured, so no chunk-level "
                    "attribution could be recovered; both tracking namespaces "
                    "are left empty."
                )

        report.entities_without_evidence = len(graph_entities - set(entity_rows))
        report.relations_without_evidence = len(graph_relations - set(relation_rows))

        # Everything above is read-only. The destructive half starts here, so a
        # failure while reading doc_status / chunks / cache leaves the stored
        # rows exactly as they were.
        await self._drop_tracking_namespace(self.entity_chunks, "entity_chunks")
        await self._drop_tracking_namespace(self.relation_chunks, "relation_chunks")

        report.entity_rows_written = await self._write_tracking_rows(
            self.entity_chunks, entity_rows, "entity_chunks"
        )
        report.relation_rows_written = await self._write_tracking_rows(
            self.relation_chunks, relation_rows, "relation_chunks"
        )

        self._warn_on_degraded_repair(
            report, len(graph_entities), len(graph_relations)
        )

        logger.info(
            "Chunk tracking repair completed: "
            f"{report.entity_rows_written} entity row(s), "
            f"{report.relation_rows_written} relation row(s) rebuilt from "
            f"{report.chunks_with_cache}/{report.scanned_chunks} cached chunk(s)"
        )
        return report

    async def _collect_corpus_chunk_ids(self) -> tuple[list[str], int]:
        """Return every chunk id the corpus still knows about, plus doc count.

        ``text_chunks`` cannot be enumerated (``BaseKVStorage`` has no such
        API), so the chunk universe comes from ``doc_status.chunks_list`` — the
        same indirection ``_migrate_entity_relation_data`` uses. Every status
        is included, not just PROCESSED: a document that failed or is mid-flight
        still owns real chunks whose cached extraction is real evidence, and
        omitting them would drop legitimate rows.

        The read is ``strict=True`` on purpose. This mapping decides which rows
        survive the repair; a relaxed read that skipped an unparseable document
        would silently drop the tracking rows of every object only that
        document's chunks support, and the operator would see a "success".
        """
        docs = await self.doc_status.get_docs_by_statuses(list(DocStatus), strict=True)

        # An insertion-ordered dict, so the rebuilt rows list their chunks
        # deterministically instead of in set order.
        ordered: dict[str, None] = {}
        contributing_docs = 0
        for doc in docs.values():
            chunks_list = getattr(doc, "chunks_list", None) or []
            if not chunks_list:
                continue
            contributing_docs += 1
            for chunk_id in chunks_list:
                if chunk_id:
                    ordered.setdefault(chunk_id, None)
        return list(ordered), contributing_docs

    async def _collect_graph_relation_keys(self) -> set[str]:
        """Tracking keys of the edges that currently exist in the graph.

        Existence filter only: an object absent from the graph must not get a
        tracking row, or the repair would resurrect exactly the orphan rows it
        is meant to remove. The edge payload's ``source_id`` is never read.
        """
        keys: set[str] = set()
        for edge in await self.chunk_entity_relation_graph.get_all_edges():
            src = edge.get("source") or edge.get("src_id") or edge.get("src")
            tgt = edge.get("target") or edge.get("tgt_id") or edge.get("tgt")
            if not src or not tgt:
                continue
            keys.add(make_relation_chunk_key(src, tgt))
        return keys

    async def _accumulate_cached_attribution(
        self,
        chunk_ids: list[str],
        graph_entities: set[str],
        graph_relations: set[str],
        entity_rows: dict[str, list[str]],
        relation_rows: dict[str, list[str]],
    ) -> tuple[int, int]:
        """Fill ``entity_rows`` / ``relation_rows`` from the extraction cache.

        Returns ``(chunks_with_cache, chunks_without_cache)``. Chunks are read
        in bounded batches; each batch's cached extraction results are parsed
        with the SAME helpers the deletion rebuild path uses, so the repair and
        the rebuild agree on what a chunk attributed to whom.
        """
        # Imported lazily: `lightrag.operate` pulls in the whole extraction
        # stack, which this module has no reason to import at definition time.
        from lightrag.operate import (
            _get_cached_extraction_results,
            _rebuild_from_extraction_result,
        )

        with_cache = 0
        for start in range(0, len(chunk_ids), _REPAIR_CHUNK_SCAN_BATCH):
            batch = chunk_ids[start : start + _REPAIR_CHUNK_SCAN_BATCH]
            # A doc_status ``chunks_list`` can name chunks that were never
            # written (a document that failed mid-ingestion). Drop them first:
            # the cache lookup reaches its entries THROUGH the chunk row, so a
            # missing chunk can only produce a per-chunk "invalid or None"
            # warning and no attribution.
            missing = await self.text_chunks.filter_keys(set(batch))
            batch = [chunk_id for chunk_id in batch if chunk_id not in missing]
            if not batch:
                continue
            cached_results, chunk_data_by_id = await _get_cached_extraction_results(
                self.llm_response_cache,
                set(batch),
                text_chunks_storage=self.text_chunks,
            )
            for chunk_id, results in cached_results.items():
                parsed_any = False
                for extraction_result, timestamp in results:
                    try:
                        entities, relationships = await _rebuild_from_extraction_result(
                            text_chunks_storage=self.text_chunks,
                            chunk_id=chunk_id,
                            extraction_result=extraction_result,
                            timestamp=timestamp,
                            chunk_data=chunk_data_by_id.get(chunk_id),
                        )
                    except Exception as exc:
                        # One unparseable cache entry must not abort the repair:
                        # the chunk simply contributes no attribution, which is
                        # the documented "no row" degradation, not a failure.
                        logger.warning(
                            "Chunk tracking repair: unusable cached extraction "
                            f"result for chunk {chunk_id}: {exc}"
                        )
                        continue
                    parsed_any = True
                    for entity_name in entities:
                        if entity_name in graph_entities:
                            _append_chunk_id(entity_rows, entity_name, chunk_id)
                    for src, tgt in relationships:
                        storage_key = make_relation_chunk_key(src, tgt)
                        if storage_key in graph_relations:
                            _append_chunk_id(relation_rows, storage_key, chunk_id)
                if parsed_any:
                    with_cache += 1
        return with_cache, len(chunk_ids) - with_cache

    @staticmethod
    async def _drop_tracking_namespace(storage, label: str) -> None:
        """``drop()`` one tracking namespace, complete-or-raise.

        ``drop`` reports failure in its return value rather than raising, and a
        drop that silently did nothing would leave the stale rows in place
        while the repair reported success.
        """
        result = await storage.drop()
        if isinstance(result, dict) and result.get("status") != "success":
            message = result.get("message", "unknown error")
            raise RuntimeError(f"Failed to drop {label} during repair: {message}")

    @staticmethod
    async def _write_tracking_rows(
        storage, rows: dict[str, list[str]], label: str
    ) -> int:
        """Write the rebuilt rows for one namespace and persist them."""
        if not rows:
            # The namespace was dropped: leaving it empty is the documented
            # "no evidence, no row" outcome, not an incomplete write.
            await storage.index_done_callback()
            return 0

        keys = list(rows)
        written = 0
        for start in range(0, len(keys), _REPAIR_UPSERT_BATCH):
            payload = {
                key: {"chunk_ids": rows[key], "count": len(rows[key])}
                for key in keys[start : start + _REPAIR_UPSERT_BATCH]
            }
            await storage.upsert(payload)
            written += len(payload)
        await storage.index_done_callback()
        logger.info(f"Chunk tracking repair: rebuilt {written} {label} row(s)")
        return written

    @staticmethod
    def _warn_on_degraded_repair(
        report: ChunkTrackingRepairReport,
        graph_entity_count: int,
        graph_relation_count: int,
    ) -> None:
        """Record the degradations an operator has to know about."""
        if report.chunks_without_cache:
            report.warnings.append(
                f"{report.chunks_without_cache} of {report.scanned_chunks} chunk(s) "
                "had no usable cached extraction result; objects supported only "
                "by them were left without a tracking row (the purge classifier "
                "falls back to the graph source_id for those)."
            )
        if report.entities_without_evidence or report.relations_without_evidence:
            report.warnings.append(
                f"{report.entities_without_evidence} entity/entities and "
                f"{report.relations_without_evidence} relation(s) present in the "
                "graph got no tracking row because no cached chunk named them."
            )
        # Only a namespace that SHOULD have rows and has none is worth the
        # warning: a corpus with no relations at all legitimately leaves
        # relation_chunks empty, and the startup migration finds nothing to
        # re-seed from either.
        emptied = [
            name
            for name, written, expected in (
                ("entity_chunks", report.entity_rows_written, graph_entity_count),
                ("relation_chunks", report.relation_rows_written, graph_relation_count),
            )
            if expected and not written
        ]
        if emptied:
            report.warnings.append(
                f"{' and '.join(emptied)} was left empty although the graph still "
                "holds such objects. The next startup's chunk-tracking migration "
                "is gated on is_empty() and would re-seed it from the graph "
                "source_id, which is KEEP-truncated; restore the extraction cache "
                "before restarting if that matters."
            )
        for warning in report.warnings:
            logger.warning(f"Chunk tracking repair: {warning}")


def _append_chunk_id(rows: dict[str, list[str]], key: str, chunk_id: str) -> None:
    """Append ``chunk_id`` to ``rows[key]`` preserving order, without duplicates."""
    bucket = rows.setdefault(key, [])
    if chunk_id not in bucket:
        bucket.append(chunk_id)
