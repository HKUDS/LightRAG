"""Deferred batch deletion coordinator with restart-safe recovery semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from lightrag.deletion_journal import (
    DeferredDeletionJournal,
    DeferredDeletionJournalStore,
    DeferredDeletionStage,
)
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.operate import rebuild_knowledge_from_chunks


@dataclass
class DeferredDeletionResult:
    batch_id: str
    stage: DeferredDeletionStage
    error_detail: str | None = None


def _store(rag: Any) -> DeferredDeletionJournalStore:
    return DeferredDeletionJournalStore(rag.working_dir, rag.workspace)


async def _snapshot_document_metadata(rag: Any, doc_id: str) -> dict[str, Any]:
    """Persist enough source metadata to finish cleanup after a restart."""
    return {
        "doc_status": await rag.doc_status.get_by_id(doc_id),
        "full_doc": await rag.full_docs.get_by_id(doc_id),
        "full_entities": await rag.full_entities.get_by_id(doc_id),
        "full_relations": await rag.full_relations.get_by_id(doc_id),
    }


async def _finalize_metadata(rag: Any, journal: DeferredDeletionJournal) -> None:
    doc_ids = journal.document_ids
    if journal.delete_llm_cache:
        cache_ids = set()
        for doc_id in doc_ids:
            snapshot = journal.document_metadata.get(doc_id, {})
            snapshot_status = snapshot.get("doc_status") or {}
            snapshot_metadata = snapshot_status.get("metadata")
            if isinstance(snapshot_metadata, dict):
                cache_ids.update(snapshot_metadata.get("deletion_llm_cache_ids", []))
            doc_status = await rag.doc_status.get_by_id(doc_id)
            if not doc_status:
                continue
            # The regular deletion path persists retryable cache IDs in nested
            # status metadata. Keep the legacy top-level lookup for journals
            # created by earlier versions.
            cache_ids.update(doc_status.get("deletion_llm_cache_ids", []))
            metadata = doc_status.get("metadata")
            if isinstance(metadata, dict):
                cache_ids.update(metadata.get("deletion_llm_cache_ids", []))
        if cache_ids:
            if not rag.llm_response_cache:
                raise RuntimeError("LLM cache storage is unavailable for finalization")
            await rag.llm_response_cache.delete(sorted(cache_ids))
            remaining = await rag._get_existing_llm_cache_ids(sorted(cache_ids))
            if remaining:
                raise RuntimeError("LLM cache entries remain after finalization")
    await rag.full_entities.delete(doc_ids)
    await rag.full_relations.delete(doc_ids)
    await rag.doc_status.delete(doc_ids)
    await rag.full_docs.delete(doc_ids)
    await rag._insert_done()

    remaining = await asyncio_gather_get_by_id(
        rag,
        doc_ids,
        (rag.full_entities, rag.full_relations, rag.doc_status, rag.full_docs),
    )
    if any(value is not None for value in remaining):
        raise RuntimeError(
            "document metadata remains after deferred deletion finalization"
        )


async def asyncio_gather_get_by_id(
    rag: Any, doc_ids: list[str], stores: tuple[Any, ...]
) -> list[Any]:
    import asyncio

    return await asyncio.gather(
        *(store.get_by_id(doc_id) for store in stores for doc_id in doc_ids)
    )


async def _rebuild_once(rag: Any, journal: DeferredDeletionJournal) -> None:
    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    await rebuild_knowledge_from_chunks(
        entities_to_rebuild=journal.entities_to_rebuild,
        relationships_to_rebuild=journal.relationships_to_rebuild,
        knowledge_graph_inst=rag.chunk_entity_relation_graph,
        entities_vdb=rag.entities_vdb,
        relationships_vdb=rag.relationships_vdb,
        text_chunks_storage=rag.text_chunks,
        llm_response_cache=rag.llm_response_cache,
        global_config=rag._build_global_config(),
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        entity_chunks_storage=rag.entity_chunks,
        relation_chunks_storage=rag.relation_chunks,
    )


async def _continue(
    rag: Any, journal: DeferredDeletionJournal
) -> DeferredDeletionResult:
    store = _store(rag)
    if journal.stage is DeferredDeletionStage.FAILED_RETRYABLE:
        journal.stage = journal.recovery_stage or DeferredDeletionStage.PREPARED

    if journal.stage is DeferredDeletionStage.PREPARED:
        for doc_id in journal.document_ids:
            if doc_id not in journal.document_metadata:
                journal.document_metadata[doc_id] = await _snapshot_document_metadata(
                    rag, doc_id
                )
                file_path = (
                    journal.document_metadata[doc_id].get("doc_status") or {}
                ).get("file_path")
                if isinstance(file_path, str):
                    journal.source_file_paths[doc_id] = file_path
                store.save(journal)
        batch_chunk_ids = {
            chunk_id
            for metadata in journal.document_metadata.values()
            for chunk_id in (metadata.get("doc_status") or {}).get("chunks_list", [])
        }
        while journal.current_document_index < len(journal.document_ids):
            doc_id = journal.document_ids[journal.current_document_index]
            checkpoint = journal.document_delete_checkpoints.get(doc_id)
            if checkpoint is not None:
                if checkpoint.get("state") == "INTENT":
                    # A process may have died before or after the storage call.
                    # No transaction spans this filesystem journal and arbitrary
                    # storage backends, so retrying risks a second delete.
                    journal.mark_failed_retryable(
                        "document deletion outcome is unknown after durable intent; "
                        "refusing to repeat destructive deletion"
                    )
                    store.save(journal)
                    return DeferredDeletionResult(
                        journal.batch_id, journal.stage, journal.error_detail
                    )
                if checkpoint.get("state") != "COMPLETED":
                    journal.mark_failed_retryable(
                        f"invalid document deletion checkpoint for {doc_id!r}"
                    )
                    store.save(journal)
                    return DeferredDeletionResult(
                        journal.batch_id, journal.stage, journal.error_detail
                    )
                # A completed result and index are persisted together. Support
                # a future/partially-written journal which has only the result.
                journal.current_document_index += 1
                store.save(journal)
                continue
            journal.mark_document_delete_intent(doc_id)
            store.save(journal)
            try:
                result = await rag.adelete_by_doc_id(
                    doc_id,
                    delete_llm_cache=journal.delete_llm_cache,
                    skip_rebuild=True,
                    batch_chunk_ids=batch_chunk_ids,
                )
            except Exception as exc:
                journal.mark_failed_retryable(f"document deletion failed: {exc}")
                store.save(journal)
                return DeferredDeletionResult(
                    journal.batch_id, journal.stage, journal.error_detail
                )
            if result.status != "success":
                journal.mark_failed_retryable(result.message)
                store.save(journal)
                return DeferredDeletionResult(
                    journal.batch_id, journal.stage, journal.error_detail
                )
            journal.record_document_delete_completed(
                doc_id,
                entities=result.entities_to_rebuild or {},
                relationships=result.relationships_to_rebuild or {},
            )
            journal.current_document_index += 1
            store.save(journal)
        journal.mark_rebuild_pending()
        store.save(journal)

    if journal.stage is DeferredDeletionStage.REBUILD_PENDING:
        try:
            await _rebuild_once(rag, journal)
        except Exception as exc:
            journal.mark_failed_retryable(f"rebuild failed: {exc}")
            store.save(journal)
            return DeferredDeletionResult(
                journal.batch_id, journal.stage, journal.error_detail
            )
        journal.mark_finalization_pending()
        store.save(journal)

    if journal.stage is DeferredDeletionStage.FINALIZATION_PENDING:
        try:
            await _finalize_metadata(rag, journal)
        except Exception as exc:
            journal.mark_failed_retryable(f"metadata finalization failed: {exc}")
            store.save(journal)
            return DeferredDeletionResult(
                journal.batch_id, journal.stage, journal.error_detail
            )
        journal.mark_committed()
        store.save(journal)

    return DeferredDeletionResult(journal.batch_id, journal.stage, journal.error_detail)


async def run_deferred_batch_delete(
    rag: Any,
    document_ids: list[str],
    delete_llm_cache: bool = False,
    delete_file: bool = False,
) -> DeferredDeletionResult:
    """Record intent before destructive work, then delete/rebuild/finalize once."""
    journal = DeferredDeletionJournal.new(
        batch_id=str(uuid4()),
        document_ids=document_ids,
        delete_llm_cache=delete_llm_cache,
        delete_file=delete_file,
    )
    store = _store(rag)
    # Persist intent before taking destructive action. The per-batch lock then
    # covers every state transition through finalization, including rebuild.
    store.save(journal)
    async with store.batch_lock(journal.batch_id):
        persisted = store.load(journal.batch_id)
        if persisted is None:  # defensive: an external operator removed intent
            raise RuntimeError(f"deletion journal {journal.batch_id!r} disappeared")
        return await _continue(rag, persisted)


async def resume_deferred_batch_delete(
    rag: Any, batch_id: str
) -> DeferredDeletionResult:
    """Resume a failed batch from durable state; never silently report success."""
    store = _store(rag)
    # Load *inside* the lock: a concurrent resume may have completed the
    # journal while this caller was waiting, in which case it must not rebuild.
    async with store.batch_lock(batch_id):
        journal = store.load(batch_id)
        if journal is None:
            raise ValueError(f"deletion journal {batch_id!r} was not found")
        if journal.stage is DeferredDeletionStage.COMMITTED:
            return DeferredDeletionResult(journal.batch_id, journal.stage)
        return await _continue(rag, journal)
