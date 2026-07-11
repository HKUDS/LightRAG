"""Behavioral tests for production-safe deferred batch deletion orchestration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from lightrag.base import DeletionResult
from lightrag.deletion_journal import (
    DeferredDeletionJournal,
    DeferredDeletionJournalStore,
    DeferredDeletionStage,
)

pytestmark = pytest.mark.offline


def _pipeline_status() -> dict:
    return {"busy": True, "destructive_busy": True, "history_messages": []}


def _rag(tmp_path):
    rag = MagicMock()
    rag.working_dir = str(tmp_path)
    rag.workspace = "default"
    rag.full_docs = AsyncMock()
    rag.full_entities = AsyncMock()
    rag.full_relations = AsyncMock()
    rag.doc_status = AsyncMock()
    rag.entity_chunks = AsyncMock()
    rag.relation_chunks = AsyncMock()
    rag.chunk_entity_relation_graph = AsyncMock()
    rag.entities_vdb = AsyncMock()
    rag.relationships_vdb = AsyncMock()
    rag.text_chunks = AsyncMock()
    rag.text_chunks.get_by_ids = AsyncMock(return_value=[])
    rag.llm_response_cache = AsyncMock()
    rag.llm_response_cache.delete = AsyncMock()
    rag._build_global_config.return_value = {}
    rag._insert_done = AsyncMock()
    rag.full_docs.get_by_id = AsyncMock(return_value=None)
    rag.full_entities.get_by_id = AsyncMock(return_value=None)
    rag.full_relations.get_by_id = AsyncMock(return_value=None)

    def make_delete(storage):
        async def clear_metadata_after_delete(doc_ids):
            """Model the post-delete reads a real storage backend would return."""
            storage.get_by_id = AsyncMock(return_value=None)

        return clear_metadata_after_delete

    for storage in (
        rag.full_docs,
        rag.full_entities,
        rag.full_relations,
        rag.doc_status,
    ):
        storage.delete.side_effect = make_delete(storage)
    rag.adelete_by_doc_id = AsyncMock(
        side_effect=[
            DeletionResult(
                status="success",
                doc_id="doc-a",
                message="prepared",
                entities_to_rebuild={"Shared": ["chunk-b"]},
                relationships_to_rebuild={("Shared", "Other"): ["chunk-b"]},
            ),
            DeletionResult(
                status="success",
                doc_id="doc-b",
                message="prepared",
                entities_to_rebuild={"Shared": ["chunk-c"]},
                relationships_to_rebuild={("Other", "Shared"): ["chunk-c"]},
            ),
        ]
    )
    return rag


@pytest.mark.asyncio
async def test_two_document_batch_runs_one_rebuild_then_finalizes_metadata(tmp_path):
    from lightrag.deferred_delete import run_deferred_batch_delete

    rag = _rag(tmp_path)
    rag.doc_status.get_by_id = AsyncMock(
        side_effect=[
            {"file_path": "a.pdf", "chunks_list": ["chunk-a"]},
            {"file_path": "b.pdf", "chunks_list": ["chunk-b"]},
        ]
    )
    status = _pipeline_status()
    lock = asyncio.Lock()

    with (
        patch(
            "lightrag.deferred_delete.get_namespace_data",
            new_callable=AsyncMock,
            return_value=status,
        ),
        patch(
            "lightrag.deferred_delete.get_namespace_lock",
            return_value=lock,
        ),
        patch(
            "lightrag.deferred_delete.rebuild_knowledge_from_chunks",
            new_callable=AsyncMock,
        ) as rebuild,
    ):
        result = await run_deferred_batch_delete(rag, ["doc-a", "doc-b"])

    assert result.stage is DeferredDeletionStage.COMMITTED
    rebuild.assert_awaited_once()
    rag.adelete_by_doc_id.assert_has_awaits(
        [
            call(
                "doc-a",
                delete_llm_cache=False,
                skip_rebuild=True,
                batch_chunk_ids={"chunk-a", "chunk-b"},
            ),
            call(
                "doc-b",
                delete_llm_cache=False,
                skip_rebuild=True,
                batch_chunk_ids={"chunk-a", "chunk-b"},
            ),
        ]
    )
    rag.full_entities.delete.assert_awaited_once_with(["doc-a", "doc-b"])
    rag.full_relations.delete.assert_awaited_once_with(["doc-a", "doc-b"])
    rag.doc_status.delete.assert_awaited_once_with(["doc-a", "doc-b"])
    rag.full_docs.delete.assert_awaited_once_with(["doc-a", "doc-b"])
    persisted = DeferredDeletionJournalStore(tmp_path, "default").load(result.batch_id)
    assert persisted is not None and persisted.stage is DeferredDeletionStage.COMMITTED


@pytest.mark.asyncio
async def test_not_found_document_is_checkpointed_and_batch_still_rebuilds(tmp_path):
    """A stale ID is a completed no-op, not an ambiguous destructive failure."""
    from lightrag.deferred_delete import run_deferred_batch_delete

    rag = _rag(tmp_path)
    rag.doc_status.get_by_id = AsyncMock(
        side_effect=[
            {"file_path": "a.pdf", "chunks_list": ["chunk-a"]},
            None,
        ]
    )
    rag.adelete_by_doc_id = AsyncMock(
        side_effect=[
            DeletionResult(
                status="success",
                doc_id="doc-a",
                message="prepared",
                entities_to_rebuild={"Shared": ["chunk-a"]},
                relationships_to_rebuild={},
            ),
            DeletionResult(
                status="not_found",
                doc_id="stale-doc",
                message="Document stale-doc not found.",
                status_code=404,
            ),
        ]
    )

    with (
        patch(
            "lightrag.deferred_delete.get_namespace_data",
            new_callable=AsyncMock,
            return_value=_pipeline_status(),
        ),
        patch(
            "lightrag.deferred_delete.get_namespace_lock",
            return_value=asyncio.Lock(),
        ),
        patch(
            "lightrag.deferred_delete.rebuild_knowledge_from_chunks",
            new_callable=AsyncMock,
        ) as rebuild,
    ):
        result = await run_deferred_batch_delete(rag, ["doc-a", "stale-doc"])

    assert result.stage is DeferredDeletionStage.COMMITTED
    rebuild.assert_awaited_once()
    journal = DeferredDeletionJournalStore(tmp_path, "default").load(result.batch_id)
    assert journal is not None
    assert journal.current_document_index == 2
    assert journal.entities_to_rebuild == {"Shared": ["chunk-a"]}
    assert journal.document_delete_checkpoints["stale-doc"]["state"] == "COMPLETED"


@pytest.mark.asyncio
async def test_cancellation_stops_before_next_destructive_delete(tmp_path):
    from lightrag.deferred_delete import run_deferred_batch_delete

    rag = _rag(tmp_path)
    rag.doc_status.get_by_id = AsyncMock(
        side_effect=[
            {"file_path": "a.pdf", "chunks_list": ["chunk-a"]},
            {"file_path": "b.pdf", "chunks_list": ["chunk-b"]},
        ]
    )
    checks = iter([False, True])

    async def cancellation_requested() -> bool:
        return next(checks)

    result = await run_deferred_batch_delete(
        rag,
        ["doc-a", "doc-b"],
        cancellation_requested=cancellation_requested,
    )

    assert result.stage is DeferredDeletionStage.FAILED_RETRYABLE
    assert result.error_detail == "document deletion cancelled by user"
    rag.adelete_by_doc_id.assert_awaited_once()
    journal = DeferredDeletionJournalStore(tmp_path, "default").load(result.batch_id)
    assert journal is not None
    assert journal.current_document_index == 1
    assert "doc-b" not in journal.document_delete_checkpoints


@pytest.mark.asyncio
async def test_finalization_deletes_cache_ids_persisted_in_status_metadata(tmp_path):
    """The deferred path must honor LightRAG's durable retry-state schema."""
    from lightrag.deferred_delete import run_deferred_batch_delete

    rag = _rag(tmp_path)
    rag.doc_status.get_by_id = AsyncMock(
        return_value={
            "file_path": "a.pdf",
            "chunks_list": ["chunk-a"],
            "metadata": {"deletion_llm_cache_ids": ["cache-a"]},
        }
    )
    rag._get_existing_llm_cache_ids = AsyncMock(return_value=[])
    status = _pipeline_status()
    lock = asyncio.Lock()

    with (
        patch(
            "lightrag.deferred_delete.get_namespace_data",
            new_callable=AsyncMock,
            return_value=status,
        ),
        patch("lightrag.deferred_delete.get_namespace_lock", return_value=lock),
        patch(
            "lightrag.deferred_delete.rebuild_knowledge_from_chunks",
            new_callable=AsyncMock,
        ),
    ):
        result = await run_deferred_batch_delete(rag, ["doc-a"], delete_llm_cache=True)

    assert result.stage is DeferredDeletionStage.COMMITTED
    rag.llm_response_cache.delete.assert_awaited_once_with(["cache-a"])
    rag._get_existing_llm_cache_ids.assert_awaited_once_with(["cache-a"])


@pytest.mark.asyncio
async def test_rebuild_failure_is_failed_retryable_and_keeps_journal(tmp_path):
    from lightrag.deferred_delete import run_deferred_batch_delete

    rag = _rag(tmp_path)
    rag.doc_status.get_by_id = AsyncMock(
        side_effect=[
            {"file_path": "a.pdf", "chunks_list": ["chunk-a"]},
            {"file_path": "b.pdf", "chunks_list": ["chunk-b"]},
        ]
    )
    status = _pipeline_status()
    lock = asyncio.Lock()

    with (
        patch(
            "lightrag.deferred_delete.get_namespace_data",
            new_callable=AsyncMock,
            return_value=status,
        ),
        patch(
            "lightrag.deferred_delete.get_namespace_lock",
            return_value=lock,
        ),
        patch(
            "lightrag.deferred_delete.rebuild_knowledge_from_chunks",
            new_callable=AsyncMock,
            side_effect=RuntimeError("forced rebuild failure"),
        ),
    ):
        result = await run_deferred_batch_delete(rag, ["doc-a", "doc-b"])

    assert result.stage is DeferredDeletionStage.FAILED_RETRYABLE
    assert "forced rebuild failure" in result.error_detail
    rag.doc_status.delete.assert_not_awaited()
    journal = DeferredDeletionJournalStore(tmp_path, "default").load(result.batch_id)
    assert journal is not None
    assert journal.stage is DeferredDeletionStage.FAILED_RETRYABLE


@pytest.mark.asyncio
async def test_document_delete_exception_is_persisted_as_retryable(tmp_path):
    from lightrag.deferred_delete import run_deferred_batch_delete

    rag = _rag(tmp_path)
    rag.doc_status.get_by_id = AsyncMock(return_value={"file_path": "a.pdf"})
    rag.adelete_by_doc_id = AsyncMock(side_effect=RuntimeError("storage unavailable"))

    result = await run_deferred_batch_delete(rag, ["doc-a"])

    assert result.stage is DeferredDeletionStage.FAILED_RETRYABLE
    assert "storage unavailable" in (result.error_detail or "")
    journal = DeferredDeletionJournalStore(tmp_path, "default").load(result.batch_id)
    assert journal is not None
    assert journal.recovery_stage is DeferredDeletionStage.PREPARED


@pytest.mark.asyncio
async def test_failed_batch_persists_delete_file_intent_and_snapshot_paths(tmp_path):
    from lightrag.deferred_delete import run_deferred_batch_delete

    rag = _rag(tmp_path)
    rag.doc_status.get_by_id = AsyncMock(
        return_value={"file_path": "uploads/a.pdf", "chunks_list": ["chunk-a"]}
    )
    rag.adelete_by_doc_id = AsyncMock(side_effect=RuntimeError("forced failure"))

    result = await run_deferred_batch_delete(rag, ["doc-a"], delete_file=True)

    assert result.stage is DeferredDeletionStage.FAILED_RETRYABLE
    persisted = DeferredDeletionJournalStore(tmp_path, "default").load(result.batch_id)
    assert persisted is not None
    assert persisted.delete_file is True
    assert persisted.source_file_paths == {"doc-a": "uploads/a.pdf"}


@pytest.mark.asyncio
async def test_resume_uses_persisted_targets_without_calling_document_delete(tmp_path):
    from lightrag.deferred_delete import resume_deferred_batch_delete
    from lightrag.deletion_journal import DeferredDeletionJournal

    rag = _rag(tmp_path)
    journal = DeferredDeletionJournal.new("resume-001", ["doc-a"], False)
    journal.document_metadata = {"doc-a": {"chunks_list": ["chunk-a"]}}
    journal.aggregate_targets(
        entities={"Shared": ["chunk-b"]},
        relationships={("Shared", "Other"): ["chunk-b"]},
    )
    journal.mark_rebuild_pending()
    journal.mark_failed_retryable("crashed")
    DeferredDeletionJournalStore(tmp_path, "default").save(journal)
    status = _pipeline_status()
    lock = asyncio.Lock()

    with (
        patch(
            "lightrag.deferred_delete.get_namespace_data",
            new_callable=AsyncMock,
            return_value=status,
        ),
        patch(
            "lightrag.deferred_delete.get_namespace_lock",
            return_value=lock,
        ),
        patch(
            "lightrag.deferred_delete.rebuild_knowledge_from_chunks",
            new_callable=AsyncMock,
        ) as rebuild,
    ):
        result = await resume_deferred_batch_delete(rag, journal.batch_id)

    assert result.stage is DeferredDeletionStage.COMMITTED
    rebuild.assert_awaited_once()
    rag.adelete_by_doc_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_restart_from_prepared_resumes_actual_document_deletion_in_batch_mode(
    tmp_path,
):
    """A restart at PREPARED must retain the no-per-document-rebuild guard."""
    from lightrag.deferred_delete import resume_deferred_batch_delete

    rag = _rag(tmp_path)
    rag.doc_status.get_by_id = AsyncMock(
        return_value={"file_path": "a.pdf", "chunks_list": ["chunk-a"]}
    )
    journal = DeferredDeletionJournal.new("restart-prepared-001", ["doc-a"], False)
    DeferredDeletionJournalStore(tmp_path, "default").save(journal)
    status = _pipeline_status()
    lock = asyncio.Lock()

    with (
        patch(
            "lightrag.deferred_delete.get_namespace_data",
            new_callable=AsyncMock,
            return_value=status,
        ),
        patch("lightrag.deferred_delete.get_namespace_lock", return_value=lock),
        patch(
            "lightrag.deferred_delete.rebuild_knowledge_from_chunks",
            new_callable=AsyncMock,
        ) as rebuild,
    ):
        result = await resume_deferred_batch_delete(rag, journal.batch_id)

    assert result.stage is DeferredDeletionStage.COMMITTED
    rag.adelete_by_doc_id.assert_awaited_once_with(
        "doc-a",
        delete_llm_cache=False,
        skip_rebuild=True,
        batch_chunk_ids={"chunk-a"},
    )
    rebuild.assert_awaited_once()


@pytest.mark.asyncio
async def test_crash_after_document_delete_before_result_checkpoint_never_repeats_delete(
    tmp_path,
):
    """An intent left without a durable result is ambiguous, never retryable."""
    from lightrag.deferred_delete import (
        resume_deferred_batch_delete,
        run_deferred_batch_delete,
    )

    rag = _rag(tmp_path)
    rag.doc_status.get_by_id = AsyncMock(
        return_value={"file_path": "a.pdf", "chunks_list": ["chunk-a"]}
    )
    original_save = DeferredDeletionJournalStore.save

    def crash_before_completed_checkpoint(store, journal):
        checkpoint = journal.document_delete_checkpoints.get("doc-a", {})
        if checkpoint.get("state") == "COMPLETED":
            raise SystemExit("simulated process crash")
        original_save(store, journal)

    with patch.object(
        DeferredDeletionJournalStore,
        "save",
        new=crash_before_completed_checkpoint,
    ):
        with pytest.raises(SystemExit, match="simulated process crash"):
            await run_deferred_batch_delete(rag, ["doc-a"])

    journal_path = next((tmp_path / "deletion_journals" / "default").glob("*.json"))
    journal = DeferredDeletionJournalStore(tmp_path, "default").load(journal_path.stem)
    assert journal is not None
    assert journal.document_delete_checkpoints == {"doc-a": {"state": "INTENT"}}

    result = await resume_deferred_batch_delete(rag, journal.batch_id)

    assert result.stage is DeferredDeletionStage.FAILED_RETRYABLE
    assert "outcome is unknown" in (result.error_detail or "")
    rag.adelete_by_doc_id.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_resume_serializes_same_batch_and_rebuilds_once(tmp_path):
    """A second operator/process must observe the first committed outcome."""
    from lightrag.deferred_delete import resume_deferred_batch_delete

    rag = _rag(tmp_path)
    journal = DeferredDeletionJournal.new("resume-lock-001", ["doc-a"], False)
    journal.aggregate_targets(entities={"Shared": ["chunk-a"]}, relationships={})
    journal.mark_rebuild_pending()
    DeferredDeletionJournalStore(tmp_path, "default").save(journal)
    status = _pipeline_status()
    lock = asyncio.Lock()
    rebuild_started = asyncio.Event()
    let_rebuild_finish = asyncio.Event()

    async def blocked_rebuild(*_args, **_kwargs):
        rebuild_started.set()
        await let_rebuild_finish.wait()

    with (
        patch(
            "lightrag.deferred_delete.get_namespace_data",
            new_callable=AsyncMock,
            return_value=status,
        ),
        patch("lightrag.deferred_delete.get_namespace_lock", return_value=lock),
        patch(
            "lightrag.deferred_delete.rebuild_knowledge_from_chunks",
            side_effect=blocked_rebuild,
        ) as rebuild,
    ):
        first = asyncio.create_task(resume_deferred_batch_delete(rag, journal.batch_id))
        await rebuild_started.wait()
        second = asyncio.create_task(
            resume_deferred_batch_delete(rag, journal.batch_id)
        )
        await asyncio.sleep(0)
        assert not second.done()
        let_rebuild_finish.set()
        first_result, second_result = await asyncio.gather(first, second)

    assert first_result.stage is DeferredDeletionStage.COMMITTED
    assert second_result.stage is DeferredDeletionStage.COMMITTED
    rebuild.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_finalization_retry_does_not_repeat_completed_rebuild(tmp_path):
    """A crash after rebuild resumes only metadata finalization, not a second rebuild."""
    from lightrag.deferred_delete import resume_deferred_batch_delete

    rag = _rag(tmp_path)
    journal = DeferredDeletionJournal.new("finalization-retry-001", ["doc-a"], False)
    journal.mark_rebuild_pending()
    journal.mark_finalization_pending()
    journal.mark_failed_retryable("process crashed after rebuild")
    DeferredDeletionJournalStore(tmp_path, "default").save(journal)

    with patch(
        "lightrag.deferred_delete.rebuild_knowledge_from_chunks",
        new_callable=AsyncMock,
    ) as rebuild:
        result = await resume_deferred_batch_delete(rag, journal.batch_id)

    assert result.stage is DeferredDeletionStage.COMMITTED
    rebuild.assert_not_awaited()
    rag.adelete_by_doc_id.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_name", ["full_entities", "full_relations", "doc_status"]
)
async def test_finalization_requires_every_metadata_store_to_be_empty(
    tmp_path, store_name
):
    from lightrag.deferred_delete import resume_deferred_batch_delete

    rag = _rag(tmp_path)
    journal = DeferredDeletionJournal.new("verify-all-stores", ["doc-a"], False)
    journal.mark_rebuild_pending()
    journal.mark_finalization_pending()
    DeferredDeletionJournalStore(tmp_path, "default").save(journal)
    getattr(rag, store_name).get_by_id = AsyncMock(return_value={"still": "here"})
    # Simulate a storage backend silently ignoring this delete; the coordinator
    # must reject it instead of treating the awaited delete call as success.
    getattr(rag, store_name).delete = AsyncMock()

    result = await resume_deferred_batch_delete(rag, journal.batch_id)

    assert result.stage is DeferredDeletionStage.FAILED_RETRYABLE
    assert "metadata remains" in (result.error_detail or "")
    persisted = DeferredDeletionJournalStore(tmp_path, "default").load(journal.batch_id)
    assert persisted is not None
    assert persisted.stage is DeferredDeletionStage.FAILED_RETRYABLE


@pytest.mark.asyncio
async def test_finalization_retry_is_idempotent_after_partial_metadata_deletion(
    tmp_path,
):
    from lightrag.deferred_delete import resume_deferred_batch_delete

    rag = _rag(tmp_path)
    journal = DeferredDeletionJournal.new("partial-finalization", ["doc-a"], False)
    journal.mark_rebuild_pending()
    journal.mark_finalization_pending()
    DeferredDeletionJournalStore(tmp_path, "default").save(journal)
    rag.doc_status.get_by_id = AsyncMock(side_effect=[{"still": "here"}, None])
    rag.doc_status.delete = AsyncMock()

    first = await resume_deferred_batch_delete(rag, journal.batch_id)

    async def clear_doc_status(_doc_ids):
        rag.doc_status.get_by_id = AsyncMock(return_value=None)

    rag.doc_status.delete.side_effect = clear_doc_status
    second = await resume_deferred_batch_delete(rag, journal.batch_id)

    assert first.stage is DeferredDeletionStage.FAILED_RETRYABLE
    assert second.stage is DeferredDeletionStage.COMMITTED
    assert rag.full_docs.delete.await_count == 2
