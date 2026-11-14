"""Deletion status is visible, durable, and excluded from ingestion recovery."""

from uuid import uuid4

import pytest

from lightrag.base import DocStatus
from lightrag.constants import (
    CUSTOM_CHUNK_PATCH_METADATA_KEY,
    KG_PURGE_METADATA_KEY,
    KG_PURGE_PHASE_COMPLETED,
    KG_PURGE_PHASE_DERIVED_COMMITTED,
    KG_PURGE_PHASE_PREPARED,
    KG_PURGE_SCHEMA_VERSION,
)
from lightrag.pipeline import _AUTO_RESUME_DOC_STATUSES
from lightrag.utils_pipeline import (
    make_custom_chunk_id,
    make_custom_chunk_operation_id,
    make_kg_purge_operation_id,
)

from .test_delete_fail_closed import _build_rag, _ingest
from .test_purge_primitive import _DocStatus, _Graph, _KV, _make_rag, _status

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.mark.parametrize("stamp", [False, True])
async def test_prepared_status_and_journal_share_a_write_before_mutation(stamp):
    writes = []

    class Status(_DocStatus):
        async def update_doc_status_fields(self, doc_id, fields, **kwargs):
            writes.append(fields)
            await super().update_doc_status_fields(doc_id, fields, **kwargs)

    rag = _make_rag(
        graph=_Graph(),
        doc_status=Status({"d1": {"status": DocStatus.PROCESSED, "metadata": {}}}),
        full_entities=_KV({"d1": {"entity_names": []}}),
        full_relations=_KV({"d1": {"relation_pairs": []}}),
    )

    async def fail_derived(*args, **kwargs):
        assert writes[0]["metadata"][KG_PURGE_METADATA_KEY]["phase"] == "prepared"
        assert (writes[0].get("status") == DocStatus.DELETING) is stamp
        raise RuntimeError("interrupted after prepared")

    rag._purge_derived_kg_contributions = fail_derived
    status, lock = _status()
    with pytest.raises(RuntimeError, match="interrupted"):
        await rag._purge_kg_contributions(
            "d1",
            [],
            stamp_deleting=stamp,
            pipeline_status=status,
            pipeline_status_lock=lock,
        )
    row = await rag.doc_status.get_by_id("d1")
    assert row["status"] == (DocStatus.DELETING if stamp else DocStatus.PROCESSED)


@pytest.mark.parametrize("retry_via_route", [False, True])
async def test_failed_delete_remains_visible_and_retry_completes(
    tmp_path, monkeypatch, retry_via_route
):
    rag = await _build_rag(tmp_path, f"deleting-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)
        original = rag._purge_derived_kg_contributions

        async def fail(*args, **kwargs):
            raise RuntimeError("injected purge interruption")

        monkeypatch.setattr(rag, "_purge_derived_kg_contributions", fail)
        result = await rag.adelete_by_doc_id(doc_id)
        assert result.status == "fail"
        row = await rag.doc_status.get_by_id(doc_id)
        assert row["status"] == DocStatus.DELETING
        assert row["metadata"]["deletion_failed"] is True
        assert "injected purge interruption" in row["error_msg"]
        assert DocStatus.DELETING not in _AUTO_RESUME_DOC_STATUSES

        # A process/retry request cannot silently turn deletion into ingestion.
        from .conftest import request_failed_retry

        await request_failed_retry(rag)
        await rag.apipeline_process_enqueue_documents()
        assert (await rag.doc_status.get_by_id(doc_id))["status"] == DocStatus.DELETING

        counts = await rag.doc_status.get_all_status_counts()
        assert counts["deleting"] == 1
        docs, total = await rag.doc_status.get_docs_paginated(
            status_filter=DocStatus.DELETING
        )
        assert total == 1
        assert docs[0][0] == doc_id

        # Existing content/source must remain blocked, with actionable diagnostics.
        for candidate_id, source in [
            (doc_id, "d.txt"),
            ("new-doc", "d.txt"),
            ("same-content", "another.txt"),
        ]:
            await rag.apipeline_enqueue_documents(
                "alice works at acme", ids=[candidate_id], file_paths=[source]
            )
            if candidate_id != doc_id:
                assert await rag.doc_status.get_by_id(candidate_id) is None
        failures = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
        assert any(
            "re-run deletion first" in doc.error_msg for doc in failures.values()
        )
        assert (await rag.doc_status.get_by_id(doc_id))["status"] == DocStatus.DELETING

        with pytest.raises(RuntimeError, match="re-run deletion first"):
            await rag.ainsert_custom_chunks("new body", ["new chunk"], doc_id=doc_id)

        monkeypatch.setattr(rag, "_purge_derived_kg_contributions", original)
        if retry_via_route:
            import asyncio
            import importlib
            import sys

            with monkeypatch.context() as import_patch:
                import_patch.setattr(sys, "argv", [sys.argv[0]])
                routes = importlib.import_module("lightrag.api.routers.document_routes")
            router = routes.create_document_routes(
                rag, routes.DocumentManager(str(tmp_path / "inputs"))
            )
            endpoint = next(
                route.endpoint
                for route in router.routes
                if route.path == "/documents/delete_document"
            )
            tasks = set()
            result = await endpoint(
                routes.DeleteDocRequest(doc_ids=[doc_id]), managed_tasks=tasks
            )
            assert result.status == "deletion_started"
            await asyncio.gather(*tasks)
        else:
            assert (await rag.adelete_by_doc_id(doc_id)).status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
    finally:
        await rag.finalize_storages()


def _journal(doc_id: str, chunk_ids: list[str], phase: str) -> dict:
    return {
        "schema_version": KG_PURGE_SCHEMA_VERSION,
        "operation_id": make_kg_purge_operation_id(doc_id, chunk_ids),
        "phase": phase,
        "chunk_count": len(chunk_ids),
        "updated_at": 0,
    }


@pytest.mark.parametrize(
    "phase",
    [
        KG_PURGE_PHASE_PREPARED,
        KG_PURGE_PHASE_DERIVED_COMMITTED,
        KG_PURGE_PHASE_COMPLETED,
    ],
)
@pytest.mark.parametrize("stamp", [False, True])
async def test_retry_with_existing_journal_stamps_deleting(phase, stamp):
    """A journal written without the stamp (older code, or an ingestion purge
    that a deletion now takes over) is projected into DELETING before any
    resumed work — and only when the caller is deletion."""
    writes = []

    class Status(_DocStatus):
        async def update_doc_status_fields(self, doc_id, fields, **kwargs):
            writes.append(dict(fields))
            await super().update_doc_status_fields(doc_id, fields, **kwargs)

    rag = _make_rag(
        graph=_Graph(),
        doc_status=Status(
            {
                "d1": {
                    "status": DocStatus.PROCESSED,
                    "metadata": {KG_PURGE_METADATA_KEY: _journal("d1", [], phase)},
                }
            }
        ),
        full_entities=_KV({"d1": {"entity_names": []}}),
        full_relations=_KV({"d1": {"relation_pairs": []}}),
    )

    async def fail_derived(*args, **kwargs):
        raise RuntimeError("interrupted while resuming")

    rag._purge_derived_kg_contributions = fail_derived
    status, lock = _status()
    call = rag._purge_kg_contributions(
        "d1",
        [],
        stamp_deleting=stamp,
        pipeline_status=status,
        pipeline_status_lock=lock,
    )
    if phase == KG_PURGE_PHASE_PREPARED:
        # Only a `prepared` journal re-runs the derived pass.
        with pytest.raises(RuntimeError, match="interrupted while resuming"):
            await call
    else:
        await call

    row = await rag.doc_status.get_by_id("d1")
    assert row["status"] == (DocStatus.DELETING if stamp else DocStatus.PROCESSED)
    if stamp:
        # The projection is the FIRST write, and it does not move the journal.
        first = writes[0]
        assert first["status"] == DocStatus.DELETING
        assert first["metadata"][KG_PURGE_METADATA_KEY]["phase"] == phase
        assert isinstance(first.get("updated_at"), str)
    else:
        assert all("status" not in fields for fields in writes)


async def test_deletion_takes_over_unstamped_journal_and_stays_visible(
    tmp_path, monkeypatch
):
    """End to end: a PROCESSED row carrying an unstamped `prepared` journal
    becomes DELETING on the first deletion attempt, even when it fails."""
    rag = await _build_rag(tmp_path, f"takeover-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)
        row = await rag.doc_status.get_by_id(doc_id)
        chunk_ids = list(row["chunks_list"])
        await rag._write_kg_purge_phase(
            doc_id,
            KG_PURGE_PHASE_PREPARED,
            operation_id=make_kg_purge_operation_id(doc_id, chunk_ids),
            chunk_count=len(chunk_ids),
        )
        assert (await rag.doc_status.get_by_id(doc_id))["status"] == (
            DocStatus.PROCESSED
        )

        original = rag._purge_derived_kg_contributions

        async def fail(*args, **kwargs):
            raise RuntimeError("injected purge interruption")

        monkeypatch.setattr(rag, "_purge_derived_kg_contributions", fail)
        assert (await rag.adelete_by_doc_id(doc_id)).status == "fail"
        row = await rag.doc_status.get_by_id(doc_id)
        assert row["status"] == DocStatus.DELETING
        assert row["metadata"][KG_PURGE_METADATA_KEY]["phase"] == (
            KG_PURGE_PHASE_PREPARED
        )

        monkeypatch.setattr(rag, "_purge_derived_kg_contributions", original)
        assert (await rag.adelete_by_doc_id(doc_id)).status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
    finally:
        await rag.finalize_storages()


async def test_custom_chunks_refuse_deleting_row_with_patch_journal(
    tmp_path, monkeypatch
):
    """A DELETING row whose metadata still holds a matching custom-chunk patch
    journal must not resume the patch: that path bypasses the generic
    PROCESSED-only status check, so only the DELETING guard stops it."""
    rag = await _build_rag(tmp_path, f"patch-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)

        async def fail(*args, **kwargs):
            raise RuntimeError("injected purge interruption")

        monkeypatch.setattr(rag, "_purge_derived_kg_contributions", fail)
        assert (await rag.adelete_by_doc_id(doc_id)).status == "fail"

        new_chunk = "patched chunk"
        operation_id = make_custom_chunk_operation_id(
            doc_id, [make_custom_chunk_id(doc_id, new_chunk)]
        )
        row = await rag.doc_status.get_by_id(doc_id)
        patch_journal = {"operation_id": operation_id, "mode": "patch"}
        metadata = {**row["metadata"], CUSTOM_CHUNK_PATCH_METADATA_KEY: patch_journal}
        await rag.doc_status.update_doc_status_fields(doc_id, {"metadata": metadata})

        with pytest.raises(RuntimeError, match="re-run deletion first"):
            await rag.ainsert_custom_chunks("patched body", [new_chunk], doc_id=doc_id)

        row = await rag.doc_status.get_by_id(doc_id)
        assert row["status"] == DocStatus.DELETING
        assert row["metadata"][CUSTOM_CHUNK_PATCH_METADATA_KEY] == patch_journal
        assert row["metadata"][KG_PURGE_METADATA_KEY]["phase"] == (
            KG_PURGE_PHASE_PREPARED
        )
    finally:
        await rag.finalize_storages()
