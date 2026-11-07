"""Deletion status is visible, durable, and excluded from ingestion recovery."""

from uuid import uuid4

import pytest

from lightrag.base import DocStatus
from lightrag.constants import KG_PURGE_METADATA_KEY
from lightrag.pipeline import _AUTO_RESUME_DOC_STATUSES

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
