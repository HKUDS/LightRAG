import sys
from uuid import uuid4

import pytest

from lightrag.base import DocStatus
from lightrag.constants import PARSED_DIR_NAME

from .test_delete_fail_closed import _build_rag


pytestmark = pytest.mark.offline


@pytest.mark.parametrize("from_scan", [False, True])
@pytest.mark.parametrize("filename", ["__tmp__report.txt", "report.txt"])
async def test_deferred_parser_preserves_source_until_it_is_parsed(
    tmp_path, monkeypatch, from_scan, filename
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LIGHTRAG_PARSER", "txt:legacy")
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    from lightrag.api.routers.document_routes import pipeline_enqueue_file

    rag = await _build_rag(tmp_path, f"deferred-source-{uuid4().hex}")
    source = tmp_path / filename
    content = "alice works at acme"
    source.write_text(content, encoding="utf-8")
    try:
        success, _ = await pipeline_enqueue_file(
            rag, source, "test-deferred-source", from_scan=from_scan
        )
        assert success
        await rag.apipeline_process_enqueue_documents()

        match = await rag.doc_status.get_doc_by_file_basename(filename)
        assert match is not None
        _, row = match
        assert row["status"] == DocStatus.PROCESSED.value, row
        assert (tmp_path / PARSED_DIR_NAME / filename).read_text(
            encoding="utf-8"
        ) == content
    finally:
        await rag.finalize_storages()
