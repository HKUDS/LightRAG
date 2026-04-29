import importlib
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_lightrag = importlib.import_module("lightrag.lightrag")
_base = importlib.import_module("lightrag.base")
_constants = importlib.import_module("lightrag.constants")
_utils = importlib.import_module("lightrag.utils")
sys.argv = _original_argv

DocStatus = _base.DocStatus
FULL_DOCS_FORMAT_LIGHTRAG = _constants.FULL_DOCS_FORMAT_LIGHTRAG
FULL_DOCS_FORMAT_PENDING_PARSE = _constants.FULL_DOCS_FORMAT_PENDING_PARSE
PARSED_DIR_NAME = _constants.PARSED_DIR_NAME
compute_mdhash_id = _utils.compute_mdhash_id
LightRAG = _lightrag.LightRAG
pipeline_index_file = _document_routes.pipeline_index_file
pipeline_index_files = _document_routes.pipeline_index_files
pipeline_enqueue_file = _document_routes.pipeline_enqueue_file
run_scanning_process = _document_routes.run_scanning_process
DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


class _FakeDocStatus:
    def __init__(self):
        self.docs = {}

    async def get_by_id(self, doc_id):
        return self.docs.get(doc_id)


class _FakeRag:
    def __init__(self, final_status=DocStatus.PROCESSED):
        self.doc_status = _FakeDocStatus()
        self.final_status = final_status
        self.enqueued = []
        self.errors = []

    async def apipeline_enqueue_documents(
        self, input, ids=None, file_paths=None, track_id=None, docs_format=None
    ):
        self.enqueued.append(
            {
                "input": input,
                "file_path": file_paths,
                "track_id": track_id,
                "docs_format": docs_format,
            }
        )
        return track_id

    async def apipeline_process_enqueue_documents(self):
        for item in self.enqueued:
            file_path = item["file_path"]
            doc_id = compute_mdhash_id(file_path, prefix="doc-")
            self.doc_status.docs[doc_id] = {
                "status": self.final_status,
                "file_path": file_path,
                "track_id": item["track_id"],
            }

    async def apipeline_enqueue_error_documents(self, error_files, track_id=None):
        self.errors.append((error_files, track_id))


class _ScanDocStatus:
    def __init__(self, docs_by_path):
        self.docs_by_path = docs_by_path

    async def get_doc_by_file_path(self, file_path):
        return self.docs_by_path.get(file_path)


class _ScanRag:
    def __init__(self, docs_by_path):
        self.doc_status = _ScanDocStatus(docs_by_path)
        self.process_calls = 0

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1


class _DuplicateUploadRag:
    def __init__(self, docs_by_path):
        self.doc_status = _ScanDocStatus(docs_by_path)


class _ArchiveFailureRag:
    _archive_docx_source_after_full_docs_sync = (
        LightRAG._archive_docx_source_after_full_docs_sync
    )


class _ParseFullDocs:
    def __init__(self, source_path):
        self.source_path = source_path
        self.events = []
        self.data = {}

    async def upsert(self, data):
        self.events.append("upsert")
        self.data.update(data)

    async def index_done_callback(self):
        self.events.append("index_done")
        assert self.source_path.exists()


class _ParseRag:
    _archive_docx_source_after_full_docs_sync = (
        LightRAG._archive_docx_source_after_full_docs_sync
    )

    def __init__(self, working_dir, source_path):
        self.working_dir = str(working_dir)
        self.full_docs = _ParseFullDocs(source_path)

    def _resolve_source_file_for_parser(self, file_path):
        return file_path


async def test_pipeline_index_file_leaves_lightrag_document_docx_for_parser_archive(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DOCX_PARSING_DEFAULT_METHOD", "lightrag_document")
    file_path = tmp_path / "sample.[docling].docx"
    file_path.write_bytes(b"docx bytes")
    rag = _FakeRag()

    await pipeline_index_file(rag, file_path, "track-docx")

    assert file_path.exists()
    assert not (tmp_path / PARSED_DIR_NAME / file_path.name).exists()
    assert rag.enqueued[0]["file_path"] == str(file_path)
    assert rag.enqueued[0]["docs_format"] == FULL_DOCS_FORMAT_PENDING_PARSE


async def test_pipeline_enqueue_lightrag_document_docx_does_not_move_source(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DOCX_PARSING_DEFAULT_METHOD", "lightrag_document")
    file_path = tmp_path / "pending.docx"
    file_path.write_bytes(b"docx bytes")
    rag = _FakeRag()

    success, returned_track_id = await pipeline_enqueue_file(
        rag, file_path, "track-docx"
    )

    assert success is True
    assert returned_track_id == "track-docx"
    assert file_path.exists()
    assert not (tmp_path / PARSED_DIR_NAME / file_path.name).exists()
    assert rag.enqueued[0]["file_path"] == str(file_path)
    assert rag.enqueued[0]["docs_format"] == FULL_DOCS_FORMAT_PENDING_PARSE


async def test_pipeline_enqueue_docx_plain_text_extracts_before_enqueue(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DOCX_PARSING_DEFAULT_METHOD", "plain_text")
    monkeypatch.setattr(
        _document_routes, "_extract_docx", lambda file_bytes: "plain docx content"
    )
    file_path = tmp_path / "plain.docx"
    file_path.write_bytes(b"docx bytes")
    rag = _FakeRag()

    success, returned_track_id = await pipeline_enqueue_file(
        rag, file_path, "track-docx"
    )

    assert success is True
    assert returned_track_id == "track-docx"
    assert rag.enqueued == [
        {
            "input": "plain docx content",
            "file_path": file_path.name,
            "track_id": "track-docx",
            "docs_format": None,
        }
    ]
    assert not file_path.exists()
    assert (tmp_path / PARSED_DIR_NAME / file_path.name).exists()


async def test_pipeline_enqueue_md_moves_after_enqueue(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCX_PARSING_DEFAULT_METHOD", "plain_text")
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Notes\n\nmarkdown content", encoding="utf-8")
    rag = _FakeRag()

    success, returned_track_id = await pipeline_enqueue_file(rag, file_path, "track-md")

    assert success is True
    assert returned_track_id == "track-md"
    assert rag.enqueued == [
        {
            "input": "# Notes\n\nmarkdown content",
            "file_path": file_path.name,
            "track_id": "track-md",
            "docs_format": None,
        }
    ]
    assert not file_path.exists()
    assert (tmp_path / PARSED_DIR_NAME / file_path.name).exists()


async def test_pipeline_index_files_leaves_lightrag_document_docx_batch(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DOCX_PARSING_DEFAULT_METHOD", "lightrag_document")
    first = tmp_path / "first.docx"
    second = tmp_path / "second.[mineru].docx"
    first.write_bytes(b"first docx bytes")
    second.write_bytes(b"second docx bytes")
    rag = _FakeRag()

    await pipeline_index_files(rag, [second, first], "track-scan")

    assert first.exists()
    assert second.exists()
    assert not (tmp_path / PARSED_DIR_NAME / first.name).exists()
    assert not (tmp_path / PARSED_DIR_NAME / second.name).exists()
    assert all(
        item["docs_format"] == FULL_DOCS_FORMAT_PENDING_PARSE for item in rag.enqueued
    )


async def test_scan_existing_full_path_docx_does_not_reenqueue(tmp_path, monkeypatch):
    file_path = tmp_path / "already-parsed.docx"
    file_path.write_bytes(b"docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            str(file_path): {
                "status": DocStatus.PARSING.value,
                "file_path": str(file_path),
                "track_id": "track-existing",
            }
        }
    )

    async def fail_if_reenqueue(*args, **kwargs):
        raise AssertionError("existing docx should not be re-enqueued")

    monkeypatch.setattr(_document_routes, "pipeline_index_files", fail_if_reenqueue)

    await run_scanning_process(rag, doc_manager, "track-scan")

    assert rag.process_calls == 1


async def test_upload_rejects_same_name_failed_doc_status_without_full_docs(
    tmp_path, monkeypatch
):
    # Other tests (e.g. test_auth.py) may replace global_args with a SimpleNamespace
    # that lacks max_upload_size; pin a known state so the upload endpoint runs.
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _DuplicateUploadRag(
        {
            "failed.docx": {
                "status": DocStatus.FAILED.value,
                "file_path": "failed.docx",
                "track_id": "track-failed",
                "metadata": {"error_type": "file_extraction_error"},
            }
        }
    )
    router = create_document_routes(rag, doc_manager)
    upload_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "upload_to_input_dir"
    ][-1]
    upload_file = _document_routes.UploadFile(
        filename="failed.docx",
        file=BytesIO(b"replacement docx bytes"),
    )

    response = await upload_endpoint(_document_routes.BackgroundTasks(), upload_file)

    assert response.status == "duplicated"
    assert response.track_id == "track-failed"
    assert "Status: failed" in response.message
    assert not (tmp_path / "failed.docx").exists()


async def test_docx_archive_failure_is_best_effort(tmp_path, monkeypatch):
    file_path = tmp_path / "archive-failure.docx"
    file_path.write_bytes(b"docx bytes")
    rag = _ArchiveFailureRag()

    async def _raise_archive_failure(*args, **kwargs):
        raise OSError("simulated archive failure")

    monkeypatch.setattr(_lightrag, "move_file_to_parsed_dir", _raise_archive_failure)

    archived_path = await LightRAG._archive_docx_source_after_full_docs_sync(
        rag, str(file_path)
    )

    assert archived_path is None
    assert file_path.exists()


async def test_parse_native_archives_docx_after_full_docs_sync(tmp_path, monkeypatch):
    source_path = tmp_path / "parsed-after-sync.docx"
    source_path.write_bytes(b"docx bytes")
    rag = _ParseRag(tmp_path / "work", source_path)

    parse_document = importlib.import_module("lightrag.extraction.parse_document")
    monkeypatch.setattr(
        parse_document,
        "parse_docx_to_interchange_jsonl",
        lambda file_bytes,
        source_file,
        doc_id,
        output_dir: '{"type":"meta"}\n{"type":"content","content":"parsed"}',
    )

    result = await LightRAG.parse_native(
        rag,
        "doc-test",
        str(source_path),
        {"format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
    )

    assert result["content"]
    assert rag.full_docs.events == ["upsert", "index_done"]
    assert not source_path.exists()
    assert (tmp_path / PARSED_DIR_NAME / source_path.name).exists()
    assert rag.full_docs.data["doc-test"]["parsed_engine"] == "native"


async def test_parse_native_docx_interchange_failure_raises_without_fallback(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "interchange-failure.docx"
    source_path.write_bytes(b"docx bytes")
    rag = _ParseRag(tmp_path / "work", source_path)
    parse_document = importlib.import_module("lightrag.extraction.parse_document")

    def _raise_parser(file_bytes, source_file, doc_id, output_dir):
        raise RuntimeError("interchange boom")

    def _fail_fallback(file_bytes):
        raise AssertionError("plain text fallback should not run")

    monkeypatch.setattr(
        parse_document, "parse_docx_to_interchange_jsonl", _raise_parser
    )
    monkeypatch.setattr(_document_routes, "_extract_docx", _fail_fallback)

    with pytest.raises(RuntimeError, match="interchange boom"):
        await LightRAG.parse_native(
            rag,
            "doc-test",
            str(source_path),
            {"format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )

    assert source_path.exists()
    assert rag.full_docs.events == []


async def test_parse_native_docx_empty_interchange_result_raises_without_fallback(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "empty-interchange.docx"
    source_path.write_bytes(b"docx bytes")
    rag = _ParseRag(tmp_path / "work", source_path)
    parse_document = importlib.import_module("lightrag.extraction.parse_document")

    def _fail_fallback(file_bytes):
        raise AssertionError("plain text fallback should not run")

    monkeypatch.setattr(
        parse_document, "parse_docx_to_interchange_jsonl", lambda *args: " \n\t"
    )
    monkeypatch.setattr(_document_routes, "_extract_docx", _fail_fallback)

    with pytest.raises(ValueError, match="empty content"):
        await LightRAG.parse_native(
            rag,
            "doc-test",
            str(source_path),
            {"format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )

    assert source_path.exists()
    assert rag.full_docs.events == []


def test_lightrag_document_reprocess_uses_full_docs_without_reparse():
    engine = LightRAG._resolve_parser_engine(
        object(),
        "report.[mineru].docx",
        {
            "format": FULL_DOCS_FORMAT_LIGHTRAG,
            "lightrag_document_path": "report.blocks.jsonl",
            "parsed_engine": "mineru",
        },
    )

    assert engine == "native"
