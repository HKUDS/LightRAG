import importlib
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_lightrag = importlib.import_module("lightrag.lightrag")
_pipeline = importlib.import_module("lightrag.pipeline")
_base = importlib.import_module("lightrag.base")
_constants = importlib.import_module("lightrag.constants")
_utils = importlib.import_module("lightrag.utils")
_parser_routing = importlib.import_module("lightrag.parser_routing")
sys.argv = _original_argv

DocStatus = _base.DocStatus
DeletionResult = _base.DeletionResult
FULL_DOCS_FORMAT_LIGHTRAG = _constants.FULL_DOCS_FORMAT_LIGHTRAG
FULL_DOCS_FORMAT_PENDING_PARSE = _constants.FULL_DOCS_FORMAT_PENDING_PARSE
PARSED_DIR_NAME = _constants.PARSED_DIR_NAME
compute_mdhash_id = _utils.compute_mdhash_id
LightRAG = _lightrag.LightRAG
resolve_stored_document_parser_engine = (
    _parser_routing.resolve_stored_document_parser_engine
)
pipeline_index_file = _document_routes.pipeline_index_file
pipeline_index_files = _document_routes.pipeline_index_files
pipeline_enqueue_file = _document_routes.pipeline_enqueue_file
run_scanning_process = _document_routes.run_scanning_process
DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    """Initialize the shared_storage module-level dicts before each test.

    The scan endpoint and pipeline-busy guards introduced for the
    reentrancy / resume work read ``pipeline_status`` via
    ``get_namespace_data``, which raises if shared dicts have never been
    initialized.  Tests using mocked ``LightRAG`` instances don't run
    ``initialize_storages``, so we set up the shared store here and reset
    pipeline_status state per-test to avoid leakage.
    """
    import importlib

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    yield
    # Reset pipeline_status to a clean state so subsequent tests don't
    # inherit ``busy`` / ``scanning`` flags set by prior runs.
    if shared_storage._shared_dicts is not None:
        for key in list(shared_storage._shared_dicts.keys()):
            if key.endswith("pipeline_status") or key == "pipeline_status":
                ns = shared_storage._shared_dicts[key]
                if isinstance(ns, dict):
                    ns["busy"] = False
                    ns["scanning"] = False


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
        self,
        input,
        ids=None,
        file_paths=None,
        track_id=None,
        docs_format=None,
        parsed_engine=None,
        process_options=None,
        from_scan=False,
    ):
        item = {
            "input": input,
            "file_path": file_paths,
            "track_id": track_id,
            "docs_format": docs_format,
            "parsed_engine": parsed_engine,
            "process_options": process_options,
            "from_scan": from_scan,
        }
        self.enqueued.append(item)
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


class _DuplicateEnqueueRag(_FakeRag):
    async def apipeline_enqueue_documents(self, *args, **kwargs):
        self.enqueued.append({"args": args, "kwargs": kwargs})
        return None


class _ScanDocStatus:
    def __init__(self, docs_by_path):
        self.docs_by_path = docs_by_path
        self.deleted_ids: list[str] = []

    async def get_doc_by_file_path(self, file_path):
        return self.docs_by_path.get(file_path)

    async def get_doc_by_file_basename(self, basename):
        from pathlib import Path as _Path

        for stored_path, doc in self.docs_by_path.items():
            if _Path(stored_path).name == basename:
                return stored_path, doc
        return None

    async def delete(self, ids):
        for doc_id in ids:
            self.docs_by_path.pop(doc_id, None)
            self.deleted_ids.append(doc_id)


class _ScanFullDocs:
    """Minimal full_docs double for run_scanning_process.

    The scan now consults ``full_docs.get_by_id`` to distinguish a
    resumable FAILED row (content was stored, only a downstream step
    failed) from an extraction-error stub recorded by
    ``apipeline_enqueue_error_documents`` (no full_docs entry exists).
    """

    def __init__(self, docs_by_id):
        self.docs_by_id = docs_by_id

    async def get_by_id(self, doc_id):
        return self.docs_by_id.get(doc_id)


class _ScanRag:
    def __init__(self, docs_by_path, full_docs_by_id=None):
        self.doc_status = _ScanDocStatus(docs_by_path)
        # Default: every doc_status row has a corresponding full_docs entry,
        # i.e. the "resumable" FAILED case.  Tests simulating extraction-error
        # stubs pass ``full_docs_by_id={}`` (or omit specific doc_ids) so the
        # scan classifies them as retry-as-new instead of resume.  The mock
        # uses the doc_status path-as-doc_id convention from _ScanDocStatus.
        if full_docs_by_id is None:
            full_docs_by_id = {path: {"content": ""} for path in docs_by_path}
        self.full_docs = _ScanFullDocs(full_docs_by_id)
        self.process_calls = 0
        self.workspace = "scan-test"

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1


class _DuplicateUploadRag:
    def __init__(self, docs_by_path):
        self.doc_status = _ScanDocStatus(docs_by_path)
        self.workspace = f"upload-test-{uuid4().hex}"


class _DeleteRag:
    def __init__(self, result):
        self.result = result
        self.workspace = f"delete-test-{uuid4().hex}"
        self.deleted_doc_ids = []

    async def adelete_by_doc_id(self, doc_id, delete_llm_cache=False):
        self.deleted_doc_ids.append((doc_id, delete_llm_cache))
        return self.result

    async def apipeline_process_enqueue_documents(self):
        return None


class _ParseFullDocs:
    def __init__(self, source_path):
        self.source_path = source_path
        self.events = []
        self.data = {}

    async def get_by_id(self, doc_id):
        # ``_persist_parsed_full_docs`` merges with the existing pending_parse
        # record so metadata seeded at enqueue time (process_options,
        # canonical_basename, ...) survives the parse-result overwrite. These
        # tests only seed the row via the parser, so returning None is fine.
        record = self.data.get(doc_id)
        return dict(record) if record is not None else None

    async def upsert(self, data):
        self.events.append("upsert")
        self.data.update(data)

    async def index_done_callback(self):
        self.events.append("index_done")
        assert self.source_path.exists()


class _ParseDocStatus:
    """Minimal doc_status double for the parse_* archive tests.

    ``_persist_parsed_full_docs`` reads the existing record and patches its
    ``content_hash``; with no record present the helper short-circuits, which
    is what these tests want — they only assert on full_docs side effects.
    """

    async def get_by_id(self, doc_id):
        return None

    async def upsert(self, data):
        return None


class _ParseRag:
    _persist_parsed_full_docs = LightRAG._persist_parsed_full_docs
    # parse_native now delegates to the LightRAG Document writer, which the
    # tests need to exercise to validate archive + full_docs side effects.
    _write_lightrag_document_from_content_list = (
        LightRAG._write_lightrag_document_from_content_list
    )

    def __init__(self, working_dir, source_path):
        self.working_dir = str(working_dir)
        self.full_docs = _ParseFullDocs(source_path)
        self.doc_status = _ParseDocStatus()

    def _resolve_source_file_for_parser(self, file_path):
        return file_path


async def test_pipeline_index_file_leaves_lightrag_document_docx_for_parser_archive(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    file_path = tmp_path / "sample.[docling].docx"
    file_path.write_bytes(b"docx bytes")
    rag = _FakeRag()

    await pipeline_index_file(rag, file_path, "track-docx")

    assert file_path.exists()
    assert not (tmp_path / PARSED_DIR_NAME / file_path.name).exists()
    assert rag.enqueued[0]["file_path"] == str(file_path)
    assert rag.enqueued[0]["docs_format"] == FULL_DOCS_FORMAT_PENDING_PARSE
    assert rag.enqueued[0]["parsed_engine"] == "native"


async def test_pipeline_enqueue_lightrag_document_docx_does_not_move_source(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
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
    assert rag.enqueued[0]["parsed_engine"] == "native"


async def test_pipeline_enqueue_docx_plain_text_extracts_before_enqueue(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:legacy")
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
            "parsed_engine": "legacy",
            "process_options": None,
            "from_scan": False,
        }
    ]
    assert not file_path.exists()
    assert (tmp_path / PARSED_DIR_NAME / file_path.name).exists()


async def test_pipeline_enqueue_md_moves_after_enqueue(tmp_path, monkeypatch):
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
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
            "parsed_engine": "legacy",
            "process_options": None,
            "from_scan": False,
        }
    ]
    assert not file_path.exists()
    assert (tmp_path / PARSED_DIR_NAME / file_path.name).exists()


async def test_pipeline_enqueue_legacy_duplicate_archives_with_unique_name(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    file_path = tmp_path / "duplicate.md"
    file_path.write_text("duplicate content", encoding="utf-8")
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / file_path.name).write_text("existing", encoding="utf-8")
    rag = _DuplicateEnqueueRag()

    success, returned_track_id = await pipeline_enqueue_file(
        rag, file_path, "track-dup"
    )

    assert success is False
    assert returned_track_id == "track-dup"
    assert not file_path.exists()
    assert (parsed_dir / file_path.name).read_text(encoding="utf-8") == "existing"
    assert (parsed_dir / "duplicate_001.md").read_text(
        encoding="utf-8"
    ) == "duplicate content"


async def test_pipeline_enqueue_parser_routed_pdf_defers_without_extraction(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LIGHTRAG_PARSER", "pdf:mineru,*:legacy")
    monkeypatch.setenv("MINERU_ENDPOINT", "http://fake-mineru")

    def _fail_pdf_extract(*args, **kwargs):
        raise AssertionError("parser-routed PDF should not be extracted before enqueue")

    monkeypatch.setattr(_document_routes, "_extract_pdf_pypdf", _fail_pdf_extract)
    file_path = tmp_path / "paper.pdf"
    file_path.write_bytes(b"fake-pdf")
    rag = _FakeRag()

    success, returned_track_id = await pipeline_enqueue_file(
        rag, file_path, "track-pdf"
    )

    assert success is True
    assert returned_track_id == "track-pdf"
    assert file_path.exists()
    assert rag.enqueued == [
        {
            "input": "",
            "file_path": str(file_path),
            "track_id": "track-pdf",
            "docs_format": FULL_DOCS_FORMAT_PENDING_PARSE,
            "parsed_engine": "mineru",
            "process_options": None,
            "from_scan": False,
        }
    ]


async def test_pipeline_enqueue_passes_process_options_from_filename_hint(
    tmp_path, monkeypatch
):
    """Filename hint ``[native-iet]`` flows into apipeline_enqueue_documents."""
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    file_path = tmp_path / "report.[native-iet].docx"
    file_path.write_bytes(b"docx-bytes")
    rag = _FakeRag()

    success, returned_track_id = await pipeline_enqueue_file(
        rag, file_path, "track-options"
    )

    assert success is True
    assert returned_track_id == "track-options"
    assert rag.enqueued == [
        {
            "input": "",
            "file_path": str(file_path),
            "track_id": "track-options",
            "docs_format": FULL_DOCS_FORMAT_PENDING_PARSE,
            "parsed_engine": "native",
            "process_options": "iet",
            "from_scan": False,
        }
    ]
    # Native engine deferral keeps the source file in place for the parser.
    assert file_path.exists()


async def test_pipeline_enqueue_lightrag_parser_rule_provides_default_options(
    tmp_path, monkeypatch
):
    """LIGHTRAG_PARSER ``docx:native-iet`` becomes the default ``process_options``."""
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native-iet,*:legacy")
    file_path = tmp_path / "rule_default.docx"
    file_path.write_bytes(b"docx-bytes")
    rag = _FakeRag()

    success, _ = await pipeline_enqueue_file(rag, file_path, "track-rule-default")

    assert success is True
    assert len(rag.enqueued) == 1
    enqueued = rag.enqueued[0]
    assert enqueued["parsed_engine"] == "native"
    assert enqueued["process_options"] == "iet"


async def test_pipeline_index_files_leaves_lightrag_document_docx_batch(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
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
    assert all(item["parsed_engine"] == "native" for item in rag.enqueued)


async def test_scan_processed_same_name_archives_with_unique_name(
    tmp_path, monkeypatch
):
    file_path = tmp_path / "already-parsed.docx"
    file_path.write_bytes(b"docx bytes")
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / file_path.name).write_bytes(b"previous parsed file")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            str(file_path): {
                "status": DocStatus.PROCESSED.value,
                "file_path": str(file_path),
                "track_id": "track-existing",
            }
        }
    )

    async def fail_if_reenqueue(*args, **kwargs):
        raise AssertionError("existing docx should not be re-enqueued")

    monkeypatch.setattr(_document_routes, "pipeline_index_files", fail_if_reenqueue)

    await run_scanning_process(rag, doc_manager, "track-scan")

    assert not file_path.exists()
    assert (parsed_dir / file_path.name).read_bytes() == b"previous parsed file"
    assert (parsed_dir / "already-parsed_001.docx").read_bytes() == b"docx bytes"


async def test_scan_processed_canonical_name_archives_hinted_file(
    tmp_path, monkeypatch
):
    file_path = tmp_path / "already-parsed.[native].docx"
    file_path.write_bytes(b"docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            "already-parsed.docx": {
                "status": DocStatus.PROCESSED.value,
                "file_path": "already-parsed.docx",
                "track_id": "track-existing",
            }
        }
    )

    async def fail_if_reenqueue(*args, **kwargs):
        raise AssertionError("canonical duplicate should not be re-enqueued")

    monkeypatch.setattr(_document_routes, "pipeline_index_files", fail_if_reenqueue)

    await run_scanning_process(rag, doc_manager, "track-scan")

    assert not file_path.exists()
    assert (tmp_path / PARSED_DIR_NAME / file_path.name).read_bytes() == b"docx bytes"


async def test_scan_archives_same_batch_canonical_duplicates(tmp_path, monkeypatch):
    plain_file = tmp_path / "same.docx"
    hinted_file = tmp_path / "same.[native].docx"
    plain_file.write_bytes(b"plain docx bytes")
    hinted_file.write_bytes(b"hinted docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag({})
    calls = []

    async def capture_pipeline(rag_arg, file_paths, track_id, **kwargs):
        calls.append(
            {
                "rag": rag_arg,
                "file_paths": file_paths,
                "track_id": track_id,
                "kwargs": kwargs,
            }
        )

    monkeypatch.setattr(_document_routes, "pipeline_index_files", capture_pipeline)

    await run_scanning_process(rag, doc_manager, "track-scan")

    # Hinted variant is preferred so the user's explicit engine choice wins;
    # the plain variant is the one that gets archived.
    assert len(calls) == 1
    assert calls[0]["file_paths"] == [hinted_file]
    # The scan-owned background task forwards from_scan=True so per-file
    # enqueues bypass the scanning busy guard the scan itself holds.
    assert calls[0]["kwargs"] == {"from_scan": True}
    archived_names = {
        path.name for path in (tmp_path / PARSED_DIR_NAME).iterdir() if path.is_file()
    }
    assert archived_names == {"same.docx"}
    assert hinted_file.exists()
    assert not plain_file.exists()


async def test_scan_existing_non_processed_reprocesses_file(tmp_path, monkeypatch):
    file_path = tmp_path / "retry.docx"
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
    calls = []

    async def capture_pipeline(rag_arg, file_paths, track_id, **kwargs):
        calls.append(
            {
                "rag": rag_arg,
                "file_paths": file_paths,
                "track_id": track_id,
                "kwargs": kwargs,
            }
        )

    monkeypatch.setattr(_document_routes, "pipeline_index_files", capture_pipeline)

    await run_scanning_process(rag, doc_manager, "track-scan")

    # Resume targets bypass pipeline_index_files entirely: routing them
    # through apipeline_enqueue_documents would treat the same canonical
    # basename as a duplicate (returning None), causing the source to be
    # archived as if it were a duplicate while leaving the unfinished
    # doc_status row untouched.  Instead, the scan kicks off
    # apipeline_process_enqueue_documents directly so the existing PARSING
    # row is picked up by the pipeline's resume logic, and the source file
    # stays in place for any pending-parse engine that still needs it.
    assert calls == []
    assert rag.process_calls == 1
    assert file_path.exists()


async def test_scan_mixed_new_and_resume_routes_only_new_through_enqueue(
    tmp_path, monkeypatch
):
    """When a scan finds both a new file and one matching an unfinished
    doc_status row, only the new file goes through pipeline_index_files;
    the resume target stays in place.  run_scanning_process always
    triggers apipeline_process_enqueue_documents whenever resume targets
    exist — even when new files were also enqueued — because
    pipeline_index_files only runs that call after at least one new file
    successfully enqueues.  Without the unconditional trigger, an all-
    failed batch of new files would silently strand the resume rows.
    """
    new_file = tmp_path / "fresh.docx"
    resume_file = tmp_path / "retry.docx"
    new_file.write_bytes(b"fresh docx bytes")
    resume_file.write_bytes(b"retry docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            str(resume_file): {
                "status": DocStatus.FAILED.value,
                "file_path": str(resume_file),
                "track_id": "track-existing",
            }
        }
    )
    calls = []

    async def capture_pipeline(rag_arg, file_paths, track_id, **kwargs):
        calls.append(
            {
                "rag": rag_arg,
                "file_paths": file_paths,
                "track_id": track_id,
                "kwargs": kwargs,
            }
        )

    monkeypatch.setattr(_document_routes, "pipeline_index_files", capture_pipeline)

    await run_scanning_process(rag, doc_manager, "track-scan")

    # Only the new file goes through the enqueue path; the resume file
    # stays in input/ for any pending-parse engine that still needs the
    # source on disk.
    assert len(calls) == 1
    assert calls[0]["file_paths"] == [new_file]
    assert calls[0]["kwargs"] == {"from_scan": True}
    # The unconditional trigger fires once — guaranteeing the resume row
    # advances even if pipeline_index_files's internal trigger were to be
    # skipped (e.g. if every new file was rejected by enqueue).
    assert rag.process_calls == 1
    assert resume_file.exists()
    assert new_file.exists()


async def test_scan_failed_extraction_record_without_full_docs_is_retried(
    tmp_path, monkeypatch
):
    """Stub doc_status rows recorded by apipeline_enqueue_error_documents
    have no full_docs entry — _validate_and_fix_document_consistency
    preserves them for manual review and excludes them from processing,
    so the resume path can never advance them.  When the user fixes the
    file and re-scans we must drop the stale stub and route the file
    through the normal new-file enqueue, otherwise the fix never lands.
    """
    file_path = tmp_path / "fixed.docx"
    file_path.write_bytes(b"now-readable bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            str(file_path): {
                "status": DocStatus.FAILED.value,
                "file_path": str(file_path),
                "track_id": "track-old",
                "metadata": {"error_type": "file_extraction_error"},
            }
        },
        full_docs_by_id={},  # Extraction error: no full_docs entry was ever written.
    )
    calls = []

    async def capture_pipeline(rag_arg, file_paths, track_id, **kwargs):
        calls.append(
            {
                "rag": rag_arg,
                "file_paths": file_paths,
                "track_id": track_id,
                "kwargs": kwargs,
            }
        )

    monkeypatch.setattr(_document_routes, "pipeline_index_files", capture_pipeline)

    await run_scanning_process(rag, doc_manager, "track-scan")

    # The stale FAILED stub is deleted and the file is routed as new so
    # the standard enqueue path can re-extract content.  No resume
    # trigger fires because there are no resume targets.
    assert rag.doc_status.deleted_ids == [str(file_path)]
    assert len(calls) == 1
    assert calls[0]["file_paths"] == [file_path]
    assert calls[0]["kwargs"] == {"from_scan": True}
    assert rag.process_calls == 0
    assert file_path.exists()


async def test_scan_failed_with_full_docs_resumes_normally(tmp_path, monkeypatch):
    """A FAILED row that DOES have a full_docs entry came from a downstream
    failure after content was successfully stored; the pipeline's resume
    logic resets it to PENDING and replays.  The scan must not delete it.
    """
    file_path = tmp_path / "downstream-failed.docx"
    file_path.write_bytes(b"docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            str(file_path): {
                "status": DocStatus.FAILED.value,
                "file_path": str(file_path),
                "track_id": "track-old",
            }
        }
        # full_docs default-seeded for the path → resumable FAILED.
    )
    calls = []

    async def capture_pipeline(rag_arg, file_paths, track_id, **kwargs):
        calls.append(file_paths)

    monkeypatch.setattr(_document_routes, "pipeline_index_files", capture_pipeline)

    await run_scanning_process(rag, doc_manager, "track-scan")

    # No stub deletion; resume path runs.
    assert rag.doc_status.deleted_ids == []
    assert calls == []
    assert rag.process_calls == 1
    assert file_path.exists()


async def test_scan_resume_runs_when_all_new_files_fail_to_enqueue(
    tmp_path, monkeypatch
):
    """The exact P2 scenario: a scan batch contains a resume target plus
    new files that all fail / are rejected during enqueue.
    pipeline_index_files's internal process_enqueue is gated on at least
    one successful enqueue; without the unconditional resume trigger in
    run_scanning_process the PARSING/FAILED row would stay stuck.

    We simulate "all new files rejected" with a pipeline_index_files mock
    that records its invocation but does not call process_enqueue (mirroring
    the real helper's behaviour when every per-file enqueue returns False).
    """
    new_file = tmp_path / "fresh.docx"
    resume_file = tmp_path / "retry.docx"
    new_file.write_bytes(b"fresh docx bytes")
    resume_file.write_bytes(b"retry docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            str(resume_file): {
                "status": DocStatus.PARSING.value,
                "file_path": str(resume_file),
                "track_id": "track-existing",
            }
        }
    )

    async def index_files_all_rejected(rag_arg, file_paths, track_id, **kwargs):
        # Real pipeline_index_files skips its internal
        # apipeline_process_enqueue_documents call when no per-file enqueue
        # succeeded; the mock omits the call entirely to mirror that.
        return None

    monkeypatch.setattr(
        _document_routes, "pipeline_index_files", index_files_all_rejected
    )

    await run_scanning_process(rag, doc_manager, "track-scan")

    # Even though pipeline_index_files's internal trigger never fired, the
    # scan still kicks process_enqueue once so the resume row advances.
    assert rag.process_calls == 1
    assert resume_file.exists()
    assert new_file.exists()


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

    # Strict name pre-check: same-canonical record in doc_status now raises 409
    # rather than returning a "duplicated" 200 response.  Clients must delete
    # the existing record before re-uploading.
    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await upload_endpoint(_document_routes.BackgroundTasks(), upload_file)
    assert excinfo.value.status_code == 409
    assert "failed.docx" in excinfo.value.detail
    assert "Status: failed" in excinfo.value.detail
    assert not (tmp_path / "failed.docx").exists()


async def test_upload_rejects_parser_hinted_filesystem_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    (tmp_path / "existing.docx").write_bytes(b"existing docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _DuplicateUploadRag({})
    router = create_document_routes(rag, doc_manager)
    upload_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "upload_to_input_dir"
    ][-1]
    upload_file = _document_routes.UploadFile(
        filename="existing.[native].docx",
        file=BytesIO(b"replacement docx bytes"),
    )

    # Strict name pre-check: an INPUT directory file with the same canonical
    # basename now blocks the upload with 409.
    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await upload_endpoint(_document_routes.BackgroundTasks(), upload_file)
    assert excinfo.value.status_code == 409
    assert "existing.docx" in excinfo.value.detail
    assert not (tmp_path / "existing.[native].docx").exists()


async def test_upload_returns_409_when_pipeline_busy(tmp_path, monkeypatch):
    """Upload must refuse with 409 while ``pipeline_status["busy"]`` is set,
    independent of any name dedup.  The strict name pre-check happens AFTER
    the busy guard, so the 409 detail is about the pipeline, not the file.
    """
    import importlib

    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _DuplicateUploadRag({})

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = True

    router = create_document_routes(rag, doc_manager)
    upload_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "upload_to_input_dir"
    ][-1]
    upload_file = _document_routes.UploadFile(
        filename="while_busy.docx",
        file=BytesIO(b"docx bytes"),
    )

    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await upload_endpoint(_document_routes.BackgroundTasks(), upload_file)
    assert excinfo.value.status_code == 409
    assert "busy" in excinfo.value.detail.lower()
    assert not (tmp_path / "while_busy.docx").exists()


async def test_upload_returns_409_when_scanning(tmp_path, monkeypatch):
    """Upload must refuse with 409 when a scan is in progress."""
    import importlib

    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _DuplicateUploadRag({})

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["scanning"] = True

    router = create_document_routes(rag, doc_manager)
    upload_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "upload_to_input_dir"
    ][-1]
    upload_file = _document_routes.UploadFile(
        filename="while_scanning.docx",
        file=BytesIO(b"docx bytes"),
    )

    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await upload_endpoint(_document_routes.BackgroundTasks(), upload_file)
    assert excinfo.value.status_code == 409
    assert "scan" in excinfo.value.detail.lower()
    assert not (tmp_path / "while_scanning.docx").exists()


async def test_scan_endpoint_returns_skipped_when_pipeline_busy(tmp_path):
    """Scan endpoint must return ``scanning_skipped_pipeline_busy`` and NOT
    schedule a background task while the pipeline is busy."""
    import importlib

    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag({})

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = True

    router = create_document_routes(rag, doc_manager)
    scan_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "scan_for_new_documents"
    ][-1]

    bg = _document_routes.BackgroundTasks()
    response = await scan_endpoint(bg)

    assert response.status == "scanning_skipped_pipeline_busy"
    # No background task should have been scheduled.
    assert len(bg.tasks) == 0
    # And ``scanning`` is left unchanged at False (we didn't acquire it).
    assert pipeline_status.get("scanning") is False


async def test_scan_endpoint_returns_skipped_when_already_scanning(tmp_path):
    """Scan endpoint must reject overlapping scans by checking the
    ``scanning`` flag, not just ``busy``."""
    import importlib

    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag({})

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["scanning"] = True

    router = create_document_routes(rag, doc_manager)
    scan_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "scan_for_new_documents"
    ][-1]

    bg = _document_routes.BackgroundTasks()
    response = await scan_endpoint(bg)

    assert response.status == "scanning_skipped_pipeline_busy"
    assert len(bg.tasks) == 0


async def test_scan_endpoint_acquires_and_releases_scanning_flag(tmp_path, monkeypatch):
    """The scan endpoint must atomically set ``scanning=True`` and
    ``run_scanning_process`` must clear it in finally — even when the body
    raises — so successive scans aren't permanently blocked.
    """
    import importlib

    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag({})

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False

    router = create_document_routes(rag, doc_manager)
    scan_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "scan_for_new_documents"
    ][-1]

    bg = _document_routes.BackgroundTasks()
    response = await scan_endpoint(bg)

    # Endpoint scheduled the task and acquired the flag synchronously.
    assert response.status == "scanning_started"
    assert pipeline_status["scanning"] is True
    assert len(bg.tasks) == 1

    # Run the scheduled task; finally-block must clear the flag.
    task = bg.tasks[0]
    await task.func(*task.args, **task.kwargs)
    assert pipeline_status["scanning"] is False


async def test_scan_endpoint_returns_skipped_when_enqueue_pending(tmp_path):
    """The preflight-to-background race: an upload/insert endpoint may
    have passed the idle check, reserved a pending-enqueue slot, and
    returned success — but its bg task has not yet called
    apipeline_enqueue_documents.  A scan that arrives in this window must
    refuse, otherwise it would set scanning=True and the bg enqueue would
    fail the busy guard after the client already saw success.
    """
    import importlib

    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag({})

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    # The "scan-test" workspace is shared across tests; reset all guarded
    # flags so we start from a clean idle state.
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0
    # Simulate a reservation made by /upload that has not yet released.
    pipeline_status["pending_enqueues"] = 1

    router = create_document_routes(rag, doc_manager)
    scan_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "scan_for_new_documents"
    ][-1]

    bg = _document_routes.BackgroundTasks()
    response = await scan_endpoint(bg)

    assert response.status == "scanning_skipped_pipeline_busy"
    # No background task scheduled; scanning flag untouched.
    assert len(bg.tasks) == 0
    assert pipeline_status.get("scanning") is False
    # Reservation count is preserved — only the owning bg task may release it.
    assert pipeline_status["pending_enqueues"] == 1


async def test_reserve_enqueue_slot_blocks_concurrent_scan_until_release(tmp_path):
    """End-to-end on the reservation primitive: reserving a slot makes
    the scan endpoint refuse; releasing it lets the next scan in.  This
    is the contract the upload/text endpoints rely on to close the
    preflight-to-background race.
    """
    import importlib

    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag({})

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0

    # Reserve a slot — mirrors what /upload, /text and /texts do
    # synchronously before scheduling their bg tasks.
    reserved = await _document_routes._reserve_enqueue_slot(rag)
    assert reserved is True
    assert pipeline_status["pending_enqueues"] == 1

    router = create_document_routes(rag, doc_manager)
    scan_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "scan_for_new_documents"
    ][-1]

    bg = _document_routes.BackgroundTasks()
    blocked = await scan_endpoint(bg)
    assert blocked.status == "scanning_skipped_pipeline_busy"

    # Release: bg task wrapper would do this in finally.
    await _document_routes._release_enqueue_slot(rag)
    assert pipeline_status["pending_enqueues"] == 0

    bg2 = _document_routes.BackgroundTasks()
    allowed = await scan_endpoint(bg2)
    assert allowed.status == "scanning_started"
    assert pipeline_status["scanning"] is True


async def test_release_enqueue_slot_returns_was_last_only_for_final(tmp_path):
    """Two-reservation cohort: the first release returns False (sibling
    still pending), the second returns True (last out → drain owner).
    Closes the dual-bg-task race where both bg tasks could otherwise
    independently trigger process_enqueue and the second's enqueue
    would hit the busy guard set by the first's processing.
    """
    import importlib

    rag = _ScanRag({})
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0

    # Two concurrent reservations from /upload + /text — both pass the
    # idle preflight because busy=F, scanning=F at reservation time.
    assert await _document_routes._reserve_enqueue_slot(rag) is True
    assert await _document_routes._reserve_enqueue_slot(rag) is True
    assert pipeline_status["pending_enqueues"] == 2

    # First release: a sibling reservation is still in flight → no drain.
    assert await _document_routes._release_enqueue_slot(rag) is False
    assert pipeline_status["pending_enqueues"] == 1

    # Second (final) release: caller owns the drain.
    assert await _document_routes._release_enqueue_slot(rag) is True
    assert pipeline_status["pending_enqueues"] == 0


async def test_release_and_drain_only_runs_process_on_last_release(tmp_path):
    """End-to-end on the drain wrapper: two siblings each release once;
    only the last release calls apipeline_process_enqueue_documents,
    so the busy phase never starts while a sibling's enqueue is still
    pending.
    """
    import importlib

    rag = _ScanRag({})
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0

    await _document_routes._reserve_enqueue_slot(rag)
    await _document_routes._reserve_enqueue_slot(rag)
    assert rag.process_calls == 0

    # Sibling A finishes first → no drain.
    await _document_routes._release_enqueue_slot_and_drain_if_last(rag)
    assert rag.process_calls == 0

    # Sibling B finishes → final release triggers drain exactly once.
    await _document_routes._release_enqueue_slot_and_drain_if_last(rag)
    assert rag.process_calls == 1


async def test_release_and_drain_solo_release_drains_immediately(tmp_path):
    """Single reservation (no concurrency): the lone release is also the
    final one and must trigger drain so the enqueued doc actually gets
    processed.
    """
    import importlib

    rag = _ScanRag({})
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0

    await _document_routes._reserve_enqueue_slot(rag)
    await _document_routes._release_enqueue_slot_and_drain_if_last(rag)
    assert rag.process_calls == 1
    assert pipeline_status["pending_enqueues"] == 0


async def test_reserve_enqueue_slot_raises_409_when_busy_or_scanning(tmp_path):
    """The reservation helper must surface 409 (not silently increment)
    when the pipeline is already busy or scanning, so endpoints fail
    fast and the caller sees a normal error response.
    """
    import importlib

    rag = _ScanRag({})
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0

    pipeline_status["busy"] = True
    with pytest.raises(_document_routes.HTTPException) as exc:
        await _document_routes._reserve_enqueue_slot(rag)
    assert exc.value.status_code == 409
    # No leak when guard rejects.
    assert pipeline_status["pending_enqueues"] == 0
    pipeline_status["busy"] = False

    pipeline_status["scanning"] = True
    with pytest.raises(_document_routes.HTTPException) as exc:
        await _document_routes._reserve_enqueue_slot(rag)
    assert exc.value.status_code == 409
    assert pipeline_status["pending_enqueues"] == 0


def test_delete_file_variants_removes_canonical_hint_variants(tmp_path):
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    files_to_delete = [
        tmp_path / "report.docx",
        tmp_path / "report.[native].docx",
        parsed_dir / "report.[mineru].docx",
        parsed_dir / "report.[native]_001.docx",
        parsed_dir / "report_001.docx",
    ]
    for path in files_to_delete:
        path.write_bytes(b"delete me")
    unrelated_files = [
        tmp_path / "report_001.docx",
        tmp_path / "report_2024.docx",
        parsed_dir / "other.[native].docx",
    ]
    for path in unrelated_files:
        path.write_bytes(b"keep me")
    artifact_dirs = [
        parsed_dir / "report.docx.parsed",
        parsed_dir / "report.docx.parsed_001",
    ]
    for artifact_dir in artifact_dirs:
        artifact_dir.mkdir()
        (artifact_dir / "report.blocks.jsonl").write_text("{}", encoding="utf-8")
    unrelated_artifact_dir = parsed_dir / "other.docx.parsed"
    unrelated_artifact_dir.mkdir()
    (unrelated_artifact_dir / "other.blocks.jsonl").write_text("{}", encoding="utf-8")

    deleted_files, errors = _document_routes.delete_file_variants_by_canonical_basename(
        tmp_path,
        "report.docx",
        str(tmp_path / "report.[native].docx"),
    )

    assert errors == []
    assert set(deleted_files) == {
        "report.docx",
        "report.[native].docx",
        str(Path(PARSED_DIR_NAME) / "report.[mineru].docx"),
        str(Path(PARSED_DIR_NAME) / "report.[native]_001.docx"),
        str(Path(PARSED_DIR_NAME) / "report_001.docx"),
        str(Path(PARSED_DIR_NAME) / "report.docx.parsed"),
        str(Path(PARSED_DIR_NAME) / "report.docx.parsed_001"),
    }
    assert all(not path.exists() for path in files_to_delete)
    assert all(path.exists() for path in unrelated_files)
    assert all(not artifact_dir.exists() for artifact_dir in artifact_dirs)
    assert unrelated_artifact_dir.is_dir()


async def test_background_delete_removes_parser_hint_file_variants(tmp_path):
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    source_file = tmp_path / "paper.[native].docx"
    source_file.write_bytes(b"source")
    parsed_variant = parsed_dir / "paper.[mineru]_001.docx"
    parsed_variant.write_bytes(b"parsed")
    unrelated_file = tmp_path / "other.[native].docx"
    unrelated_file.write_bytes(b"keep")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _DeleteRag(
        DeletionResult(
            status="success",
            doc_id="doc-paper",
            message="deleted",
            file_path="paper.docx",
            source_path=str(source_file),
        )
    )
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)

    await _document_routes.background_delete_documents(
        rag,
        doc_manager,
        ["doc-paper"],
        delete_file=True,
        delete_llm_cache=True,
    )

    assert rag.deleted_doc_ids == [("doc-paper", True)]
    assert not source_file.exists()
    assert not parsed_variant.exists()
    assert unrelated_file.exists()


async def test_docx_archive_failure_is_best_effort(tmp_path, monkeypatch):
    file_path = tmp_path / "archive-failure.docx"
    file_path.write_bytes(b"docx bytes")

    async def _raise_archive_failure(*args, **kwargs):
        raise OSError("simulated archive failure")

    from lightrag.utils_pipeline import (
        archive_docx_source_after_full_docs_sync,
    )
    import lightrag.utils_pipeline as _utils_pipeline

    monkeypatch.setattr(
        _utils_pipeline, "move_file_to_parsed_dir", _raise_archive_failure
    )

    archived_path = await archive_docx_source_after_full_docs_sync(str(file_path))

    assert archived_path is None
    assert file_path.exists()


async def test_parse_native_archives_docx_after_full_docs_sync(tmp_path, monkeypatch):
    source_path = tmp_path / "parsed-after-sync.docx"
    source_path.write_bytes(b"docx bytes")
    rag = _ParseRag(tmp_path / "work", source_path)

    parse_document = importlib.import_module("lightrag.extraction.parse_document")

    def _fake_parse_docx(file_bytes, source_file, doc_id):
        # New entry returns (content_list, asset_blobs); a single text item
        # is enough to exercise the archive + full_docs side-effects tested
        # below — the LightRAG Document writer will turn it into one block.
        return [{"type": "text", "text": "parsed"}], {}

    monkeypatch.setattr(
        parse_document, "parse_docx_to_lightrag_content_list", _fake_parse_docx
    )

    result = await LightRAG.parse_native(
        rag,
        "doc-test",
        str(source_path),
        {"format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
    )

    # parse_native now returns LIGHTRAG-format parsed_data with merged_text
    # (not the {{LRdoc}} marker — that's only in the persisted full_docs row).
    assert result["content"]
    assert result["format"] == "lightrag"
    assert result["blocks_path"]
    assert rag.full_docs.events == ["upsert", "index_done"]
    assert not source_path.exists()
    assert (tmp_path / PARSED_DIR_NAME / source_path.name).exists()
    parsed_artifact_dir = tmp_path / PARSED_DIR_NAME / f"{source_path.name}.parsed"
    assert parsed_artifact_dir.is_dir()
    assert (parsed_artifact_dir / "parsed-after-sync.blocks.jsonl").is_file()
    assert rag.full_docs.data["doc-test"]["parsed_engine"] == "native"
    assert rag.full_docs.data["doc-test"]["format"] == "lightrag"
    # Per docs/FileProcessingConfiguration-zh.md, content uses the {{LRdoc}}
    # marker plus a leading-text summary derived from merged blocks.
    assert rag.full_docs.data["doc-test"]["content"].startswith("{{LRdoc}}")


def test_parsed_artifact_dir_uses_unique_suffix_when_path_is_file(tmp_path):
    from lightrag.utils_pipeline import parsed_artifact_dir_for_source

    source_path = tmp_path / "demo.docx"
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "demo.docx.parsed").write_text("legacy file", encoding="utf-8")

    artifact_dir = parsed_artifact_dir_for_source(str(source_path))

    assert artifact_dir == parsed_dir / "demo.docx.parsed_001"


def test_parsed_artifact_dir_reuses_existing_parsed_parent(tmp_path):
    from lightrag.utils_pipeline import parsed_artifact_dir_for_source

    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    source_path = parsed_dir / "demo.docx"
    source_path.write_bytes(b"docx bytes")

    artifact_dir = parsed_artifact_dir_for_source(str(source_path))

    assert artifact_dir == parsed_dir / "demo.docx.parsed"


async def test_parse_native_docx_interchange_failure_raises_without_fallback(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "interchange-failure.docx"
    source_path.write_bytes(b"docx bytes")
    rag = _ParseRag(tmp_path / "work", source_path)
    parse_document = importlib.import_module("lightrag.extraction.parse_document")

    def _raise_parser(file_bytes, source_file, doc_id):
        raise RuntimeError("interchange boom")

    def _fail_fallback(file_bytes):
        raise AssertionError("plain text fallback should not run")

    monkeypatch.setattr(
        parse_document, "parse_docx_to_lightrag_content_list", _raise_parser
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
        parse_document,
        "parse_docx_to_lightrag_content_list",
        lambda *args, **kwargs: ([], {}),
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
    engine = resolve_stored_document_parser_engine(
        "report.[mineru].docx",
        {
            "format": FULL_DOCS_FORMAT_LIGHTRAG,
            "lightrag_document_path": "report.blocks.jsonl",
            "parsed_engine": "mineru",
        },
    )

    assert engine == "native"
