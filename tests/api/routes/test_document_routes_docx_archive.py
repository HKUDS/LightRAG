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
_parser_routing = importlib.import_module("lightrag.parser.routing")
_parser_registry = importlib.import_module("lightrag.parser.registry")
_parser_base = importlib.import_module("lightrag.parser.base")
sys.argv = _original_argv

DocStatus = _base.DocStatus
DeletionResult = _base.DeletionResult
FULL_DOCS_FORMAT_LIGHTRAG = _constants.FULL_DOCS_FORMAT_LIGHTRAG
FULL_DOCS_FORMAT_PENDING_PARSE = _constants.FULL_DOCS_FORMAT_PENDING_PARSE
PARSED_DIR_NAME = _constants.PARSED_DIR_NAME
PROCESS_OPTION_CHUNK_FIXED = _constants.PROCESS_OPTION_CHUNK_FIXED
compute_mdhash_id = _utils.compute_mdhash_id
LightRAG = _lightrag.LightRAG
resolve_stored_document_parser_engine = (
    _parser_routing.resolve_stored_document_parser_engine
)
get_parser = _parser_registry.get_parser
ParseContext = _parser_base.ParseContext


async def _parse_via_registry(rag, engine, doc_id, file_path, content_data):
    """Drive a parser the way the pipeline worker does (registry dispatch)."""
    result = await get_parser(engine).parse(
        ParseContext(rag, doc_id, file_path, content_data)
    )
    return result.to_dict()


pipeline_index_file = _document_routes.pipeline_index_file
pipeline_index_files = _document_routes.pipeline_index_files
pipeline_index_texts = _document_routes.pipeline_index_texts
pipeline_enqueue_file = _document_routes.pipeline_enqueue_file
run_scanning_process = _document_routes.run_scanning_process
DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    """Initialize the shared_storage module-level dicts before each test.

    The scan endpoint and the enqueue/scanning guards read
    ``pipeline_status`` via ``get_namespace_data``, which raises if
    shared dicts have never been initialized.  Tests using mocked
    ``LightRAG`` instances don't run ``initialize_storages``, so we set
    up the shared store here and reset pipeline_status state per-test
    to avoid leakage.
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
        # _resolve_text_chunking reads addon_params; {} -> default chunker config.
        self.addon_params = {}

    async def apipeline_enqueue_documents(
        self,
        input,
        ids=None,
        file_paths=None,
        track_id=None,
        docs_format=None,
        parse_engine=None,
        process_options=None,
        chunk_options=None,
        from_scan=False,
    ):
        item = {
            "input": input,
            "file_path": file_paths,
            "track_id": track_id,
            "docs_format": docs_format,
            "parse_engine": parse_engine,
            "process_options": process_options,
            "chunk_options": chunk_options,
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
        self.enqueued = []
        self.errors = []

    async def apipeline_enqueue_documents(
        self,
        input,
        ids=None,
        file_paths=None,
        track_id=None,
        docs_format=None,
        parse_engine=None,
        process_options=None,
        from_scan=False,
    ):
        item = {
            "input": input,
            "file_path": file_paths,
            "track_id": track_id,
            "docs_format": docs_format,
            "parse_engine": parse_engine,
            "process_options": process_options,
            "from_scan": from_scan,
        }
        self.enqueued.append(item)
        return track_id

    async def apipeline_enqueue_error_documents(self, error_files, track_id=None):
        self.errors.append((error_files, track_id))

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


class _ParseTokenizer(_utils.TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


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
        self.tokenizer = _utils.Tokenizer(
            model_name="char", tokenizer=_ParseTokenizer()
        )

    def _resolve_source_file_for_parser(self, file_path):
        return file_path


async def test_pipeline_index_file_leaves_lightrag_document_docx_for_parser_archive(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    file_path = tmp_path / "sample.docx"
    file_path.write_bytes(b"docx bytes")
    rag = _FakeRag()

    await pipeline_index_file(rag, file_path, "track-docx")

    assert file_path.exists()
    assert not (tmp_path / PARSED_DIR_NAME / file_path.name).exists()
    assert rag.enqueued[0]["file_path"] == str(file_path)
    assert rag.enqueued[0]["docs_format"] == FULL_DOCS_FORMAT_PENDING_PARSE
    assert rag.enqueued[0]["parse_engine"] == "native"


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
    assert rag.enqueued[0]["parse_engine"] == "native"


async def test_pipeline_enqueue_docx_defers_to_legacy_parser(tmp_path, monkeypatch):
    # Legacy now defers extraction to the worker stage; enqueue just records a
    # PENDING_PARSE row with parse_engine=legacy (no eager extraction here).
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:legacy")
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
            "input": "",
            "file_path": str(file_path),
            "track_id": "track-docx",
            "docs_format": FULL_DOCS_FORMAT_PENDING_PARSE,
            "parse_engine": "legacy",
            "process_options": PROCESS_OPTION_CHUNK_FIXED,
            "chunk_options": None,
            "from_scan": False,
        }
    ]
    # Deferred: the source stays in place until the worker archives it.
    assert file_path.exists()


async def test_pipeline_enqueue_md_defers_to_legacy_parser(tmp_path, monkeypatch):
    # Unhinted .md defaults to the legacy engine and now defers extraction to
    # the worker stage (PENDING_PARSE), like every other engine.
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Notes\n\nmarkdown content", encoding="utf-8")
    rag = _FakeRag()

    success, returned_track_id = await pipeline_enqueue_file(rag, file_path, "track-md")

    assert success is True
    assert returned_track_id == "track-md"
    assert rag.enqueued == [
        {
            "input": "",
            "file_path": str(file_path),
            "track_id": "track-md",
            "docs_format": FULL_DOCS_FORMAT_PENDING_PARSE,
            "parse_engine": "legacy",
            "process_options": PROCESS_OPTION_CHUNK_FIXED,
            "chunk_options": None,
            "from_scan": False,
        }
    ]
    # Deferred: the source stays in place until the worker archives it.
    assert file_path.exists()


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
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")

    # Extraction is always deferred now (no enqueue-stage extraction for any
    # engine), so the pdf simply enqueues as PENDING_PARSE for mineru.
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
            "parse_engine": "mineru",
            "process_options": PROCESS_OPTION_CHUNK_FIXED,
            "chunk_options": None,
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
            "parse_engine": "native",
            "process_options": "iet",
            "chunk_options": None,
            "from_scan": False,
        }
    ]
    # Native engine deferral keeps the source file in place for the parser.
    assert file_path.exists()


async def test_pipeline_enqueue_rejects_invalid_filename_hint(tmp_path, monkeypatch):
    """Bad filename processing hints must become file-processing errors."""
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    file_path = tmp_path / "report.[abc].docx"
    file_path.write_bytes(b"docx-bytes")
    rag = _FakeRag()

    success, returned_track_id = await pipeline_enqueue_file(
        rag, file_path, "track-bad-hint"
    )

    assert success is False
    assert returned_track_id == "track-bad-hint"
    assert rag.enqueued == []
    assert len(rag.errors) == 1
    error_files, track_id = rag.errors[0]
    assert track_id == "track-bad-hint"
    assert error_files[0]["file_path"] == file_path.name
    assert error_files[0]["error_description"] == (
        "[File Extraction]Filename hint error"
    )
    assert "unsupported parser engine 'abc'" in error_files[0]["original_error"]
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
    assert enqueued["parse_engine"] == "native"
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
    assert all(item["parse_engine"] == "native" for item in rag.enqueued)
    assert all(
        item["process_options"] == PROCESS_OPTION_CHUNK_FIXED for item in rag.enqueued
    )


async def test_pipeline_index_texts_sets_api_default_process_options():
    rag = _FakeRag()

    await pipeline_index_texts(
        rag,
        ["first text", "second text"],
        file_sources=["first.txt", "second.txt"],
        track_id="track-texts",
    )

    assert len(rag.enqueued) == 1
    item = rag.enqueued[0]
    assert item["input"] == ["first text", "second text"]
    assert item["file_path"] == ["first.txt", "second.txt"]
    assert item["track_id"] == "track-texts"
    assert item["docs_format"] is None
    assert item["parse_engine"] is None
    assert item["process_options"] == PROCESS_OPTION_CHUNK_FIXED
    assert item["from_scan"] is False
    # No chunking config -> default F snapshot is still passed through.
    assert isinstance(item["chunk_options"], dict)
    assert "fixed_token" in item["chunk_options"]


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
    # enqueues bypass the scanning guard whose ``scanning`` flag the
    # scan task itself holds.
    assert calls[0]["kwargs"] == {"from_scan": True}
    archived_names = {
        path.name for path in (tmp_path / PARSED_DIR_NAME).iterdir() if path.is_file()
    }
    assert archived_names == {"same.docx"}
    assert hinted_file.exists()
    assert not plain_file.exists()


async def test_scan_rejects_invalid_filename_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    file_path = tmp_path / "bad-scan.[native-FR].docx"
    file_path.write_bytes(b"docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag({})

    await run_scanning_process(rag, doc_manager, "track-scan")

    assert rag.enqueued == []
    assert len(rag.errors) == 1
    error_files, track_id = rag.errors[0]
    assert track_id == "track-scan"
    assert error_files[0]["file_path"] == file_path.name
    assert error_files[0]["error_description"] == (
        "[File Extraction]Filename hint error"
    )
    assert "multiple chunking modes" in error_files[0]["original_error"]
    assert rag.process_calls == 0
    assert file_path.exists()


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


async def test_upload_rejects_malformed_hint_with_detail(tmp_path, monkeypatch):
    """A malformed filename hint fails the upload synchronously with the
    detailed hint error in the 400 body (it used to be accepted and only
    surface later as an error document)."""
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _DuplicateUploadRag({})
    router = create_document_routes(rag, doc_manager)
    upload_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "upload_to_input_dir"
    ][-1]
    upload_file = _document_routes.UploadFile(
        # F and R are two chunking modes -> invalid hint combination.
        filename="bad.[native-FR].docx",
        file=BytesIO(b"docx bytes"),
    )

    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await upload_endpoint(_document_routes.BackgroundTasks(), upload_file)
    assert excinfo.value.status_code == 400
    assert "multiple chunking modes" in excinfo.value.detail


async def test_upload_succeeds_concurrent_with_pipeline_busy(tmp_path, monkeypatch):
    """Under the new contract, ``busy=True`` no longer blocks uploads.
    The upload reserves a pending-enqueue slot, schedules its bg task,
    and returns success; the bg task's enqueue is permitted while the
    pipeline is busy and the running loop's request_pending mechanism
    will pick up the new doc after its current batch.
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
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0
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

    bg = _document_routes.BackgroundTasks()
    response = await upload_endpoint(bg, upload_file)

    # Endpoint accepted the upload despite busy=True.
    assert response.status == "success"
    assert (tmp_path / "while_busy.docx").exists()
    # The slot has been transferred to the bg task; it will release on
    # completion.  Until then pending_enqueues stays at 1 so a
    # concurrent /scan would refuse.
    assert pipeline_status["pending_enqueues"] == 1
    assert len(bg.tasks) == 1


async def test_upload_returns_409_when_scanning_classification(tmp_path, monkeypatch):
    """Upload must refuse with 409 when scan is in its CLASSIFICATION
    phase (``scanning_exclusive=True``).  Scan's processing phase
    (``scanning=True`` but ``scanning_exclusive=False``) is permissive
    — see ``test_upload_succeeds_during_scan_processing_phase`` below.
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
    pipeline_status["scanning"] = True
    pipeline_status["scanning_exclusive"] = True

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
    assert "classifying" in excinfo.value.detail.lower()
    assert not (tmp_path / "while_scanning.docx").exists()


async def test_upload_succeeds_during_scan_processing_phase(tmp_path, monkeypatch):
    """User-reported scenario: while pipeline is doing scan-driven
    processing (``scanning=True`` but ``scanning_exclusive=False``),
    new uploads must be accepted.  Scan's processing phase is
    behaviourally identical to busy=True — uploads coexist via
    request_pending.
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
    # Classification done; scan is now driving the processing pipeline.
    pipeline_status["scanning"] = True
    pipeline_status["scanning_exclusive"] = False
    pipeline_status["busy"] = True
    pipeline_status["pending_enqueues"] = 0

    router = create_document_routes(rag, doc_manager)
    upload_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "upload_to_input_dir"
    ][-1]
    upload_file = _document_routes.UploadFile(
        filename="upload_during_scan_processing.docx",
        file=BytesIO(b"docx bytes"),
    )

    bg = _document_routes.BackgroundTasks()
    response = await upload_endpoint(bg, upload_file)

    # Endpoint accepted the upload despite scan in progress.
    assert response.status == "success"
    assert (tmp_path / "upload_during_scan_processing.docx").exists()
    assert pipeline_status["pending_enqueues"] == 1
    assert len(bg.tasks) == 1


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
    returned success — but its bg task has not yet written to
    doc_status.  A scan that arrives in this window must refuse;
    starting it would race scan's doc_status reads against the bg
    task's still-pending writes.
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


async def test_release_enqueue_slot_decrements_per_call(tmp_path):
    """Two-reservation cohort: each release is a pure decrement.  Drain
    coordination is no longer needed because the busy guard on enqueue
    has been removed — concurrent enqueues are permitted while the
    pipeline is busy and the running loop's request_pending mechanism
    drains them.
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
    pipeline_status["scanning_exclusive"] = False
    pipeline_status["pending_enqueues"] = 0

    # Two concurrent reservations from /upload + /text — both pass the
    # idle preflight because scanning_exclusive=F at reservation time
    # (busy and the bare ``scanning`` flag are no longer gates;
    # concurrent enqueue with the processing loop and scan's
    # processing phase is explicitly allowed).
    assert await _document_routes._reserve_enqueue_slot(rag) is True
    assert await _document_routes._reserve_enqueue_slot(rag) is True
    assert pipeline_status["pending_enqueues"] == 2

    # Each release is a pure decrement; no drain coordination required
    # because each bg task triggers process_enqueue independently and
    # the running loop's request_pending mechanism collapses duplicate
    # triggers safely.
    await _document_routes._release_enqueue_slot(rag)
    assert pipeline_status["pending_enqueues"] == 1

    await _document_routes._release_enqueue_slot(rag)
    assert pipeline_status["pending_enqueues"] == 0


async def test_two_concurrent_uploads_both_succeed_when_pipeline_busy(
    tmp_path, monkeypatch
):
    """The scenario the original race report described, end-to-end:
    two upload requests arrive while the pipeline is busy.  Under the
    new contract neither is rejected; both reserve slots, both schedule
    bg tasks, and pending_enqueues stays at 2 until each bg task
    releases.  No reservation can be killed by the busy guard because
    that guard has been removed from apipeline_enqueue_documents.
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
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0
    pipeline_status["busy"] = True

    router = create_document_routes(rag, doc_manager)
    upload_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "upload_to_input_dir"
    ][-1]

    bg_a = _document_routes.BackgroundTasks()
    upload_a = _document_routes.UploadFile(filename="a.docx", file=BytesIO(b"a bytes"))
    response_a = await upload_endpoint(bg_a, upload_a)
    assert response_a.status == "success"
    assert pipeline_status["pending_enqueues"] == 1

    bg_b = _document_routes.BackgroundTasks()
    upload_b = _document_routes.UploadFile(filename="b.docx", file=BytesIO(b"b bytes"))
    response_b = await upload_endpoint(bg_b, upload_b)
    assert response_b.status == "success"
    # Both reservations coexist while bg tasks are pending.
    assert pipeline_status["pending_enqueues"] == 2
    # Both files were written to disk; both bg tasks scheduled.
    assert (tmp_path / "a.docx").exists()
    assert (tmp_path / "b.docx").exists()
    assert len(bg_a.tasks) == 1
    assert len(bg_b.tasks) == 1


async def test_reserve_enqueue_slot_allows_busy_and_scan_processing_phase(tmp_path):
    """Reservation only blocks on ``scanning_exclusive`` (scan's
    classification phase) and ``destructive_busy``.  Plain ``busy=True``
    (processing loop) and ``scanning=True`` with
    ``scanning_exclusive=False`` (scan in its processing phase) are
    BOTH permitted — that's what enables "upload while pipeline is
    working".
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
    pipeline_status["scanning_exclusive"] = False
    pipeline_status["pending_enqueues"] = 0

    # busy=True alone does NOT block.
    pipeline_status["busy"] = True
    assert await _document_routes._reserve_enqueue_slot(rag) is True
    await _document_routes._release_enqueue_slot(rag)
    pipeline_status["busy"] = False

    # scanning=True (scan processing phase) does NOT block — this is
    # the user-reported case: upload during scan-driven processing
    # must succeed.
    pipeline_status["scanning"] = True
    assert await _document_routes._reserve_enqueue_slot(rag) is True
    await _document_routes._release_enqueue_slot(rag)
    pipeline_status["scanning"] = False

    # scanning_exclusive=True (scan classification phase) STILL rejects.
    pipeline_status["scanning"] = True
    pipeline_status["scanning_exclusive"] = True
    with pytest.raises(_document_routes.HTTPException) as exc:
        await _document_routes._reserve_enqueue_slot(rag)
    assert exc.value.status_code == 409
    assert "classifying" in exc.value.detail.lower()
    assert pipeline_status["pending_enqueues"] == 0


async def test_reserve_enqueue_slot_rejects_destructive_busy(tmp_path):
    """``destructive_busy`` (set by /documents/clear and per-doc delete)
    must reject reservation: those jobs DROP storages and remove input
    files, so any concurrent enqueue would write to a storage being
    torn down and silently lose the document after the client saw 200.
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
    pipeline_status["scanning_exclusive"] = False
    pipeline_status["pending_enqueues"] = 0

    # destructive_busy=True (clear / delete in flight) → 409.
    pipeline_status["busy"] = True
    pipeline_status["destructive_busy"] = True
    with pytest.raises(_document_routes.HTTPException) as exc:
        await _document_routes._reserve_enqueue_slot(rag)
    assert exc.value.status_code == 409
    assert "clearing or deleting" in exc.value.detail.lower()
    assert pipeline_status["pending_enqueues"] == 0

    # Cleared once the destructive job finishes.
    pipeline_status["destructive_busy"] = False
    pipeline_status["busy"] = False
    assert await _document_routes._reserve_enqueue_slot(rag) is True
    await _document_routes._release_enqueue_slot(rag)


async def test_clear_documents_sets_and_clears_destructive_busy(tmp_path):
    """``/documents/clear`` must set ``destructive_busy=True`` while it is
    dropping storages (so concurrent uploads get 409, not silent loss)
    and clear the flag on completion so the pipeline returns to idle.
    """
    import importlib

    workspace = f"clear-test-{uuid4().hex}"
    observed = {"destructive_busy": None}

    class _DropSpy:
        """Mid-drop probe: snapshots ``destructive_busy`` when the clear
        endpoint calls our ``drop()``.  Concurrent reservations during
        this window MUST see destructive_busy=True.
        """

        def __init__(self, ws):
            self.namespace = "spy"
            self.workspace = ws

        async def drop(self):
            shared_storage_inner = importlib.import_module("lightrag.kg.shared_storage")
            ns = await shared_storage_inner.get_namespace_data(
                "pipeline_status", workspace=self.workspace
            )
            observed["destructive_busy"] = ns.get("destructive_busy")
            return None

    spy = _DropSpy(workspace)

    class _ClearRag:
        def __init__(self):
            self.workspace = workspace
            # Eleven storage attributes the clear endpoint iterates over.
            # Reusing the same spy is fine — each gets ``.drop()`` called
            # in turn, all observe the same destructive_busy flag.
            self.text_chunks = spy
            self.full_docs = spy
            self.full_entities = spy
            self.full_relations = spy
            self.entity_chunks = spy
            self.relation_chunks = spy
            self.entities_vdb = spy
            self.relationships_vdb = spy
            self.chunks_vdb = spy
            self.chunk_entity_relation_graph = spy
            self.doc_status = spy

        async def aclear_cache(self, modes=None):
            return None

    doc_manager = DocumentManager(str(tmp_path))
    rag = _ClearRag()

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )

    router = create_document_routes(rag, doc_manager)
    clear_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "clear_documents"
    ][-1]

    response = await clear_endpoint()
    assert response.status == "success"
    # destructive_busy was True for the duration of the storage drop.
    assert observed["destructive_busy"] is True
    # And cleared back to False after completion.
    assert pipeline_status.get("destructive_busy") is False
    assert pipeline_status.get("busy") is False


async def test_clear_documents_refuses_when_scanning_or_pending_enqueues(tmp_path):
    """``/documents/clear`` must refuse atomically when ANY exclusive
    or in-flight writer is active — not just ``busy``.  Previously
    only ``busy`` was checked, so clear could begin dropping storages
    while a /scan task was running or while an upload bg task had
    reserved a slot but not yet written its doc to doc_status.
    """
    import importlib

    workspace = f"clear-refuse-test-{uuid4().hex}"

    class _StubRag:
        def __init__(self):
            self.workspace = workspace

    rag = _StubRag()
    doc_manager = DocumentManager(str(tmp_path))

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )

    router = create_document_routes(rag, doc_manager)
    clear_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "clear_documents"
    ][-1]

    # Case 1: scanning=True must refuse.
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = True
    pipeline_status["pending_enqueues"] = 0
    response = await clear_endpoint()
    assert response.status == "busy"
    # Critical: no flag mutation occurred; scanning is still owned.
    assert pipeline_status["scanning"] is True
    assert pipeline_status.get("destructive_busy", False) is False

    # Case 2: pending_enqueues>0 must refuse.
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 1
    response = await clear_endpoint()
    assert response.status == "busy"
    assert pipeline_status["pending_enqueues"] == 1
    assert pipeline_status.get("destructive_busy", False) is False

    # Case 3: busy=True (e.g. processing loop or another destructive
    # job) must refuse — preserves existing behaviour.
    pipeline_status["pending_enqueues"] = 0
    pipeline_status["busy"] = True
    response = await clear_endpoint()
    assert response.status == "busy"
    assert pipeline_status.get("destructive_busy", False) is False


async def test_delete_document_reserves_destructive_busy_synchronously(tmp_path):
    """``/documents/delete_document`` must reserve the destructive slot
    synchronously BEFORE returning ``deletion_started``.  Otherwise
    a /scan or /upload arriving between the response and the bg task
    starting could race the destructive job.

    Acceptance criteria: after the endpoint returns success,
    pipeline_status reflects ``busy=True`` and ``destructive_busy=True``
    even though the bg task hasn't run yet.  Refusal cases for
    scanning / pending_enqueues / busy must short-circuit and return
    ``status="busy"`` without scheduling.
    """
    import importlib

    rag = _DeleteRag(DeletionResult(status="success", message="ok", doc_id="doc-1"))
    doc_manager = DocumentManager(str(tmp_path))

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )

    router = create_document_routes(rag, doc_manager)
    delete_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "delete_document"
    ][-1]

    # Build the request payload using the model class on the module.
    DeleteDocRequest = _document_routes.DeleteDocRequest

    # Case 1: reservation acquired synchronously, bg task scheduled.
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0
    bg = _document_routes.BackgroundTasks()
    response = await delete_endpoint(
        DeleteDocRequest(doc_ids=["doc-1"]),
        bg,
    )
    assert response.status == "deletion_started"
    # Synchronously reserved BEFORE returning.
    assert pipeline_status["busy"] is True
    assert pipeline_status["destructive_busy"] is True
    assert len(bg.tasks) == 1
    # Reset for next case.
    pipeline_status["busy"] = False
    pipeline_status["destructive_busy"] = False

    # Case 2: scanning=True must refuse without scheduling.
    pipeline_status["scanning"] = True
    bg = _document_routes.BackgroundTasks()
    response = await delete_endpoint(
        DeleteDocRequest(doc_ids=["doc-1"]),
        bg,
    )
    assert response.status == "busy"
    assert len(bg.tasks) == 0
    assert pipeline_status.get("destructive_busy", False) is False
    pipeline_status["scanning"] = False

    # Case 3: pending_enqueues>0 must refuse without scheduling.
    pipeline_status["pending_enqueues"] = 1
    bg = _document_routes.BackgroundTasks()
    response = await delete_endpoint(
        DeleteDocRequest(doc_ids=["doc-1"]),
        bg,
    )
    assert response.status == "busy"
    assert len(bg.tasks) == 0
    assert pipeline_status["pending_enqueues"] == 1
    assert pipeline_status.get("destructive_busy", False) is False
    pipeline_status["pending_enqueues"] = 0


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

    deleted_files, errors = _document_routes.delete_file_variants_by_file_path(
        tmp_path,
        "report.docx",
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
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    source_path = tmp_path / "parsed-after-sync.docx"
    source_path.write_bytes(b"docx bytes")
    rag = _ParseRag(tmp_path / "work", source_path)

    def _fake_extract(file_path, fixlevel=None, drawing_context=None, **kwargs):
        # extract_docx_blocks returns a list of block dicts; a single text
        # block is enough to exercise the archive + full_docs side-effects
        # tested below — the adapter will turn it into one .blocks.jsonl
        # content row.
        return [
            {
                "uuid": "p1",
                "uuid_end": "p1",
                "heading": "",
                "content": "parsed",
                "type": "text",
                "parent_headings": [],
                "level": 0,
                "table_chunk_role": "none",
            }
        ]

    monkeypatch.setattr(
        "lightrag.parser.docx.parse_document.extract_docx_blocks",
        _fake_extract,
    )

    result = await _parse_via_registry(
        rag,
        "native",
        "doc-test",
        str(source_path),
        {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
    )

    # parse_native now returns LIGHTRAG-format parsed_data with merged_text
    # (not the {{LRdoc}} marker — that's only in the persisted full_docs row).
    assert result["content"]
    assert result["parse_format"] == "lightrag"
    assert result["blocks_path"]
    assert rag.full_docs.events == ["upsert", "index_done"]
    assert not source_path.exists()
    assert (tmp_path / PARSED_DIR_NAME / source_path.name).exists()
    parsed_artifact_dir = tmp_path / PARSED_DIR_NAME / f"{source_path.name}.parsed"
    assert parsed_artifact_dir.is_dir()
    assert (parsed_artifact_dir / "parsed-after-sync.blocks.jsonl").is_file()
    assert rag.full_docs.data["doc-test"]["parse_engine"] == "native"
    assert rag.full_docs.data["doc-test"]["parse_format"] == "lightrag"
    # Per docs/FileProcessingConfiguration-zh.md, content uses the {{LRdoc}}
    # marker plus a leading-text summary derived from merged blocks.
    assert rag.full_docs.data["doc-test"]["content"].startswith("{{LRdoc}}")


def test_parsed_artifact_dir_uses_unique_suffix_when_path_is_file(
    tmp_path, monkeypatch
):
    from lightrag.utils_pipeline import parsed_artifact_dir_for

    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "demo.docx.parsed").write_text("legacy file", encoding="utf-8")

    artifact_dir = parsed_artifact_dir_for("demo.docx")

    assert artifact_dir == parsed_dir / "demo.docx.parsed_001"


def test_parsed_artifact_dir_reuses_existing_parsed_parent(tmp_path, monkeypatch):
    from lightrag.utils_pipeline import parsed_artifact_dir_for

    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()

    artifact_dir = parsed_artifact_dir_for("demo.docx")

    assert artifact_dir == parsed_dir / "demo.docx.parsed"


async def test_parse_native_docx_content_list_failure_raises_without_fallback(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "content-list-failure.docx"
    source_path.write_bytes(b"docx bytes")
    rag = _ParseRag(tmp_path / "work", source_path)

    def _raise_parser(file_path, fixlevel=None, drawing_context=None, **kwargs):
        raise RuntimeError("content list boom")

    monkeypatch.setattr(
        "lightrag.parser.docx.parse_document.extract_docx_blocks",
        _raise_parser,
    )

    with pytest.raises(RuntimeError, match="content list boom"):
        await _parse_via_registry(
            rag,
            "native",
            "doc-test",
            str(source_path),
            {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )

    assert source_path.exists()
    assert rag.full_docs.events == []


async def test_parse_native_docx_empty_content_list_result_raises_without_fallback(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "empty-content-list.docx"
    source_path.write_bytes(b"docx bytes")
    rag = _ParseRag(tmp_path / "work", source_path)

    monkeypatch.setattr(
        "lightrag.parser.docx.parse_document.extract_docx_blocks",
        lambda *args, **kwargs: [],
    )

    with pytest.raises(ValueError, match="empty content"):
        await _parse_via_registry(
            rag,
            "native",
            "doc-test",
            str(source_path),
            {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )

    assert source_path.exists()
    assert rag.full_docs.events == []


def test_lightrag_document_reprocess_uses_full_docs_without_reparse():
    engine = resolve_stored_document_parser_engine(
        "report.[mineru].docx",
        {
            "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
            "sidecar_location": "file:///tmp/report.docx.parsed/",
            "parse_engine": "mineru",
        },
    )

    # All lightrag rows route to the internal reuse handler (reuse the stored
    # sidecar without re-parsing) regardless of the originating engine.
    assert engine == "reuse"


def test_default_allowlist_equals_local_engine_suffixes(tmp_path, monkeypatch):
    """With no external endpoints and no routing rules, the registry-derived
    allowlist must equal the local engines' (legacy ∪ native) suffixes —
    i.e. exactly the historical hardcoded upload allowlist."""
    from lightrag.parser import registry

    for var in (
        "MINERU_LOCAL_ENDPOINT",
        "MINERU_API_TOKEN",
        "DOCLING_ENDPOINT",
        "LIGHTRAG_PARSER",
    ):
        monkeypatch.delenv(var, raising=False)

    dm = DocumentManager(str(tmp_path))
    local = {f".{s}" for s in registry.suffix_capabilities("legacy")} | {
        f".{s}" for s in registry.suffix_capabilities("native")
    }
    assert set(dm.supported_extensions) == local
    # External-only suffixes stay out while their endpoints are unset.
    assert ".png" not in dm.supported_extensions
    assert ".doc" not in dm.supported_extensions


def test_unroutable_suffix_needs_rule_or_hint(tmp_path, monkeypatch):
    """An endpoint-configured engine's suffix is accepted only when routing
    actually reaches that engine: a bare filename needs a LIGHTRAG_PARSER
    rule; a per-file hint works without one. Otherwise the file would pass
    upload only to fail the parse worker's legacy suffix gate."""
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")

    dm = DocumentManager(str(tmp_path))
    # Capable (endpoint up) but unroutable: bare png defaults to legacy.
    assert ".png" not in dm.supported_extensions
    assert not dm.is_supported_file("scan.png")
    # A per-file hint routes this specific name to mineru.
    assert dm.is_supported_file("scan.[mineru].png")
    # A routing rule makes the bare suffix routable (and advertised).
    monkeypatch.setenv("LIGHTRAG_PARSER", "png:mineru")
    assert ".png" in dm.supported_extensions
    assert dm.is_supported_file("scan.png")
    # docling-only suffixes stay out (endpoint unset), rule or not.
    assert ".xhtml" not in dm.supported_extensions


def test_third_party_engine_suffixes_join_allowlist_and_scan(tmp_path, monkeypatch):
    """A registered third-party engine's suffixes become uploadable and
    scannable once routable (rule for bare names, hint for individual
    files), and revert on unregister."""
    from lightrag.parser import registry

    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    dm = DocumentManager(str(tmp_path))

    # Before registration: .foo rejected by upload and invisible to scan.
    assert not dm.is_supported_file("sample.foo")
    (dm.input_dir / "sample.foo").write_text("x", encoding="utf-8")
    (dm.input_dir / "hinted.[fooengine].foo").write_text("x", encoding="utf-8")
    assert not [p for p in dm.scan_directory_for_new_files() if p.suffix == ".foo"]

    registry.register_parser(
        registry.ParserSpec(
            engine_name="fooengine",
            impl="x:Y",
            suffixes=frozenset({"foo"}),
            queue_group="fooengine",
            concurrency=1,
        )
    )
    try:
        # Registered but bare .foo is still unroutable (defaults to legacy).
        assert not dm.is_supported_file("sample.foo")
        # The hinted file routes to fooengine: uploadable AND discoverable
        # by scan (glob covers the capability surface, filter is per-name).
        assert dm.is_supported_file("hinted.[fooengine].foo")
        scanned = {p.name for p in dm.scan_directory_for_new_files()}
        assert "hinted.[fooengine].foo" in scanned
        assert "sample.foo" not in scanned
        # A routing rule makes the bare suffix routable.
        monkeypatch.setenv("LIGHTRAG_PARSER", "foo:fooengine")
        assert ".foo" in dm.supported_extensions
        assert dm.is_supported_file("sample.foo")
        assert "sample.foo" in {p.name for p in dm.scan_directory_for_new_files()}
    finally:
        registry._REGISTRY.pop("fooengine", None)
    assert not dm.is_supported_file("sample.foo")


class _DropStorage:
    """Minimal storage stub whose drop() returns a preset result dict."""

    def __init__(self, drop_result, namespace="ns", workspace="clear-test"):
        self._drop_result = drop_result
        self.namespace = namespace
        self.workspace = workspace

    async def drop(self):
        return self._drop_result


class _ClearRag:
    """Mock LightRAG exposing the storages that clear_documents drops."""

    def __init__(self, chunks_drop_result):
        self.workspace = "clear-test"
        ok = {"status": "success", "message": "data dropped"}
        self.text_chunks = _DropStorage(ok, "text_chunks")
        self.full_docs = _DropStorage(ok, "full_docs")
        self.full_entities = _DropStorage(ok, "full_entities")
        self.full_relations = _DropStorage(ok, "full_relations")
        self.entity_chunks = _DropStorage(ok, "entity_chunks")
        self.relation_chunks = _DropStorage(ok, "relation_chunks")
        self.entities_vdb = _DropStorage(ok, "entities")
        self.relationships_vdb = _DropStorage(ok, "relationships")
        # The storage under test: drop() result is configurable.
        self.chunks_vdb = _DropStorage(chunks_drop_result, "chunks")
        self.chunk_entity_relation_graph = _DropStorage(ok, "graph")
        self.doc_status = _DropStorage(ok, "doc_status")


def _clear_endpoint(router):
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "clear_documents"
    ][-1]


async def test_clear_documents_honors_drop_error_status(tmp_path):
    """A storage whose drop() returns {"status": "error"} (without raising) must
    NOT be counted as a success: the clear is reported as partial_success so the
    caller knows it is incomplete and can retry, instead of a misleading success.
    """
    import importlib

    doc_manager = DocumentManager(str(tmp_path))
    rag = _ClearRag(
        chunks_drop_result={
            "status": "error",
            "message": "legacy tagging undetermined",
        }
    )

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)

    router = create_document_routes(rag, doc_manager)
    response = await _clear_endpoint(router)()

    assert response.status == "partial_success"


async def test_clear_documents_succeeds_when_all_drops_succeed(tmp_path):
    """Baseline: when every storage drop() returns success the clear reports
    success."""
    import importlib

    doc_manager = DocumentManager(str(tmp_path))
    rag = _ClearRag(chunks_drop_result={"status": "success", "message": "data dropped"})

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)

    router = create_document_routes(rag, doc_manager)
    response = await _clear_endpoint(router)()

    assert response.status == "success"
