"""A scanned file whose basename cannot be a document source is refused (§8.3.0).

``/documents/upload`` and ``/documents/text`` reject an unsafe source at the
request boundary, but INPUT_DIR is populated by the filesystem — a volume mount,
an rsync, an operator's ``cp`` — so ``/documents/scan`` is a third ingress with
no request to validate. Nothing between ``os.scandir`` and ``doc_status``
filtered characters: discovery gates on the suffix and ``normalize_file_path``
only strips a parser hint, so the basename reached the extraction handlers and
was stamped verbatim onto every entity/relation ``file_path``
(GHSA-c922-pw4m-4wcv).

Driven through the REAL ``iter_new_files`` and the real classification loop, so
the assertions cover discovery, classification and the enqueue hand-off rather
than a hand-built candidate list.
"""

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.base import SourceAbsent  # noqa: E402
from lightrag.constants import PARSED_DIR_NAME  # noqa: E402

DocumentManager = _document_routes.DocumentManager
run_scanning_process = _document_routes.run_scanning_process
pipeline_enqueue_file = _document_routes.pipeline_enqueue_file
_ScanFileClass = _document_routes._ScanFileClass

pytestmark = pytest.mark.offline

UNSAFE_NAME = "scan\x0bsource.txt"
"""U+000B: a vertical tab is legal in a POSIX/APFS filename, illegal in XML 1.0,
and the exact character the advisory's reproduction used."""


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    importlib.import_module("lightrag.kg.shared_storage").initialize_share_data()
    yield


class _NoRowsDocStatus:
    """Every canonical key resolves to SourceAbsent: nothing is a duplicate, so
    a skipped file can only have been skipped by the guard under test."""

    def __init__(self):
        self.lookups: list[str] = []

    async def resolve_doc_source_strict(self, canonical_source_key):
        self.lookups.append(canonical_source_key)
        return SourceAbsent()

    async def get_full_docs_by_ids(self, doc_ids, *, strict=False):
        return {}


class _ScanRag:
    def __init__(self):
        self.workspace = f"unsafe-source-{uuid4().hex[:8]}"
        self.doc_status = _NoRowsDocStatus()
        self.full_docs = SimpleNamespace(
            get_by_id=self._missing,
            get_by_id_strict=self._missing,
            supports_strict_point_reads=True,
        )

    async def _missing(self, _doc_id):
        return None

    async def arollback_failed_custom_chunk_patches(self, **_kwargs):
        return {"rolled_back": [], "failed": []}

    async def apipeline_process_enqueue_documents(self):
        return None


async def _init_pipeline_status(workspace) -> None:
    """``record_scan_warning`` writes into this workspace's shared status dict and
    swallows every failure, so an uninitialised namespace would make the warning
    assertions vacuously pass."""
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=workspace)


async def _history(workspace) -> list[str]:
    """The messages ``/documents/pipeline_status`` would return."""
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    return list(status.get("history_messages", []))


async def _scan(tmp_path, monkeypatch, *, filenames) -> tuple[_ScanRag, list[str]]:
    """Run one real scan over ``filenames``; return the rag and what was enqueued."""
    rag = _ScanRag()
    await _init_pipeline_status(rag.workspace)
    doc_manager = DocumentManager(str(tmp_path))
    for name in filenames:
        (doc_manager.input_dir / name).write_text("body", encoding="utf-8")

    enqueued: list[str] = []

    async def _capture_batch(_rag, candidates, _track_id):
        enqueued.extend(candidate.path.name for candidate in candidates)
        return len(candidates)

    monkeypatch.setattr(_document_routes, "pipeline_enqueue_scan_batch", _capture_batch)
    await run_scanning_process(rag, doc_manager, "track-unsafe")
    return rag, enqueued


def test_scan_refuses_an_unsafe_basename_and_still_ingests_its_neighbour(
    tmp_path, monkeypatch
):
    """Fix proof: the unsafe file is discovered but never enqueued, and the
    refusal is scoped to that one file."""

    async def _run():
        _rag, enqueued = await _scan(
            tmp_path, monkeypatch, filenames=[UNSAFE_NAME, "clean.txt"]
        )
        assert enqueued == ["clean.txt"]

    asyncio.run(_run())


def test_the_refused_file_stays_in_the_input_directory(tmp_path, monkeypatch):
    """Not archived, not renamed, not deleted — the basename is the document
    identity, dedup key and doc_id seed, so only an operator rename fixes it.
    Leaving it put also means the next scan warns again instead of going quiet."""

    async def _run():
        await _scan(tmp_path, monkeypatch, filenames=[UNSAFE_NAME])

        assert (tmp_path / UNSAFE_NAME).exists()
        parsed_dir = tmp_path / PARSED_DIR_NAME
        archived = (
            [path.name for path in parsed_dir.iterdir()] if parsed_dir.exists() else []
        )
        assert archived == []

    asyncio.run(_run())


def test_the_refusal_happens_before_any_identity_resolution(tmp_path, monkeypatch):
    """§8.3.0 runs ahead of ``classify_scan_file``: no doc_status read for the
    rejected name, so no later exit can format the raw basename into a warning,
    a bounded job sample or a doc_status row."""

    async def _run():
        rag, _enqueued = await _scan(
            tmp_path, monkeypatch, filenames=[UNSAFE_NAME, "clean.txt"]
        )
        assert rag.doc_status.lookups == ["clean.txt"]

    asyncio.run(_run())


def test_the_scan_warning_names_the_offending_code_point(tmp_path, monkeypatch):
    """An operator has to be able to act on this: the message must identify the
    character, since the raw name looks unremarkable in a terminal."""

    async def _run():
        rag, _enqueued = await _scan(tmp_path, monkeypatch, filenames=[UNSAFE_NAME])

        history = await _history(rag.workspace)
        matching = [message for message in history if "U+000B" in message]
        assert matching, history
        assert "unsafe document source" in matching[-1]
        # The name is escaped, never interpolated raw.
        assert UNSAFE_NAME not in matching[-1]
        assert "scan\\x0bsource.txt" in matching[-1]

    asyncio.run(_run())


def test_the_recorded_message_survives_json_rendering(tmp_path, monkeypatch):
    """Filesystem-independent half of the surrogate defence.

    Starlette renders responses with ``ensure_ascii=False`` + ``utf-8``, which
    RAISES on a lone surrogate. ``pipeline_status`` history is returned by
    ``/documents/pipeline_status``, so a raw name in that history would take out
    every later read of the endpoint. Asserted on the real formatter rather than
    on a filename, because not every filesystem accepts a surrogate name.
    """
    surrogate_name = "scan\udcffsource.txt"
    assert (
        _document_routes.find_unsafe_document_source_character(surrogate_name)
        == "\udcff"
    )
    rendered = _document_routes.describe_rejected_document_source(surrogate_name)
    payload = {"history_messages": [f"Skipping {rendered}"]}
    # Would raise UnicodeEncodeError with the raw name in place of ``rendered``.
    json.dumps(payload, ensure_ascii=False).encode("utf-8")

    with pytest.raises(UnicodeEncodeError):
        json.dumps({"raw": surrogate_name}, ensure_ascii=False).encode("utf-8")


def test_a_surrogate_basename_is_refused_end_to_end(tmp_path, monkeypatch):
    """The real hazard on Linux: a directory entry whose bytes are not valid
    UTF-8 reaches Python through ``surrogateescape``. Skipped where the
    filesystem refuses such a name (APFS does)."""

    async def _run():
        rag = _ScanRag()
        await _init_pipeline_status(rag.workspace)
        doc_manager = DocumentManager(str(tmp_path))
        try:
            (doc_manager.input_dir / "scan\udcffsource.txt").write_text("body")
        except (OSError, UnicodeEncodeError) as exc:
            pytest.skip(f"filesystem rejects non-UTF-8 filenames: {exc}")

        enqueued: list[str] = []

        async def _capture_batch(_rag, candidates, _track_id):
            enqueued.extend(candidate.path.name for candidate in candidates)
            return len(candidates)

        monkeypatch.setattr(
            _document_routes, "pipeline_enqueue_scan_batch", _capture_batch
        )
        await run_scanning_process(rag, doc_manager, "track-surrogate")

        assert enqueued == []
        history = await _history(rag.workspace)
        # Every recorded message stays renderable.
        json.dumps(history, ensure_ascii=False).encode("utf-8")

    asyncio.run(_run())


def test_the_unsafe_exit_is_part_of_the_scan_taxonomy():
    """The counter key a ``/documents/scan/status`` reader sees."""
    assert _ScanFileClass.UNSAFE_SOURCE.value == "unsafe_source"


def test_pipeline_enqueue_file_refuses_an_unsafe_basename_without_a_doc_row(tmp_path):
    """Chokepoint backstop: every file-shaped ingest passes here, and a refusal
    must not create an error document — that would persist the offending name
    into ``doc_status.file_path``, which is what the guard exists to prevent."""

    error_batches: list[list[dict]] = []

    class _EnqueueRag:
        async def apipeline_enqueue_documents(self, _content, **_kwargs):
            raise AssertionError("an unsafe source must not reach doc_status")

        async def apipeline_enqueue_error_documents(self, error_files, _track_id):
            error_batches.append(error_files)

    async def _run():
        unsafe_file = tmp_path / UNSAFE_NAME
        unsafe_file.write_text("body", encoding="utf-8")

        success, track_id = await pipeline_enqueue_file(
            _EnqueueRag(), unsafe_file, "track-backstop"
        )

        assert success is False
        assert track_id == "track-backstop"
        assert error_batches == []
        # The file is left alone, same as the scan exit.
        assert unsafe_file.exists()

    asyncio.run(_run())


def test_a_safe_basename_still_reaches_the_enqueue(tmp_path):
    """Stability: the backstop rejects only what the shared rule rejects."""

    seen: list[str] = []

    class _EnqueueRag:
        addon_params: dict = {}

        async def apipeline_enqueue_documents(self, _content, **kwargs):
            seen.append(kwargs["file_paths"])
            return {"ok": True}

        async def apipeline_enqueue_error_documents(self, error_files, _track_id):
            raise AssertionError(f"unexpected error document: {error_files}")

    async def _run():
        clean_file = tmp_path / "clean.txt"
        clean_file.write_text("body", encoding="utf-8")

        success, _track_id = await pipeline_enqueue_file(
            _EnqueueRag(), clean_file, "track-clean"
        )
        assert success is True
        assert seen == [str(clean_file)]

    asyncio.run(_run())


def test_the_escaped_name_stays_recognisable():
    """The escaping is for transport, not obfuscation — an operator still has to
    be able to find the file."""
    rendered = _document_routes.describe_rejected_document_source(UNSAFE_NAME)
    assert rendered.isascii()
    assert "scan" in rendered and "source.txt" in rendered
    assert Path(UNSAFE_NAME).suffix == ".txt"
