"""Streaming scan discovery in bounded batches (LR2 Phase 4-c, §8.2/§8.4).

Discovery is a single generator pass and the scan holds at most
``SCAN_ENQUEUE_BATCH_SIZE`` claimed files: each batch is written to doc_status as
soon as it fills — while discovery is still running — so peak memory is set by
the batch, not by how many files the input directory holds. Inside one batch the
first physical file to claim a canonical source key wins; a later variant is
archived (never deleted).
"""

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.constants import PARSED_DIR_NAME  # noqa: E402

DocumentManager = _document_routes.DocumentManager
run_scanning_process = _document_routes.run_scanning_process

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    """The stub rags never run ``initialize_storages``, and the scan resolves
    ``pipeline_status`` through the module-level shared dicts."""
    importlib.import_module("lightrag.kg.shared_storage").initialize_share_data()
    yield


class _NoRowsDocStatus:
    """Every canonical basename is new (SourceAbsent-equivalent)."""

    def __init__(self):
        self.lookups = []

    async def get_doc_by_file_basename(self, basename):
        self.lookups.append(basename)
        return None


class _StreamRag:
    def __init__(self):
        self.workspace = f"scanstream-{uuid4().hex[:8]}"
        self.doc_status = _NoRowsDocStatus()
        self.full_docs = SimpleNamespace(get_by_id=self._missing)
        self.process_calls = 0

    async def _missing(self, _doc_id):
        return None

    async def arollback_failed_custom_chunk_patches(self, **_kwargs):
        return {"rolled_back": [], "failed": []}

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1


def _batch_size(monkeypatch, size: int) -> None:
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(scan_enqueue_batch_size=size)
    )


def test_batches_are_written_while_discovery_is_still_streaming(tmp_path, monkeypatch):
    """The defining property of §8.2: batch 1 reaches doc_status BEFORE the last
    file is discovered, and no batch exceeds ``SCAN_ENQUEUE_BATCH_SIZE``.

    Fix-proof: the previous implementation materialized ``scan_directory_for_new_files()``
    into a list, grouped it by canonical name and enqueued everything in ONE
    call — so the first write could not happen until discovery had finished, and
    both structures grew with the directory."""

    async def _run():
        rag = _StreamRag()
        doc_manager = DocumentManager(str(tmp_path))
        _batch_size(monkeypatch, 2)

        files = [tmp_path / f"doc{index}.txt" for index in range(5)]
        events: list[tuple[str, object]] = []

        def _iter_new_files():
            for file_path in files:
                events.append(("yield", file_path.name))
                yield file_path

        doc_manager.iter_new_files = _iter_new_files

        async def _capture_batch(_rag, file_paths, _track_id):
            events.append(("flush", [path.name for path in file_paths]))
            return len(file_paths)

        monkeypatch.setattr(
            _document_routes, "pipeline_enqueue_scan_batch", _capture_batch
        )

        await run_scanning_process(rag, doc_manager, "track-stream")

        flushes = [event for event in events if event[0] == "flush"]
        assert [len(names) for _, names in flushes] == [2, 2, 1]
        assert sorted(name for _, names in flushes for name in names) == [
            f"doc{index}.txt" for index in range(5)
        ]
        # Streaming, not batching-after-collecting: the first write lands before
        # discovery yields its last file.
        first_flush = events.index(flushes[0])
        last_yield = max(
            index for index, event in enumerate(events) if event[0] == "yield"
        )
        assert first_flush < last_yield
        # One processing drive for the whole scan (§8.1), never per batch.
        assert rag.process_calls == 1

    asyncio.run(_run())


def test_first_claim_in_a_batch_wins_and_the_alias_is_archived(tmp_path, monkeypatch):
    """§8.4: a canonical key claimed inside the current batch is not yet visible
    to the doc_status lookup, so the claim set is what stops a second variant
    from minting a duplicate document. The loser is archived, not deleted."""

    async def _run():
        rag = _StreamRag()
        doc_manager = DocumentManager(str(tmp_path))
        _batch_size(monkeypatch, 8)

        plain = doc_manager.input_dir / "same.txt"
        hinted = doc_manager.input_dir / "same.[native].txt"
        plain.write_text("plain", encoding="utf-8")
        hinted.write_text("hinted", encoding="utf-8")
        ordered = [plain, hinted]  # pin the discovery order this test reasons about
        doc_manager.iter_new_files = lambda: iter(ordered)

        batched: list[Path] = []

        async def _capture_batch(_rag, file_paths, _track_id):
            batched.extend(file_paths)
            return len(file_paths)

        monkeypatch.setattr(
            _document_routes, "pipeline_enqueue_scan_batch", _capture_batch
        )

        await run_scanning_process(rag, doc_manager, "track-claim")

        assert batched == [plain]  # first claim wins
        assert plain.exists()
        assert not hinted.exists()
        archived = {
            path.name
            for path in (doc_manager.input_dir / PARSED_DIR_NAME).iterdir()
            if path.is_file()
        }
        assert archived == {hinted.name}
        # Only ONE canonical lookup per physical file; the alias never reaches
        # the enqueue path (which would have created a ``dup-*`` row).
        assert rag.doc_status.lookups == ["same.txt", "same.txt"]

    asyncio.run(_run())


def test_iter_new_files_is_lazy_and_skips_directories(tmp_path):
    """Discovery must be a generator (no whole-directory list) and must not hand
    a directory named like a document to the enqueue path."""
    doc_manager = DocumentManager(str(tmp_path))
    (doc_manager.input_dir / "a.txt").write_text("a", encoding="utf-8")
    (doc_manager.input_dir / "b.txt").write_text("b", encoding="utf-8")
    (doc_manager.input_dir / "trap.txt").mkdir()
    (doc_manager.input_dir / "ignored.bin").write_text("x", encoding="utf-8")

    stream = doc_manager.iter_new_files()
    assert not isinstance(stream, list)
    first = next(iter(stream))  # consumes ONE entry, not the directory
    assert first.suffix == ".txt"

    names = {path.name for path in doc_manager.iter_new_files()}
    assert names == {"a.txt", "b.txt"}


def test_batch_size_falls_back_when_unconfigured(monkeypatch):
    """A rig that bypassed ``initialize_config`` must still get a BOUNDED batch —
    never "hold everything"."""
    from lightrag.constants import DEFAULT_SCAN_ENQUEUE_BATCH_SIZE

    monkeypatch.setattr(_document_routes, "global_args", SimpleNamespace())
    assert (
        _document_routes._scan_enqueue_batch_size() == DEFAULT_SCAN_ENQUEUE_BATCH_SIZE
    )

    _batch_size(monkeypatch, 0)
    assert (
        _document_routes._scan_enqueue_batch_size() == DEFAULT_SCAN_ENQUEUE_BATCH_SIZE
    )

    _batch_size(monkeypatch, 7)
    assert _document_routes._scan_enqueue_batch_size() == 7
