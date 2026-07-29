"""Disk-spooled scan ordering in bounded batches (LR2 §8.2/§8.4).

Discovery/classification is a single generator pass. New candidates and their
scan-wide canonical claims live in a disposable SQLite spool, not Python memory.
After discovery, its on-disk mtime index yields at most
``SCAN_ENQUEUE_BATCH_SIZE`` paths per enqueue call. The first physical file to
claim a canonical source key wins; a later variant is archived (never deleted).
"""

import asyncio
import importlib
import os
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

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    """The stub rags never run ``initialize_storages``, and the scan resolves
    ``pipeline_status`` through the module-level shared dicts."""
    importlib.import_module("lightrag.kg.shared_storage").initialize_share_data()
    yield


class _NoRowsDocStatus:
    """Every canonical source key resolves to SourceAbsent (all files are new)."""

    def __init__(self):
        self.lookups = []

    async def resolve_doc_source_strict(self, canonical_source_key):
        self.lookups.append(canonical_source_key)
        return SourceAbsent()

    async def get_full_docs_by_ids(self, doc_ids, *, strict=False):
        return {}


class _StreamRag:
    def __init__(self):
        self.workspace = f"scanstream-{uuid4().hex[:8]}"
        self.doc_status = _NoRowsDocStatus()
        self.full_docs = SimpleNamespace(
            get_by_id=self._missing,
            get_by_id_strict=self._missing,
            supports_strict_point_reads=True,
        )
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


def test_candidates_are_spooled_then_written_in_global_mtime_order(
    tmp_path, monkeypatch
):
    """Exact global order crosses enqueue-batch boundaries without O(files) RAM.

    Discovery order is deliberately unrelated to mtime. Nothing reaches
    doc_status until the unordered stream is exhausted; then bounded batches
    preserve one global oldest-first order.
    """

    async def _run():
        rag = _StreamRag()
        doc_manager = DocumentManager(str(tmp_path))
        _batch_size(monkeypatch, 2)

        by_age = [tmp_path / f"age-{index}.txt" for index in range(5)]
        for index, file_path in enumerate(by_age):
            file_path.write_text(str(index), encoding="utf-8")
            stamp = 1_600_000_000 + index
            os.utime(file_path, (stamp, stamp))
        files = [by_age[4], by_age[1], by_age[3], by_age[0], by_age[2]]
        events: list[tuple[str, object]] = []

        def _iter_new_files():
            for file_path in files:
                events.append(("yield", file_path.name))
                yield file_path

        doc_manager.iter_new_files = _iter_new_files

        async def _capture_batch(_rag, candidates, _track_id):
            events.append(("flush", [c.path.name for c in candidates]))
            return len(candidates)

        monkeypatch.setattr(
            _document_routes, "pipeline_enqueue_scan_batch", _capture_batch
        )

        await run_scanning_process(rag, doc_manager, "track-stream")

        flushes = [event for event in events if event[0] == "flush"]
        assert [len(names) for _, names in flushes] == [2, 2, 1]
        assert [name for _, names in flushes for name in names] == [
            path.name for path in by_age
        ]
        # Global sorting needs to see every candidate, but only the disposable
        # disk spool grows; doc_status writes start after discovery.
        first_flush = events.index(flushes[0])
        last_yield = max(
            index for index, event in enumerate(events) if event[0] == "yield"
        )
        assert first_flush > last_yield
        # One processing drive for the whole scan (§8.1), never per batch.
        assert rag.process_calls == 1

    asyncio.run(_run())


def test_first_scan_wide_claim_wins_and_the_alias_is_archived(tmp_path, monkeypatch):
    """The disk UNIQUE claim replaces both the old batch map and cross-batch
    visibility through early doc_status writes."""

    async def _run():
        rag = _StreamRag()
        doc_manager = DocumentManager(str(tmp_path))
        # The variants would occupy different enqueue batches if both survived.
        _batch_size(monkeypatch, 1)

        plain = doc_manager.input_dir / "same.txt"
        hinted = doc_manager.input_dir / "same.[native].txt"
        plain.write_text("plain", encoding="utf-8")
        hinted.write_text("hinted", encoding="utf-8")
        ordered = [plain, hinted]  # pin the discovery order this test reasons about
        doc_manager.iter_new_files = lambda: iter(ordered)

        batched: list[Path] = []

        async def _capture_batch(_rag, candidates, _track_id):
            batched.extend(c.path for c in candidates)
            return len(candidates)

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


def test_unreadable_mtime_sorts_after_readable_candidates(tmp_path, monkeypatch):
    """Losing a stat costs priority information, never the candidate itself."""

    async def _run():
        rag = _StreamRag()
        doc_manager = DocumentManager(str(tmp_path))
        _batch_size(monkeypatch, 2)

        readable = tmp_path / "middle.txt"
        readable.write_text("body", encoding="utf-8")
        missing_b = tmp_path / "z-gone.txt"
        missing_a = tmp_path / "a-gone.txt"
        doc_manager.iter_new_files = lambda: iter([missing_b, readable, missing_a])

        ordered: list[str] = []

        async def _capture_batch(_rag, candidates, _track_id):
            ordered.extend(c.path.name for c in candidates)
            return len(candidates)

        monkeypatch.setattr(
            _document_routes, "pipeline_enqueue_scan_batch", _capture_batch
        )

        await run_scanning_process(rag, doc_manager, "track-missing-mtime")

        assert ordered == ["middle.txt", "a-gone.txt", "z-gone.txt"]

    asyncio.run(_run())


def test_an_unusable_spool_directory_fails_the_scan_closed(
    tmp_path, monkeypatch, caplog
):
    """Fail-closed placement, at the route level.

    The scan reports the misconfiguration and enqueues nothing, rather than
    silently relocating its O(number of files) ordering state to a possibly
    RAM-backed /tmp and surfacing the problem as an OOM kill partway through.
    Files stay in INPUT_DIR, so fixing the setting and re-scanning loses
    nothing.
    """

    async def _run():
        rag = _StreamRag()
        doc_manager = DocumentManager(str(tmp_path))
        blocker = tmp_path / "blocked-by-a-regular-file"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setattr(
            _document_routes,
            "global_args",
            SimpleNamespace(
                scan_enqueue_batch_size=2,
                scan_spool_dir=str(blocker / "spool"),
            ),
        )

        candidate = doc_manager.input_dir / "candidate.txt"
        candidate.write_text("body", encoding="utf-8")

        batches: list[object] = []

        async def _capture_batch(_rag, candidates, _track_id):
            batches.append(candidates)
            return len(candidates)

        monkeypatch.setattr(
            _document_routes, "pipeline_enqueue_scan_batch", _capture_batch
        )

        lightrag_logger = importlib.import_module("lightrag.utils").logger
        previous = lightrag_logger.propagate
        lightrag_logger.propagate = True
        try:
            with caplog.at_level("ERROR", logger=lightrag_logger.name):
                await run_scanning_process(rag, doc_manager, "track-bad-spool")
        finally:
            lightrag_logger.propagate = previous

        assert batches == []
        assert candidate.exists()
        assert not (doc_manager.input_dir / PARSED_DIR_NAME).exists()
        # The failure names the knob the operator has to fix.
        assert "SCAN_SPOOL_DIR" in caplog.text

    asyncio.run(_run())


def test_iter_new_files_is_lazy_and_skips_directories(tmp_path, monkeypatch):
    """Discovery must not read the next directory entry before yielding the
    current file, or hand a directory named like a document to enqueue."""
    doc_manager = DocumentManager(str(tmp_path))
    (doc_manager.input_dir / "a.txt").write_text("a", encoding="utf-8")
    (doc_manager.input_dir / "b.txt").write_text("b", encoding="utf-8")
    (doc_manager.input_dir / "trap.txt").mkdir()
    (doc_manager.input_dir / "ignored.bin").write_text("x", encoding="utf-8")

    class _OneThenTrap:
        """A scandir iterator that fails if discovery reads ahead."""

        def __init__(self, directory):
            self._first = SimpleNamespace(path=str(Path(directory) / "a.txt"))
            self._yielded = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            if self._yielded:
                raise AssertionError("directory scan read ahead before yielding")
            self._yielded = True
            return self._first

    with monkeypatch.context() as patch:
        patch.setattr(_document_routes.os, "scandir", _OneThenTrap)
        stream = doc_manager.iter_new_files()
        assert not isinstance(stream, list)
        first = next(stream)
        assert first.name == "a.txt"
        stream.close()

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
