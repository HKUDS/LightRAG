"""Scan enqueues carry the file's mtime as ``created_at`` (LR2 Phase 4-e, §8.3.A/§8.5).

Processing order is ``(created_at, id)``, so a bulk scan of an existing corpus
must adopt each file's age instead of stamping the moment the directory
happened to be walked.  Upload/text keep ``now()`` — their arrival time *is*
their age.

The ordering consequence of the supplied timestamp is covered in
``tests/pipeline/test_enqueue_created_at.py``; here we pin the wiring: which
enqueue path supplies it, and what happens when the stat fails.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

pipeline_enqueue_file = _document_routes.pipeline_enqueue_file

pytestmark = pytest.mark.offline

# 2019-08-07 06:05:04 UTC — safely in the past, so a wrongly-stamped row is
# never mistaken for a correctly-stamped one.
_MTIME_EPOCH = 1565157904
_MTIME_ISO = "2019-08-07T06:05:04+00:00"


def _recording_rag() -> tuple[SimpleNamespace, dict]:
    captured: dict = {}

    async def fake_enqueue(content, **kwargs):
        captured.update(kwargs)
        return "track-xyz"  # non-None → success path

    async def fake_error(files, track_id):
        captured["error_files"] = files

    rag = SimpleNamespace(
        addon_params={},
        apipeline_enqueue_documents=fake_enqueue,
        apipeline_enqueue_error_documents=fake_error,
    )
    return rag, captured


def _aged_file(tmp_path: Path, name: str = "corpus.txt") -> Path:
    file_path = tmp_path / name
    file_path.write_text("hello world", encoding="utf-8")
    os.utime(file_path, (_MTIME_EPOCH, _MTIME_EPOCH))
    return file_path


def test_scan_enqueue_uses_the_file_mtime_as_created_at(tmp_path):
    rag, captured = _recording_rag()

    ok, _track = asyncio.run(
        pipeline_enqueue_file(rag, _aged_file(tmp_path), from_scan=True)
    )

    assert ok is True
    assert "error_files" not in captured
    assert captured["created_at"] == _MTIME_ISO


def test_upload_enqueue_leaves_created_at_to_the_pipeline(tmp_path):
    """An uploaded file's mtime is the moment it was written to disk, which is
    already ``now()`` — passing it would only add a rounding difference, and
    §8.5 scopes the mtime rule to scan."""
    rag, captured = _recording_rag()

    ok, _track = asyncio.run(
        pipeline_enqueue_file(rag, _aged_file(tmp_path), from_scan=False)
    )

    assert ok is True
    assert "created_at" not in captured


def test_scan_enqueue_survives_a_vanished_file(tmp_path):
    """The file may disappear between discovery and enqueue.  A missing mtime
    must cost us the timestamp, not the row: the enqueue proceeds and the
    pipeline stamps ``now()``."""
    rag, captured = _recording_rag()

    ok, _track = asyncio.run(
        pipeline_enqueue_file(rag, tmp_path / "gone.txt", from_scan=True)
    )

    assert ok is True
    assert "created_at" not in captured


def test_scan_created_at_is_the_mtime_not_the_scan_time(tmp_path):
    """Fix-proof for the ordering defect: before Phase 4-e every scanned file
    was stamped with the scan's own wall clock, so a decade-old corpus sorted
    behind whatever was uploaded a second earlier."""
    rag, captured = _recording_rag()

    started = datetime.now(timezone.utc)
    asyncio.run(pipeline_enqueue_file(rag, _aged_file(tmp_path), from_scan=True))

    assert datetime.fromisoformat(captured["created_at"]) < started
