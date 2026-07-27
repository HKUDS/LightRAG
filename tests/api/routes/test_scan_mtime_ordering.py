"""Scan mtime orders candidates without becoming ``doc_status.created_at``.

The scan's disposable disk spool globally orders new files by ``st_mtime`` before
enqueue. Once a file crosses into persistent doc_status, the filesystem timestamp
has no role: ``created_at`` is the actual first-write time, exactly as it is for
upload/text.

Global ordering and bounded batch reads are covered in
``test_scan_streaming_batches.py``. Here we pin the enqueue boundary: neither
scan nor upload is allowed to supply/forge a persistent creation timestamp.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

pipeline_enqueue_file = _document_routes.pipeline_enqueue_file

pytestmark = pytest.mark.offline

_MTIME_EPOCH = 1565157904


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


def test_scan_enqueue_does_not_supply_created_at(tmp_path):
    rag, captured = _recording_rag()

    ok, _track = asyncio.run(
        pipeline_enqueue_file(rag, _aged_file(tmp_path), from_scan=True)
    )

    assert ok is True
    assert "error_files" not in captured
    assert "created_at" not in captured


def test_upload_enqueue_leaves_created_at_to_the_pipeline(tmp_path):
    """Every ingress path lets the pipeline stamp the persistent row."""
    rag, captured = _recording_rag()

    ok, _track = asyncio.run(
        pipeline_enqueue_file(rag, _aged_file(tmp_path), from_scan=False)
    )

    assert ok is True
    assert "created_at" not in captured


def test_scan_enqueue_survives_a_vanished_file(tmp_path):
    """A file may disappear after spooling; enqueue still owns error handling."""
    rag, captured = _recording_rag()

    ok, _track = asyncio.run(
        pipeline_enqueue_file(rag, tmp_path / "gone.txt", from_scan=True)
    )

    assert ok is True
    assert "created_at" not in captured
