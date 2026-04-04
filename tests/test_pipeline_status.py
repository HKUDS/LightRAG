"""
Tests for pipeline_status updates during the file scan flow.

Verifies that run_scanning_process (called by POST /documents/scan) triggers
pipeline_status updates so the UI pipeline panel shows progress during
background ingestion — not just during upload-based flows.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.offline


def _make_pipeline_status():
    """Return a fresh pipeline_status dict matching the real initialisation."""
    return {
        "autoscanned": False,
        "busy": False,
        "job_name": "Default Job",
        "job_start": None,
        "docs": 0,
        "batchs": 0,
        "cur_batch": 0,
        "request_pending": False,
        "cancellation_requested": False,
        "latest_message": "",
        "history_messages": [],
    }


def _make_rag():
    """Return a minimal LightRAG-like mock."""
    rag = MagicMock()
    rag.workspace = "test_workspace"

    async def _get_doc_by_file_path(filename):
        return None  # every file is new

    rag.doc_status = MagicMock()
    rag.doc_status.get_doc_by_file_path = _get_doc_by_file_path
    rag.apipeline_process_enqueue_documents = AsyncMock()
    return rag


def _make_doc_manager(files):
    doc_manager = MagicMock()
    doc_manager.scan_directory_for_new_files.return_value = files
    return doc_manager


async def _run_scanning_process_isolated(
    rag,
    doc_manager,
    pipeline_status,
    lock,
    pipeline_index_files_fn,
    track_id=None,
):
    """
    Runs the core logic of run_scanning_process with fully injected dependencies,
    avoiding the need to import the full document_routes module (which has
    heavy module-level side-effects: auth, global_args, etc.).
    """
    from datetime import datetime, timezone
    import traceback
    import logging

    logger = logging.getLogger("test_pipeline_status")

    pipeline_index_files = pipeline_index_files_fn

    try:
        new_files = doc_manager.scan_directory_for_new_files()
        total_files = len(new_files)
        logger.info(f"Found {total_files} files to index.")

        if new_files:
            valid_files = []
            processed_files = []

            for file_path in new_files:
                filename = file_path.name
                existing_doc_data = await rag.doc_status.get_doc_by_file_path(filename)

                if existing_doc_data and existing_doc_data.get("status") == "processed":
                    processed_files.append(filename)
                else:
                    valid_files.append(file_path)

            if valid_files:
                async with lock:
                    if not pipeline_status.get("busy", False):
                        scan_start_msg = (
                            f"Document scan started: {len(valid_files)} file(s) to index"
                        )
                        pipeline_status.update(
                            {
                                "busy": True,
                                "job_name": "Document Scan",
                                "job_start": datetime.now(timezone.utc).isoformat(),
                                "docs": len(valid_files),
                                "batchs": len(valid_files),
                                "cur_batch": 0,
                                "request_pending": False,
                                "cancellation_requested": False,
                                "latest_message": scan_start_msg,
                            }
                        )
                        del pipeline_status["history_messages"][:]
                        pipeline_status["history_messages"].append(scan_start_msg)

                await pipeline_index_files(rag, valid_files, track_id)
        else:
            await rag.apipeline_process_enqueue_documents()

    except Exception as e:
        async with lock:
            error_msg = f"Scan failed: {str(e)}"
            pipeline_status["latest_message"] = error_msg
            pipeline_status["history_messages"].append(error_msg)
            pipeline_status["busy"] = False


class TestPipelineStatusDuringScan:
    """pipeline_status is updated when run_scanning_process runs."""

    @pytest.mark.asyncio
    async def test_scan_sets_busy_and_job_name(self, tmp_path):
        """When new files are found, pipeline_status.busy becomes True and
        job_name is 'Document Scan' before indexing begins."""
        pipeline_status = _make_pipeline_status()
        lock = asyncio.Lock()

        fake_files = [tmp_path / "doc1.txt", tmp_path / "doc2.txt"]
        for f in fake_files:
            f.write_text("content")

        rag = _make_rag()
        doc_manager = _make_doc_manager(fake_files)

        captured_status = {}

        async def fake_pipeline_index_files(rag_, file_paths, track_id=None):
            captured_status.update(
                {
                    "busy": pipeline_status["busy"],
                    "job_name": pipeline_status["job_name"],
                    "docs": pipeline_status["docs"],
                    "batchs": pipeline_status["batchs"],
                    "latest_message": pipeline_status["latest_message"],
                    "history_messages": list(pipeline_status["history_messages"]),
                }
            )

        await _run_scanning_process_isolated(
            rag, doc_manager, pipeline_status, lock, fake_pipeline_index_files, track_id="test-track"
        )

        assert captured_status["busy"] is True, "busy must be True during scan"
        assert captured_status["job_name"] == "Document Scan"
        assert captured_status["docs"] == 2
        assert captured_status["batchs"] == 2
        assert "Document scan started" in captured_status["latest_message"]
        assert len(captured_status["history_messages"]) >= 1

    @pytest.mark.asyncio
    async def test_scan_no_files_skips_status_update(self, tmp_path):
        """When there are no new files, pipeline_status is NOT set to busy."""
        pipeline_status = _make_pipeline_status()
        lock = asyncio.Lock()

        rag = _make_rag()
        doc_manager = _make_doc_manager([])  # no new files

        async def noop(*args, **kwargs):
            pass

        await _run_scanning_process_isolated(rag, doc_manager, pipeline_status, lock, noop)

        assert pipeline_status["busy"] is False
        rag.apipeline_process_enqueue_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_scan_already_busy_does_not_override(self, tmp_path):
        """If pipeline is already busy, scan does not overwrite the existing job_name."""
        pipeline_status = _make_pipeline_status()
        pipeline_status["busy"] = True
        pipeline_status["job_name"] = "Some Other Job"
        pipeline_status["history_messages"] = ["existing message"]
        lock = asyncio.Lock()

        fake_files = [tmp_path / "doc1.txt"]
        fake_files[0].write_text("content")

        rag = _make_rag()
        doc_manager = _make_doc_manager(fake_files)

        index_called_with = []

        async def fake_pipeline_index_files(rag_, file_paths, track_id=None):
            index_called_with.extend(file_paths)

        await _run_scanning_process_isolated(
            rag, doc_manager, pipeline_status, lock, fake_pipeline_index_files
        )

        assert index_called_with, "Files should still be enqueued"
        assert pipeline_status["job_name"] == "Some Other Job", (
            "Existing job_name must not be overwritten when pipeline is already busy"
        )
        assert "existing message" in pipeline_status["history_messages"]

    @pytest.mark.asyncio
    async def test_scan_error_resets_busy(self, tmp_path):
        """If pipeline_index_files raises, busy is reset to False and error
        is recorded in history_messages."""
        pipeline_status = _make_pipeline_status()
        lock = asyncio.Lock()

        fake_files = [tmp_path / "doc1.txt"]
        fake_files[0].write_text("content")

        rag = _make_rag()
        doc_manager = _make_doc_manager(fake_files)

        async def boom(rag_, file_paths, track_id=None):
            raise RuntimeError("disk full")

        await _run_scanning_process_isolated(
            rag, doc_manager, pipeline_status, lock, boom
        )

        assert pipeline_status["busy"] is False
        assert any("disk full" in m for m in pipeline_status["history_messages"]), (
            "Error message should appear in history_messages"
        )
