"""``/documents`` DELETE (clear all): __parsed__ is preserved by default, and
only removed when the caller passes delete_parsed_files=True.

Preserving __parsed__ by default lets re-adding the same file skip
re-parsing and keeps a deleted document's raw upload recoverable -- the
same reasoning already documented for the per-document delete_file flag.
"""

import asyncio
import importlib
import shutil
import sys
import threading
from unittest.mock import patch
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.constants import PARSED_DIR_NAME  # noqa: E402

DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


class _NoopStorage:
    namespace = "noop"

    async def drop(self):
        return {"status": "success", "message": "data dropped"}


class _ClearRag:
    def __init__(self, workspace: str):
        self.workspace = workspace
        storage = _NoopStorage()
        storage.workspace = workspace
        self.text_chunks = storage
        self.full_docs = storage
        self.full_entities = storage
        self.full_relations = storage
        self.entity_chunks = storage
        self.relation_chunks = storage
        self.entities_vdb = storage
        self.relationships_vdb = storage
        self.chunks_vdb = storage
        self.chunk_entity_relation_graph = storage
        self.doc_status = storage

    async def aclear_cache(self, modes=None):
        return None


def _clear_endpoint(rag, input_dir):
    router = create_document_routes(rag, DocumentManager(str(input_dir)))
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "clear_documents"
    ][-1]


async def _init_workspace(workspace):
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=workspace)


async def test_clear_documents_preserves_parsed_dir_by_default(tmp_path):
    workspace = f"clear-parsed-default-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "a.parsed.json").write_text("{}")

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, tmp_path)

    response = await endpoint()

    assert response.status in ("success", "partial_success")
    assert parsed_dir.exists()
    assert (parsed_dir / "a.parsed.json").exists()
    assert "__parsed__ preserved" in response.message
    assert "delete_parsed_files=true" in response.message


async def test_clear_documents_deletes_parsed_dir_when_opted_in(tmp_path):
    workspace = f"clear-parsed-optin-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "a.parsed.json").write_text("{}")

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, tmp_path)

    response = await endpoint(delete_parsed_files=True)

    assert response.status in ("success", "partial_success")
    assert not parsed_dir.exists()
    assert "__parsed__" in response.message


async def test_clear_documents_opt_in_is_a_noop_without_parsed_dir(tmp_path):
    workspace = f"clear-parsed-missing-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, tmp_path)

    response = await endpoint(delete_parsed_files=True)

    assert response.status in ("success", "partial_success")


async def test_clear_documents_deletes_parsed_dir_off_the_event_loop_thread(tmp_path):
    """__parsed__ can hold many files; shutil.rmtree on it must not block
    the event loop for the duration of a large recursive delete."""
    workspace = f"clear-parsed-thread-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "a.parsed.json").write_text("{}")

    main_thread_id = threading.get_ident()
    call_thread_id = {}
    real_rmtree = shutil.rmtree

    def fake_rmtree(path, *args, **kwargs):
        call_thread_id["id"] = threading.get_ident()
        return real_rmtree(path, *args, **kwargs)

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, tmp_path)

    with patch.object(shutil, "rmtree", side_effect=fake_rmtree):
        response = await endpoint(delete_parsed_files=True)

    assert response.status in ("success", "partial_success")
    assert not parsed_dir.exists()
    assert call_thread_id["id"] != main_thread_id


async def test_clear_documents_cancel_defers_until_rmtree_finishes_before_releasing_lock(
    tmp_path,
):
    """A bare cancel (e.g. the client disconnecting) during the rmtree
    await must not let the finally block release destructive_busy while
    the delete is still running in the background -- that would let a new
    request race an in-flight __parsed__ deletion."""
    workspace = f"clear-parsed-cancel-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "a.parsed.json").write_text("{}")

    call_started = threading.Event()
    release_call = threading.Event()
    real_rmtree = shutil.rmtree

    def fake_rmtree(path, *args, **kwargs):
        call_started.set()
        release_call.wait(timeout=5)
        return real_rmtree(path, *args, **kwargs)

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, tmp_path)
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")

    with patch.object(shutil, "rmtree", side_effect=fake_rmtree):
        task = asyncio.ensure_future(endpoint(delete_parsed_files=True))
        for _ in range(500):
            if call_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert call_started.is_set()

        task.cancel()
        # Give the cancelled task a chance to re-suspend on the deferred
        # wait before checking the lock state.
        for _ in range(10):
            await asyncio.sleep(0.01)

        pipeline_status = await shared_storage.get_namespace_data(
            "pipeline_status", workspace=workspace
        )
        assert pipeline_status["destructive_busy"] is True

        release_call.set()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert not parsed_dir.exists()
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    assert pipeline_status["destructive_busy"] is False


async def test_clear_documents_default_message_silent_without_parsed_dir(tmp_path):
    workspace = f"clear-parsed-silent-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, tmp_path)

    response = await endpoint()

    assert response.status in ("success", "partial_success")
    assert "__parsed__" not in response.message
