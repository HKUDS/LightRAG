"""``/documents`` DELETE (clear all): __parsed__ is preserved by default, and
only removed when the caller passes delete_parsed_files=True.

Preserving __parsed__ by default lets re-adding the same file skip
re-parsing and keeps a deleted document's raw upload recoverable -- the
same reasoning already documented for the per-document delete_file flag.
"""

import importlib
import sys
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
    assert "__parsed__" not in response.message


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
