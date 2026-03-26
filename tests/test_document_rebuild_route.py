import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.prompt_version_store import PromptVersionStore

pytestmark = pytest.mark.offline


class _DummyLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummyStorage:
    def __init__(self, namespace: str, workspace: str = "demo"):
        self.namespace = namespace
        self.workspace = workspace
        self.drop_calls = 0

    async def drop(self):
        self.drop_calls += 1
        return {"status": "success", "namespace": self.namespace}


class _DummyRAG:
    def __init__(self, working_dir: str):
        self.workspace = "demo"
        self.prompt_version_store = PromptVersionStore(working_dir, workspace="demo")
        self.text_chunks = _DummyStorage("text_chunks")
        self.full_docs = _DummyStorage("full_docs")
        self.full_entities = _DummyStorage("full_entities")
        self.full_relations = _DummyStorage("full_relations")
        self.entity_chunks = _DummyStorage("entity_chunks")
        self.relation_chunks = _DummyStorage("relation_chunks")
        self.entities_vdb = _DummyStorage("entities_vdb")
        self.relationships_vdb = _DummyStorage("relationships_vdb")
        self.chunks_vdb = _DummyStorage("chunks_vdb")
        self.chunk_entity_relation_graph = _DummyStorage("chunk_entity_relation_graph")
        self.llm_response_cache = _DummyStorage("llm_response_cache")
        self.doc_status = _DummyStorage("doc_status")


def test_rebuild_endpoint_preserves_source_files_and_starts_scan(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    document_routes = importlib.import_module("lightrag.api.routers.document_routes")
    from lightrag.api.routers.document_routes import DocumentManager
    from lightrag.kg import shared_storage

    input_root = tmp_path / "inputs"
    doc_manager = DocumentManager(str(input_root), workspace="demo")
    source_dir = doc_manager.input_dir
    source_dir.mkdir(parents=True, exist_ok=True)
    active_file = source_dir / "root.txt"
    active_file.write_text("root source", encoding="utf-8")
    enqueued_dir = source_dir / "__enqueued__"
    enqueued_dir.mkdir()
    archived_file = enqueued_dir / "archived.txt"
    archived_file.write_text("archived source", encoding="utf-8")

    rag = _DummyRAG(str(tmp_path))
    seeded = rag.prompt_version_store.initialize(locale="en")
    version_id = seeded["indexing"]["versions"][0]["version_id"]

    pipeline_status = {"busy": False, "history_messages": [], "latest_message": ""}
    scan_calls: list[tuple[str | None, list[str]]] = []

    async def _fake_get_namespace_data(*args, **kwargs):
        return pipeline_status

    def _fake_get_namespace_lock(*args, **kwargs):
        return _DummyLock()

    async def _fake_run_scanning_process(rag_obj, manager, track_id=None):
        scan_calls.append(
            (
                track_id,
                sorted(path.name for path in manager.input_dir.glob("*") if path.is_file()),
            )
        )

    monkeypatch.setattr(
        document_routes, "get_combined_auth_dependency", lambda *_: (lambda: None)
    )
    monkeypatch.setattr(shared_storage, "get_namespace_data", _fake_get_namespace_data)
    monkeypatch.setattr(shared_storage, "get_namespace_lock", _fake_get_namespace_lock)
    monkeypatch.setattr(document_routes, "run_scanning_process", _fake_run_scanning_process)
    monkeypatch.setattr(document_routes, "generate_track_id", lambda prefix: f"{prefix}-123")

    app = FastAPI()
    app.include_router(document_routes.create_document_routes(rag, doc_manager))
    client = TestClient(app)

    response = client.post(
        "/documents/rebuild_from_indexing_version", json={"version_id": version_id}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rebuild_started"
    assert body["track_id"] == "rebuild-123"
    assert rag.prompt_version_store.list_versions("indexing")["active_version_id"] == version_id
    assert rag.llm_response_cache.drop_calls == 1
    assert rag.doc_status.drop_calls == 1
    assert scan_calls == [("rebuild-123", ["archived.txt", "root.txt"])]
    remaining_files = sorted(
        path.name for path in source_dir.rglob("*") if path.is_file()
    )
    assert remaining_files == ["archived.txt", "root.txt"]
