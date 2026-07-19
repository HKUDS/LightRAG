"""Regression coverage for API workspace control-plane isolation."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest


@pytest.fixture(autouse=True)
def _initialize_api_config():
    """Avoid FastAPI module imports parsing pytest's command-line arguments."""

    from lightrag.api import config

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        config.initialize_config(config.parse_args(), force=True)
        yield
    finally:
        sys.argv = original_argv
        config._global_args = None
        config._initialized = False


class _DropStorage:
    async def drop(self):
        return {"status": "success"}


class _FakeRag:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.initialized = False
        self.finalized = False
        for name in (
            "text_chunks",
            "full_docs",
            "full_entities",
            "full_relations",
            "entity_chunks",
            "relation_chunks",
            "entities_vdb",
            "relationships_vdb",
            "chunks_vdb",
            "chunk_entity_relation_graph",
            "llm_response_cache",
            "doc_status",
        ):
            setattr(self, name, _DropStorage())

    async def initialize_storages(self):
        self.initialized = True

    async def finalize_storages(self):
        self.finalized = True


@pytest.mark.asyncio
async def test_registry_and_runtime_keep_workspace_values_independent(tmp_path: Path):
    from lightrag.api.routers.document_routes import DocumentManager
    from lightrag.api.workspaces import (
        InMemoryWorkspaceRegistry,
        WorkspaceRuntimeManager,
    )

    inputs = tmp_path / "inputs"
    default_rag = _FakeRag("")
    manager = WorkspaceRuntimeManager(
        registry=InMemoryWorkspaceRegistry(),
        default_rag=default_rag,
        default_doc_manager=DocumentManager(str(inputs)),
        rag_factory=_FakeRag,
        input_dir=inputs,
    )

    default = await manager.initialize()
    created = await manager.create_workspace("Finance knowledge base", "FY2025")
    context = await manager.get_context(created.id)

    assert default.is_default is True
    assert default.storage_key == ""
    assert context.workspace.storage_key.startswith("ws_")
    assert context.rag.workspace == context.workspace.storage_key
    assert context.doc_manager.input_dir == inputs / context.workspace.storage_key

    token = manager.activate(context)
    try:
        assert manager.rag_proxy().workspace == context.workspace.storage_key
        assert (
            manager.document_manager_proxy().workspace == context.workspace.storage_key
        )
    finally:
        manager.deactivate(token)

    assert manager.rag_proxy().workspace == ""


@pytest.mark.asyncio
async def test_workspace_delete_drops_data_and_only_its_file_prefix(tmp_path: Path):
    from lightrag.api.routers.document_routes import DocumentManager
    from lightrag.api.workspaces import (
        InMemoryWorkspaceRegistry,
        WorkspaceRuntimeManager,
    )

    inputs = tmp_path / "inputs"
    manager = WorkspaceRuntimeManager(
        registry=InMemoryWorkspaceRegistry(),
        default_rag=_FakeRag(""),
        default_doc_manager=DocumentManager(str(inputs)),
        rag_factory=_FakeRag,
        input_dir=inputs,
    )
    await manager.initialize()
    workspace = await manager.create_workspace("Temporary workspace")
    workspace_dir = inputs / workspace.storage_key
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "source.pdf").write_bytes(b"placeholder")

    _, job_id = await manager.request_workspace_deletion(workspace.id)
    await manager.delete_workspace(workspace.id, job_id)

    deleted = await manager.registry.get(workspace.id, include_deleted=True)
    assert deleted.status == "deleted"
    assert not workspace_dir.exists()
    assert inputs.exists()


def test_workspace_header_is_exposed_in_openapi_and_selects_runtime(tmp_path: Path):
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    from lightrag.api.routers.document_routes import DocumentManager
    from lightrag.api.workspaces import (
        WORKSPACE_HEADER,
        InMemoryWorkspaceRegistry,
        WorkspaceRuntimeManager,
        bind_workspace_context,
    )

    inputs = tmp_path / "inputs"
    default_rag = _FakeRag("")
    manager = WorkspaceRuntimeManager(
        registry=InMemoryWorkspaceRegistry(),
        default_rag=default_rag,
        default_doc_manager=DocumentManager(str(inputs)),
        rag_factory=_FakeRag,
        input_dir=inputs,
    )
    asyncio.run(manager.initialize())
    workspace = asyncio.run(manager.create_workspace("Header selected"))

    app = FastAPI()
    app.state.workspace_manager = manager
    proxy = manager.rag_proxy()

    @app.get("/probe", dependencies=[Depends(bind_workspace_context)])
    async def probe():
        return {"workspace": proxy.workspace}

    with TestClient(app) as client:
        default_response = client.get("/probe")
        selected_response = client.get(
            "/probe", headers={WORKSPACE_HEADER: workspace.id}
        )
        schema = client.get("/openapi.json").json()

    assert default_response.json() == {"workspace": ""}
    assert selected_response.json() == {"workspace": workspace.storage_key}
    parameters = schema["paths"]["/probe"]["get"]["parameters"]
    assert any(
        parameter["in"] == "header" and parameter["name"] == WORKSPACE_HEADER
        for parameter in parameters
    )


def test_registry_backend_follows_configured_database(monkeypatch: pytest.MonkeyPatch):
    from lightrag.api.workspaces import (
        MongoWorkspaceRegistry,
        PostgresWorkspaceRegistry,
        create_workspace_registry,
    )

    monkeypatch.delenv("LIGHTRAG_WORKSPACE_REGISTRY_BACKEND", raising=False)
    postgres = create_workspace_registry(
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="Neo4JStorage",
    )
    mongo = create_workspace_registry(
        kv_storage="MongoKVStorage",
        vector_storage="MongoVectorDBStorage",
        doc_status_storage="MongoDocStatusStorage",
        graph_storage="MongoGraphStorage",
    )

    assert isinstance(postgres, PostgresWorkspaceRegistry)
    assert isinstance(mongo, MongoWorkspaceRegistry)


def test_registry_never_falls_back_to_sqlite(monkeypatch: pytest.MonkeyPatch):
    from lightrag.api.workspaces import WorkspaceStateError, create_workspace_registry

    monkeypatch.delenv("LIGHTRAG_WORKSPACE_REGISTRY_BACKEND", raising=False)
    with pytest.raises(WorkspaceStateError, match="SQLite is not supported"):
        create_workspace_registry(
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            doc_status_storage="JsonDocStatusStorage",
            graph_storage="NetworkXStorage",
        )
