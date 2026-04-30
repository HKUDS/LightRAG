from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from lightrag_enterprise.little_bull.models import (
    LittleBullEmbeddingCostEstimateRequest,
    LittleBullKnowledgeBaseReindexRequest,
    LittleBullKnowledgeBaseRollbackRequest,
    LittleBullKnowledgeBaseUpsertRequest,
    LittleBullQueryRequest,
)
from lightrag_enterprise.little_bull.service import LittleBullService
from lightrag_enterprise.system import (
    AccessControlService,
    ApprovalService,
    AuditService,
    SystemAuthService,
    Workspace,
)
from lightrag_enterprise.system.policy_keys import PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY
from lightrag_enterprise.system.repositories import (
    InMemorySystemRepository,
    membership_for_master,
)


class FakeDocStatus:
    async def get_docs_paginated(self, **_kwargs):
        doc = SimpleNamespace(
            file_path="manual.pdf",
            status="processed",
            content_summary="Manual",
            content_length=42,
            updated_at="2026-04-27T00:00:00Z",
            created_at="2026-04-27T00:00:00Z",
            track_id="trk",
            chunks_count=1,
            metadata={},
        )
        return [("doc-1", doc)], 1

    async def get_all_status_counts(self):
        return {"processed": 1}


class FakeLlmCache:
    def __init__(self):
        self.global_config = {"enable_llm_cache": True}


class FakeDroppableStorage:
    def __init__(self, path: Path | None = None):
        self.dropped = False
        self.path = path

    async def drop(self):
        self.dropped = True
        if self.path and self.path.exists():
            if self.path.is_dir():
                for child in self.path.iterdir():
                    if child.is_file():
                        child.unlink()
            else:
                self.path.unlink()
        return {"status": "success", "message": "data dropped"}


class FakeRag:
    def __init__(
        self,
        *,
        llm_binding: str = "ollama",
        llm_model: str = "qwen-local",
        llm_host: str = "",
        working_dir: str = "./rag_storage",
    ):
        self.doc_status = FakeDocStatus()
        self.workspace = "default"
        self.working_dir = working_dir
        self.little_bull_llm_binding = llm_binding
        self.little_bull_llm_model = llm_model
        self.little_bull_llm_host = llm_host
        self.llm_response_cache = FakeLlmCache()
        self.query_calls = 0
        self.last_query_param = None
        self.cache_states_during_query: list[bool] = []

    def clone_for_workspace(self, workspace_id):
        clone = FakeRag(
            llm_binding=self.little_bull_llm_binding,
            llm_model=self.little_bull_llm_model,
            llm_host=self.little_bull_llm_host,
            working_dir=self.working_dir,
        )
        clone.workspace = workspace_id
        for storage_name in (
            "full_docs",
            "text_chunks",
            "full_entities",
            "full_relations",
            "entity_chunks",
            "relation_chunks",
            "entities_vdb",
            "relationships_vdb",
            "chunks_vdb",
            "chunk_entity_relation_graph",
            "doc_status",
        ):
            setattr(clone, storage_name, FakeDroppableStorage())
        clone.llm_response_cache = FakeDroppableStorage()
        return clone

    async def aquery_llm(self, query, param):
        self.query_calls += 1
        self.last_query_param = param
        self.cache_states_during_query.append(self.llm_response_cache.global_config["enable_llm_cache"])
        return {
            "llm_response": {"content": f"Answer for {query} in {param.mode}"},
            "data": {"references": [{"reference_id": "1", "file_path": "manual.pdf"}]},
        }


class FakeDocManager:
    def __init__(self, tmp_path: Path):
        self.input_dir = tmp_path

    def is_supported_file(self, _filename: str) -> bool:
        return True


class FakeAdminStore:
    def __init__(self):
        self.models: dict[str, dict] = {}

    async def list_model_settings(self, *, tenant_id: str | None, workspace_id: str | None):
        return [
            model
            for model in self.models.values()
            if (model.get("tenant_id") is None or model.get("tenant_id") == tenant_id)
            and (model.get("workspace_id") is None or model.get("workspace_id") == workspace_id)
        ]

    async def upsert_model_setting(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str | None,
        user_id: str,
    ):
        model_setting_id = payload.get("model_setting_id") or f"model-{len(self.models) + 1}"
        if payload.get("is_default"):
            for model in self.models.values():
                if (
                    model.get("usage") == payload.get("usage")
                    and model.get("tenant_id") == tenant_id
                    and model.get("workspace_id") == workspace_id
                ):
                    model["is_default"] = False
        row = {
            **payload,
            "model_setting_id": model_setting_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.models[model_setting_id] = row
        return row


async def _principal_and_service(tmp_path: Path, *, rag: FakeRag | None = None):
    repo = InMemorySystemRepository()
    auth = SystemAuthService(repo, secret="test-secret")
    user = await auth.bootstrap_master(username="master", password="secret123")
    await repo.create_membership(membership_for_master(user.user_id))
    principal = await auth.principal_for_user(user)
    base_rag = rag or FakeRag(working_dir=str(tmp_path / "rag_storage"))
    base_rag.working_dir = str(tmp_path / "rag_storage")
    service = LittleBullService(
        rag=base_rag,
        doc_manager=FakeDocManager(tmp_path),
        repository=repo,
        access=AccessControlService(),
        audit=AuditService(repo),
        approvals=ApprovalService(repo),
    )
    return principal, service


async def _principal_and_service_with_admin_store(tmp_path: Path):
    principal, service = await _principal_and_service(tmp_path)
    service.admin_store = FakeAdminStore()
    return principal, service


@pytest.mark.asyncio
async def test_little_bull_lists_documents_and_audits(tmp_path):
    principal, service = await _principal_and_service(tmp_path)

    response = await service.list_documents(principal, workspace_id="default")

    assert response.total_count == 1
    assert response.documents[0].title == "manual.pdf"
    assert (await service.list_activity(principal, workspace_id="default"))[0].result == "success"


@pytest.mark.asyncio
async def test_little_bull_query_blocks_hosted_profile_for_private_data(tmp_path):
    rag = FakeRag(llm_binding="openai", llm_model="gpt-hosted")
    principal, service = await _principal_and_service(tmp_path, rag=rag)

    with pytest.raises(Exception) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                query="private question",
                confidentiality="privado",
                model_profile="equilibrado",
            ),
        )

    assert "Private/local" in str(exc.value)
    assert rag.query_calls == 0
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    assert events[0].result == "blocked"
    assert events[0].metadata["reason"] == "private_local_required"


@pytest.mark.asyncio
async def test_little_bull_query_private_profile_unavailable_blocks_before_rag(tmp_path):
    rag = FakeRag(llm_binding="openai", llm_model="gpt-hosted")
    principal, service = await _principal_and_service(tmp_path, rag=rag)

    with pytest.raises(HTTPException) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                query="private question",
                confidentiality="privado",
                model_profile="privado",
            ),
        )

    assert exc.value.status_code == 503
    assert "Private/local model is unavailable" in exc.value.detail
    assert rag.query_calls == 0
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    assert events[0].result == "blocked"
    assert events[0].metadata["reason"] == "private_local_unavailable"


@pytest.mark.asyncio
async def test_little_bull_query_private_profile_uses_configured_local_model_and_disables_cache(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("LITTLE_BULL_PRIVATE_LOCAL_MODEL", "qwen-local")
    rag = FakeRag(llm_binding="openai", llm_model="gpt-hosted")
    principal, service = await _principal_and_service(tmp_path, rag=rag)

    response = await service.query(
        principal,
        LittleBullQueryRequest(
            workspace_id="default",
            query="private question",
            confidentiality="privado",
            model_profile="privado",
        ),
    )

    assert response.response.startswith("Answer for private question")
    assert rag.query_calls == 1
    assert rag.last_query_param.model_func is not None
    assert rag.cache_states_during_query == [False]
    assert rag.llm_response_cache.global_config["enable_llm_cache"] is True
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    assert events[0].result == "success"
    assert events[1].result == "allowed"
    assert events[1].metadata["selected_model_id"] == "ollama/qwen-local"


@pytest.mark.asyncio
async def test_little_bull_query_blocks_hosted_profile_when_workspace_has_private_data(tmp_path):
    rag = FakeRag(llm_binding="openai", llm_model="gpt-hosted")
    principal, service = await _principal_and_service(tmp_path, rag=rag)
    await service.repository.set_policy(
        "little_bull.workspace_contains_private_data",
        True,
        tenant_id="default",
        workspace_id="default",
    )

    with pytest.raises(HTTPException) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                query="normal looking question",
                confidentiality="normal",
                model_profile="equilibrado",
            ),
        )

    assert exc.value.status_code == 403
    assert rag.query_calls == 0
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    assert events[0].metadata["workspace_contains_private_data"] is True


@pytest.mark.asyncio
async def test_little_bull_query_allows_master_policy_hosted_private_exception_and_audits(
    tmp_path,
):
    rag = FakeRag(
        llm_binding="openai",
        llm_model="openai/gpt-4o-mini",
        llm_host="https://openrouter.ai/api/v1",
    )
    principal, service = await _principal_and_service(tmp_path, rag=rag)
    await service.repository.set_policy(
        PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY,
        {
            "schema_version": 1,
            "enabled": True,
            "provider": "openrouter",
            "binding": "openai",
            "binding_host": "https://openrouter.ai/api/v1",
            "allowed_model_ids": ["openai/gpt-4o-mini"],
            "allowed_confidentiality": ["sensivel", "privado"],
            "expires_at": "2099-01-01T00:00:00Z",
            "approved_by": principal.user_id,
            "approved_at": "2026-04-28T00:00:00Z",
            "approval_id": "apr_openrouter_private",
            "reason": "MASTER approved OpenRouter for this private workspace.",
        },
        tenant_id="default",
        workspace_id="default",
    )

    response = await service.query(
        principal,
        LittleBullQueryRequest(
            workspace_id="default",
            query="private question",
            confidentiality="privado",
            model_profile="equilibrado",
        ),
    )

    assert response.response.startswith("Answer for private question")
    assert rag.query_calls == 1
    assert rag.last_query_param.model_func is None
    assert rag.cache_states_during_query == [False]
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    success = next(event for event in events if event.result == "success")
    allowed = next(event for event in events if event.result == "allowed")
    success_gateway = success.metadata["private_gateway"]
    allowed_gateway = allowed.metadata
    assert success_gateway["hosted_private_exception"] is True
    assert success_gateway["hosted_private_provider"] == "openrouter"
    assert success_gateway["hosted_private_approval_id"] == "apr_openrouter_private"
    assert success_gateway["hosted_private_policy_status"] == "valid"
    assert success_gateway["requires_private_runtime"] is False
    assert success_gateway["cache_disabled"] is True
    assert allowed_gateway["hosted_private_exception"] is True
    assert allowed_gateway["cache_disabled"] is True


@pytest.mark.asyncio
async def test_little_bull_query_policy_does_not_allow_private_profile_hosted_bypass(
    tmp_path,
):
    rag = FakeRag(
        llm_binding="openai",
        llm_model="openai/gpt-4o-mini",
        llm_host="https://openrouter.ai/api/v1",
    )
    principal, service = await _principal_and_service(tmp_path, rag=rag)
    await service.repository.set_policy(
        PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY,
        {
            "schema_version": 1,
            "enabled": True,
            "provider": "openrouter",
            "binding": "openai",
            "binding_host": "https://openrouter.ai/api/v1",
            "allowed_model_ids": ["openai/gpt-4o-mini"],
            "allowed_confidentiality": ["sensivel", "privado"],
            "expires_at": "2099-01-01T00:00:00Z",
            "approved_by": principal.user_id,
            "approved_at": "2026-04-28T00:00:00Z",
            "reason": "MASTER approved OpenRouter for hosted profiles only.",
        },
        tenant_id="default",
        workspace_id="default",
    )

    with pytest.raises(HTTPException) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                query="private question",
                confidentiality="privado",
                model_profile="privado",
            ),
        )

    assert exc.value.status_code == 503
    assert rag.query_calls == 0


@pytest.mark.asyncio
async def test_little_bull_query_expired_hosted_private_policy_fails_closed(tmp_path):
    rag = FakeRag(
        llm_binding="openai",
        llm_model="openai/gpt-4o-mini",
        llm_host="https://openrouter.ai/api/v1",
    )
    principal, service = await _principal_and_service(tmp_path, rag=rag)
    await service.repository.set_policy(
        PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY,
        {
            "schema_version": 1,
            "enabled": True,
            "provider": "openrouter",
            "binding": "openai",
            "binding_host": "https://openrouter.ai/api/v1",
            "allowed_model_ids": ["openai/gpt-4o-mini"],
            "allowed_confidentiality": ["sensivel", "privado"],
            "expires_at": "2020-01-01T00:00:00Z",
            "approved_by": principal.user_id,
            "approved_at": "2026-04-28T00:00:00Z",
            "reason": "Expired policy should not authorize hosted private data.",
        },
        tenant_id="default",
        workspace_id="default",
    )

    with pytest.raises(HTTPException) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                query="private question",
                confidentiality="privado",
                model_profile="equilibrado",
            ),
        )

    assert exc.value.status_code == 403
    assert rag.query_calls == 0


@pytest.mark.asyncio
async def test_little_bull_blocks_unbacked_workspace_data_plane(tmp_path):
    principal, service = await _principal_and_service(tmp_path)
    await service.repository.create_workspace(
        Workspace(
            workspace_id="other",
            tenant_id="default",
            name="Other",
            slug="other",
        )
    )
    principal = await SystemAuthService(service.repository, secret="test-secret").principal_for_user(
        await service.repository.get_user(principal.user_id)
    )

    with pytest.raises(HTTPException) as exc:
        await service.list_documents(principal, workspace_id="other")

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_little_bull_delete_creates_pending_approval(tmp_path):
    principal, service = await _principal_and_service(tmp_path)

    response = await service.delete_document(
        principal,
        workspace_id="default",
        document_id="doc-1",
    )

    assert response["status"] == "pending_approval"
    assert response["approval"]["status"] == "pending"


@pytest.mark.asyncio
async def test_little_bull_delete_reuses_pending_approval(tmp_path):
    principal, service = await _principal_and_service(tmp_path)

    first = await service.delete_document(principal, workspace_id="default", document_id="doc-1")
    second = await service.delete_document(principal, workspace_id="default", document_id="doc-1")
    approvals = await service.approvals.list(tenant_id="default", workspace_id="default")

    assert first["approval"]["approval_id"] == second["approval"]["approval_id"]
    assert len(approvals) == 1


@pytest.mark.asyncio
async def test_little_bull_reindex_archived_copies_files_and_queues(tmp_path):
    principal, service = await _principal_and_service(tmp_path)
    archived_dir = tmp_path / "__enqueued__"
    archived_dir.mkdir()
    source = archived_dir / "manual.md"
    source.write_text("# Manual\n\nConteudo da base.", encoding="utf-8")
    queued: dict[str, object] = {}

    def fake_queue(_background_tasks, copied_paths, track_id, *, rag):
        queued["paths"] = copied_paths
        queued["track_id"] = track_id
        queued["workspace"] = rag.workspace

    service._queue_pipeline_index_files = fake_queue

    response = await service.reindex_archived_documents(
        principal,
        workspace_id="default",
        background_tasks=BackgroundTasks(),
    )

    assert response.status == "queued"
    assert response.recovered_count == 1
    assert source.exists()
    assert (tmp_path / "manual.md").exists()
    assert queued["track_id"] == response.track_id
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    assert events[0].result == "reindex_queued"


@pytest.mark.asyncio
async def test_little_bull_lists_embedding_catalog_and_estimates_cost(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)

    catalog = await service.list_embedding_catalog(principal)
    qwen = next(model for model in catalog if model.model_id == "qwen/qwen3-embedding-8b")
    estimate = await service.estimate_embedding_cost_for_workspace(
        principal,
        request=LittleBullEmbeddingCostEstimateRequest(
            workspace_id="default",
            model_id=qwen.model_id,
            estimated_tokens=200_000,
        ),
    )

    assert qwen.prompt_cost_per_million_tokens == 0.01
    assert estimate.estimated_cost_usd == 0.002
    assert estimate.recommended_chunk_tokens == 3000


@pytest.mark.asyncio
async def test_little_bull_upserts_knowledge_base_with_default_embedding_and_reindex_flag(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)

    base = await service.upsert_knowledge_base(
        principal,
        LittleBullKnowledgeBaseUpsertRequest(
            name="Jurídico",
            slug="juridico",
            description="Base jurídica",
            embedding_model_id="baai/bge-m3",
            estimated_tokens=120_000,
        ),
    )

    assert base.workspace_id == "juridico"
    assert base.embedding_model is not None
    assert base.embedding_model.model_id == "baai/bge-m3"
    assert base.embedding_reindex_required is True
    assert base.embedding_model.config["estimated_reindex_cost_usd"] == 0.0012
    assert base.data_plane_attached is False


@pytest.mark.asyncio
async def test_little_bull_attaches_knowledge_base_data_plane(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    await service.upsert_knowledge_base(
        principal,
        LittleBullKnowledgeBaseUpsertRequest(name="Artigos", slug="artigos"),
    )

    response = await service.attach_knowledge_base_data_plane(
        principal,
        workspace_id="artigos",
    )
    bases = await service.list_knowledge_bases(principal)
    base = next(item for item in bases if item.workspace_id == "artigos")

    assert response.status == "attached"
    assert response.data_plane_attached is True
    assert base.data_plane_attached is True
    assert service.rag._little_bull_workspace_rags["artigos"].workspace == "artigos"


@pytest.mark.asyncio
async def test_little_bull_reindex_knowledge_base_requests_approval(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    await service.upsert_knowledge_base(
        principal,
        LittleBullKnowledgeBaseUpsertRequest(name="Artigos", slug="artigos"),
    )
    await service.attach_knowledge_base_data_plane(principal, workspace_id="artigos")

    response = await service.reindex_knowledge_base(
        principal,
        workspace_id="artigos",
        request=LittleBullKnowledgeBaseReindexRequest(),
        background_tasks=BackgroundTasks(),
    )

    assert response.status == "pending_approval"
    assert response.approval is not None
    assert response.approval["action"] == "little_bull.documents.reindex"


@pytest.mark.asyncio
async def test_little_bull_reindex_approval_rejects_payload_drift(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    await service.upsert_knowledge_base(
        principal,
        LittleBullKnowledgeBaseUpsertRequest(name="Artigos", slug="artigos"),
    )
    await service.attach_knowledge_base_data_plane(principal, workspace_id="artigos")
    response = await service.reindex_knowledge_base(
        principal,
        workspace_id="artigos",
        request=LittleBullKnowledgeBaseReindexRequest(
            include_archived=True,
            include_input_root=True,
        ),
        background_tasks=BackgroundTasks(),
    )
    approval_id = response.approval["approval_id"]
    await service.approvals.approve(approval_id, principal)

    with pytest.raises(HTTPException) as exc:
        await service.reindex_knowledge_base(
            principal,
            workspace_id="artigos",
            request=LittleBullKnowledgeBaseReindexRequest(
                approval_id=approval_id,
                include_archived=False,
                include_input_root=True,
            ),
            background_tasks=BackgroundTasks(),
        )

    assert exc.value.status_code == 409
    assert "approved payload" in exc.value.detail


@pytest.mark.asyncio
async def test_little_bull_reindex_knowledge_base_queues_workspace_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("LITTLE_BULL_APPROVALS_ENFORCED", "false")
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    await service.upsert_knowledge_base(
        principal,
        LittleBullKnowledgeBaseUpsertRequest(
            name="Artigos",
            slug="artigos",
            embedding_model_id="baai/bge-m3",
        ),
    )
    await service.attach_knowledge_base_data_plane(principal, workspace_id="artigos")
    workspace_dir = tmp_path / "artigos"
    workspace_dir.mkdir(exist_ok=True)
    (workspace_dir / "fonte.md").write_text("# Fonte\n\nConteudo.", encoding="utf-8")
    archived_dir = workspace_dir / "__enqueued__"
    archived_dir.mkdir()
    (archived_dir / "antigo.md").write_text("# Antigo\n\nConteudo.", encoding="utf-8")
    queued: dict[str, object] = {}

    def fake_queue(_background_tasks, copied_paths, track_id, *, rag):
        queued["paths"] = copied_paths
        queued["track_id"] = track_id
        queued["workspace"] = rag.workspace

    service._queue_pipeline_index_files = fake_queue

    response = await service.reindex_knowledge_base(
        principal,
        workspace_id="artigos",
        request=LittleBullKnowledgeBaseReindexRequest(),
        background_tasks=BackgroundTasks(),
    )

    assert response.status == "queued"
    assert response.queued_count == 2
    assert queued["workspace"] == "artigos"
    assert "fonte.md" in response.files
    assert any(file.startswith("antigo") for file in response.files)
    settings = await service.admin_store.list_model_settings(
        tenant_id="default",
        workspace_id="artigos",
    )
    embedding = next(model for model in settings if model["usage"] == "embedding")
    assert embedding["config"]["last_reindex_track_id"] == response.track_id


@pytest.mark.asyncio
async def test_little_bull_destructive_rebuild_requires_approval_even_when_approvals_disabled(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("LITTLE_BULL_APPROVALS_ENFORCED", "false")
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    await service.upsert_knowledge_base(
        principal,
        LittleBullKnowledgeBaseUpsertRequest(name="Artigos", slug="artigos"),
    )
    await service.attach_knowledge_base_data_plane(principal, workspace_id="artigos")

    response = await service.reindex_knowledge_base(
        principal,
        workspace_id="artigos",
        request=LittleBullKnowledgeBaseReindexRequest(destructive_rebuild=True),
        background_tasks=BackgroundTasks(),
    )

    assert response.status == "pending_approval"
    assert response.approval is not None
    assert response.approval["metadata"]["destructive_rebuild"] is True


@pytest.mark.asyncio
async def test_little_bull_destructive_rebuild_snapshots_storage_and_queues_sources(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("LITTLE_BULL_APPROVALS_ENFORCED", "false")
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    await service.upsert_knowledge_base(
        principal,
        LittleBullKnowledgeBaseUpsertRequest(
            name="Artigos",
            slug="artigos",
            embedding_model_id="baai/bge-m3",
        ),
    )
    await service.attach_knowledge_base_data_plane(principal, workspace_id="artigos")
    input_dir = tmp_path / "artigos"
    input_dir.mkdir(exist_ok=True)
    (input_dir / "fonte.md").write_text("# Fonte\n\nConteudo.", encoding="utf-8")
    storage_dir = tmp_path / "rag_storage" / "artigos"
    storage_dir.mkdir(parents=True, exist_ok=True)
    stored_doc = storage_dir / "kv_store_full_docs.json"
    stored_doc.write_text('{"old": {"content": "old embedding"}}', encoding="utf-8")
    workspace_rag = service.rag._little_bull_workspace_rags["artigos"]
    workspace_rag.full_docs = FakeDroppableStorage(stored_doc)
    queued: dict[str, object] = {}

    def fake_queue(_background_tasks, copied_paths, track_id, *, rag):
        queued["paths"] = copied_paths
        queued["track_id"] = track_id
        queued["workspace"] = rag.workspace

    service._queue_pipeline_index_files = fake_queue

    pending = await service.reindex_knowledge_base(
        principal,
        workspace_id="artigos",
        request=LittleBullKnowledgeBaseReindexRequest(destructive_rebuild=True),
        background_tasks=BackgroundTasks(),
    )
    approval_id = pending.approval["approval_id"]
    await service.approvals.approve(approval_id, principal)
    response = await service.reindex_knowledge_base(
        principal,
        workspace_id="artigos",
        request=LittleBullKnowledgeBaseReindexRequest(
            approval_id=approval_id,
            destructive_rebuild=True,
        ),
        background_tasks=BackgroundTasks(),
    )

    assert response.status == "queued"
    assert response.destructive_rebuild is True
    assert response.snapshot_id
    assert response.snapshot_path
    assert response.rollback_available is True
    assert queued["workspace"] == "artigos"
    assert response.files == ["fonte.md"]
    assert (Path(response.snapshot_path) / "storage" / "kv_store_full_docs.json").exists()
    assert workspace_rag.full_docs.dropped is True
    assert not stored_doc.exists()
    settings = await service.admin_store.list_model_settings(
        tenant_id="default",
        workspace_id="artigos",
    )
    embedding = next(model for model in settings if model["usage"] == "embedding")
    assert embedding["config"]["last_reindex_track_id"] == response.track_id
    assert embedding["config"]["last_reindex_file_count"] == 1


@pytest.mark.asyncio
async def test_little_bull_rollback_restores_snapshot_for_attached_workspace(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("LITTLE_BULL_APPROVALS_ENFORCED", "false")
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    await service.upsert_knowledge_base(
        principal,
        LittleBullKnowledgeBaseUpsertRequest(name="Artigos", slug="artigos"),
    )
    await service.attach_knowledge_base_data_plane(principal, workspace_id="artigos")
    input_dir = tmp_path / "artigos"
    input_dir.mkdir(exist_ok=True)
    (input_dir / "fonte.md").write_text("# Fonte\n\nConteudo.", encoding="utf-8")
    storage_dir = tmp_path / "rag_storage" / "artigos"
    storage_dir.mkdir(parents=True, exist_ok=True)
    stored_doc = storage_dir / "kv_store_full_docs.json"
    stored_doc.write_text('{"old": {"content": "old embedding"}}', encoding="utf-8")
    workspace_rag = service.rag._little_bull_workspace_rags["artigos"]
    workspace_rag.full_docs = FakeDroppableStorage(stored_doc)
    service._queue_pipeline_index_files = lambda *_args, **_kwargs: None
    pending = await service.reindex_knowledge_base(
        principal,
        workspace_id="artigos",
        request=LittleBullKnowledgeBaseReindexRequest(destructive_rebuild=True),
        background_tasks=BackgroundTasks(),
    )
    approval_id = pending.approval["approval_id"]
    await service.approvals.approve(approval_id, principal)
    rebuilt = await service.reindex_knowledge_base(
        principal,
        workspace_id="artigos",
        request=LittleBullKnowledgeBaseReindexRequest(
            approval_id=approval_id,
            destructive_rebuild=True,
        ),
        background_tasks=BackgroundTasks(),
    )
    (storage_dir / "new_index.json").write_text('{"new": true}', encoding="utf-8")

    response = await service.rollback_knowledge_base_snapshot(
        principal,
        workspace_id="artigos",
        request=LittleBullKnowledgeBaseRollbackRequest(snapshot_id=rebuilt.snapshot_id),
    )

    assert response.status == "restored"
    assert response.preserved_current_snapshot_id
    assert (storage_dir / "kv_store_full_docs.json").exists()
    assert not (storage_dir / "new_index.json").exists()
    assert "artigos" not in service.rag._little_bull_workspace_rags


@pytest.mark.asyncio
async def test_little_bull_reindex_pending_approval_reuse_is_payload_scoped(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("LITTLE_BULL_APPROVALS_ENFORCED", "true")
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    await service.upsert_knowledge_base(
        principal,
        LittleBullKnowledgeBaseUpsertRequest(name="Artigos", slug="artigos"),
    )
    await service.attach_knowledge_base_data_plane(principal, workspace_id="artigos")
    non_destructive = await service.reindex_knowledge_base(
        principal,
        workspace_id="artigos",
        request=LittleBullKnowledgeBaseReindexRequest(),
        background_tasks=BackgroundTasks(),
    )
    destructive = await service.reindex_knowledge_base(
        principal,
        workspace_id="artigos",
        request=LittleBullKnowledgeBaseReindexRequest(destructive_rebuild=True),
        background_tasks=BackgroundTasks(),
    )

    assert non_destructive.approval is not None
    assert destructive.approval is not None
    assert destructive.approval["approval_id"] != non_destructive.approval["approval_id"]
