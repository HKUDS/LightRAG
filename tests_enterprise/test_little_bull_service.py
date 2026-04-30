import asyncio
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException, UploadFile

from lightrag_enterprise.little_bull.models import (
    LittleBullBacklinkRequest,
    LittleBullCanvasBoardRequest,
    LittleBullCanvasEdgeRequest,
    LittleBullCanvasNodeRequest,
    LittleBullContentMapRequest,
    LittleBullConversationSaveRequest,
    LittleBullContextEstimateRequest,
    LittleBullCuratorSuggestionRequest,
    LittleBullDailyNoteRequest,
    LittleBullEmbeddingCostEstimateRequest,
    LittleBullAgentBuilderPublishRequest,
    LittleBullAgentBuilderSessionRequest,
    LittleBullAgentConfig,
    LittleBullAgentContextBudgetRequest,
    LittleBullInboxItemRequest,
    LittleBullInboxItemStatusRequest,
    LittleBullKnowledgeBaseReindexRequest,
    LittleBullKnowledgeBaseRollbackRequest,
    LittleBullKnowledgeBaseUpsertRequest,
    LittleBullKnowledgeGroupRequest,
    LittleBullKnowledgeSubgroupRequest,
    LittleBullKnowledgeTrailRequest,
    LittleBullKnowledgeTrailStepRequest,
    LittleBullMarkdownNoteRequest,
    LittleBullModelSetting,
    LittleBullOperationalChatRequest,
    LittleBullQueryRequest,
    LittleBullSourceProvenanceRequest,
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
from lightrag_enterprise.system.models import utc_now


class FakeDocStatus:
    def __init__(self, docs: list[tuple[str, SimpleNamespace]] | None = None):
        self.docs = docs

    async def get_docs_paginated(self, **_kwargs):
        if self.docs is None:
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
        return self.docs, len(self.docs)

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
        docs: list[tuple[str, SimpleNamespace]] | None = None,
    ):
        self.doc_status = FakeDocStatus(docs)
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


class SlowFakeRag(FakeRag):
    async def aquery_llm(self, query, param):
        await asyncio.sleep(0.05)
        return await super().aquery_llm(query, param)


class FakeDocManager:
    def __init__(self, tmp_path: Path):
        self.input_dir = tmp_path

    def is_supported_file(self, _filename: str) -> bool:
        return True


class FakeAdminStore:
    def __init__(self):
        self.models: dict[str, dict] = {}
        self.agents: dict[str, dict] = {}
        self.agent_builder_sessions: dict[str, dict] = {}
        self.agent_context_budgets: dict[str, dict] = {}
        self.usage_ledger: list[dict] = []
        self.budget_lock = asyncio.Lock()
        self.groups: dict[str, dict] = {}
        self.subgroups: dict[str, dict] = {}
        self.documents: dict[str, dict] = {}
        self.notes: dict[str, dict] = {}
        self.markdown_notes: dict[str, list[dict]] = {}
        self.wiki_links: dict[str, list[dict]] = {}
        self.tags: dict[str, dict] = {}
        self.backlinks: dict[str, dict] = {}
        self.provenance: dict[str, dict] = {}
        self.canvas_boards: dict[str, dict] = {}
        self.canvas_nodes: dict[str, dict] = {}
        self.canvas_edges: dict[str, dict] = {}
        self.dossiers: dict[str, dict] = {}
        self.content_maps: dict[str, dict] = {}
        self.trails: dict[str, dict] = {}
        self.trail_steps: dict[str, dict] = {}
        self.inbox_items: dict[str, dict] = {}
        self.daily_notes: dict[str, dict] = {}
        self.conversations: dict[str, dict] = {}
        self.suggestions: dict[str, dict] = {}

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

    async def list_agent_configs(self, *, tenant_id: str | None, workspace_id: str | None):
        return [
            agent
            for agent in self.agents.values()
            if agent["tenant_id"] == tenant_id and agent["workspace_id"] == workspace_id
        ]

    async def upsert_agent_config(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str | None,
        user_id: str,
    ):
        agent_id = payload.get("agent_id") or f"agent-{len(self.agents) + 1}"
        row = {
            "agent_id": agent_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "name": payload["name"],
            "description": payload.get("description", ""),
            "enabled": payload.get("enabled", True),
            "model_setting_id": payload.get("model_setting_id"),
            "system_prompt": payload.get("system_prompt", ""),
            "response_rules": payload.get("response_rules", []),
            "tools": payload.get("tools", []),
            "config": payload.get("config", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.agents[agent_id] = row
        return row

    async def list_agent_builder_sessions(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str | None = None,
        status: str | None = None,
    ):
        return [
            session
            for session in self.agent_builder_sessions.values()
            if session["tenant_id"] == tenant_id
            and session["workspace_id"] == workspace_id
            and (user_id is None or session["user_id"] == user_id)
            and (status is None or session["status"] == status)
        ]

    async def get_agent_builder_session(
        self,
        agent_builder_session_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
    ):
        session = self.agent_builder_sessions.get(agent_builder_session_id)
        if session and session["tenant_id"] == tenant_id and session["workspace_id"] == workspace_id:
            return session
        return None

    async def upsert_agent_builder_session(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        agent_builder_session_id = payload.get("agent_builder_session_id") or f"builder-{len(self.agent_builder_sessions) + 1}"
        row = {
            "agent_builder_session_id": agent_builder_session_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "user_id": payload.get("user_id") or user_id,
            "agent_id": payload.get("agent_id"),
            "model_setting_id": payload.get("model_setting_id"),
            "status": payload.get("status", "draft"),
            "current_step": payload.get("current_step", "intake"),
            "builder_transcript": payload.get("builder_transcript", []),
            "generated_config": payload.get("generated_config", {}),
            "readiness_score": payload.get("readiness_score", 0),
            "requires_review": payload.get("requires_review", True),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.agent_builder_sessions[agent_builder_session_id] = row
        return row

    async def list_agent_context_budgets(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        agent_id: str | None = None,
    ):
        return [
            budget
            for budget in self.agent_context_budgets.values()
            if budget["tenant_id"] == tenant_id
            and budget["workspace_id"] == workspace_id
            and (agent_id is None or budget["agent_id"] == agent_id)
        ]

    async def upsert_agent_context_budget(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        budget_id = payload.get("agent_context_budget_id") or f"budget-{len(self.agent_context_budgets) + 1}"
        row = {
            "agent_context_budget_id": budget_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "agent_id": payload["agent_id"],
            "model_setting_id": payload.get("model_setting_id"),
            "max_context_tokens": payload.get("max_context_tokens", 0),
            "reserved_response_tokens": payload.get("reserved_response_tokens", 0),
            "max_prompt_tokens": payload.get("max_prompt_tokens", 0),
            "daily_cost_limit_usd": payload.get("daily_cost_limit_usd"),
            "monthly_cost_limit_usd": payload.get("monthly_cost_limit_usd"),
            "policy": payload.get("policy", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.agent_context_budgets[budget_id] = row
        return row

    async def sum_llm_usage_cost(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        agent_id: str | None = None,
        since=None,
    ):
        total = 0.0
        for entry in self.usage_ledger:
            if entry.get("tenant_id") != tenant_id or entry.get("workspace_id") != workspace_id:
                continue
            if agent_id is not None and entry.get("agent_id") != agent_id:
                continue
            created_at = entry.get("created_at")
            if since is not None and created_at is not None and hasattr(created_at, "tzinfo") and created_at < since:
                continue
            total += float(entry.get("actual_cost_usd") or entry.get("estimated_cost_usd") or 0)
        return total

    async def list_llm_usage_ledger(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        model_id: str | None = None,
        operation: str | None = None,
        group_id: str | None = None,
        subgroup_id: str | None = None,
    ):
        return [
            entry
            for entry in self.usage_ledger
            if entry.get("tenant_id") == tenant_id
            and entry.get("workspace_id") == workspace_id
            and (user_id is None or entry.get("user_id") == user_id)
            and (agent_id is None or entry.get("agent_id") == agent_id)
            and (model_id is None or entry.get("model_id") == model_id)
            and (operation is None or entry.get("operation") == operation)
            and (group_id is None or entry.get("group_id") == group_id or entry.get("metadata", {}).get("group_id") == group_id)
            and (
                subgroup_id is None
                or entry.get("subgroup_id") == subgroup_id
                or entry.get("metadata", {}).get("subgroup_id") == subgroup_id
            )
        ]

    async def insert_llm_usage_ledger(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        previous_hash = self.usage_ledger[-1]["ledger_hash"] if self.usage_ledger else ""
        row = {
            **payload,
            "usage_ledger_id": payload.get("usage_ledger_id") or f"ledger-{len(self.usage_ledger) + 1}",
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "group_id": payload.get("group_id"),
            "subgroup_id": payload.get("subgroup_id"),
            "user_id": payload.get("user_id") or user_id,
            "previous_ledger_hash": previous_hash,
            "ledger_hash": f"sha256:ledger-{len(self.usage_ledger) + 1}",
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": payload.get("created_at") or "2026-04-29T00:00:00Z",
            "updated_at": payload.get("created_at") or "2026-04-29T00:00:00Z",
        }
        self.usage_ledger.append(row)
        return row

    async def reserve_llm_usage_budget(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
        agent_id: str | None,
        daily_limit_usd: float | None,
        monthly_limit_usd: float | None,
        daily_since,
        monthly_since,
    ):
        async with self.budget_lock:
            estimate = float(payload.get("estimated_cost_usd") or 0)
            daily_used = await self.sum_llm_usage_cost(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                agent_id=agent_id,
                since=daily_since,
            )
            if daily_limit_usd is not None and daily_used + estimate > daily_limit_usd:
                raise ValueError("daily_cost_budget_exceeded")
            monthly_used = await self.sum_llm_usage_cost(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                agent_id=agent_id,
                since=monthly_since,
            )
            if monthly_limit_usd is not None and monthly_used + estimate > monthly_limit_usd:
                raise ValueError("monthly_cost_budget_exceeded")
            return await self.insert_llm_usage_ledger(
                payload,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )

    async def list_knowledge_groups(self, *, tenant_id: str | None, workspace_id: str):
        return [
            group
            for group in self.groups.values()
            if group["tenant_id"] == tenant_id and group["workspace_id"] == workspace_id
        ]

    async def get_knowledge_group(self, group_id: str, *, tenant_id: str | None, workspace_id: str):
        group = self.groups.get(group_id)
        if group and group["tenant_id"] == tenant_id and group["workspace_id"] == workspace_id:
            return group
        return None

    async def upsert_knowledge_group(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        group_id = payload.get("group_id") or f"group-{len(self.groups) + 1}"
        row = {
            "group_id": group_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "slug": payload["slug"],
            "name": payload["name"],
            "description": payload.get("description", ""),
            "privacy": payload.get("privacy", "team"),
            "color": payload.get("color", "#2563EB"),
            "metadata": payload.get("metadata", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.groups[group_id] = row
        return row

    async def list_knowledge_subgroups(
        self, *, tenant_id: str | None, workspace_id: str, group_id: str | None = None
    ):
        return [
            subgroup
            for subgroup in self.subgroups.values()
            if subgroup["tenant_id"] == tenant_id
            and subgroup["workspace_id"] == workspace_id
            and (group_id is None or subgroup["group_id"] == group_id)
        ]

    async def get_knowledge_subgroup(
        self,
        subgroup_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
        group_id: str | None = None,
    ):
        subgroup = self.subgroups.get(subgroup_id)
        if (
            subgroup
            and subgroup["tenant_id"] == tenant_id
            and subgroup["workspace_id"] == workspace_id
            and (group_id is None or subgroup["group_id"] == group_id)
        ):
            return subgroup
        return None

    async def upsert_knowledge_subgroup(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        subgroup_id = payload.get("subgroup_id") or f"subgroup-{len(self.subgroups) + 1}"
        row = {
            "subgroup_id": subgroup_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "group_id": payload["group_id"],
            "slug": payload["slug"],
            "name": payload["name"],
            "description": payload.get("description", ""),
            "privacy": payload.get("privacy", "team"),
            "metadata": payload.get("metadata", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.subgroups[subgroup_id] = row
        return row

    async def register_document(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        if payload.get("source_kind", "upload") == "upload" and (
            not payload.get("group_id") or not payload.get("subgroup_id")
        ):
            raise ValueError("Upload documents require group_id and subgroup_id.")
        document_id = payload.get("document_id") or f"registry-doc-{len(self.documents) + 1}"
        row = {
            "document_id": document_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "embedding_version_id": payload.get("embedding_version_id"),
            "chunk_count": payload.get("chunk_count", 0),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
            **payload,
        }
        self.documents[document_id] = row
        return row

    async def list_document_registry(self, *, tenant_id: str | None, workspace_id: str):
        return [
            document
            for document in self.documents.values()
            if document["tenant_id"] == tenant_id and document["workspace_id"] == workspace_id
        ]

    async def get_document_registry(self, document_id: str, *, tenant_id: str | None, workspace_id: str):
        document = self.documents.get(document_id)
        if document and document["tenant_id"] == tenant_id and document["workspace_id"] == workspace_id:
            return document
        return None

    async def list_note_registry(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        group_id: str | None = None,
        subgroup_id: str | None = None,
    ):
        return [
            note
            for note in self.notes.values()
            if note["tenant_id"] == tenant_id
            and note["workspace_id"] == workspace_id
            and (group_id is None or note["group_id"] == group_id)
            and (subgroup_id is None or note["subgroup_id"] == subgroup_id)
        ]

    async def get_note_registry(self, note_id: str, *, tenant_id: str | None, workspace_id: str):
        note = self.notes.get(note_id)
        if note and note["tenant_id"] == tenant_id and note["workspace_id"] == workspace_id:
            return note
        return None

    async def find_note_by_slug_or_title(
        self,
        *,
        slug: str,
        title: str,
        tenant_id: str | None,
        workspace_id: str,
    ):
        for note in self.notes.values():
            if (
                note["tenant_id"] == tenant_id
                and note["workspace_id"] == workspace_id
                and (note["slug"] == slug or note["title"].lower() == title.lower())
            ):
                return note
        return None

    async def upsert_note_registry(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        note_id = payload.get("note_id")
        if not note_id:
            for note in self.notes.values():
                if note["workspace_id"] == workspace_id and note["slug"] == payload["slug"]:
                    note_id = note["note_id"]
                    break
        note_id = note_id or f"note-{len(self.notes) + 1}"
        row = {
            "note_id": note_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "group_id": payload.get("group_id"),
            "subgroup_id": payload.get("subgroup_id"),
            "title": payload["title"],
            "slug": payload["slug"],
            "note_type": payload.get("note_type", "markdown"),
            "privacy": payload.get("privacy", "team"),
            "status": payload.get("status", "active"),
            "metadata": payload.get("metadata", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.notes[note_id] = row
        return row

    async def insert_markdown_note_version(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        note_id = payload["note_id"]
        versions = self.markdown_notes.setdefault(note_id, [])
        for version in versions:
            if version["status"] == "current":
                version["status"] = "superseded"
        version_number = len(versions) + 1
        row = {
            "markdown_note_id": f"md-{note_id}-{version_number}",
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "note_id": note_id,
            "version_number": version_number,
            "markdown": payload["markdown"],
            "rendered_summary": payload.get("rendered_summary", ""),
            "content_hash": payload["content_hash"],
            "status": payload.get("status", "current"),
            "source_document_id": payload.get("source_document_id"),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        versions.append(row)
        return row

    async def get_latest_markdown_note(self, note_id: str, *, tenant_id: str | None, workspace_id: str):
        versions = [
            note
            for note in self.markdown_notes.get(note_id, [])
            if note["tenant_id"] == tenant_id and note["workspace_id"] == workspace_id
        ]
        return versions[-1] if versions else None

    async def replace_wiki_links(
        self,
        *,
        source_note_id: str,
        links: list[dict],
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        rows = []
        for index, link in enumerate(links, start=1):
            rows.append(
                {
                    "wiki_link_id": f"wiki-{source_note_id}-{index}",
                    "tenant_id": tenant_id,
                    "workspace_id": workspace_id,
                    "source_note_id": source_note_id,
                    "target_note_id": link.get("target_note_id"),
                    "target_label": link["target_label"],
                    "link_text": link.get("link_text", ""),
                    "link_status": link.get("link_status", "unresolved"),
                    "metadata": link.get("metadata", {}),
                    "created_by": user_id,
                    "updated_by": user_id,
                    "created_at": "2026-04-29T00:00:00Z",
                    "updated_at": "2026-04-29T00:00:00Z",
                }
            )
        self.wiki_links[source_note_id] = rows
        return rows

    async def list_wiki_links(self, *, source_note_id: str, tenant_id: str | None, workspace_id: str):
        return [
            link
            for link in self.wiki_links.get(source_note_id, [])
            if link["tenant_id"] == tenant_id and link["workspace_id"] == workspace_id
        ]

    async def upsert_tag_registry(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        tag = payload["tag"]
        row = {
            "tag_id": self.tags.get(tag, {}).get("tag_id") or f"tag-{len(self.tags) + 1}",
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "tag": tag,
            "label": payload["label"],
            "description": payload.get("description", ""),
            "color": payload.get("color", "#64748B"),
            "metadata": payload.get("metadata", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.tags[tag] = row
        return row

    async def list_tag_registry(self, *, tenant_id: str | None, workspace_id: str):
        return [
            tag
            for tag in self.tags.values()
            if tag["tenant_id"] == tenant_id and tag["workspace_id"] == workspace_id
        ]

    async def upsert_backlink(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        backlink_id = payload.get("backlink_id") or f"backlink-{len(self.backlinks) + 1}"
        row = {
            "backlink_id": backlink_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "source_kind": payload["source_kind"],
            "source_id": payload["source_id"],
            "target_kind": payload["target_kind"],
            "target_id": payload["target_id"],
            "link_text": payload.get("link_text", ""),
            "origin_type": payload.get("origin_type", "manual"),
            "graph_edge_origin_id": payload.get("graph_edge_origin_id"),
            "confidence": payload.get("confidence"),
            "metadata": payload.get("metadata", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        for existing_id, existing in self.backlinks.items():
            if (
                existing["workspace_id"] == workspace_id
                and existing["source_kind"] == row["source_kind"]
                and existing["source_id"] == row["source_id"]
                and existing["target_kind"] == row["target_kind"]
                and existing["target_id"] == row["target_id"]
                and existing["origin_type"] == row["origin_type"]
            ):
                row["backlink_id"] = existing_id
                break
        self.backlinks[row["backlink_id"]] = row
        return row

    async def replace_backlinks_for_source(
        self,
        *,
        source_kind: str,
        source_id: str,
        origin_type: str,
        backlinks: list[dict],
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        self.backlinks = {
            backlink_id: backlink
            for backlink_id, backlink in self.backlinks.items()
            if not (
                backlink["tenant_id"] == tenant_id
                and backlink["workspace_id"] == workspace_id
                and backlink["source_kind"] == source_kind
                and backlink["source_id"] == source_id
                and backlink["origin_type"] == origin_type
            )
        }
        rows = []
        for backlink in backlinks:
            rows.append(
                await self.upsert_backlink(
                    backlink,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                )
            )
        return rows

    async def list_backlinks(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        source_kind: str | None = None,
        source_id: str | None = None,
        target_kind: str | None = None,
        target_id: str | None = None,
    ):
        return [
            backlink
            for backlink in self.backlinks.values()
            if backlink["tenant_id"] == tenant_id
            and backlink["workspace_id"] == workspace_id
            and (source_kind is None or backlink["source_kind"] == source_kind)
            and (source_id is None or backlink["source_id"] == source_id)
            and (target_kind is None or backlink["target_kind"] == target_kind)
            and (target_id is None or backlink["target_id"] == target_id)
        ]

    async def insert_source_provenance(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        source_provenance_id = payload.get("source_provenance_id") or f"prov-{len(self.provenance) + 1}"
        row = {
            "source_provenance_id": source_provenance_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "source_kind": payload["source_kind"],
            "source_id": payload["source_id"],
            "document_id": payload.get("document_id"),
            "note_id": payload.get("note_id"),
            "chunk_id": payload.get("chunk_id", ""),
            "model_id": payload.get("model_id", ""),
            "agent_id": payload.get("agent_id"),
            "usage_ledger_id": payload.get("usage_ledger_id"),
            "confidence": payload.get("confidence"),
            "locator": payload.get("locator", {}),
            "metadata": payload.get("metadata", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.provenance[source_provenance_id] = row
        return row

    async def list_source_provenance(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        source_kind: str | None = None,
        source_id: str | None = None,
        document_id: str | None = None,
        note_id: str | None = None,
    ):
        return [
            provenance
            for provenance in self.provenance.values()
            if provenance["tenant_id"] == tenant_id
            and provenance["workspace_id"] == workspace_id
            and (source_kind is None or provenance["source_kind"] == source_kind)
            and (source_id is None or provenance["source_id"] == source_id)
            and (document_id is None or provenance["document_id"] == document_id)
            and (note_id is None or provenance["note_id"] == note_id)
        ]

    async def list_canvas_boards(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        group_id: str | None = None,
        subgroup_id: str | None = None,
    ):
        return [
            board
            for board in self.canvas_boards.values()
            if board["tenant_id"] == tenant_id
            and board["workspace_id"] == workspace_id
            and (group_id is None or board["group_id"] == group_id)
            and (subgroup_id is None or board["subgroup_id"] == subgroup_id)
        ]

    async def get_canvas_board(self, canvas_board_id: str, *, tenant_id: str | None, workspace_id: str):
        board = self.canvas_boards.get(canvas_board_id)
        if board and board["tenant_id"] == tenant_id and board["workspace_id"] == workspace_id:
            return board
        return None

    async def upsert_canvas_board(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        canvas_board_id = payload.get("canvas_board_id") or f"canvas-{len(self.canvas_boards) + 1}"
        row = {
            "canvas_board_id": canvas_board_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "group_id": payload["group_id"],
            "subgroup_id": payload["subgroup_id"],
            "title": payload["title"],
            "slug": payload["slug"],
            "layout": payload.get("layout", {}),
            "status": payload.get("status", "active"),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.canvas_boards[canvas_board_id] = row
        return row

    async def upsert_canvas_node(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        canvas_node_id = payload.get("canvas_node_id") or f"node-{len(self.canvas_nodes) + 1}"
        row = {
            "canvas_node_id": canvas_node_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "canvas_board_id": payload["canvas_board_id"],
            "node_kind": payload["node_kind"],
            "ref_kind": payload.get("ref_kind", ""),
            "ref_id": payload.get("ref_id", ""),
            "x": payload.get("x", 0),
            "y": payload.get("y", 0),
            "width": payload.get("width", 280),
            "height": payload.get("height", 160),
            "content": payload.get("content", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.canvas_nodes[canvas_node_id] = row
        return row

    async def get_canvas_node(self, canvas_node_id: str, *, tenant_id: str | None, workspace_id: str):
        node = self.canvas_nodes.get(canvas_node_id)
        if node and node["tenant_id"] == tenant_id and node["workspace_id"] == workspace_id:
            return node
        return None

    async def list_canvas_nodes(self, *, canvas_board_id: str, tenant_id: str | None, workspace_id: str):
        return [
            node
            for node in self.canvas_nodes.values()
            if node["tenant_id"] == tenant_id
            and node["workspace_id"] == workspace_id
            and node["canvas_board_id"] == canvas_board_id
        ]

    async def upsert_canvas_edge(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        canvas_edge_id = payload.get("canvas_edge_id") or f"edge-{len(self.canvas_edges) + 1}"
        row = {
            "canvas_edge_id": canvas_edge_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "canvas_board_id": payload["canvas_board_id"],
            "source_node_id": payload["source_node_id"],
            "target_node_id": payload["target_node_id"],
            "edge_kind": payload.get("edge_kind", "manual"),
            "label": payload.get("label", ""),
            "metadata": payload.get("metadata", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.canvas_edges[canvas_edge_id] = row
        return row

    async def list_canvas_edges(self, *, canvas_board_id: str, tenant_id: str | None, workspace_id: str):
        return [
            edge
            for edge in self.canvas_edges.values()
            if edge["tenant_id"] == tenant_id
            and edge["workspace_id"] == workspace_id
            and edge["canvas_board_id"] == canvas_board_id
        ]

    async def upsert_knowledge_dossier(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        knowledge_dossier_id = payload.get("knowledge_dossier_id") or f"dossier-{len(self.dossiers) + 1}"
        row = {
            "knowledge_dossier_id": knowledge_dossier_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "group_id": payload.get("group_id"),
            "subgroup_id": payload.get("subgroup_id"),
            "title": payload["title"],
            "slug": payload["slug"],
            "dossier_kind": payload.get("dossier_kind", "knowledge"),
            "status": payload.get("status", "draft"),
            "content_refs": payload.get("content_refs", []),
            "export_policy": payload.get("export_policy", {}),
            "approval_id": payload.get("approval_id"),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.dossiers[knowledge_dossier_id] = row
        return row

    async def list_content_maps(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        group_id: str | None = None,
        subgroup_id: str | None = None,
    ):
        return [
            content_map
            for content_map in self.content_maps.values()
            if content_map["tenant_id"] == tenant_id
            and content_map["workspace_id"] == workspace_id
            and (group_id is None or content_map["group_id"] == group_id)
            and (subgroup_id is None or content_map["subgroup_id"] == subgroup_id)
        ]

    async def upsert_content_map(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        content_map_id = payload.get("content_map_id")
        if not content_map_id:
            content_map_id = next(
                (
                    content_map["content_map_id"]
                    for content_map in self.content_maps.values()
                    if content_map["tenant_id"] == tenant_id
                    and content_map["workspace_id"] == workspace_id
                    and content_map["slug"] == payload["slug"]
                ),
                f"moc-{len(self.content_maps) + 1}",
            )
        row = {
            "content_map_id": content_map_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "group_id": payload["group_id"],
            "subgroup_id": payload["subgroup_id"],
            "title": payload["title"],
            "slug": payload["slug"],
            "root_note_id": payload.get("root_note_id"),
            "description": payload.get("description", ""),
            "map_body": payload.get("map_body", {}),
            "status": payload.get("status", "draft"),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.content_maps[content_map_id] = row
        return row

    async def list_knowledge_trails(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        group_id: str | None = None,
        subgroup_id: str | None = None,
    ):
        return [
            trail
            for trail in self.trails.values()
            if trail["tenant_id"] == tenant_id
            and trail["workspace_id"] == workspace_id
            and (group_id is None or trail["group_id"] == group_id)
            and (subgroup_id is None or trail["subgroup_id"] == subgroup_id)
        ]

    async def get_knowledge_trail(self, knowledge_trail_id: str, *, tenant_id: str | None, workspace_id: str):
        trail = self.trails.get(knowledge_trail_id)
        if trail and trail["tenant_id"] == tenant_id and trail["workspace_id"] == workspace_id:
            return trail
        return None

    async def upsert_knowledge_trail(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        knowledge_trail_id = payload.get("knowledge_trail_id")
        if not knowledge_trail_id:
            knowledge_trail_id = next(
                (
                    trail["knowledge_trail_id"]
                    for trail in self.trails.values()
                    if trail["tenant_id"] == tenant_id
                    and trail["workspace_id"] == workspace_id
                    and trail["slug"] == payload["slug"]
                ),
                f"trail-{len(self.trails) + 1}",
            )
        row = {
            "knowledge_trail_id": knowledge_trail_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "group_id": payload["group_id"],
            "subgroup_id": payload["subgroup_id"],
            "title": payload["title"],
            "slug": payload["slug"],
            "trail_type": payload.get("trail_type", "study"),
            "description": payload.get("description", ""),
            "status": payload.get("status", "draft"),
            "metadata": payload.get("metadata", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.trails[knowledge_trail_id] = row
        return row

    async def list_knowledge_trail_steps(
        self,
        *,
        knowledge_trail_id: str,
        tenant_id: str | None,
        workspace_id: str,
    ):
        return sorted(
            [
                step
                for step in self.trail_steps.values()
                if step["tenant_id"] == tenant_id
                and step["workspace_id"] == workspace_id
                and step["knowledge_trail_id"] == knowledge_trail_id
            ],
            key=lambda step: (step["step_order"], step["created_at"]),
        )

    async def upsert_knowledge_trail_step(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        knowledge_trail_step_id = payload.get("knowledge_trail_step_id") or f"step-{len(self.trail_steps) + 1}"
        row = {
            "knowledge_trail_step_id": knowledge_trail_step_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "knowledge_trail_id": payload["knowledge_trail_id"],
            "step_order": payload["step_order"],
            "title": payload["title"],
            "step_kind": payload.get("step_kind", "note"),
            "note_id": payload.get("note_id"),
            "document_id": payload.get("document_id"),
            "canvas_board_id": payload.get("canvas_board_id"),
            "instructions": payload.get("instructions", ""),
            "metadata": payload.get("metadata", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        for existing_id, existing in self.trail_steps.items():
            if (
                existing["knowledge_trail_id"] == row["knowledge_trail_id"]
                and existing["step_order"] == row["step_order"]
            ):
                row["knowledge_trail_step_id"] = existing_id
                break
        self.trail_steps[row["knowledge_trail_step_id"]] = row
        return row

    async def list_knowledge_inbox_items(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        status: str | None = None,
        group_id: str | None = None,
        subgroup_id: str | None = None,
        limit: int = 100,
    ):
        items = [
            item
            for item in self.inbox_items.values()
            if item["tenant_id"] == tenant_id
            and item["workspace_id"] == workspace_id
            and (status is None or item["status"] == status)
            and (group_id is None or item["group_id"] == group_id)
            and (subgroup_id is None or item["subgroup_id"] == subgroup_id)
        ]
        return items[:limit]

    async def get_knowledge_inbox_item(self, inbox_item_id: str, *, tenant_id: str | None, workspace_id: str):
        item = self.inbox_items.get(inbox_item_id)
        if item and item["tenant_id"] == tenant_id and item["workspace_id"] == workspace_id:
            return item
        return None

    async def upsert_knowledge_inbox_item(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        inbox_item_id = payload.get("inbox_item_id") or f"inbox-{len(self.inbox_items) + 1}"
        row = {
            "inbox_item_id": inbox_item_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "group_id": payload.get("group_id"),
            "subgroup_id": payload.get("subgroup_id"),
            "item_kind": payload["item_kind"],
            "title": payload["title"],
            "body": payload.get("body", ""),
            "source_kind": payload.get("source_kind", ""),
            "source_id": payload.get("source_id", ""),
            "status": payload.get("status", "open"),
            "priority": payload.get("priority", "normal"),
            "metadata": payload.get("metadata", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.inbox_items[inbox_item_id] = row
        return row

    async def update_knowledge_inbox_item_status(
        self,
        inbox_item_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
        status: str,
        metadata: dict | None,
        user_id: str,
    ):
        item = await self.get_knowledge_inbox_item(
            inbox_item_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if not item:
            return None
        item = {
            **item,
            "status": status,
            "metadata": {**item.get("metadata", {}), **(metadata or {})},
            "updated_by": user_id,
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.inbox_items[inbox_item_id] = item
        return item

    async def list_daily_notes(self, *, tenant_id: str | None, workspace_id: str, limit: int = 30):
        rows = [
            daily_note
            for daily_note in self.daily_notes.values()
            if daily_note["tenant_id"] == tenant_id and daily_note["workspace_id"] == workspace_id
        ]
        return rows[:limit]

    async def get_daily_note(self, note_date: str, *, tenant_id: str | None, workspace_id: str):
        daily_note = self.daily_notes.get(note_date)
        if daily_note and daily_note["tenant_id"] == tenant_id and daily_note["workspace_id"] == workspace_id:
            return daily_note
        return None

    async def upsert_daily_note(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        note_date = payload["note_date"]
        daily_note_id = payload.get("daily_note_id") or self.daily_notes.get(note_date, {}).get("daily_note_id")
        row = {
            "daily_note_id": daily_note_id or f"daily-{len(self.daily_notes) + 1}",
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "note_id": payload["note_id"],
            "note_date": note_date,
            "summary": payload.get("summary", ""),
            "decisions": payload.get("decisions", []),
            "pending_items": payload.get("pending_items", []),
            "cost_snapshot": payload.get("cost_snapshot", {}),
            "created_by": user_id,
            "updated_by": user_id,
            "created_at": "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.daily_notes[note_date] = row
        return row

    async def get_conversation(self, conversation_id: str):
        return self.conversations.get(conversation_id)

    async def save_conversation(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        conversation_id = payload.get("conversation_id") or f"conversation-{len(self.conversations) + 1}"
        existing = self.conversations.get(conversation_id)
        if existing and (
            existing["tenant_id"] != tenant_id
            or existing["workspace_id"] != workspace_id
            or existing["user_id"] != user_id
            or existing.get("scope_snapshot", {}) != (payload.get("scope_snapshot") or {})
        ):
            raise ValueError("conversation_scope_mismatch")
        messages = []
        for index, message in enumerate(payload.get("messages") or [], start=1):
            messages.append(
                {
                    "message_id": message.get("message_id") or message.get("id") or f"message-{index}",
                    "role": message["role"],
                    "content": message["content"],
                    "references": message.get("references", []),
                    "metadata": message.get("metadata", {}),
                    "created_at": "2026-04-29T00:00:00Z",
                }
            )
        row = {
            "conversation_id": conversation_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "user_id": user_id,
            "title": payload.get("title") or "Conversa Little Bull",
            "agent_id": payload.get("agent_id"),
            "model_profile": payload.get("model_profile", "equilibrado"),
            "confidentiality": payload.get("confidentiality", "normal"),
            "scope_snapshot": payload.get("scope_snapshot") or {},
            "message_count": len(messages),
            "messages": messages,
            "created_at": existing.get("created_at") if existing else "2026-04-29T00:00:00Z",
            "updated_at": "2026-04-29T00:00:00Z",
        }
        self.conversations[conversation_id] = row
        return row

    async def list_conversations(self, *, tenant_id: str | None, workspace_id: str, user_id: str | None = None):
        return [
            conversation
            for conversation in self.conversations.values()
            if conversation["tenant_id"] == tenant_id
            and conversation["workspace_id"] == workspace_id
            and (user_id is None or conversation["user_id"] == user_id)
        ]

    async def get_correlation_suggestion(self, suggestion_id: str):
        return self.suggestions.get(suggestion_id)

    async def create_correlation_suggestion(
        self,
        payload: dict,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ):
        suggestion_id = payload.get("suggestion_id") or f"suggestion-{len(self.suggestions) + 1}"
        row = {
            "suggestion_id": suggestion_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "user_id": user_id,
            "source_label": payload["source_label"],
            "target_label": payload["target_label"],
            "reason": payload.get("reason", ""),
            "status": payload.get("status", "pending"),
            "metadata": payload.get("metadata", {}),
            "created_at": "2026-04-29T00:00:00Z",
            "decided_at": None,
            "decided_by": None,
        }
        self.suggestions[suggestion_id] = row
        return row

    async def list_correlation_suggestions(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        status: str | None = None,
    ):
        return [
            suggestion
            for suggestion in self.suggestions.values()
            if suggestion["tenant_id"] == tenant_id
            and suggestion["workspace_id"] == workspace_id
            and (status is None or suggestion["status"] == status)
        ]


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


async def _create_group_and_subgroup(service: LittleBullService, principal):
    group = await service.upsert_knowledge_group(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeGroupRequest(
            name="Jurídico",
            slug="juridico",
        ),
    )
    subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Inicial",
            slug="inicial",
        ),
    )
    return group, subgroup


@pytest.mark.asyncio
async def test_little_bull_lists_documents_and_audits(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)

    response = await service.list_documents(principal, workspace_id="default")

    assert response.total_count == 0
    assert response.documents == []
    assert (await service.list_activity(principal, workspace_id="default"))[0].result == "success"


@pytest.mark.asyncio
async def test_little_bull_upload_requires_existing_group_and_subgroup(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)

    with pytest.raises(HTTPException) as exc:
        await service.upload_document(
            principal,
            workspace_id="default",
            group_id="missing-group",
            subgroup_id="missing-subgroup",
            file=UploadFile(filename="novo.md", file=BytesIO(b"# Novo")),
            background_tasks=BackgroundTasks(),
        )

    assert exc.value.status_code == 404
    assert not (tmp_path / "novo.md").exists()
    assert service.admin_store.documents == {}


@pytest.mark.asyncio
async def test_little_bull_upload_requires_group_and_subgroup_values(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)

    with pytest.raises(HTTPException) as exc:
        await service.upload_document(
            principal,
            workspace_id="default",
            group_id="",
            subgroup_id="",
            file=UploadFile(filename="sem-classe.md", file=BytesIO(b"# Sem classe")),
            background_tasks=BackgroundTasks(),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.documents == {}


@pytest.mark.asyncio
async def test_little_bull_upload_rejects_subgroup_outside_group(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_group = await service.upsert_knowledge_group(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeGroupRequest(name="Outro", slug="outro"),
    )

    with pytest.raises(HTTPException) as exc:
        await service.upload_document(
            principal,
            workspace_id="default",
            group_id=other_group.group_id,
            subgroup_id=subgroup.subgroup_id,
            file=UploadFile(filename="fora.md", file=BytesIO(b"# Fora")),
            background_tasks=BackgroundTasks(),
        )

    assert exc.value.status_code == 404
    assert group.group_id != other_group.group_id
    assert not (tmp_path / "fora.md").exists()
    assert service.admin_store.documents == {}


@pytest.mark.asyncio
async def test_little_bull_upload_registers_document_with_group_subgroup(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    background_tasks = BackgroundTasks()
    queued: dict[str, object] = {}

    def fake_queue(background_tasks, file_path, track_id, *, rag):
        queued["file_path"] = file_path
        queued["track_id"] = track_id
        queued["workspace"] = rag.workspace
        background_tasks.add_task(lambda: None)

    service._queue_pipeline_index_file = fake_queue

    response = await service.upload_document(
        principal,
        workspace_id="default",
        group_id=group.group_id,
        subgroup_id=subgroup.subgroup_id,
        file=UploadFile(filename="peticao.md", file=BytesIO(b"# Peticao inicial")),
        background_tasks=background_tasks,
        confidentiality="normal",
    )

    assert response.status == "success"
    assert response.group_id == group.group_id
    assert response.subgroup_id == subgroup.subgroup_id
    assert response.registry_document_id
    assert len(background_tasks.tasks) == 1
    assert queued["workspace"] == "default"
    registry = service.admin_store.documents[response.registry_document_id]
    assert registry["group_id"] == group.group_id
    assert registry["subgroup_id"] == subgroup.subgroup_id
    assert registry["source_uri"] == "peticao.md"
    assert registry["source_kind"] == "upload"
    assert registry["content_hash"]
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    upload_event = next(event for event in events if event.action == "little_bull.documents.upload")
    assert upload_event.metadata["registry_document_id"] == response.registry_document_id
    assert upload_event.metadata["group_id"] == group.group_id


@pytest.mark.asyncio
async def test_little_bull_user_resubmission_creates_new_registry_entry(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    service._queue_pipeline_index_file = lambda background_tasks, file_path, track_id, *, rag: background_tasks.add_task(
        lambda: None
    )

    first = await service.upload_document(
        principal,
        workspace_id="default",
        group_id=group.group_id,
        subgroup_id=subgroup.subgroup_id,
        file=UploadFile(filename="reenvio.md", file=BytesIO(b"primeira versao")),
        background_tasks=BackgroundTasks(),
    )
    archived_dir = tmp_path / "__enqueued__"
    archived_dir.mkdir()
    (tmp_path / "reenvio.md").rename(archived_dir / "reenvio.md")
    second = await service.upload_document(
        principal,
        workspace_id="default",
        group_id=group.group_id,
        subgroup_id=subgroup.subgroup_id,
        file=UploadFile(filename="reenvio.md", file=BytesIO(b"segunda versao")),
        background_tasks=BackgroundTasks(),
    )

    assert first.registry_document_id != second.registry_document_id
    assert len(service.admin_store.documents) == 2
    titles = {document["title"] for document in service.admin_store.documents.values()}
    assert "reenvio.md" in titles
    assert any(title.startswith("reenvio_") for title in titles)
    assert (tmp_path / "reenvio_reindex_001.md").exists()


@pytest.mark.asyncio
async def test_little_bull_list_documents_filters_legacy_when_registry_exists(tmp_path):
    registered = SimpleNamespace(
        file_path="manual.pdf",
        status="processed",
        content_summary="Manual",
        content_length=42,
        updated_at="2026-04-27T00:00:00Z",
        created_at="2026-04-27T00:00:00Z",
        track_id="trk-registered",
        chunks_count=1,
        metadata={},
    )
    legacy = SimpleNamespace(
        file_path="legacy.pdf",
        status="processed",
        content_summary="Legacy",
        content_length=99,
        updated_at="2026-04-27T00:00:00Z",
        created_at="2026-04-27T00:00:00Z",
        track_id="trk-legacy",
        chunks_count=1,
        metadata={},
    )
    rag = FakeRag(
        working_dir=str(tmp_path / "rag_storage"),
        docs=[("doc-legacy", legacy), ("doc-registered", registered)],
    )
    principal, service = await _principal_and_service(tmp_path, rag=rag)
    service.admin_store = FakeAdminStore()
    group, subgroup = await _create_group_and_subgroup(service, principal)
    registry = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": subgroup.subgroup_id,
            "title": "manual.pdf",
            "source_uri": str(tmp_path / "manual.pdf"),
            "source_kind": "upload",
            "mime_type": "application/pdf",
            "content_hash": "hash",
            "confidentiality": "normal",
            "status": "processed",
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )

    response = await service.list_documents(principal, workspace_id="default", page_size=1)

    assert response.total_count == 1
    assert [document.title for document in response.documents] == ["manual.pdf"]
    assert response.documents[0].file_path == "manual.pdf"
    assert str(tmp_path) not in response.documents[0].file_path
    assert response.documents[0].group_id == group.group_id
    assert response.documents[0].subgroup_id == subgroup.subgroup_id
    assert response.documents[0].registry_document_id == registry["document_id"]


@pytest.mark.asyncio
async def test_little_bull_markdown_note_extracts_wikilinks_tags_and_versions(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)

    target = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Mapa de Conteudo",
            slug="mapa",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Mapa\n#juridico",
        ),
    )
    source = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Peticao Inicial",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Peticao\nLeia [[Mapa de Conteudo|MOC]] #juridico #processual",
        ),
    )

    assert source.note.version_number == 1
    assert source.registry.group_id == group.group_id
    assert source.wiki_links[0].target_note_id == target.registry.note_id
    assert source.wiki_links[0].link_status == "resolved"
    assert {tag.tag for tag in source.tags} == {"#juridico", "#processual"}

    updated = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            note_id=source.registry.note_id,
            title="Peticao Inicial",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Peticao atualizada\nVoltar para [[Mapa de Conteudo]] #juridico",
        ),
    )

    assert updated.note.version_number == 2
    versions = service.admin_store.markdown_notes[source.registry.note_id]
    assert [version["status"] for version in versions] == ["superseded", "current"]
    fetched = await service.get_markdown_note(principal, workspace_id="default", note_id=source.registry.note_id)
    assert fetched.note.markdown.startswith("# Peticao atualizada")
    assert [note.note_id for note in await service.list_notes(principal, workspace_id="default")] == [
        target.registry.note_id,
        source.registry.note_id,
    ]


@pytest.mark.asyncio
async def test_little_bull_markdown_note_requires_group_and_subgroup(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)

    with pytest.raises(HTTPException) as exc:
        await service.upsert_markdown_note(
            principal,
            workspace_id="default",
            payload=LittleBullMarkdownNoteRequest(
                title="Sem Classe",
                group_id="",
                subgroup_id="",
                markdown="# Sem classe",
            ),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.notes == {}


@pytest.mark.asyncio
async def test_little_bull_markdown_note_rejects_cross_subgroup_source_document(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": other_subgroup.subgroup_id,
            "title": "origem.md",
            "source_uri": "origem.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash",
            "confidentiality": "normal",
            "status": "processed",
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )
    with pytest.raises(HTTPException) as exc:
        await service.upsert_markdown_note(
            principal,
            workspace_id="default",
            payload=LittleBullMarkdownNoteRequest(
                title="Nota",
                group_id=group.group_id,
                subgroup_id=subgroup.subgroup_id,
                source_document_id=document["document_id"],
                markdown="# Nota #juridico",
            ),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.notes == {}


@pytest.mark.asyncio
async def test_little_bull_wikilinks_create_backlinks_and_panel(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    target = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Mapa do Caso",
            slug="mapa-caso",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Mapa do caso",
        ),
    )
    source = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Linha Argumentativa",
            slug="linha",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="Ver [[Mapa do Caso]] para contexto #juridico",
        ),
    )

    backlinks = await service.list_backlinks(
        principal,
        workspace_id="default",
        target_kind="note",
        target_id=target.registry.note_id,
    )
    assert len(backlinks) == 1
    assert backlinks[0].source_id == source.registry.note_id
    assert backlinks[0].origin_type == "wikilink"
    assert backlinks[0].metadata["target_label"] == "Mapa do Caso"

    panel = await service.get_provenance_panel(
        principal,
        workspace_id="default",
        target_kind="note",
        target_id=target.registry.note_id,
    )
    assert [item.backlink_id for item in panel.mentioned_in] == [backlinks[0].backlink_id]
    assert [item.backlink_id for item in panel.cited_by] == [backlinks[0].backlink_id]
    assert panel.used_in_responses == []

    await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            note_id=source.registry.note_id,
            title="Linha Argumentativa",
            slug="linha",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="Sem wikilink nesta versao.",
        ),
    )
    assert await service.list_backlinks(
        principal,
        workspace_id="default",
        source_kind="note",
        source_id=source.registry.note_id,
    ) == []


@pytest.mark.asyncio
async def test_little_bull_manual_backlink_validates_refs_and_audits(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Nota",
        ),
    )
    document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": subgroup.subgroup_id,
            "title": "fonte.md",
            "source_uri": "fonte.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash",
            "confidentiality": "normal",
            "status": "processed",
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )

    backlink = await service.upsert_backlink(
        principal,
        workspace_id="default",
        payload=LittleBullBacklinkRequest(
            source_kind="note",
            source_id=note.registry.note_id,
            target_kind="document",
            target_id=document["document_id"],
            link_text="cita fonte",
            origin_type="manual",
            confidence=0.9,
        ),
    )

    assert backlink.target_id == document["document_id"]
    assert backlink.confidence == 0.9
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    assert any(event.result == "backlink_upserted" for event in events)

    with pytest.raises(HTTPException) as exc:
        await service.upsert_backlink(
            principal,
            workspace_id="default",
            payload=LittleBullBacklinkRequest(
                source_kind="note",
                source_id="missing-note",
                target_kind="document",
                target_id=document["document_id"],
            ),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_little_bull_backlink_rejects_cross_subgroup_refs(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Nota",
        ),
    )
    document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": other_subgroup.subgroup_id,
            "title": "outra-fonte.md",
            "source_uri": "outra-fonte.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash",
            "confidentiality": "normal",
            "status": "processed",
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )

    with pytest.raises(HTTPException) as exc:
        await service.upsert_backlink(
            principal,
            workspace_id="default",
            payload=LittleBullBacklinkRequest(
                source_kind="note",
                source_id=note.registry.note_id,
                target_kind="document",
                target_id=document["document_id"],
            ),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.backlinks == {}


@pytest.mark.asyncio
async def test_little_bull_backlink_rejects_unscoped_graph_edge_origin(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Nota",
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await service.upsert_backlink(
            principal,
            workspace_id="default",
            payload=LittleBullBacklinkRequest(
                source_kind="note",
                source_id=note.registry.note_id,
                target_kind="note",
                target_id=note.registry.note_id,
                graph_edge_origin_id="edge-origin-from-other-scope",
            ),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.backlinks == {}


@pytest.mark.parametrize("target_kind", ["canvas", "trail", "content_map", "conversation", "agent"])
@pytest.mark.asyncio
async def test_little_bull_backlink_rejects_missing_graph_node_refs(tmp_path, target_kind):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Nota",
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await service.upsert_backlink(
            principal,
            workspace_id="default",
            payload=LittleBullBacklinkRequest(
                source_kind="note",
                source_id=note.registry.note_id,
                target_kind=target_kind,
                target_id="missing-ref",
            ),
        )

    assert exc.value.status_code == 404
    assert service.admin_store.backlinks == {}


@pytest.mark.asyncio
async def test_little_bull_cross_subgroup_wikilink_stays_unresolved(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Mapa Externo",
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
            markdown="# Mapa",
        ),
    )

    source = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota Fonte",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="Ver [[Mapa Externo]]",
        ),
    )

    assert source.wiki_links[0].target_note_id is None
    assert source.wiki_links[0].link_status == "unresolved"
    backlinks = await service.list_backlinks(
        principal,
        workspace_id="default",
        source_kind="note",
        source_id=source.registry.note_id,
    )
    assert backlinks[0].target_kind == "note_label"


@pytest.mark.asyncio
async def test_little_bull_source_provenance_tracks_used_in_response(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": subgroup.subgroup_id,
            "title": "fonte.md",
            "source_uri": "fonte.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash",
            "confidentiality": "normal",
            "status": "processed",
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )

    provenance = await service.record_source_provenance(
        principal,
        workspace_id="default",
        payload=LittleBullSourceProvenanceRequest(
            source_kind="answer",
            source_id="answer-1",
            document_id=document["document_id"],
            chunk_id="chunk-1",
            model_id="openrouter/test-model",
            confidence=0.82,
            locator={"page": 1},
            metadata={"cost_usd": "0.0001"},
        ),
    )

    assert provenance.document_id == document["document_id"]
    assert provenance.chunk_id == "chunk-1"
    note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota Fonte",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Nota",
        ),
    )
    await service.record_source_provenance(
        principal,
        workspace_id="default",
        payload=LittleBullSourceProvenanceRequest(
            source_kind="answer",
            source_id="answer-note-only",
            note_id=note.registry.note_id,
        ),
    )
    panel = await service.get_provenance_panel(
        principal,
        workspace_id="default",
        target_kind=" Document ",
        target_id=document["document_id"],
    )
    assert panel.target_kind == "document"
    assert [item.source_id for item in panel.used_in_responses] == ["answer-1"]

    with pytest.raises(HTTPException) as exc:
        await service.record_source_provenance(
            principal,
            workspace_id="default",
            payload=LittleBullSourceProvenanceRequest(
                source_kind="answer",
                source_id="answer-2",
                document_id="missing-doc",
            ),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_little_bull_source_provenance_rejects_cross_subgroup_refs(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Nota",
        ),
    )
    document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": other_subgroup.subgroup_id,
            "title": "outra-fonte.md",
            "source_uri": "outra-fonte.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash",
            "confidentiality": "normal",
            "status": "processed",
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )

    with pytest.raises(HTTPException) as exc:
        await service.record_source_provenance(
            principal,
            workspace_id="default",
            payload=LittleBullSourceProvenanceRequest(
                source_kind="answer",
                source_id="answer-1",
                document_id=document["document_id"],
                note_id=note.registry.note_id,
            ),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.provenance == {}


@pytest.mark.asyncio
async def test_little_bull_source_provenance_rejects_unscoped_agent_and_ledger(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": subgroup.subgroup_id,
            "title": "fonte.md",
            "source_uri": "fonte.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash",
            "confidentiality": "normal",
            "status": "processed",
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )

    with pytest.raises(HTTPException) as exc:
        await service.record_source_provenance(
            principal,
            workspace_id="default",
            payload=LittleBullSourceProvenanceRequest(
                source_kind="answer",
                source_id="answer-1",
                document_id=document["document_id"],
                agent_id="agent-from-other-scope",
                usage_ledger_id="ledger-from-other-scope",
            ),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.provenance == {}


@pytest.mark.asyncio
async def test_little_bull_provenance_panel_rejects_unscoped_target_kind(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)

    with pytest.raises(HTTPException) as exc:
        await service.get_provenance_panel(
            principal,
            workspace_id="default",
            target_kind="answer",
            target_id="answer-1",
        )

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_little_bull_canvas_board_nodes_edges_analysis_and_dossier(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Nota",
        ),
    )
    document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": subgroup.subgroup_id,
            "title": "fonte.md",
            "source_uri": "fonte.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash",
            "confidentiality": "normal",
            "status": "processed",
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )

    board = await service.upsert_canvas_board(
        principal,
        workspace_id="default",
        payload=LittleBullCanvasBoardRequest(
            title="Canvas do Caso",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            layout={"zoom": 1},
        ),
    )
    note_node = await service.upsert_canvas_node(
        principal,
        workspace_id="default",
        canvas_board_id=board.canvas_board_id,
        payload=LittleBullCanvasNodeRequest(
            node_kind="note",
            ref_kind="note",
            ref_id=note.registry.note_id,
            x=10,
            y=20,
        ),
    )
    doc_node = await service.upsert_canvas_node(
        principal,
        workspace_id="default",
        canvas_board_id=board.canvas_board_id,
        payload=LittleBullCanvasNodeRequest(
            node_kind="document",
            ref_kind="document",
            ref_id=document["document_id"],
            x=320,
            y=20,
        ),
    )
    edge = await service.upsert_canvas_edge(
        principal,
        workspace_id="default",
        canvas_board_id=board.canvas_board_id,
        payload=LittleBullCanvasEdgeRequest(
            source_node_id=note_node.canvas_node_id,
            target_node_id=doc_node.canvas_node_id,
            label="cita",
        ),
    )

    detail = await service.get_canvas_board(principal, workspace_id="default", canvas_board_id=board.canvas_board_id)
    assert [node.canvas_node_id for node in detail.nodes] == [note_node.canvas_node_id, doc_node.canvas_node_id]
    assert [item.canvas_edge_id for item in detail.edges] == [edge.canvas_edge_id]
    analysis = await service.analyze_canvas_board(
        principal,
        workspace_id="default",
        canvas_board_id=board.canvas_board_id,
    )
    assert analysis.node_kind_counts == {"note": 1, "document": 1}
    assert analysis.edge_count == 1
    assert len(analysis.clusters) == 1
    dossier = await service.export_canvas_board_dossier(
        principal,
        workspace_id="default",
        canvas_board_id=board.canvas_board_id,
    )
    assert dossier.dossier_kind == "canvas"
    assert dossier.export_policy["requires_lgpd_review"] is True
    assert dossier.export_policy["analysis"]["node_count"] == 2


@pytest.mark.asyncio
async def test_little_bull_canvas_rejects_cross_subgroup_ref_and_cross_board_edge(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota Outro",
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
            markdown="# Nota",
        ),
    )
    board = await service.upsert_canvas_board(
        principal,
        workspace_id="default",
        payload=LittleBullCanvasBoardRequest(
            title="Canvas",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await service.upsert_canvas_node(
            principal,
            workspace_id="default",
            canvas_board_id=board.canvas_board_id,
            payload=LittleBullCanvasNodeRequest(
                node_kind="note",
                ref_kind="note",
                ref_id=note.registry.note_id,
            ),
        )

    assert exc.value.status_code == 422

    card_node = await service.upsert_canvas_node(
        principal,
        workspace_id="default",
        canvas_board_id=board.canvas_board_id,
        payload=LittleBullCanvasNodeRequest(node_kind="card", content={"text": "A"}),
    )
    other_board = await service.upsert_canvas_board(
        principal,
        workspace_id="default",
        payload=LittleBullCanvasBoardRequest(
            title="Outro Canvas",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
        ),
    )
    other_node = await service.upsert_canvas_node(
        principal,
        workspace_id="default",
        canvas_board_id=other_board.canvas_board_id,
        payload=LittleBullCanvasNodeRequest(node_kind="card", content={"text": "B"}),
    )

    with pytest.raises(HTTPException) as exc:
        await service.upsert_canvas_edge(
            principal,
            workspace_id="default",
            canvas_board_id=board.canvas_board_id,
            payload=LittleBullCanvasEdgeRequest(
                source_node_id=card_node.canvas_node_id,
                target_node_id=other_node.canvas_node_id,
            ),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_little_bull_canvas_rejects_cross_board_node_and_edge_id_mutation(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    board = await service.upsert_canvas_board(
        principal,
        workspace_id="default",
        payload=LittleBullCanvasBoardRequest(
            title="Canvas A",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
        ),
    )
    other_board = await service.upsert_canvas_board(
        principal,
        workspace_id="default",
        payload=LittleBullCanvasBoardRequest(
            title="Canvas B",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
        ),
    )
    node_a = await service.upsert_canvas_node(
        principal,
        workspace_id="default",
        canvas_board_id=board.canvas_board_id,
        payload=LittleBullCanvasNodeRequest(node_kind="card", content={"text": "A"}),
    )
    node_b = await service.upsert_canvas_node(
        principal,
        workspace_id="default",
        canvas_board_id=board.canvas_board_id,
        payload=LittleBullCanvasNodeRequest(node_kind="card", content={"text": "B"}),
    )
    other_node = await service.upsert_canvas_node(
        principal,
        workspace_id="default",
        canvas_board_id=other_board.canvas_board_id,
        payload=LittleBullCanvasNodeRequest(node_kind="card", content={"text": "Other"}),
    )
    other_edge = await service.upsert_canvas_edge(
        principal,
        workspace_id="default",
        canvas_board_id=other_board.canvas_board_id,
        payload=LittleBullCanvasEdgeRequest(
            source_node_id=other_node.canvas_node_id,
            target_node_id=other_node.canvas_node_id,
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await service.upsert_canvas_node(
            principal,
            workspace_id="default",
            canvas_board_id=board.canvas_board_id,
            payload=LittleBullCanvasNodeRequest(
                canvas_node_id=other_node.canvas_node_id,
                node_kind="card",
                content={"text": "mutate"},
            ),
        )

    assert exc.value.status_code == 404
    assert service.admin_store.canvas_nodes[other_node.canvas_node_id]["content"] == {"text": "Other"}

    with pytest.raises(HTTPException) as exc:
        await service.upsert_canvas_edge(
            principal,
            workspace_id="default",
            canvas_board_id=board.canvas_board_id,
            payload=LittleBullCanvasEdgeRequest(
                canvas_edge_id=other_edge.canvas_edge_id,
                source_node_id=node_a.canvas_node_id,
                target_node_id=node_b.canvas_node_id,
            ),
        )

    assert exc.value.status_code == 404
    assert service.admin_store.canvas_edges[other_edge.canvas_edge_id]["canvas_board_id"] == other_board.canvas_board_id


@pytest.mark.asyncio
async def test_little_bull_canvas_board_scope_cannot_move_by_slug(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    board = await service.upsert_canvas_board(
        principal,
        workspace_id="default",
        payload=LittleBullCanvasBoardRequest(
            title="Canvas Movel",
            slug="canvas-movel",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await service.upsert_canvas_board(
            principal,
            workspace_id="default",
            payload=LittleBullCanvasBoardRequest(
                title="Canvas Movel",
                slug="canvas-movel",
                group_id=group.group_id,
                subgroup_id=other_subgroup.subgroup_id,
            ),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.canvas_boards[board.canvas_board_id]["subgroup_id"] == subgroup.subgroup_id

    renamed = await service.upsert_canvas_board(
        principal,
        workspace_id="default",
        payload=LittleBullCanvasBoardRequest(
            canvas_board_id=board.canvas_board_id,
            title="Canvas Renomeado",
            slug="canvas-renomeado",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
        ),
    )
    assert renamed.canvas_board_id == board.canvas_board_id
    assert renamed.slug == "canvas_renomeado"


@pytest.mark.asyncio
async def test_little_bull_content_map_root_note_scope_and_slug_guard(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    root = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota Raiz",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Nota raiz",
        ),
    )
    other_note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota Externa",
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
            markdown="# Nota externa",
        ),
    )

    content_map = await service.upsert_content_map(
        principal,
        workspace_id="default",
        payload=LittleBullContentMapRequest(
            title="MOC do Caso",
            slug="moc-caso",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            root_note_id=root.registry.note_id,
            map_body={"sections": [{"title": "Leia isto antes"}]},
        ),
    )

    assert content_map.root_note_id == root.registry.note_id
    assert content_map.map_body["sections"][0]["title"] == "Leia isto antes"
    listed = await service.list_content_maps(
        principal,
        workspace_id="default",
        group_id=group.group_id,
        subgroup_id=subgroup.subgroup_id,
    )
    assert [item.content_map_id for item in listed] == [content_map.content_map_id]

    with pytest.raises(HTTPException) as exc:
        await service.upsert_content_map(
            principal,
            workspace_id="default",
            payload=LittleBullContentMapRequest(
                title="MOC Escopo Errado",
                group_id=group.group_id,
                subgroup_id=subgroup.subgroup_id,
                root_note_id=other_note.registry.note_id,
            ),
        )

    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        await service.upsert_content_map(
            principal,
            workspace_id="default",
            payload=LittleBullContentMapRequest(
                title="MOC do Caso",
                slug="moc-caso",
                group_id=group.group_id,
                subgroup_id=other_subgroup.subgroup_id,
                root_note_id=other_note.registry.note_id,
            ),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.content_maps[content_map.content_map_id]["subgroup_id"] == subgroup.subgroup_id

    with pytest.raises(HTTPException) as exc:
        await service.upsert_content_map(
            principal,
            workspace_id="default",
            payload=LittleBullContentMapRequest(
                content_map_id=content_map.content_map_id,
                title="MOC Movido por ID",
                slug="moc-movido-por-id",
                group_id=group.group_id,
                subgroup_id=other_subgroup.subgroup_id,
                root_note_id=other_note.registry.note_id,
            ),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.content_maps[content_map.content_map_id]["subgroup_id"] == subgroup.subgroup_id

    other_content_map = await service.upsert_content_map(
        principal,
        workspace_id="default",
        payload=LittleBullContentMapRequest(
            title="MOC Outro Escopo",
            slug="moc-outro-escopo",
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
            root_note_id=other_note.registry.note_id,
        ),
    )
    listed = await service.list_content_maps(
        principal,
        workspace_id="default",
        group_id=group.group_id,
        subgroup_id=subgroup.subgroup_id,
    )
    other_listed = await service.list_content_maps(
        principal,
        workspace_id="default",
        group_id=group.group_id,
        subgroup_id=other_subgroup.subgroup_id,
    )
    assert [item.content_map_id for item in listed] == [content_map.content_map_id]
    assert [item.content_map_id for item in other_listed] == [other_content_map.content_map_id]

    renamed = await service.upsert_content_map(
        principal,
        workspace_id="default",
        payload=LittleBullContentMapRequest(
            content_map_id=content_map.content_map_id,
            title="MOC do Caso Renomeado",
            slug="moc-caso-renomeado",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            root_note_id=root.registry.note_id,
        ),
    )
    assert renamed.content_map_id == content_map.content_map_id
    assert renamed.slug == "moc_caso_renomeado"


@pytest.mark.asyncio
async def test_little_bull_knowledge_trail_steps_validate_refs_and_order(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota Guia",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Nota guia",
        ),
    )
    other_note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota de Outro Escopo",
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
            markdown="# Fora",
        ),
    )
    other_document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": other_subgroup.subgroup_id,
            "title": "outra-fonte.md",
            "source_uri": "outra-fonte.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "other-hash",
            "confidentiality": "normal",
            "status": "processed",
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )
    document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": subgroup.subgroup_id,
            "title": "fonte.md",
            "source_uri": "fonte.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash",
            "confidentiality": "normal",
            "status": "processed",
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )
    board = await service.upsert_canvas_board(
        principal,
        workspace_id="default",
        payload=LittleBullCanvasBoardRequest(
            title="Canvas Linha Argumentativa",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
        ),
    )
    other_board = await service.upsert_canvas_board(
        principal,
        workspace_id="default",
        payload=LittleBullCanvasBoardRequest(
            title="Canvas Outro Escopo",
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
        ),
    )
    trail = await service.upsert_knowledge_trail(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeTrailRequest(
            title="Leia Isto Antes",
            slug="leia-isto-antes",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            trail_type="study",
        ),
    )

    await service.upsert_knowledge_trail_step(
        principal,
        workspace_id="default",
        knowledge_trail_id=trail.knowledge_trail_id,
        payload=LittleBullKnowledgeTrailStepRequest(
            step_order=2,
            title="Analise o canvas",
            step_kind="canvas",
            canvas_board_id=board.canvas_board_id,
        ),
    )
    await service.upsert_knowledge_trail_step(
        principal,
        workspace_id="default",
        knowledge_trail_id=trail.knowledge_trail_id,
        payload=LittleBullKnowledgeTrailStepRequest(
            step_order=0,
            title="Comece pela nota",
            step_kind="note",
            note_id=note.registry.note_id,
        ),
    )
    await service.upsert_knowledge_trail_step(
        principal,
        workspace_id="default",
        knowledge_trail_id=trail.knowledge_trail_id,
        payload=LittleBullKnowledgeTrailStepRequest(
            step_order=1,
            title="Confira a fonte",
            step_kind="document",
            document_id=document["document_id"],
        ),
    )

    detail = await service.get_knowledge_trail(
        principal,
        workspace_id="default",
        knowledge_trail_id=trail.knowledge_trail_id,
    )
    assert [step.title for step in detail.steps] == [
        "Comece pela nota",
        "Confira a fonte",
        "Analise o canvas",
    ]
    assert detail.steps[0].note_id == note.registry.note_id
    assert detail.steps[1].document_id == document["document_id"]
    assert detail.steps[2].canvas_board_id == board.canvas_board_id

    with pytest.raises(HTTPException) as exc:
        await service.upsert_knowledge_trail_step(
            principal,
            workspace_id="default",
            knowledge_trail_id=trail.knowledge_trail_id,
            payload=LittleBullKnowledgeTrailStepRequest(
                step_order=3,
                title="Fora do escopo",
                note_id=other_note.registry.note_id,
            ),
        )

    assert exc.value.status_code == 422
    assert len(service.admin_store.trail_steps) == 3

    with pytest.raises(HTTPException) as exc:
        await service.upsert_knowledge_trail_step(
            principal,
            workspace_id="default",
            knowledge_trail_id=trail.knowledge_trail_id,
            payload=LittleBullKnowledgeTrailStepRequest(
                step_order=3,
                title="Documento fora do escopo",
                document_id=other_document["document_id"],
            ),
        )

    assert exc.value.status_code == 422
    assert len(service.admin_store.trail_steps) == 3

    with pytest.raises(HTTPException) as exc:
        await service.upsert_knowledge_trail_step(
            principal,
            workspace_id="default",
            knowledge_trail_id=trail.knowledge_trail_id,
            payload=LittleBullKnowledgeTrailStepRequest(
                step_order=3,
                title="Canvas fora do escopo",
                canvas_board_id=other_board.canvas_board_id,
            ),
        )

    assert exc.value.status_code == 422
    assert len(service.admin_store.trail_steps) == 3


@pytest.mark.asyncio
async def test_little_bull_knowledge_trail_scope_and_step_ids_cannot_cross_trails(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    trail = await service.upsert_knowledge_trail(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeTrailRequest(
            title="Linha Argumentativa",
            slug="linha-argumentativa",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
        ),
    )
    other_trail = await service.upsert_knowledge_trail(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeTrailRequest(
            title="Linha do Tempo",
            slug="linha-do-tempo",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
        ),
    )
    scoped_other_trail = await service.upsert_knowledge_trail(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeTrailRequest(
            title="Trilha Outro Escopo",
            slug="trilha-outro-escopo",
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
        ),
    )
    other_step = await service.upsert_knowledge_trail_step(
        principal,
        workspace_id="default",
        knowledge_trail_id=other_trail.knowledge_trail_id,
        payload=LittleBullKnowledgeTrailStepRequest(
            step_order=0,
            title="Outro passo",
            instructions="Passo de outra trilha",
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await service.upsert_knowledge_trail(
            principal,
            workspace_id="default",
            payload=LittleBullKnowledgeTrailRequest(
                title="Linha Argumentativa",
                slug="linha-argumentativa",
                group_id=group.group_id,
                subgroup_id=other_subgroup.subgroup_id,
            ),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.trails[trail.knowledge_trail_id]["subgroup_id"] == subgroup.subgroup_id

    with pytest.raises(HTTPException) as exc:
        await service.upsert_knowledge_trail(
            principal,
            workspace_id="default",
            payload=LittleBullKnowledgeTrailRequest(
                knowledge_trail_id=trail.knowledge_trail_id,
                title="Linha Argumentativa por ID",
                slug="linha-argumentativa-por-id",
                group_id=group.group_id,
                subgroup_id=other_subgroup.subgroup_id,
            ),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.trails[trail.knowledge_trail_id]["subgroup_id"] == subgroup.subgroup_id

    renamed = await service.upsert_knowledge_trail(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeTrailRequest(
            knowledge_trail_id=trail.knowledge_trail_id,
            title="Linha Argumentativa Renomeada",
            slug="linha-argumentativa-renomeada",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
        ),
    )
    assert renamed.knowledge_trail_id == trail.knowledge_trail_id
    assert renamed.slug == "linha_argumentativa_renomeada"

    listed = await service.list_knowledge_trails(
        principal,
        workspace_id="default",
        group_id=group.group_id,
        subgroup_id=subgroup.subgroup_id,
    )
    other_listed = await service.list_knowledge_trails(
        principal,
        workspace_id="default",
        group_id=group.group_id,
        subgroup_id=other_subgroup.subgroup_id,
    )
    assert {item.knowledge_trail_id for item in listed} == {
        trail.knowledge_trail_id,
        other_trail.knowledge_trail_id,
    }
    assert [item.knowledge_trail_id for item in other_listed] == [scoped_other_trail.knowledge_trail_id]

    with pytest.raises(HTTPException) as exc:
        await service.upsert_knowledge_trail_step(
            principal,
            workspace_id="default",
            knowledge_trail_id=trail.knowledge_trail_id,
            payload=LittleBullKnowledgeTrailStepRequest(
                knowledge_trail_step_id=other_step.knowledge_trail_step_id,
                step_order=0,
                title="Tentativa de mutacao",
            ),
        )

    assert exc.value.status_code == 404
    assert service.admin_store.trail_steps[other_step.knowledge_trail_step_id][
        "knowledge_trail_id"
    ] == other_trail.knowledge_trail_id


@pytest.mark.asyncio
async def test_little_bull_obsidian_graph_scopes_filters_and_focus_without_data_plane(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    note_a = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Alpha",
            slug="alpha",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Alpha",
        ),
    )
    note_b = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Beta",
            slug="beta",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Beta",
        ),
    )
    other_note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Outro Escopo",
            slug="outro-escopo",
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
            markdown="# Fora",
        ),
    )
    document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": subgroup.subgroup_id,
            "title": "fonte.md",
            "source_uri": "fonte.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash",
            "confidentiality": "normal",
            "status": "processed",
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )
    await service.upsert_backlink(
        principal,
        workspace_id="default",
        payload=LittleBullBacklinkRequest(
            source_kind="note",
            source_id=note_a.registry.note_id,
            target_kind="note",
            target_id=note_b.registry.note_id,
            link_text="Alpha para Beta",
            origin_type="manual",
            confidence=0.91,
        ),
    )
    await service.upsert_backlink(
        principal,
        workspace_id="default",
        payload=LittleBullBacklinkRequest(
            source_kind="note",
            source_id=note_a.registry.note_id,
            target_kind="document",
            target_id=document["document_id"],
            link_text="Fonte",
            origin_type="wikilink",
        ),
    )
    await service.admin_store.upsert_backlink(
        {
            "source_kind": "note",
            "source_id": note_a.registry.note_id,
            "target_kind": "note",
            "target_id": other_note.registry.note_id,
            "link_text": "Fora",
            "origin_type": "manual",
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )
    trail = await service.upsert_knowledge_trail(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeTrailRequest(
            title="Leia Isto Antes",
            slug="leia-isto-antes",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
        ),
    )
    await service.upsert_knowledge_trail_step(
        principal,
        workspace_id="default",
        knowledge_trail_id=trail.knowledge_trail_id,
        payload=LittleBullKnowledgeTrailStepRequest(
            step_order=0,
            title="Comece pela Alpha",
            step_kind="note",
            note_id=note_a.registry.note_id,
        ),
    )

    async def fail_data_plane(*_args, **_kwargs):
        raise AssertionError("Obsidian graph must not activate the data plane")

    service._require_data_plane = fail_data_plane
    central_node_id = f"note:{note_a.registry.note_id}"
    graph = await service.get_obsidian_graph(
        principal,
        workspace_id="default",
        scope="subgroup",
        group_id=group.group_id,
        subgroup_id=subgroup.subgroup_id,
        central_node_id=central_node_id,
    )

    assert graph.scope == "subgroup"
    assert service.rag.query_calls == 0
    assert {node.node_id for node in graph.nodes} == {
        central_node_id,
        f"note:{note_b.registry.note_id}",
        f"document:{document['document_id']}",
        f"trail:{trail.knowledge_trail_id}",
    }
    assert f"note:{other_note.registry.note_id}" not in {node.node_id for node in graph.nodes}
    assert {edge.origin_type for edge in graph.edges} == {"manual", "wikilink", "trail_step"}
    assert graph.chat_context == {
        "enabled": True,
        "focus_node_id": central_node_id,
        "focus_label": "Alpha",
        "neighbor_count": 3,
        "edge_count": 3,
        "context_kind": "obsidian_graph",
    }
    assert len(graph.clusters) == 1
    assert graph.clusters[0].node_count == 4
    assert graph.trails == [
        {
            "knowledge_trail_id": trail.knowledge_trail_id,
            "title": "Leia Isto Antes",
            "trail_type": "study",
            "node_ids": [central_node_id],
        }
    ]

    group_graph = await service.get_obsidian_graph(
        principal,
        workspace_id="default",
        scope="group",
        group_id=group.group_id,
    )
    group_node_ids = {node.node_id for node in group_graph.nodes}
    assert f"note:{other_note.registry.note_id}" in group_node_ids
    assert central_node_id in group_node_ids
    assert group_graph.filters["group_id"] == group.group_id

    workspace_graph = await service.get_obsidian_graph(
        principal,
        workspace_id="default",
        scope="workspace",
    )
    assert group_node_ids.issubset({node.node_id for node in workspace_graph.nodes})

    manual_graph = await service.get_obsidian_graph(
        principal,
        workspace_id="default",
        scope="subgroup",
        group_id=group.group_id,
        subgroup_id=subgroup.subgroup_id,
        central_node_id=central_node_id,
        origin_type="manual",
    )

    assert {edge.origin_type for edge in manual_graph.edges} == {"manual"}
    assert {node.node_id for node in manual_graph.nodes} == {
        central_node_id,
        f"note:{note_b.registry.note_id}",
    }


@pytest.mark.asyncio
async def test_little_bull_obsidian_graph_rejects_invalid_scoped_requests(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, _subgroup = await _create_group_and_subgroup(service, principal)

    with pytest.raises(HTTPException) as exc:
        await service.get_obsidian_graph(
            principal,
            workspace_id="default",
            scope="group",
        )
    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        await service.get_obsidian_graph(
            principal,
            workspace_id="default",
            scope="subgroup",
            group_id=group.group_id,
        )
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_little_bull_inbox_validates_refs_status_and_filters(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": subgroup.subgroup_id,
            "title": "triagem.md",
            "source_uri": "triagem.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash",
            "confidentiality": "normal",
            "status": "processed",
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )
    other_note = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota Outro Escopo",
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
            markdown="# Fora",
        ),
    )

    item = await service.upsert_inbox_item(
        principal,
        workspace_id="default",
        payload=LittleBullInboxItemRequest(
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            item_kind="document",
            title="Triar documento",
            source_kind="document",
            source_id=document["document_id"],
            priority="high",
        ),
    )

    assert item.source_kind == "document"
    listed = await service.list_inbox_items(
        principal,
        workspace_id="default",
        status_filter="open",
        group_id=group.group_id,
        subgroup_id=subgroup.subgroup_id,
    )
    assert [inbox_item.inbox_item_id for inbox_item in listed] == [item.inbox_item_id]

    updated = await service.update_inbox_item_status(
        principal,
        workspace_id="default",
        inbox_item_id=item.inbox_item_id,
        payload=LittleBullInboxItemStatusRequest(status="done", metadata={"resolved_by": "test"}),
    )
    assert updated.status == "done"
    assert updated.metadata["resolved_by"] == "test"
    assert await service.list_inbox_items(
        principal,
        workspace_id="default",
        status_filter="open",
        group_id=group.group_id,
        subgroup_id=subgroup.subgroup_id,
    ) == []

    target_open = await service.upsert_inbox_item(
        principal,
        workspace_id="default",
        payload=LittleBullInboxItemRequest(
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            item_kind="quick_note",
            title="Aberto no alvo",
        ),
    )
    await service.upsert_inbox_item(
        principal,
        workspace_id="default",
        payload=LittleBullInboxItemRequest(
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
            item_kind="quick_note",
            title="Aberto em outro subgrupo",
        ),
    )
    listed = await service.list_inbox_items(
        principal,
        workspace_id="default",
        status_filter="open",
        group_id=group.group_id,
        subgroup_id=subgroup.subgroup_id,
    )
    assert [inbox_item.inbox_item_id for inbox_item in listed] == [target_open.inbox_item_id]

    with pytest.raises(HTTPException) as exc:
        await service.upsert_inbox_item(
            principal,
            workspace_id="default",
            payload=LittleBullInboxItemRequest(
                group_id=group.group_id,
                subgroup_id=subgroup.subgroup_id,
                item_kind="note",
                title="Fonte errada",
                source_kind="note",
                source_id=other_note.registry.note_id,
            ),
        )

    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        await service.upsert_inbox_item(
            principal,
            workspace_id="default",
            payload=LittleBullInboxItemRequest(
                inbox_item_id=target_open.inbox_item_id,
                item_kind="quick_note",
                title="Remover escopo",
            ),
        )

    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        await service.upsert_inbox_item(
            principal,
            workspace_id="default",
            payload=LittleBullInboxItemRequest(
                item_kind="document",
                title="Fonte sem escopo",
                source_kind="document",
                source_id=document["document_id"],
            ),
        )

    assert exc.value.status_code == 422

    service.admin_store.conversations["conversation-1"] = {
        "conversation_id": "conversation-1",
        "tenant_id": "default",
        "workspace_id": "default",
        "user_id": principal.user_id,
    }
    service.admin_store.suggestions["suggestion-1"] = {
        "suggestion_id": "suggestion-1",
        "tenant_id": "default",
        "workspace_id": "default",
        "user_id": principal.user_id,
    }
    conversation_item = await service.upsert_inbox_item(
        principal,
        workspace_id="default",
        payload=LittleBullInboxItemRequest(
            item_kind="conversation",
            title="Rever conversa",
            source_kind="conversation",
            source_id="conversation-1",
        ),
    )
    suggestion_item = await service.upsert_inbox_item(
        principal,
        workspace_id="default",
        payload=LittleBullInboxItemRequest(
            item_kind="suggestion",
            title="Rever sugestao",
            source_kind="suggestion",
            source_id="suggestion-1",
        ),
    )
    assert conversation_item.source_kind == "conversation"
    assert suggestion_item.source_kind == "suggestion"

    with pytest.raises(HTTPException) as exc:
        await service.upsert_inbox_item(
            principal,
            workspace_id="default",
            payload=LittleBullInboxItemRequest(
                item_kind="conversation",
                title="Conversa ausente",
                source_kind="conversation",
                source_id="missing-conversation",
            ),
        )

    assert exc.value.status_code == 404

    with pytest.raises(HTTPException) as exc:
        await service.upsert_inbox_item(
            principal,
            workspace_id="default",
            payload=LittleBullInboxItemRequest(
                item_kind="suggestion",
                title="Sugestao ausente",
                source_kind="suggestion",
                source_id="missing-suggestion",
            ),
        )

    assert exc.value.status_code == 404

    with pytest.raises(HTTPException) as exc:
        await service.upsert_inbox_item(
            principal,
            workspace_id="default",
            payload=LittleBullInboxItemRequest(
                subgroup_id=subgroup.subgroup_id,
                item_kind="quick_note",
                title="Sem grupo",
            ),
        )

    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        await service.upsert_inbox_item(
            principal,
            workspace_id="default",
            payload=LittleBullInboxItemRequest(
                inbox_item_id=item.inbox_item_id,
                group_id=group.group_id,
                subgroup_id=other_subgroup.subgroup_id,
                item_kind="quick_note",
                title="Mover item",
            ),
        )

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_little_bull_curator_suggestions_are_pending_and_do_not_mutate_graph(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    source = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Fonte",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Fonte",
        ),
    )
    target = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Alvo",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Alvo",
        ),
    )
    conversation = await service.save_conversation(
        principal,
        LittleBullConversationSaveRequest(
            workspace_id="default",
            scope_snapshot={
                "group_id": group.group_id,
                "subgroup_id": subgroup.subgroup_id,
            },
            messages=[{"role": "user", "content": "Transforme em nota"}],
        ),
    )
    board = await service.upsert_canvas_board(
        principal,
        workspace_id="default",
        payload=LittleBullCanvasBoardRequest(
            title="Canvas Curador",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
        ),
    )

    backlink = await service.create_curator_suggestion(
        principal,
        LittleBullCuratorSuggestionRequest(
            workspace_id="default",
            suggestion_kind="backlink",
            source_kind="note",
            source_id=source.registry.note_id,
            target_kind="note",
            target_id=target.registry.note_id,
            title="Ligar Fonte ao Alvo",
        ),
    )
    moc = await service.create_curator_suggestion(
        principal,
        LittleBullCuratorSuggestionRequest(
            workspace_id="default",
            suggestion_kind="content_map",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            title="Criar MOC do caso",
        ),
    )
    subgroup_suggestion = await service.create_curator_suggestion(
        principal,
        LittleBullCuratorSuggestionRequest(
            workspace_id="default",
            suggestion_kind="subgroup",
            group_id=group.group_id,
            title="Criar subgrupo de prazos",
            metadata={"proposed_slug": "prazos"},
        ),
    )
    conversation_note = await service.create_curator_suggestion(
        principal,
        LittleBullCuratorSuggestionRequest(
            workspace_id="default",
            suggestion_kind="conversation_note",
            source_kind="conversation",
            source_id=conversation.conversation_id,
        ),
    )
    canvas_dossier = await service.create_curator_suggestion(
        principal,
        LittleBullCuratorSuggestionRequest(
            workspace_id="default",
            suggestion_kind="canvas_dossier",
            source_kind="canvas",
            source_id=board.canvas_board_id,
        ),
    )

    assert backlink.requires_approval is True
    assert backlink.inbox_item["metadata"]["critical_graph_mutation"] is True
    assert backlink.inbox_item["metadata"]["target_id"] == target.registry.note_id
    assert moc.inbox_item["metadata"]["curator_kind"] == "content_map"
    assert subgroup_suggestion.inbox_item["metadata"]["curator_kind"] == "subgroup"
    assert conversation_note.inbox_item["metadata"]["conversation_id"] == conversation.conversation_id
    assert canvas_dossier.inbox_item["metadata"]["canvas_board_id"] == board.canvas_board_id
    assert service.admin_store.backlinks == {}
    assert service.admin_store.content_maps == {}
    assert service.admin_store.dossiers == {}
    listed = await service.list_curator_suggestions(
        principal,
        workspace_id="default",
    )
    assert {item.inbox_item_id for item in listed} == {
        backlink.inbox_item["inbox_item_id"],
        moc.inbox_item["inbox_item_id"],
        subgroup_suggestion.inbox_item["inbox_item_id"],
        conversation_note.inbox_item["inbox_item_id"],
        canvas_dossier.inbox_item["inbox_item_id"],
    }
    with pytest.raises(HTTPException) as exc:
        await service.apply_curator_suggestion(
            principal,
            workspace_id="default",
            inbox_item_id=backlink.inbox_item["inbox_item_id"],
        )

    assert exc.value.status_code == 409
    assert service.admin_store.backlinks == {}
    assert service.admin_store.content_maps == {}
    assert service.admin_store.dossiers == {}


@pytest.mark.asyncio
async def test_little_bull_curator_rejects_cross_scope_backlink_suggestion_without_mutation(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro-curador",
        ),
    )
    source = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Fonte",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            markdown="# Fonte",
        ),
    )
    target = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Alvo externo",
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
            markdown="# Alvo",
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await service.create_curator_suggestion(
            principal,
            LittleBullCuratorSuggestionRequest(
                workspace_id="default",
                suggestion_kind="backlink",
                source_kind="note",
                source_id=source.registry.note_id,
                target_kind="note",
                target_id=target.registry.note_id,
            ),
        )

    assert exc.value.status_code == 422
    assert service.admin_store.inbox_items == {}
    assert service.admin_store.backlinks == {}


@pytest.mark.asyncio
async def test_little_bull_daily_note_creates_markdown_and_uses_open_inbox(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    inbox_item = await service.upsert_inbox_item(
        principal,
        workspace_id="default",
        payload=LittleBullInboxItemRequest(
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            item_kind="quick_note",
            title="Revisar pendencia",
        ),
    )
    done_item = await service.upsert_inbox_item(
        principal,
        workspace_id="default",
        payload=LittleBullInboxItemRequest(
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            item_kind="quick_note",
            title="Pendencia concluida",
        ),
    )
    await service.update_inbox_item_status(
        principal,
        workspace_id="default",
        inbox_item_id=done_item.inbox_item_id,
        payload=LittleBullInboxItemStatusRequest(status="done"),
    )
    await service.upsert_inbox_item(
        principal,
        workspace_id="default",
        payload=LittleBullInboxItemRequest(
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
            item_kind="quick_note",
            title="Pendencia outro subgrupo",
        ),
    )

    daily = await service.ensure_daily_note(
        principal,
        workspace_id="default",
        payload=LittleBullDailyNoteRequest(
            note_date="2026-04-30",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            summary="Dia de triagem",
            decisions=[{"title": "Manter base limpa"}],
            cost_snapshot={"today_usd": "0.00"},
        ),
    )

    assert daily.note_date == "2026-04-30"
    assert [item["inbox_item_id"] for item in daily.pending_items] == [inbox_item.inbox_item_id]
    latest = await service.admin_store.get_latest_markdown_note(
        daily.note_id,
        tenant_id="default",
        workspace_id="default",
    )
    assert "Dia de triagem" in latest["markdown"]
    assert "Revisar pendencia" in latest["markdown"]
    assert service.admin_store.notes[daily.note_id]["metadata"]["daily_note"] is True

    updated = await service.ensure_daily_note(
        principal,
        workspace_id="default",
        payload=LittleBullDailyNoteRequest(
            note_date="2026-04-30",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            summary="Dia atualizado",
            pending_items=[{"title": "Pendencia manual"}],
        ),
    )

    assert updated.daily_note_id == daily.daily_note_id
    assert updated.note_id == daily.note_id
    assert updated.pending_items == [{"title": "Pendencia manual"}]
    assert service.admin_store.markdown_notes[daily.note_id][-1]["version_number"] == 2
    listed = await service.list_daily_notes(principal, workspace_id="default")
    assert [item.daily_note_id for item in listed] == [daily.daily_note_id]

    with pytest.raises(HTTPException) as exc:
        await service.ensure_daily_note(
            principal,
            workspace_id="default",
            payload=LittleBullDailyNoteRequest(
                note_date="2026-04-30",
                group_id=group.group_id,
                subgroup_id=other_subgroup.subgroup_id,
                summary="Mover daily",
            ),
        )

    assert exc.value.status_code == 422

    conflicting = await service.upsert_markdown_note(
        principal,
        workspace_id="default",
        payload=LittleBullMarkdownNoteRequest(
            title="Nota com slug de daily",
            slug="daily-2026-05-01",
            group_id=group.group_id,
            subgroup_id=other_subgroup.subgroup_id,
            markdown="# Nao mover",
        ),
    )
    with pytest.raises(HTTPException) as exc:
        await service.ensure_daily_note(
            principal,
            workspace_id="default",
            payload=LittleBullDailyNoteRequest(
                note_date="2026-05-01",
                group_id=group.group_id,
                subgroup_id=subgroup.subgroup_id,
                summary="Conflito de slug",
            ),
        )

    assert exc.value.status_code == 409
    assert service.admin_store.notes[conflicting.registry.note_id]["subgroup_id"] == other_subgroup.subgroup_id

    with pytest.raises(HTTPException) as exc:
        await service.ensure_daily_note(
            principal,
            workspace_id="default",
            payload=LittleBullDailyNoteRequest(
                note_date="30/04/2026",
                group_id=group.group_id,
                subgroup_id=subgroup.subgroup_id,
            ),
        )

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_little_bull_agent_builder_requires_review_before_publish(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    builder_model = await service.upsert_model_setting(
        principal,
        workspace_id="default",
        payload=LittleBullModelSetting(
            usage="agent_builder",
            provider="openrouter",
            binding="openai",
            binding_host="https://openrouter.ai/api/v1",
            model_id="openai/builder-test",
            display_name="Builder Test",
        ),
    )
    runtime_model = await service.upsert_model_setting(
        principal,
        workspace_id="default",
        payload=LittleBullModelSetting(
            usage="agent",
            provider="openrouter",
            binding="openai",
            binding_host="https://openrouter.ai/api/v1",
            model_id="openai/runtime-test",
            display_name="Runtime Test",
        ),
    )

    session = await service.upsert_agent_builder_session(
        principal,
        workspace_id="default",
        payload=LittleBullAgentBuilderSessionRequest(
            model_setting_id=builder_model.model_setting_id,
            user_message="Agente juridico para resumir documentos com fontes e pedir aprovacao humana.",
            generated_config={"model_setting_id": runtime_model.model_setting_id},
        ),
    )

    assert session.status == "draft"
    assert session.model_setting_id == builder_model.model_setting_id
    assert session.requires_review is True
    assert session.readiness_score >= 80
    assert session.generated_config["enabled"] is False
    assert session.generated_config["model_setting_id"] == runtime_model.model_setting_id
    assert [entry["role"] for entry in session.builder_transcript] == ["user", "assistant"]
    listed = await service.list_agent_builder_sessions(principal, workspace_id="default", status_filter="draft")
    assert [item.agent_builder_session_id for item in listed] == [session.agent_builder_session_id]
    decoy = await service.upsert_agent_builder_session(
        principal,
        workspace_id="default",
        payload=LittleBullAgentBuilderSessionRequest(
            user_message="Agente decoy para validar filtro de drafts.",
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await service.publish_agent_builder_session(
            principal,
            workspace_id="default",
            agent_builder_session_id=session.agent_builder_session_id,
            payload=LittleBullAgentBuilderPublishRequest(approved=False),
        )

    assert exc.value.status_code == 409

    published = await service.publish_agent_builder_session(
        principal,
        workspace_id="default",
        agent_builder_session_id=session.agent_builder_session_id,
        payload=LittleBullAgentBuilderPublishRequest(approved=True, enabled=True),
    )

    assert published.status == "published"
    assert published.requires_review is False
    assert published.agent_id in service.admin_store.agents
    assert service.admin_store.agents[published.agent_id]["enabled"] is True
    assert service.admin_store.agents[published.agent_id]["model_setting_id"] == runtime_model.model_setting_id
    listed = await service.list_agent_builder_sessions(principal, workspace_id="default", status_filter="draft")
    assert [item.agent_builder_session_id for item in listed] == [decoy.agent_builder_session_id]

    not_ready = await service.upsert_agent_builder_session(
        principal,
        workspace_id="default",
        payload=LittleBullAgentBuilderSessionRequest(
            user_message="Agente sem principios.",
            generated_config={
                "name": "Agente Sem Principios",
                "config": {"ethics": {"principles": []}},
            },
        ),
    )
    agent_count = len(service.admin_store.agents)
    with pytest.raises(HTTPException) as exc:
        await service.publish_agent_builder_session(
            principal,
            workspace_id="default",
            agent_builder_session_id=not_ready.agent_builder_session_id,
            payload=LittleBullAgentBuilderPublishRequest(approved=True),
        )

    assert exc.value.status_code == 422
    assert len(service.admin_store.agents) == agent_count

    with pytest.raises(HTTPException) as exc:
        await service.upsert_agent_builder_session(
            principal,
            workspace_id="default",
            payload=LittleBullAgentBuilderSessionRequest(
                model_setting_id="missing-model",
                user_message="Use modelo inexistente.",
            ),
        )

    assert exc.value.status_code == 404

    embedding_model = await service.upsert_model_setting(
        principal,
        workspace_id="default",
        payload=LittleBullModelSetting(
            usage="embedding",
            provider="openrouter",
            binding="openai",
            binding_host="https://openrouter.ai/api/v1",
            model_id="openai/embedding-test",
            display_name="Embedding Test",
        ),
    )
    with pytest.raises(HTTPException) as exc:
        await service.upsert_agent_builder_session(
            principal,
            workspace_id="default",
            payload=LittleBullAgentBuilderSessionRequest(
                model_setting_id=embedding_model.model_setting_id,
                user_message="Nao aceite modelo de embedding no builder.",
            ),
        )

    assert exc.value.status_code == 422
    with pytest.raises(HTTPException) as exc:
        await service.upsert_agent_builder_session(
            principal,
            workspace_id="default",
            payload=LittleBullAgentBuilderSessionRequest(
                model_setting_id=builder_model.model_setting_id,
                user_message="Nao aceite embedding como modelo runtime do agente.",
                generated_config={"model_setting_id": embedding_model.model_setting_id},
            ),
        )

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_little_bull_agent_config_cannot_update_foreign_workspace_agent(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    own_agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Local",
            description="Agente do workspace default",
            enabled=False,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder no workspace atual."},
                "tests": [{"name": "escopo", "input": "Pergunta", "expected_behavior": "Manter workspace"}],
            },
        ),
    )
    service.admin_store.agents["foreign-agent"] = {
        **service.admin_store.agents[own_agent.agent_id],
        "agent_id": "foreign-agent",
        "workspace_id": "other-workspace",
    }

    with pytest.raises(HTTPException) as exc:
        await service.upsert_agent_config(
            principal,
            workspace_id="default",
            payload=LittleBullAgentConfig(
                agent_id="foreign-agent",
                name="Tentativa de takeover",
                description="Nao deve atualizar agente de outro workspace",
                enabled=False,
                tools=["query_knowledge"],
                config={
                    "identity": {"mission": "Invadir escopo."},
                    "tests": [{"name": "escopo", "input": "Pergunta", "expected_behavior": "Bloquear"}],
                },
            ),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_little_bull_model_setting_cannot_update_foreign_workspace_setting(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    service.admin_store.models["foreign-model"] = {
        "model_setting_id": "foreign-model",
        "tenant_id": "default",
        "workspace_id": "other-workspace",
        "usage": "chat",
        "provider": "openrouter",
        "binding": "openai",
        "binding_host": "https://openrouter.ai/api/v1",
        "model_id": "openai/foreign",
        "display_name": "Foreign Model",
        "enabled": True,
        "is_default": False,
        "config": {},
        "created_by": principal.user_id,
        "updated_by": principal.user_id,
        "created_at": "2026-04-29T00:00:00Z",
        "updated_at": "2026-04-29T00:00:00Z",
    }

    with pytest.raises(HTTPException) as exc:
        await service.upsert_model_setting(
            principal,
            workspace_id="default",
            payload=LittleBullModelSetting(
                model_setting_id="foreign-model",
                usage="chat",
                provider="openrouter",
                binding="openai",
                binding_host="https://openrouter.ai/api/v1",
                model_id="openai/takeover",
                display_name="Takeover",
            ),
        )

    assert exc.value.status_code == 404
    assert service.admin_store.models["foreign-model"]["model_id"] == "openai/foreign"


@pytest.mark.asyncio
async def test_little_bull_agent_runtime_models_must_be_chat_or_agent(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    builder_model = await service.upsert_model_setting(
        principal,
        workspace_id="default",
        payload=LittleBullModelSetting(
            usage="agent_builder",
            provider="openrouter",
            binding="openai",
            binding_host="https://openrouter.ai/api/v1",
            model_id="openai/builder-only",
            display_name="Builder Only",
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await service.upsert_agent_config(
            principal,
            workspace_id="default",
            payload=LittleBullAgentConfig(
                name="Agente Runtime Invalido",
                description="Nao deve usar modelo builder em runtime",
                enabled=True,
                model_setting_id=builder_model.model_setting_id,
                tools=["query_knowledge"],
                config={
                    "identity": {"mission": "Responder com fontes."},
                    "tests": [{"name": "runtime", "input": "Pergunta", "expected_behavior": "Bloquear"}],
                },
            ),
        )

    assert exc.value.status_code == 422

    service.admin_store.agents["legacy-builder-agent"] = {
        "agent_id": "legacy-builder-agent",
        "tenant_id": "default",
        "workspace_id": "default",
        "name": "Legacy Builder Agent",
        "description": "Persistido antes da regra runtime",
        "enabled": True,
        "model_setting_id": builder_model.model_setting_id,
        "system_prompt": "",
        "response_rules": [],
        "tools": ["query_knowledge"],
        "config": {
            "identity": {"mission": "Responder com fontes."},
            "tests": [{"name": "runtime", "input": "Pergunta", "expected_behavior": "Bloquear"}],
        },
        "created_by": principal.user_id,
        "updated_by": principal.user_id,
        "created_at": "2026-04-29T00:00:00Z",
        "updated_at": "2026-04-29T00:00:00Z",
    }
    with pytest.raises(HTTPException) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                agent_id="legacy-builder-agent",
                query="Explique com fontes.",
            ),
        )

    assert exc.value.status_code == 422
    assert service.rag.query_calls == 0


@pytest.mark.asyncio
async def test_little_bull_query_uses_scoped_runtime_agent_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    runtime_model = await service.upsert_model_setting(
        principal,
        workspace_id="default",
        payload=LittleBullModelSetting(
            usage="agent",
            provider="openrouter",
            binding="openai",
            binding_host="https://openrouter.ai/api/v1",
            model_id="openai/runtime-positive",
            display_name="Runtime Positive",
        ),
    )
    agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Runtime Positivo",
            description="Agente com runtime model scoped",
            enabled=True,
            model_setting_id=runtime_model.model_setting_id,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder com fontes."},
                "tests": [{"name": "runtime", "input": "Pergunta", "expected_behavior": "Usar modelo scoped"}],
            },
        ),
    )

    response = await service.query(
        principal,
        LittleBullQueryRequest(
            workspace_id="default",
            agent_id=agent.agent_id,
            query="Explique com fontes.",
        ),
    )

    assert response.response.startswith("Answer for")
    assert service.rag.last_query_param.model_func is not None
    assert service.rag.last_query_param.model_func.args[0] == "openai/runtime-positive"


@pytest.mark.asyncio
async def test_little_bull_query_clamps_agent_max_tokens_to_reserved_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    runtime_model = await service.upsert_model_setting(
        principal,
        workspace_id="default",
        payload=LittleBullModelSetting(
            usage="agent",
            provider="openrouter",
            binding="openai",
            binding_host="https://openrouter.ai/api/v1",
            model_id="openai/runtime-clamp",
            display_name="Runtime Clamp",
        ),
    )
    agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Clamp",
            description="Agente com max_tokens maior que budget reservado",
            enabled=True,
            model_setting_id=runtime_model.model_setting_id,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder com fontes."},
                "model": {"max_tokens": 2000},
                "tests": [{"name": "clamp", "input": "Pergunta", "expected_behavior": "Limitar resposta"}],
            },
        ),
    )
    await service.upsert_agent_context_budget(
        principal,
        workspace_id="default",
        payload=LittleBullAgentContextBudgetRequest(
            agent_id=agent.agent_id,
            max_context_tokens=9000,
            reserved_response_tokens=120,
        ),
    )

    await service.query(
        principal,
        LittleBullQueryRequest(
            workspace_id="default",
            agent_id=agent.agent_id,
            query="Explique com fontes.",
        ),
    )

    assert service.rag.last_query_param.model_func is not None
    assert service.rag.last_query_param.model_func.keywords["max_tokens"] == 120


@pytest.mark.asyncio
async def test_little_bull_agent_context_budget_validates_agent_and_window(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Custos",
            description="Agente com budget controlado",
            enabled=False,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder com fontes."},
                "tests": [{"name": "fontes", "input": "Pergunta", "expected_behavior": "Citar fontes"}],
            },
        ),
    )

    budget = await service.upsert_agent_context_budget(
        principal,
        workspace_id="default",
        payload=LittleBullAgentContextBudgetRequest(
            agent_id=agent.agent_id,
            max_context_tokens=8000,
            reserved_response_tokens=1000,
            max_prompt_tokens=6000,
            daily_cost_limit_usd=5,
            policy={"overflow": "block"},
        ),
    )

    assert budget.agent_id == agent.agent_id
    assert budget.policy["overflow"] == "block"
    other_agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Decoy",
            description="Outro agente para testar filtros de budget",
            enabled=False,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder com fontes."},
                "tests": [{"name": "fontes", "input": "Pergunta", "expected_behavior": "Citar fontes"}],
            },
        ),
    )
    other_budget = await service.upsert_agent_context_budget(
        principal,
        workspace_id="default",
        payload=LittleBullAgentContextBudgetRequest(
            agent_id=other_agent.agent_id,
            max_context_tokens=4000,
            reserved_response_tokens=500,
        ),
    )
    listed = await service.list_agent_context_budgets(principal, workspace_id="default", agent_id=agent.agent_id)
    assert [item.agent_context_budget_id for item in listed] == [budget.agent_context_budget_id]
    listed = await service.list_agent_context_budgets(
        principal,
        workspace_id="default",
        agent_id=other_agent.agent_id,
    )
    assert [item.agent_context_budget_id for item in listed] == [other_budget.agent_context_budget_id]

    with pytest.raises(HTTPException) as exc:
        await service.upsert_agent_context_budget(
            principal,
            workspace_id="default",
            payload=LittleBullAgentContextBudgetRequest(
                agent_context_budget_id=budget.agent_context_budget_id,
                agent_id=other_agent.agent_id,
                max_context_tokens=8000,
            ),
        )

    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        await service.upsert_agent_context_budget(
            principal,
            workspace_id="default",
            payload=LittleBullAgentContextBudgetRequest(
                agent_id="missing-agent",
                max_context_tokens=8000,
                reserved_response_tokens=1000,
            ),
        )

    assert exc.value.status_code == 404

    service.admin_store.models["foreign-model"] = {
        "model_setting_id": "foreign-model",
        "tenant_id": "default",
        "workspace_id": "other-workspace",
        "usage": "chat",
        "provider": "openrouter",
        "binding": "openai",
        "binding_host": "https://openrouter.ai/api/v1",
        "model_id": "openai/test",
        "display_name": "Foreign",
        "enabled": True,
        "is_default": False,
        "config": {},
        "created_by": principal.user_id,
        "updated_by": principal.user_id,
        "created_at": "2026-04-29T00:00:00Z",
        "updated_at": "2026-04-29T00:00:00Z",
    }
    with pytest.raises(HTTPException) as exc:
        await service.upsert_agent_context_budget(
            principal,
            workspace_id="default",
            payload=LittleBullAgentContextBudgetRequest(
                agent_id=agent.agent_id,
                model_setting_id="foreign-model",
                max_context_tokens=8000,
            ),
        )

    assert exc.value.status_code == 404

    with pytest.raises(HTTPException) as exc:
        await service.upsert_agent_context_budget(
            principal,
            workspace_id="default",
            payload=LittleBullAgentContextBudgetRequest(
                agent_id=agent.agent_id,
                max_context_tokens=1000,
                reserved_response_tokens=1200,
            ),
        )

    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        await service.upsert_agent_context_budget(
            principal,
            workspace_id="default",
            payload=LittleBullAgentContextBudgetRequest(
                agent_id=agent.agent_id,
                max_context_tokens=1000,
                reserved_response_tokens=200,
                max_prompt_tokens=900,
            ),
        )

    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        await service.upsert_agent_context_budget(
            principal,
            workspace_id="default",
            payload=LittleBullAgentContextBudgetRequest(
                agent_id=agent.agent_id,
                daily_cost_limit_usd=1,
                policy={"estimated_request_cost_usd": 0.01},
            ),
        )

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_little_bull_query_enforces_agent_context_budget_before_rag(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Budget Runtime",
            description="Agente com teto de contexto em runtime",
            enabled=True,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder com fontes e respeitar budget."},
                "tests": [{"name": "budget", "input": "Pergunta", "expected_behavior": "Respeitar budget"}],
            },
        ),
    )
    budget = await service.upsert_agent_context_budget(
        principal,
        workspace_id="default",
        payload=LittleBullAgentContextBudgetRequest(
            agent_id=agent.agent_id,
            max_context_tokens=8,
            reserved_response_tokens=0,
            max_prompt_tokens=1,
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                agent_id=agent.agent_id,
                query="Explique o contrato com detalhes e cite as fontes.",
            ),
        )

    assert exc.value.status_code == 422
    assert service.rag.query_calls == 0
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    blocked = next(event for event in events if event.result == "blocked")
    assert blocked.metadata["reason"] == "agent_context_budget"
    assert blocked.metadata["agent_id"] == agent.agent_id

    service.admin_store.agent_context_budgets[budget.agent_context_budget_id]["max_prompt_tokens"] = 8000
    service.admin_store.agent_context_budgets[budget.agent_context_budget_id]["max_context_tokens"] = 9000
    response = await service.query(
        principal,
        LittleBullQueryRequest(
            workspace_id="default",
            agent_id=agent.agent_id,
            query="Explique o contrato com fontes.",
            top_k=1,
        ),
    )
    assert response.response.startswith("Answer for")
    assert service.rag.last_query_param.max_total_tokens < 9000
    assert service.rag.last_query_param.top_k == 1
    assert service.rag.last_query_param.chunk_top_k == 1
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    success = next(event for event in events if event.result == "success")
    assert success.metadata["agent_context_budget"]["agent_context_budget_id"] == budget.agent_context_budget_id
    assert success.metadata["usage_ledger_id"] == "ledger-1"
    assert service.admin_store.usage_ledger[0]["operation"] == "agent_query"


@pytest.mark.asyncio
async def test_little_bull_query_blocks_agent_context_budget_cost_overflow(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    runtime_model = await service.upsert_model_setting(
        principal,
        workspace_id="default",
        payload=LittleBullModelSetting(
            usage="agent",
            provider="openrouter",
            binding="openai",
            binding_host="https://openrouter.ai/api/v1",
            model_id="openai/runtime-cost-budget",
            display_name="Runtime Cost Budget",
        ),
    )
    agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Budget Custo",
            description="Agente com teto de custo em runtime",
            enabled=True,
            model_setting_id=runtime_model.model_setting_id,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder com fontes e respeitar custo."},
                "tests": [{"name": "custo", "input": "Pergunta", "expected_behavior": "Respeitar custo"}],
            },
        ),
    )
    await service.upsert_agent_context_budget(
        principal,
        workspace_id="default",
        payload=LittleBullAgentContextBudgetRequest(
            agent_id=agent.agent_id,
            max_context_tokens=9000,
            reserved_response_tokens=1000,
            daily_cost_limit_usd=0.03,
            policy={"estimated_request_cost_usd": 0.02},
        ),
    )

    response = await service.query(
        principal,
        LittleBullQueryRequest(
            workspace_id="default",
            agent_id=agent.agent_id,
            query="Explique o contrato com fontes.",
        ),
    )
    assert response.response.startswith("Answer for")
    assert service.rag.query_calls == 1
    assert service.admin_store.usage_ledger[0]["estimated_cost_usd"] == 0.02

    with pytest.raises(HTTPException) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                agent_id=agent.agent_id,
                query="Explique o contrato com fontes.",
            ),
        )

    assert exc.value.status_code == 422
    assert "daily cost budget" in exc.value.detail
    assert service.rag.query_calls == 1
    assert len(service.admin_store.usage_ledger) == 1


@pytest.mark.asyncio
async def test_little_bull_query_reserves_cost_budget_before_concurrent_rag(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    rag = SlowFakeRag()
    principal, service = await _principal_and_service(tmp_path, rag=rag)
    service.admin_store = FakeAdminStore()
    runtime_model = await service.upsert_model_setting(
        principal,
        workspace_id="default",
        payload=LittleBullModelSetting(
            usage="agent",
            provider="openrouter",
            binding="openai",
            binding_host="https://openrouter.ai/api/v1",
            model_id="openai/runtime-concurrent-budget",
            display_name="Runtime Concurrent Budget",
        ),
    )
    agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Budget Concorrente",
            description="Agente com teto de uma requisicao",
            enabled=True,
            model_setting_id=runtime_model.model_setting_id,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder com fontes e respeitar custo."},
                "tests": [{"name": "concorrencia", "input": "Pergunta", "expected_behavior": "Reservar antes"}],
            },
        ),
    )
    await service.upsert_agent_context_budget(
        principal,
        workspace_id="default",
        payload=LittleBullAgentContextBudgetRequest(
            agent_id=agent.agent_id,
            max_context_tokens=9000,
            reserved_response_tokens=1000,
            daily_cost_limit_usd=0.02,
            policy={"estimated_request_cost_usd": 0.02},
        ),
    )

    results = await asyncio.gather(
        service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                agent_id=agent.agent_id,
                query="Explique o contrato com fontes.",
            ),
        ),
        service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                agent_id=agent.agent_id,
                query="Explique o contrato com fontes.",
            ),
        ),
        return_exceptions=True,
    )

    successes = [result for result in results if not isinstance(result, Exception)]
    failures = [result for result in results if isinstance(result, HTTPException)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert failures[0].status_code == 422
    assert rag.query_calls == 1
    assert len(service.admin_store.usage_ledger) == 1
    assert service.admin_store.usage_ledger[0]["metadata"]["status"] == "reserved"


@pytest.mark.asyncio
async def test_little_bull_query_blocks_legacy_cost_budget_without_context_cap(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Budget Legado",
            description="Agente com budget legado sem teto de contexto",
            enabled=True,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder com fontes e respeitar custo."},
                "tests": [{"name": "legado", "input": "Pergunta", "expected_behavior": "Bloquear"}],
            },
        ),
    )
    service.admin_store.agent_context_budgets["legacy-budget"] = {
        "agent_context_budget_id": "legacy-budget",
        "tenant_id": "default",
        "workspace_id": "default",
        "agent_id": agent.agent_id,
        "model_setting_id": None,
        "max_context_tokens": 0,
        "reserved_response_tokens": 0,
        "max_prompt_tokens": 0,
        "daily_cost_limit_usd": 1,
        "monthly_cost_limit_usd": None,
        "policy": {"estimated_request_cost_usd": 0.01},
        "created_by": principal.user_id,
        "updated_by": principal.user_id,
        "created_at": "2026-04-29T00:00:00Z",
        "updated_at": "2026-04-29T00:00:00Z",
    }

    with pytest.raises(HTTPException) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                agent_id=agent.agent_id,
                query="Explique o contrato com fontes.",
            ),
        )

    assert exc.value.status_code == 422
    assert "max_context_tokens" in exc.value.detail
    assert service.rag.query_calls == 0


@pytest.mark.asyncio
async def test_little_bull_operational_chat_saves_context_and_transforms_to_note(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    service.rag.little_bull_scoped_query_supported = True
    group, subgroup = await _create_group_and_subgroup(service, principal)
    document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": subgroup.subgroup_id,
            "title": "contrato.md",
            "source_uri": "contrato.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "chat-hash",
            "confidentiality": "normal",
            "status": "processed",
            "content_length": 1200,
            "chunk_count": 3,
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro Chat",
            slug="outro-chat",
        ),
    )
    other_document = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": other_subgroup.subgroup_id,
            "title": "fora.md",
            "source_uri": "fora.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "outside-chat-hash",
            "confidentiality": "normal",
            "status": "processed",
            "content_length": 300,
            "chunk_count": 1,
            "metadata": {},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )
    agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Chat",
            description="Agente para chat operacional",
            enabled=True,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder com fontes."},
                "tests": [{"name": "fontes", "input": "Pergunta", "expected_behavior": "Citar fontes"}],
            },
        ),
    )
    await service.upsert_agent_context_budget(
        principal,
        workspace_id="default",
        payload=LittleBullAgentContextBudgetRequest(
            agent_id=agent.agent_id,
            max_context_tokens=9000,
            policy={"estimated_request_cost_usd": 0.012},
        ),
    )

    response = await service.operational_chat(
        principal,
        LittleBullOperationalChatRequest(
            workspace_id="default",
            agent_id=agent.agent_id,
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            document_ids=[document["document_id"]],
            top_k=2,
            query="Explique o contrato com fontes.",
            title="Analise do contrato",
            transform_to="note",
            note_title="Nota da conversa",
        ),
    )

    assert response.response.startswith("Answer for")
    assert response.sources[0]["file_path"] == "manual.pdf"
    assert response.context["agent_id"] == agent.agent_id
    assert response.context["estimate"]["document_count"] == 1
    assert response.cost_estimate["estimated_request_cost_usd"] == 0.012
    assert response.conversation is not None
    assert response.conversation.message_count == 2
    assert response.conversation.agent_id == agent.agent_id
    assert response.conversation.scope_snapshot == {
        "group_id": group.group_id,
        "subgroup_id": subgroup.subgroup_id,
        "document_ids": [document["document_id"]],
    }
    assert response.note is not None
    assert response.note["registry"]["title"] == "Nota da conversa"
    assert response.note["registry"]["metadata"]["source_kind"] == "conversation"
    assert service.rag.query_calls == 1

    conversation_count = len(service.admin_store.conversations)
    note_count = len(service.admin_store.notes)
    with pytest.raises(HTTPException) as exc:
        await service.operational_chat(
            principal,
            LittleBullOperationalChatRequest(
                workspace_id="default",
                agent_id=agent.agent_id,
                group_id=group.group_id,
                subgroup_id=subgroup.subgroup_id,
                document_ids=[other_document["document_id"]],
                query="Tente usar documento de outro subgrupo.",
                transform_to="note",
            ),
        )

    assert exc.value.status_code == 422
    assert service.rag.query_calls == 1
    assert len(service.admin_store.conversations) == conversation_count
    assert len(service.admin_store.notes) == note_count


@pytest.mark.asyncio
async def test_little_bull_operational_chat_transforms_to_suggestion(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)

    response = await service.operational_chat(
        principal,
        LittleBullOperationalChatRequest(
            workspace_id="default",
            query="Relacione o memorando com a tese principal.",
            transform_to="suggestion",
            suggestion_target_label="Tese principal",
        ),
    )

    assert response.conversation is not None
    assert response.suggestion is not None
    assert response.suggestion.target_label == "Tese principal"
    assert response.suggestion.metadata["conversation_id"] == response.conversation.conversation_id
    assert service.rag.query_calls == 1


@pytest.mark.asyncio
async def test_little_bull_operational_chat_note_transform_requires_scope_before_rag(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)

    with pytest.raises(HTTPException) as exc:
        await service.operational_chat(
            principal,
            LittleBullOperationalChatRequest(
                workspace_id="default",
                query="Transforme esta conversa em nota.",
                transform_to="note",
            ),
        )

    assert exc.value.status_code == 422
    assert service.rag.query_calls == 0


@pytest.mark.asyncio
async def test_little_bull_save_conversation_rejects_unvalidated_agent(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)

    with pytest.raises(HTTPException) as exc:
        await service.save_conversation(
            principal,
            LittleBullConversationSaveRequest(
                workspace_id="default",
                agent_id="missing-agent",
                messages=[{"role": "user", "content": "Oi"}],
            ),
        )

    assert exc.value.status_code == 404
    assert service.admin_store.conversations == {}


@pytest.mark.asyncio
async def test_little_bull_cost_summary_aggregates_periods_and_dimensions(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Custos",
            description="Agente para sumarizar custos",
            enabled=True,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder com fontes."},
                "tests": [{"name": "custos", "input": "Pergunta", "expected_behavior": "Somar"}],
            },
        ),
    )
    now = utc_now()
    service.admin_store.usage_ledger.extend(
        [
            {
                "usage_ledger_id": "ledger-today",
                "tenant_id": "default",
                "workspace_id": "default",
                "group_id": group.group_id,
                "subgroup_id": subgroup.subgroup_id,
                "user_id": principal.user_id,
                "agent_id": agent.agent_id,
                "provider": "openrouter",
                "model_id": "openai/model-a",
                "operation": "agent_query",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "estimated_cost_usd": 0.02,
                "actual_cost_usd": None,
                "currency": "USD",
                "metadata": {"group_id": group.group_id, "subgroup_id": subgroup.subgroup_id},
                "created_at": now - timedelta(hours=1),
            },
            {
                "usage_ledger_id": "ledger-week",
                "tenant_id": "default",
                "workspace_id": "default",
                "group_id": group.group_id,
                "subgroup_id": subgroup.subgroup_id,
                "user_id": principal.user_id,
                "agent_id": agent.agent_id,
                "provider": "openrouter",
                "model_id": "openai/model-b",
                "operation": "agent_query",
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
                "estimated_cost_usd": 0.06,
                "actual_cost_usd": 0.05,
                "currency": "USD",
                "metadata": {"group_id": group.group_id, "subgroup_id": subgroup.subgroup_id},
                "created_at": now - timedelta(days=3),
            },
            {
                "usage_ledger_id": "ledger-legacy-metadata-scope",
                "tenant_id": "default",
                "workspace_id": "default",
                "group_id": None,
                "subgroup_id": None,
                "user_id": principal.user_id,
                "agent_id": agent.agent_id,
                "provider": "openrouter",
                "model_id": "openai/model-legacy-metadata",
                "operation": "agent_query",
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "estimated_cost_usd": 0.03,
                "actual_cost_usd": None,
                "currency": "USD",
                "metadata": {"group_id": group.group_id, "subgroup_id": subgroup.subgroup_id},
                "created_at": now - timedelta(days=2),
            },
            {
                "usage_ledger_id": "ledger-old",
                "tenant_id": "default",
                "workspace_id": "default",
                "group_id": None,
                "subgroup_id": None,
                "user_id": "other-user",
                "agent_id": None,
                "provider": "openrouter",
                "model_id": "openai/model-old",
                "operation": "embedding_reindex",
                "prompt_tokens": 300,
                "completion_tokens": 0,
                "total_tokens": 300,
                "estimated_cost_usd": 0.10,
                "actual_cost_usd": None,
                "currency": "USD",
                "metadata": {},
                "created_at": now - timedelta(days=40),
            },
            {
                "usage_ledger_id": "ledger-other-workspace",
                "tenant_id": "default",
                "workspace_id": "other",
                "group_id": group.group_id,
                "subgroup_id": subgroup.subgroup_id,
                "user_id": principal.user_id,
                "agent_id": agent.agent_id,
                "provider": "openrouter",
                "model_id": "openai/model-a",
                "operation": "agent_query",
                "prompt_tokens": 999,
                "completion_tokens": 999,
                "total_tokens": 1998,
                "estimated_cost_usd": 9.99,
                "actual_cost_usd": None,
                "currency": "USD",
                "metadata": {"group_id": group.group_id, "subgroup_id": subgroup.subgroup_id},
                "created_at": now,
            },
        ]
    )

    summary = await service.summarize_costs(principal, workspace_id="default")

    assert summary.periods["total"].request_count == 4
    assert summary.periods["total"].cost_usd == 0.20
    assert summary.periods["total"].estimated_cost_usd == 0.21
    assert summary.periods["total"].actual_cost_usd == 0.05
    assert summary.periods["month"].cost_usd == 0.10
    assert summary.periods["last_7_days"].cost_usd == 0.10
    assert summary.periods["today"].cost_usd == 0.02
    principal_user = next(item for item in summary.by_user if item.key == principal.user_id)
    assert principal_user.cost_usd == 0.10
    agent_bucket = next(item for item in summary.by_agent if item.key == agent.agent_id)
    assert agent_bucket.cost_usd == 0.10
    assert summary.by_model[0].key == "openai/model-old"
    scoped_group = next(
        item for item in summary.by_group_subgroup if item.key == f"{group.group_id}:{subgroup.subgroup_id}"
    )
    assert scoped_group.cost_usd == 0.10
    assert {item.key for item in summary.by_operation} == {"agent_query", "embedding_reindex"}

    scoped = await service.summarize_costs(
        principal,
        workspace_id="default",
        group_id=group.group_id,
        subgroup_id=subgroup.subgroup_id,
    )

    assert scoped.periods["total"].request_count == 3
    assert scoped.periods["total"].cost_usd == 0.10
    assert scoped.by_operation[0].key == "agent_query"
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    assert any(event.result == "cost_summary" for event in events)


@pytest.mark.asyncio
async def test_little_bull_query_budget_fails_closed_when_reserved_response_cannot_be_enforced(tmp_path):
    rag = FakeRag(
        llm_binding="openai",
        llm_model="openai/gpt-4o-mini",
        llm_host="https://openrouter.ai/api/v1",
    )
    principal, service = await _principal_and_service(tmp_path, rag=rag)
    service.admin_store = FakeAdminStore()
    await service.repository.set_policy(
        PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY,
        {
            "schema_version": 1,
            "enabled": True,
            "provider": "openrouter",
            "binding": "openai",
            "binding_host": "https://openrouter.ai/api/v1",
            "allowed_model_ids": ["openai/gpt-4o-mini"],
            "allowed_confidentiality": ["privado"],
            "expires_at": "2099-01-01T00:00:00Z",
            "approved_by": principal.user_id,
            "approved_at": "2026-04-28T00:00:00Z",
            "approval_id": "apr_reserved_response",
            "reason": "Hosted exception for fail-closed reserved response test.",
        },
        tenant_id="default",
        workspace_id="default",
    )
    agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Sem Func",
            description="Agente com budget mas rota hospedada sem model_func enforceable",
            enabled=True,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder com fontes."},
                "tests": [{"name": "reserved", "input": "Pergunta", "expected_behavior": "Bloquear"}],
            },
        ),
    )
    await service.upsert_agent_context_budget(
        principal,
        workspace_id="default",
        payload=LittleBullAgentContextBudgetRequest(
            agent_id=agent.agent_id,
            max_context_tokens=9000,
            reserved_response_tokens=120,
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                agent_id=agent.agent_id,
                query="Explique com fontes.",
                confidentiality="privado",
                model_profile="equilibrado",
            ),
        )

    assert exc.value.status_code == 422
    assert "reserved response token limit" in exc.value.detail
    assert rag.query_calls == 0


@pytest.mark.asyncio
async def test_little_bull_context_estimate_scopes_documents_and_flags_overflow(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro",
        ),
    )
    runtime_model = await service.upsert_model_setting(
        principal,
        workspace_id="default",
        payload=LittleBullModelSetting(
            usage="agent",
            provider="openrouter",
            binding="openai",
            binding_host="https://openrouter.ai/api/v1",
            model_id="openai/context-window",
            display_name="Context Window",
            config={"context_window": 200},
        ),
    )
    agent = await service.upsert_agent_config(
        principal,
        workspace_id="default",
        payload=LittleBullAgentConfig(
            name="Agente Contexto",
            description="Agente para estimar contexto",
            enabled=True,
            model_setting_id=runtime_model.model_setting_id,
            tools=["query_knowledge"],
            config={
                "identity": {"mission": "Responder com fontes."},
                "model": {"max_tokens": 40},
                "tests": [{"name": "contexto", "input": "Pergunta", "expected_behavior": "Calcular"}],
            },
        ),
    )
    await service.upsert_agent_context_budget(
        principal,
        workspace_id="default",
        payload=LittleBullAgentContextBudgetRequest(
            agent_id=agent.agent_id,
            max_context_tokens=120,
            reserved_response_tokens=30,
        ),
    )
    doc = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": subgroup.subgroup_id,
            "title": "contrato.md",
            "source_uri": "contrato.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash-context",
            "confidentiality": "normal",
            "status": "processed",
            "chunk_count": 4,
            "metadata": {"estimated_tokens": 400},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )
    other_doc = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": other_subgroup.subgroup_id,
            "title": "fora.md",
            "source_uri": "fora.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash-other-context",
            "confidentiality": "normal",
            "status": "processed",
            "chunk_count": 10,
            "metadata": {"estimated_tokens": 2000},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )

    estimate = await service.estimate_context(
        principal,
        LittleBullContextEstimateRequest(
            workspace_id="default",
            agent_id=agent.agent_id,
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            query="Explique o contrato com fontes.",
            conversation_history=[{"role": "user", "content": "Histórico " * 40}],
            top_k=2,
        ),
    )

    assert estimate.model_setting_id == runtime_model.model_setting_id
    assert estimate.context_window_tokens == 200
    assert estimate.document_count == 1
    assert estimate.chunk_count == 2
    assert estimate.chunk_tokens == 200
    assert estimate.reserved_response_tokens == 30
    assert estimate.overflow is True
    assert estimate.overflow_tokens > 0
    assert estimate.available_context_tokens == 120 - estimate.total_estimated_tokens
    assert estimate.overflow_tokens == estimate.total_estimated_tokens - 120
    assert any("budget caps" in note for note in estimate.notes)
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    assert events[0].result == "context_estimate"

    with pytest.raises(HTTPException) as exc:
        await service.estimate_context(
            principal,
            LittleBullContextEstimateRequest(
                workspace_id="default",
                group_id=group.group_id,
                subgroup_id=subgroup.subgroup_id,
                document_ids=[other_doc["document_id"]],
                query="Explique documento fora do subgrupo.",
            ),
        )
    assert exc.value.status_code == 422

    narrow = await service.estimate_context(
        principal,
        LittleBullContextEstimateRequest(
            workspace_id="default",
            group_id=group.group_id,
            subgroup_id=subgroup.subgroup_id,
            document_ids=[doc["document_id"]],
            query="Resumo.",
            top_k=1,
            reserved_response_tokens=10,
        ),
    )
    assert narrow.document_count == 1
    assert narrow.chunk_count == 1
    assert narrow.retrieval_chunk_limit == 1
    assert narrow.reserved_response_tokens == 10


@pytest.mark.asyncio
async def test_little_bull_query_validates_scope_and_fails_closed_until_filters_exist(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)
    group, subgroup = await _create_group_and_subgroup(service, principal)
    other_subgroup = await service.upsert_knowledge_subgroup(
        principal,
        workspace_id="default",
        payload=LittleBullKnowledgeSubgroupRequest(
            group_id=group.group_id,
            name="Outro",
            slug="outro-query",
        ),
    )
    doc = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": subgroup.subgroup_id,
            "title": "contrato-query.md",
            "source_uri": "contrato-query.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash-query-scope",
            "confidentiality": "normal",
            "status": "processed",
            "chunk_count": 2,
            "metadata": {"estimated_tokens": 120},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )
    other_doc = await service.admin_store.register_document(
        {
            "group_id": group.group_id,
            "subgroup_id": other_subgroup.subgroup_id,
            "title": "fora-query.md",
            "source_uri": "fora-query.md",
            "source_kind": "upload",
            "mime_type": "text/markdown",
            "content_hash": "hash-query-other",
            "confidentiality": "normal",
            "status": "processed",
            "chunk_count": 2,
            "metadata": {"estimated_tokens": 120},
        },
        tenant_id="default",
        workspace_id="default",
        user_id=principal.user_id,
    )

    with pytest.raises(HTTPException) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                query="Explique somente este documento.",
                group_id=group.group_id,
                subgroup_id=subgroup.subgroup_id,
                document_ids=[doc["document_id"]],
            ),
        )

    assert exc.value.status_code == 422
    assert "Scoped Little Bull queries require data-plane filter support" in exc.value.detail
    assert service.rag.query_calls == 0
    events = await service.audit.list(tenant_id="default", workspace_id="default")
    blocked = next(event for event in events if event.result == "blocked")
    assert blocked.metadata["reason"] == "scoped_query_filters_unavailable"
    assert blocked.metadata["scope"]["document_ids"] == [doc["document_id"]]

    with pytest.raises(HTTPException) as exc:
        await service.query(
            principal,
            LittleBullQueryRequest(
                workspace_id="default",
                query="Explique documento fora.",
                group_id=group.group_id,
                subgroup_id=subgroup.subgroup_id,
                document_ids=[other_doc["document_id"]],
            ),
        )

    assert exc.value.status_code == 422
    assert "outside subgroup scope" in exc.value.detail
    assert service.rag.query_calls == 0


@pytest.mark.asyncio
async def test_little_bull_context_estimate_requires_group_for_subgroup(tmp_path):
    principal, service = await _principal_and_service_with_admin_store(tmp_path)

    with pytest.raises(HTTPException) as exc:
        await service.estimate_context(
            principal,
            LittleBullContextEstimateRequest(
                workspace_id="default",
                subgroup_id="subgroup-1",
                query="Pergunta com contexto.",
            ),
        )

    assert exc.value.status_code == 422


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
