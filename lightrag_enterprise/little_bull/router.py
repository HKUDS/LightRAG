from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile

from lightrag_enterprise.system.runtime import (
    get_access_service,
    get_approval_service,
    get_audit_service,
    get_system_repository,
    require_principal,
)

from .models import (
    LittleBullAgentConfig,
    LittleBullAgentBuilderPublishRequest,
    LittleBullAgentBuilderSessionRequest,
    LittleBullAgentContextBudgetRequest,
    LittleBullAgentStudioPreviewRequest,
    LittleBullConversationSaveRequest,
    LittleBullCorrelationSuggestionRequest,
    LittleBullBacklinkRequest,
    LittleBullCanvasBoardRequest,
    LittleBullCanvasEdgeRequest,
    LittleBullCanvasNodeRequest,
    LittleBullContentMapRequest,
    LittleBullDailyNoteRequest,
    LittleBullEmbeddingCostEstimateRequest,
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
    LittleBullQueryRequest,
    LittleBullSourceProvenanceRequest,
)
from .service import LittleBullService


def create_little_bull_router(rag, doc_manager) -> APIRouter:
    router = APIRouter(prefix="/little-bull", tags=["little-bull"])

    def service() -> LittleBullService:
        return LittleBullService(
            rag=rag,
            doc_manager=doc_manager,
            repository=get_system_repository(),
            access=get_access_service(),
            audit=get_audit_service(),
            approvals=get_approval_service(),
        )

    @router.get("/areas")
    async def list_areas(principal=Depends(require_principal)):
        return {"areas": [area.model_dump() for area in await service().list_areas(principal)]}

    @router.get("/knowledge-groups")
    async def list_knowledge_groups(workspace_id: str, principal=Depends(require_principal)):
        return {
            "groups": [
                item.model_dump()
                for item in await service().list_knowledge_groups(
                    principal,
                    workspace_id=workspace_id,
                )
            ]
        }

    @router.post("/knowledge-groups")
    async def upsert_knowledge_group(
        request: LittleBullKnowledgeGroupRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_knowledge_group(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/knowledge-subgroups")
    async def list_knowledge_subgroups(
        workspace_id: str,
        group_id: str | None = None,
        principal=Depends(require_principal),
    ):
        return {
            "subgroups": [
                item.model_dump()
                for item in await service().list_knowledge_subgroups(
                    principal,
                    workspace_id=workspace_id,
                    group_id=group_id,
                )
            ]
        }

    @router.post("/knowledge-subgroups")
    async def upsert_knowledge_subgroup(
        request: LittleBullKnowledgeSubgroupRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_knowledge_subgroup(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/notes")
    async def list_notes(
        workspace_id: str,
        group_id: str | None = None,
        subgroup_id: str | None = None,
        principal=Depends(require_principal),
    ):
        return {
            "notes": [
                item.model_dump()
                for item in await service().list_notes(
                    principal,
                    workspace_id=workspace_id,
                    group_id=group_id,
                    subgroup_id=subgroup_id,
                )
            ]
        }

    @router.post("/notes/markdown")
    async def upsert_markdown_note(
        request: LittleBullMarkdownNoteRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_markdown_note(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/notes/{note_id}/markdown")
    async def get_markdown_note(note_id: str, workspace_id: str, principal=Depends(require_principal)):
        return (
            await service().get_markdown_note(
                principal,
                workspace_id=workspace_id,
                note_id=note_id,
            )
        ).model_dump()

    @router.get("/tags")
    async def list_tags(workspace_id: str, principal=Depends(require_principal)):
        return {
            "tags": [
                item.model_dump()
                for item in await service().list_tags(
                    principal,
                    workspace_id=workspace_id,
                )
            ]
        }

    @router.get("/backlinks")
    async def list_backlinks(
        workspace_id: str,
        source_kind: str | None = None,
        source_id: str | None = None,
        target_kind: str | None = None,
        target_id: str | None = None,
        principal=Depends(require_principal),
    ):
        return {
            "backlinks": [
                item.model_dump()
                for item in await service().list_backlinks(
                    principal,
                    workspace_id=workspace_id,
                    source_kind=source_kind,
                    source_id=source_id,
                    target_kind=target_kind,
                    target_id=target_id,
                )
            ]
        }

    @router.post("/backlinks")
    async def upsert_backlink(
        request: LittleBullBacklinkRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_backlink(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/provenance/panel")
    async def get_provenance_panel(
        workspace_id: str,
        target_kind: str,
        target_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().get_provenance_panel(
                principal,
                workspace_id=workspace_id,
                target_kind=target_kind,
                target_id=target_id,
            )
        ).model_dump()

    @router.get("/source-provenance")
    async def list_source_provenance(
        workspace_id: str,
        source_kind: str | None = None,
        source_id: str | None = None,
        document_id: str | None = None,
        note_id: str | None = None,
        principal=Depends(require_principal),
    ):
        return {
            "provenance": [
                item.model_dump()
                for item in await service().list_source_provenance(
                    principal,
                    workspace_id=workspace_id,
                    source_kind=source_kind,
                    source_id=source_id,
                    document_id=document_id,
                    note_id=note_id,
                )
            ]
        }

    @router.post("/source-provenance")
    async def record_source_provenance(
        request: LittleBullSourceProvenanceRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().record_source_provenance(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/canvas/boards")
    async def list_canvas_boards(
        workspace_id: str,
        group_id: str | None = None,
        subgroup_id: str | None = None,
        principal=Depends(require_principal),
    ):
        return {
            "boards": [
                item.model_dump()
                for item in await service().list_canvas_boards(
                    principal,
                    workspace_id=workspace_id,
                    group_id=group_id,
                    subgroup_id=subgroup_id,
                )
            ]
        }

    @router.post("/canvas/boards")
    async def upsert_canvas_board(
        request: LittleBullCanvasBoardRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_canvas_board(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/canvas/boards/{canvas_board_id}")
    async def get_canvas_board(canvas_board_id: str, workspace_id: str, principal=Depends(require_principal)):
        return (
            await service().get_canvas_board(
                principal,
                workspace_id=workspace_id,
                canvas_board_id=canvas_board_id,
            )
        ).model_dump()

    @router.post("/canvas/boards/{canvas_board_id}/nodes")
    async def upsert_canvas_node(
        canvas_board_id: str,
        request: LittleBullCanvasNodeRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_canvas_node(
                principal,
                workspace_id=workspace_id,
                canvas_board_id=canvas_board_id,
                payload=request,
            )
        ).model_dump()

    @router.post("/canvas/boards/{canvas_board_id}/edges")
    async def upsert_canvas_edge(
        canvas_board_id: str,
        request: LittleBullCanvasEdgeRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_canvas_edge(
                principal,
                workspace_id=workspace_id,
                canvas_board_id=canvas_board_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/canvas/boards/{canvas_board_id}/analysis")
    async def analyze_canvas_board(canvas_board_id: str, workspace_id: str, principal=Depends(require_principal)):
        return (
            await service().analyze_canvas_board(
                principal,
                workspace_id=workspace_id,
                canvas_board_id=canvas_board_id,
            )
        ).model_dump()

    @router.post("/canvas/boards/{canvas_board_id}/dossier")
    async def export_canvas_board_dossier(
        canvas_board_id: str,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().export_canvas_board_dossier(
                principal,
                workspace_id=workspace_id,
                canvas_board_id=canvas_board_id,
            )
        ).model_dump()

    @router.get("/content-maps")
    async def list_content_maps(
        workspace_id: str,
        group_id: str | None = None,
        subgroup_id: str | None = None,
        principal=Depends(require_principal),
    ):
        return {
            "content_maps": [
                item.model_dump()
                for item in await service().list_content_maps(
                    principal,
                    workspace_id=workspace_id,
                    group_id=group_id,
                    subgroup_id=subgroup_id,
                )
            ]
        }

    @router.post("/content-maps")
    async def upsert_content_map(
        request: LittleBullContentMapRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_content_map(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/knowledge-trails")
    async def list_knowledge_trails(
        workspace_id: str,
        group_id: str | None = None,
        subgroup_id: str | None = None,
        principal=Depends(require_principal),
    ):
        return {
            "trails": [
                item.model_dump()
                for item in await service().list_knowledge_trails(
                    principal,
                    workspace_id=workspace_id,
                    group_id=group_id,
                    subgroup_id=subgroup_id,
                )
            ]
        }

    @router.post("/knowledge-trails")
    async def upsert_knowledge_trail(
        request: LittleBullKnowledgeTrailRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_knowledge_trail(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/knowledge-trails/{knowledge_trail_id}")
    async def get_knowledge_trail(
        knowledge_trail_id: str,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().get_knowledge_trail(
                principal,
                workspace_id=workspace_id,
                knowledge_trail_id=knowledge_trail_id,
            )
        ).model_dump()

    @router.post("/knowledge-trails/{knowledge_trail_id}/steps")
    async def upsert_knowledge_trail_step(
        knowledge_trail_id: str,
        request: LittleBullKnowledgeTrailStepRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_knowledge_trail_step(
                principal,
                workspace_id=workspace_id,
                knowledge_trail_id=knowledge_trail_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/inbox")
    async def list_inbox_items(
        workspace_id: str,
        status: str | None = None,
        group_id: str | None = None,
        subgroup_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        principal=Depends(require_principal),
    ):
        return {
            "items": [
                item.model_dump()
                for item in await service().list_inbox_items(
                    principal,
                    workspace_id=workspace_id,
                    status_filter=status,
                    group_id=group_id,
                    subgroup_id=subgroup_id,
                    limit=limit,
                )
            ]
        }

    @router.post("/inbox")
    async def upsert_inbox_item(
        request: LittleBullInboxItemRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_inbox_item(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.post("/inbox/{inbox_item_id}/status")
    async def update_inbox_item_status(
        inbox_item_id: str,
        request: LittleBullInboxItemStatusRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().update_inbox_item_status(
                principal,
                workspace_id=workspace_id,
                inbox_item_id=inbox_item_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/daily-notes")
    async def list_daily_notes(
        workspace_id: str,
        limit: int = Query(default=30, ge=1, le=365),
        principal=Depends(require_principal),
    ):
        return {
            "daily_notes": [
                item.model_dump()
                for item in await service().list_daily_notes(
                    principal,
                    workspace_id=workspace_id,
                    limit=limit,
                )
            ]
        }

    @router.post("/daily-notes/ensure")
    async def ensure_daily_note(
        request: LittleBullDailyNoteRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().ensure_daily_note(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/documents")
    async def list_documents(
        workspace_id: str,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
        principal=Depends(require_principal),
    ):
        return (await service().list_documents(
            principal,
            workspace_id=workspace_id,
            page=page,
            page_size=page_size,
        )).model_dump()

    @router.post("/documents/upload")
    async def upload_document(
        background_tasks: BackgroundTasks,
        workspace_id: str,
        group_id: str,
        subgroup_id: str,
        confidentiality: str = "normal",
        file: UploadFile = File(...),
        principal=Depends(require_principal),
    ):
        return (await service().upload_document(
            principal,
            workspace_id=workspace_id,
            group_id=group_id,
            subgroup_id=subgroup_id,
            file=file,
            background_tasks=background_tasks,
            confidentiality=confidentiality,
        )).model_dump()

    @router.post("/documents/reindex-archived")
    async def reindex_archived_documents(
        background_tasks: BackgroundTasks,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (await service().reindex_archived_documents(
            principal,
            workspace_id=workspace_id,
            background_tasks=background_tasks,
        )).model_dump()

    @router.delete("/documents/{document_id}")
    async def delete_document(
        document_id: str,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return await service().delete_document(
            principal,
            workspace_id=workspace_id,
            document_id=document_id,
        )

    @router.post("/query")
    async def query(request: LittleBullQueryRequest, principal=Depends(require_principal)):
        return (await service().query(principal, request)).model_dump()

    @router.get("/activity")
    async def list_activity(
        workspace_id: str,
        limit: int = Query(default=50, ge=1, le=200),
        principal=Depends(require_principal),
    ):
        return {
            "activity": [
                item.model_dump()
                for item in await service().list_activity(
                    principal,
                    workspace_id=workspace_id,
                    limit=limit,
                )
            ]
        }

    @router.get("/assistants")
    async def list_assistants(workspace_id: str, principal=Depends(require_principal)):
        return {
            "assistants": [
                item.model_dump()
                for item in await service().list_assistants(
                    principal,
                    workspace_id=workspace_id,
                )
            ]
        }

    @router.get("/graph")
    async def get_knowledge_graph(
        workspace_id: str,
        label: str = Query(..., description="Label to get knowledge graph for"),
        max_depth: int = Query(3, description="Maximum depth of graph", ge=1),
        max_nodes: int = Query(1000, description="Maximum nodes to return", ge=1),
        principal=Depends(require_principal),
    ):
        return await service().get_knowledge_graph(
            principal,
            workspace_id=workspace_id,
            label=label,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

    @router.get("/graph/label/list")
    async def list_graph_labels(workspace_id: str, principal=Depends(require_principal)):
        return await service().list_graph_labels(principal, workspace_id=workspace_id)

    @router.get("/graph/label/popular")
    async def list_popular_graph_labels(
        workspace_id: str,
        limit: int = Query(
            300, description="Maximum number of popular labels to return", ge=1, le=1000
        ),
        principal=Depends(require_principal),
    ):
        return await service().list_popular_graph_labels(
            principal,
            workspace_id=workspace_id,
            limit=limit,
        )

    @router.get("/graph/label/search")
    async def search_graph_labels(
        workspace_id: str,
        q: str = Query(..., description="Search query string"),
        limit: int = Query(
            50, description="Maximum number of search results to return", ge=1, le=100
        ),
        principal=Depends(require_principal),
    ):
        return await service().search_graph_labels(
            principal,
            workspace_id=workspace_id,
            query=q,
            limit=limit,
        )

    @router.get("/admin/models")
    async def list_model_settings(workspace_id: str, principal=Depends(require_principal)):
        return {
            "models": [
                item.model_dump()
                for item in await service().list_model_settings(
                    principal,
                    workspace_id=workspace_id,
                )
            ]
        }

    @router.post("/admin/models")
    async def upsert_model_setting(
        request: LittleBullModelSetting,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_model_setting(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/admin/embedding-models")
    async def list_embedding_catalog(principal=Depends(require_principal)):
        return {
            "models": [
                item.model_dump()
                for item in await service().list_embedding_catalog(principal)
            ]
        }

    @router.get("/admin/knowledge-bases")
    async def list_knowledge_bases(principal=Depends(require_principal)):
        return {
            "knowledge_bases": [
                item.model_dump()
                for item in await service().list_knowledge_bases(principal)
            ]
        }

    @router.post("/admin/knowledge-bases")
    async def upsert_knowledge_base(
        request: LittleBullKnowledgeBaseUpsertRequest,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_knowledge_base(
                principal,
                request,
            )
        ).model_dump()

    @router.post("/admin/knowledge-bases/{workspace_id}/attach-data-plane")
    async def attach_knowledge_base_data_plane(
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().attach_knowledge_base_data_plane(
                principal,
                workspace_id=workspace_id,
            )
        ).model_dump()

    @router.post("/admin/knowledge-bases/{workspace_id}/reindex")
    async def reindex_knowledge_base(
        background_tasks: BackgroundTasks,
        workspace_id: str,
        request: LittleBullKnowledgeBaseReindexRequest,
        principal=Depends(require_principal),
    ):
        return (
            await service().reindex_knowledge_base(
                principal,
                workspace_id=workspace_id,
                request=request,
                background_tasks=background_tasks,
            )
        ).model_dump()

    @router.post("/admin/knowledge-bases/{workspace_id}/rollback")
    async def rollback_knowledge_base_snapshot(
        workspace_id: str,
        request: LittleBullKnowledgeBaseRollbackRequest,
        principal=Depends(require_principal),
    ):
        return (
            await service().rollback_knowledge_base_snapshot(
                principal,
                workspace_id=workspace_id,
                request=request,
            )
        ).model_dump()

    @router.post("/admin/embedding-cost-estimate")
    async def estimate_embedding_cost(
        request: LittleBullEmbeddingCostEstimateRequest,
        principal=Depends(require_principal),
    ):
        return (
            await service().estimate_embedding_cost_for_workspace(
                principal,
                request,
            )
        ).model_dump()

    @router.get("/admin/agents")
    async def list_agent_configs(workspace_id: str, principal=Depends(require_principal)):
        return {
            "agents": [
                item.model_dump()
                for item in await service().list_agent_configs(
                    principal,
                    workspace_id=workspace_id,
                )
            ]
        }

    @router.post("/admin/agents")
    async def upsert_agent_config(
        request: LittleBullAgentConfig,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_agent_config(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.post("/admin/agents/preview")
    async def preview_agent_studio(
        request: LittleBullAgentStudioPreviewRequest,
        principal=Depends(require_principal),
    ):
        return (await service().preview_agent_studio(principal, request)).model_dump()

    @router.get("/admin/agent-builder/sessions")
    async def list_agent_builder_sessions(
        workspace_id: str,
        status: str | None = None,
        principal=Depends(require_principal),
    ):
        return {
            "sessions": [
                item.model_dump()
                for item in await service().list_agent_builder_sessions(
                    principal,
                    workspace_id=workspace_id,
                    status_filter=status,
                )
            ]
        }

    @router.post("/admin/agent-builder/sessions")
    async def upsert_agent_builder_session(
        request: LittleBullAgentBuilderSessionRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_agent_builder_session(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.post("/admin/agent-builder/sessions/{agent_builder_session_id}/publish")
    async def publish_agent_builder_session(
        agent_builder_session_id: str,
        request: LittleBullAgentBuilderPublishRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().publish_agent_builder_session(
                principal,
                workspace_id=workspace_id,
                agent_builder_session_id=agent_builder_session_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/admin/agents/context-budgets")
    async def list_agent_context_budgets(
        workspace_id: str,
        agent_id: str | None = None,
        principal=Depends(require_principal),
    ):
        return {
            "budgets": [
                item.model_dump()
                for item in await service().list_agent_context_budgets(
                    principal,
                    workspace_id=workspace_id,
                    agent_id=agent_id,
                )
            ]
        }

    @router.post("/admin/agents/context-budgets")
    async def upsert_agent_context_budget(
        request: LittleBullAgentContextBudgetRequest,
        workspace_id: str,
        principal=Depends(require_principal),
    ):
        return (
            await service().upsert_agent_context_budget(
                principal,
                workspace_id=workspace_id,
                payload=request,
            )
        ).model_dump()

    @router.get("/conversations")
    async def list_conversations(workspace_id: str, principal=Depends(require_principal)):
        return {
            "conversations": [
                item.model_dump()
                for item in await service().list_conversations(
                    principal,
                    workspace_id=workspace_id,
                )
            ]
        }

    @router.post("/conversations")
    async def save_conversation(
        request: LittleBullConversationSaveRequest,
        principal=Depends(require_principal),
    ):
        return (await service().save_conversation(principal, request)).model_dump()

    @router.get("/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str, principal=Depends(require_principal)):
        return (
            await service().get_conversation(
                principal,
                conversation_id=conversation_id,
            )
        ).model_dump()

    @router.get("/conversations/{conversation_id}/export")
    async def export_conversation(
        conversation_id: str,
        format: str = Query(default="md", pattern="^(md|txt|docx)$"),
        principal=Depends(require_principal),
    ):
        return await service().export_conversation(
            principal,
            conversation_id=conversation_id,
            export_format=format,
        )

    @router.get("/correlation-suggestions")
    async def list_correlation_suggestions(
        workspace_id: str,
        status: str | None = Query(default=None),
        principal=Depends(require_principal),
    ):
        return {
            "suggestions": [
                item.model_dump()
                for item in await service().list_correlation_suggestions(
                    principal,
                    workspace_id=workspace_id,
                    suggestion_status=status,
                )
            ]
        }

    @router.post("/correlation-suggestions")
    async def create_correlation_suggestion(
        request: LittleBullCorrelationSuggestionRequest,
        principal=Depends(require_principal),
    ):
        return (await service().create_correlation_suggestion(principal, request)).model_dump()

    @router.post("/correlation-suggestions/{suggestion_id}/approve")
    async def approve_correlation_suggestion(suggestion_id: str, principal=Depends(require_principal)):
        return (
            await service().decide_correlation_suggestion(
                principal,
                suggestion_id=suggestion_id,
                decision="approved",
            )
        ).model_dump()

    @router.post("/correlation-suggestions/{suggestion_id}/reject")
    async def reject_correlation_suggestion(suggestion_id: str, principal=Depends(require_principal)):
        return (
            await service().decide_correlation_suggestion(
                principal,
                suggestion_id=suggestion_id,
                decision="rejected",
            )
        ).model_dump()

    return router
