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
    LittleBullAgentStudioPreviewRequest,
    LittleBullConversationSaveRequest,
    LittleBullCorrelationSuggestionRequest,
    LittleBullEmbeddingCostEstimateRequest,
    LittleBullKnowledgeBaseReindexRequest,
    LittleBullKnowledgeBaseRollbackRequest,
    LittleBullKnowledgeBaseUpsertRequest,
    LittleBullModelSetting,
    LittleBullQueryRequest,
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
        confidentiality: str = "normal",
        file: UploadFile = File(...),
        principal=Depends(require_principal),
    ):
        return (await service().upload_document(
            principal,
            workspace_id=workspace_id,
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
