from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile

from lightrag_enterprise.system.runtime import (
    get_access_service,
    get_approval_service,
    get_audit_service,
    get_system_repository,
    require_principal,
)

from .models import LittleBullQueryRequest
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

    return router

