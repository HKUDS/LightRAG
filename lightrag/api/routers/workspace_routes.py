"""Workspace control-plane CRUD routes.

Data-plane routes deliberately retain their existing paths. Clients select a
workspace for documents, query, graph, and Ollama APIs through the shared
``X-LightRAG-Workspace-ID`` header dependency.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.api.workspaces import (
    WorkspaceBusyError,
    WorkspaceNotFoundError,
    WorkspaceRecord,
    WorkspaceRuntimeManager,
    WorkspaceStateError,
)


class WorkspaceCreateRequest(BaseModel):
    display_name: str = Field(
        min_length=1,
        max_length=255,
        description="Human-readable workspace name. It can be changed later.",
    )
    description: str | None = Field(default=None, max_length=10_000)


class WorkspaceUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=10_000)

    @model_validator(mode="after")
    def require_at_least_one_field(self) -> "WorkspaceUpdateRequest":
        if (
            "display_name" not in self.model_fields_set
            and "description" not in self.model_fields_set
        ):
            raise ValueError(
                "At least one of display_name or description must be supplied"
            )
        return self


class WorkspaceResponse(BaseModel):
    id: UUID
    display_name: str
    description: str | None
    status: str
    is_default: bool
    created_at: str
    updated_at: str
    deleted_at: str | None = None


class WorkspaceDeleteResponse(BaseModel):
    status: str
    workspace_id: UUID
    job_id: UUID
    message: str


def _response(record: WorkspaceRecord) -> WorkspaceResponse:
    return WorkspaceResponse(
        id=UUID(record.id),
        display_name=record.display_name,
        description=record.description,
        status=record.status,
        is_default=record.is_default,
        created_at=record.created_at,
        updated_at=record.updated_at,
        deleted_at=record.deleted_at,
    )


def create_workspace_routes(
    manager: WorkspaceRuntimeManager, api_key: str | None = None
) -> APIRouter:
    router = APIRouter(prefix="/v1/workspaces", tags=["workspaces"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "",
        response_model=WorkspaceResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(combined_auth)],
    )
    async def create_workspace(request: WorkspaceCreateRequest) -> WorkspaceResponse:
        try:
            workspace = await manager.create_workspace(
                request.display_name, request.description
            )
            return _response(workspace)
        except (ValueError, WorkspaceStateError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get(
        "",
        response_model=list[WorkspaceResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_workspaces(include_deleted: bool = False) -> list[WorkspaceResponse]:
        records = await manager.registry.list(include_deleted=include_deleted)
        return [_response(record) for record in records]

    @router.get(
        "/{workspace_id}",
        response_model=WorkspaceResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_workspace(workspace_id: UUID) -> WorkspaceResponse:
        try:
            return _response(await manager.registry.get(workspace_id))
        except WorkspaceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.patch(
        "/{workspace_id}",
        response_model=WorkspaceResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def update_workspace(
        workspace_id: UUID, request: WorkspaceUpdateRequest
    ) -> WorkspaceResponse:
        try:
            workspace = await manager.update_workspace(
                workspace_id,
                display_name=request.display_name,
                description=request.description,
                description_provided="description" in request.model_fields_set,
            )
            return _response(workspace)
        except WorkspaceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, WorkspaceStateError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.delete(
        "/{workspace_id}",
        response_model=WorkspaceDeleteResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_workspace(
        workspace_id: UUID, background_tasks: BackgroundTasks
    ) -> WorkspaceDeleteResponse:
        try:
            workspace, job_id = await manager.request_workspace_deletion(workspace_id)
            background_tasks.add_task(manager.delete_workspace, workspace.id, job_id)
            return WorkspaceDeleteResponse(
                status="deleting",
                workspace_id=UUID(workspace.id),
                job_id=UUID(job_id),
                message="Workspace deletion has been scheduled.",
            )
        except WorkspaceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except WorkspaceBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except WorkspaceStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return router
