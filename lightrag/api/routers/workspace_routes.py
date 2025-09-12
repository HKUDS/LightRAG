"""
Workspace (Project) routes for the LightRAG API.
Treats "workspace" and "project" as the same entity.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, Request, Path, Body
from fastapi import Header, Query as Q
from sqlalchemy import select, update, delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from pydantic import BaseModel, Field, ConfigDict
from ascii_colors import trace_exception

from ..utils_api import get_combined_auth_dependency
from ..database import get_db
from ..models import User, Project

try:
    import httpx
except ImportError:
    httpx = None

router = APIRouter(
    prefix="/workspace",
    tags=["workspace"]
)


class WorkspaceCreateRequest(BaseModel):
    name: str = Field(min_length=1, description="Workspace (project) name.")
    user_id: str = Field(min_length=1, description="Owner user id (UUID).")
    instructions: Optional[str] = Field(
        default=None, description="Optional instructions string."
    )


class WorkspaceUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, description="New name.")
    user_id: Optional[str] = Field(default=None, description="New owner user id.")
    instructions: Optional[str] = Field(default=None, description="New instructions.")


class InstructionsUpdateRequest(BaseModel):
    instructions: str = Field(min_length=1, description="Instructions content.")


class WorkspaceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: Optional[str]
    user_id: str
    instructions: Optional[str] = None
    created_at: Any


class ApiSuccess(BaseModel):
    success: bool = True
    message: Optional[str] = None


class HealthResponse(BaseModel):
    success: bool
    status: Optional[int] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    details: Optional[Any] = None


class WorkspaceSingleResponse(BaseModel):
    success: bool = True
    project: WorkspaceOut


class WorkspaceListResponse(BaseModel):
    success: bool = True
    projects: List[WorkspaceOut]


async def _ensure_rag_initialized(request: Request, user_id: str, workspace_id: str) -> None:
    """
    Ensure LightRAG instance exists/ready for (user_id, workspace_id).
    Uses app.state.instance_manager like in query routes.
    """
    try:
        manager = request.app.state.instance_manager
    except Exception as e:
        # If instance manager is not available, this is a soft failure.
        logging.warning(f"Instance manager not available: {e}")
        return

    try:
        # get_instance should create if missing and return (rag, doc_manager)
        await manager.get_instance(user_id, workspace_id)
    except Exception as e:
        # Propagate to caller (Node code returned 500 in this case)
        raise RuntimeError(f"Failed to initialize LightRAG instance: {e}") from e
    

def create_workspace_routes(api_key: Optional[str] = None) -> APIRouter:
    """
    Factory to build workspace/project routes with shared auth dependency.
    """
    combined_auth = get_combined_auth_dependency(api_key)


    @router.post(
        "/create",
        response_model=WorkspaceSingleResponse,
        status_code=201,
        dependencies=[Depends(combined_auth)],
        summary="Create a new workspace (project) and initialize LightRAG",
    )
    async def create_new(
        request: Request,
        payload: WorkspaceCreateRequest,
        db: Session = Depends(get_db),
    ):
        # Validate user existence
        owner = db.get(User, payload.user_id)
        if owner is None:
            raise HTTPException(status_code=400, detail="Invalid user_id: user not found.")

        # Create project
        project_id = str(uuid4())
        project = Project(
            id=project_id,
            name=payload.name,
            user_id=payload.user_id,
            instructions=payload.instructions,
        )
        try:
            db.add(project)
            db.commit()
            db.refresh(project)
        except Exception as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=500, detail="Failed to create project.")

        # Initialize LightRAG instance for this project (workspace)
        try:
            await _ensure_rag_initialized(request, user_id=payload.user_id, workspace_id=project_id)
        except Exception as e:
            # Match Node behavior: created but failed to start lightrag -> 500
            trace_exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Project created, but failed to initialize LightRAG. {e}",
            )

        return WorkspaceSingleResponse(
            success=True,
            project=WorkspaceOut.model_validate(project),
        )


    @router.post(
        "/init/{id}",
        response_model=ApiSuccess,
        dependencies=[Depends(combined_auth)],
        summary="Initialize (or re-initialize) LightRAG for the workspace",
    )
    async def init_rag(
        request: Request,
        id: str = Path(..., description="Workspace/Project ID"),
        db: Session = Depends(get_db),
    ):
        # Verify project exists
        project = db.get(Project, id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")

        try:
            await _ensure_rag_initialized(request, user_id=project.user_id, workspace_id=id)
        except Exception as e:
            trace_exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize LightRAG for workspace {id}. {e}",
            )

        return ApiSuccess(message=f"LightRAG initialized for workspace {id}.")


    @router.get(
        "/user/{user_id}",
        response_model=WorkspaceListResponse,
        dependencies=[Depends(combined_auth)],
        summary="List workspaces for a user",
    )
    async def get_by_user(
        user_id: str = Path(..., description="Owner user id"),
        db: Session = Depends(get_db),
    ):
        stmt = (
            select(Project)
            .where(Project.user_id == user_id)
            .order_by(Project.created_at.desc())
        )
        projects = db.execute(stmt).scalars().all()
        return WorkspaceListResponse(
            success=True,
            projects=[WorkspaceOut.model_validate(p) for p in projects],
        )


    @router.get(
        "/{id}",
        response_model=WorkspaceSingleResponse,
        dependencies=[Depends(combined_auth)],
        summary="Fetch a workspace by id",
    )
    async def get_by_id(
        id: str = Path(..., description="Workspace/Project ID"),
        db: Session = Depends(get_db),
    ):
        project = db.get(Project, id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        return WorkspaceSingleResponse(success=True, project=WorkspaceOut.model_validate(project))


    @router.put(
        "/{id}",
        response_model=ApiSuccess,
        dependencies=[Depends(combined_auth)],
        summary="Update a workspace",
    )
    async def update_by_id(
        id: str = Path(..., description="Workspace/Project ID"),
        payload: WorkspaceUpdateRequest = Body(...),
        db: Session = Depends(get_db),
    ):
        # Nothing to update?
        if (
            payload.name is None
            and payload.user_id is None
            and payload.instructions is None
        ):
            raise HTTPException(
                status_code=400,
                detail="Nothing to update. Provide at least `name`, `user_id`, or `instructions`.",
            )

        # If user_id provided, validate exists
        if payload.user_id is not None and db.get(User, payload.user_id) is None:
            raise HTTPException(status_code=400, detail="Invalid user_id: user not found.")

        values: Dict[str, Any] = {}
        if payload.name is not None:
            values["name"] = payload.name
        if payload.user_id is not None:
            values["user_id"] = payload.user_id
        if payload.instructions is not None:
            values["instructions"] = payload.instructions

        stmt = (
            update(Project)
            .where(Project.id == id)
            .values(**values)
        )
        try:
            res = db.execute(stmt)
            db.commit()
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="Project not found.")
        except IntegrityError as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=400, detail="Integrity error while updating project.")
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=500, detail="Failed to update project.")

        return ApiSuccess(message="Project updated successfully.")


    @router.delete(
        "/{id}",
        response_model=ApiSuccess,
        dependencies=[Depends(combined_auth)],
        summary="Delete a workspace",
    )
    async def delete_by_id(
        id: str = Path(..., description="Workspace/Project ID"),
        db: Session = Depends(get_db),
    ):
        try:
            stmt = delete(Project).where(Project.id == id)
            res = db.execute(stmt)
            db.commit()
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="Project not found.")
        except IntegrityError as e:
            db.rollback()
            trace_exception(e)
            # Likely prevented by FK RESTRICT (files/chat_sessions/questions still present)
            raise HTTPException(
                status_code=409,
                detail="Cannot delete project due to existing references."
            )
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=500, detail="Failed to delete project.")

        return ApiSuccess(message="Project deleted successfully.")


    @router.put(
        "/{id}/instructions",
        response_model=ApiSuccess,
        dependencies=[Depends(combined_auth)],
        summary="Save/update instructions for a workspace",
    )
    async def save_instructions(
        id: str = Path(..., description="Workspace/Project ID"),
        payload: InstructionsUpdateRequest = Body(...),
        db: Session = Depends(get_db),
    ):
        stmt = (
            update(Project)
            .where(Project.id == id)
            .values(instructions=payload.instructions)
        )
        try:
            res = db.execute(stmt)
            db.commit()
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="Project not found.")
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=500, detail="Failed to save instructions.")

        return ApiSuccess(message="Instructions saved successfully.")

    return router