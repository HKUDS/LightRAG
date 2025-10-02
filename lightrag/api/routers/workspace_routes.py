"""
Workspaces routes (CRUD-first) for the LightRAG API.
Treats "workspace" and "project" as the same entity.
Old routes mapped to:
- POST   /workspaces                         (was POST /workspace/create)
- POST   /workspaces/{id}/initializations    (was POST /workspace/init/{id})
- GET    /workspaces?user_id=…&limit=…&sort= (was GET  /workspace/user/{user_id})
- GET    /workspaces/{id}                    (was GET  /workspace/{id})
- PATCH  /workspaces/{id}                    (was PUT  /workspace/{id} and PUT /workspace/{id}/instructions)
- DELETE /workspaces/{id}                    (was DELETE /workspace/{id})
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

# -------- Schemas --------

class WorkspaceCreateRequest(BaseModel):
    name: str = Field(min_length=1, description="Workspace (project) name.")
    user_id: str = Field(min_length=1, description="Owner user id (UUID).")
    instructions: Optional[str] = Field(
        default=None, description="Optional instructions string."
    )
    initialize: bool = Field(
        default=False,
        description="If true, initialize LightRAG after create."
    )

class WorkspacePatchRequest(BaseModel):
    name: Optional[str] = Field(default=None, description="New name.")
    user_id: Optional[str] = Field(default=None, description="New owner user id.")
    instructions: Optional[str] = Field(default=None, description="New instructions.")
    # Optional side-effect flag (zero-extra-route variant, if you want it):
    reinitialize: Optional[bool] = Field(
        default=None,
        description="If true, (re)initialize LightRAG after updating."
    )

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

class WorkspaceSingleResponse(BaseModel):
    success: bool = True
    project: WorkspaceOut

class WorkspaceListResponse(BaseModel):
    success: bool = True
    projects: List[WorkspaceOut]

class InitRunOut(BaseModel):
    run_id: str
    workspace_id: str
    status: str = Field(description='queued|running|ready|failed')

# -------------------- Shared helpers --------------------

async def _ensure_rag_initialized(request: Request, user_id: str, workspace_id: str) -> None:
    """
    Ensure LightRAG instance exists/ready for (user_id, workspace_id).
    Uses app.state.instance_manager like in query routes.
    """
    try:
        manager = request.app.state.instance_manager
    except Exception as e:
        logging.warning(f"Instance manager not available: {e}")
        return

    try:
        await manager.get_instance(user_id, workspace_id)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LightRAG instance: {e}") from e


# -------------------- Router factory --------------------

def create_workspace_routes(api_key: Optional[str] = None) -> APIRouter:
    combined_auth = get_combined_auth_dependency(api_key)

    router = APIRouter(prefix="/workspaces", tags=["workspaces"])

    # -------- Routes --------

    @router.post(
        "",
        response_model=WorkspaceSingleResponse,
        status_code=201,
        dependencies=[Depends(combined_auth)],
        summary="Create a new workspace. Optionally initialize LightRAG."
    )
    async def create_workspace(
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

        # Optional initialize
        if payload.initialize:
            try:
                await _ensure_rag_initialized(request, user_id=payload.user_id, workspace_id=project_id)
            except Exception as e:
                trace_exception(e)
                # Match previous behavior: created but init failed -> 500
                raise HTTPException(
                    status_code=500,
                    detail=f"Project created, but failed to initialize LightRAG. {e}",
                )

        return WorkspaceSingleResponse(success=True, project=WorkspaceOut.model_validate(project))

    @router.get(
        "",
        response_model=WorkspaceListResponse,
        dependencies=[Depends(combined_auth)],
        summary="List workspaces (filter by user_id, pagination, sorting)."
    )
    async def list_workspaces(
        user_id: Optional[str] = Q(default=None, description="Filter by owner user id"),
        limit: int = Q(default=30, ge=1, le=100, description="Max rows to return"),
        sort: str = Q(default="-created_at", description="Field to sort by, e.g., -created_at or created_at"),
        db: Session = Depends(get_db),
    ):
        # Base query
        stmt = select(Project)
        if user_id:
            stmt = stmt.where(Project.user_id == user_id)

        # Sorting
        sort_field = sort.lstrip("-")
        desc = sort.startswith("-")
        sort_col = getattr(Project, sort_field, None)
        if sort_col is None:
            raise HTTPException(status_code=400, detail=f"Unsupported sort field: {sort_field}")
        stmt = stmt.order_by(sort_col.desc() if desc else sort_col.asc())

        # Limit (simple page)
        stmt = stmt.limit(limit)

        projects = db.execute(stmt).scalars().all()
        return WorkspaceListResponse(
            success=True,
            projects=[WorkspaceOut.model_validate(p) for p in projects],
        )

    @router.get(
        "/{id}",
        response_model=WorkspaceSingleResponse,
        dependencies=[Depends(combined_auth)],
        summary="Fetch a workspace by id"
    )
    async def get_workspace(
        id: str = Path(..., description="Workspace/Project ID"),
        db: Session = Depends(get_db),
    ):
        project = db.get(Project, id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        return WorkspaceSingleResponse(success=True, project=WorkspaceOut.model_validate(project))

    @router.patch(
        "/{id}",
        response_model=WorkspaceSingleResponse,
        dependencies=[Depends(combined_auth)],
        summary="Patch a workspace (name, user_id, instructions). Optionally reinitialize."
    )
    async def patch_workspace(
        request: Request,
        id: str = Path(..., description="Workspace/Project ID"),
        payload: WorkspacePatchRequest = Body(...),
        db: Session = Depends(get_db),
    ):
        # No-op?
        if (
            payload.name is None
            and payload.user_id is None
            and payload.instructions is None
            and not payload.reinitialize
        ):
            raise HTTPException(
                status_code=400,
                detail="Nothing to update. Provide at least one of: name, user_id, instructions, or reinitialize=true.",
            )

        # Validate user if changing
        if payload.user_id is not None and db.get(User, payload.user_id) is None:
            raise HTTPException(status_code=400, detail="Invalid user_id: user not found.")

        # Build values
        values: Dict[str, Any] = {}
        if payload.name is not None:
            values["name"] = payload.name
        if payload.user_id is not None:
            values["user_id"] = payload.user_id
        if payload.instructions is not None:
            values["instructions"] = payload.instructions

        # Update row
        if values:
            stmt = update(Project).where(Project.id == id).values(**values)
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

        # Optionally (re)initialize after update
        if payload.reinitialize:
            # Need current project to know user_id (after possible change)
            project = db.get(Project, id)
            if project is None:
                raise HTTPException(status_code=404, detail="Project not found.")
            try:
                await _ensure_rag_initialized(request, user_id=project.user_id, workspace_id=id)
            except Exception as e:
                trace_exception(e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to (re)initialize LightRAG for workspace {id}. {e}",
                )

        # Return fresh row
        project = db.get(Project, id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        return WorkspaceSingleResponse(success=True, project=WorkspaceOut.model_validate(project))

    @router.delete(
        "/{id}",
        response_model=ApiSuccess,
        dependencies=[Depends(combined_auth)],
        summary="Delete a workspace"
    )
    async def delete_workspace(
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

    @router.post(
        "/{id}/initializations",
        response_model=InitRunOut,
        status_code=202,
        dependencies=[Depends(combined_auth)],
        summary="(Re)Initialize LightRAG for the workspace (explicit subresource)"
    )
    async def start_initialization(
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
            # Expose a failed run status while keeping a consistent shape
            return InitRunOut(
                run_id=f"init_{uuid4().hex[:8]}",
                workspace_id=id,
                status="failed",
            )

        return InitRunOut(
            run_id=f"init_{uuid4().hex[:8]}",
            workspace_id=id,
            status="ready",  # manager.get_instance completed without raising
        )

    return router