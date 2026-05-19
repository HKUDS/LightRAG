"""Workspace-related routes for the LightRAG API."""

from typing import Optional, List
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.api.workspace_registry import get_workspace_registry

router = APIRouter(
    prefix="/workspaces",
    tags=["workspaces"],
)


class WorkspaceInfo(BaseModel):
    """Workspace information model.

    Attributes:
        name: The workspace name.
        first_seen: ISO timestamp when the workspace was first seen.
        last_seen: ISO timestamp when the workspace was last accessed.
    """

    name: str = Field(description="The workspace name")
    first_seen: str = Field(
        description="ISO timestamp when the workspace was first seen"
    )
    last_seen: str = Field(
        description="ISO timestamp when the workspace was last accessed"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "my-workspace",
                "first_seen": "2026-04-30T10:00:00+00:00",
                "last_seen": "2026-04-30T11:30:00+00:00",
            }
        }
    )


class WorkspacesResponse(BaseModel):
    """Response model for listing workspaces.

    Attributes:
        workspaces: List of workspace information.
    """

    workspaces: List[WorkspaceInfo] = Field(description="List of registered workspaces")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workspaces": [
                    {
                        "name": "my-workspace",
                        "first_seen": "2026-04-30T10:00:00+00:00",
                        "last_seen": "2026-04-30T11:30:00+00:00",
                    },
                    {
                        "name": "research-docs",
                        "first_seen": "2026-04-29T08:00:00+00:00",
                        "last_seen": "2026-04-30T09:15:00+00:00",
                    },
                ]
            }
        }
    )


def create_workspace_routes(
    api_key: Optional[str] = None, working_dir: Optional[str] = None
):
    """Create workspace routes with the given API key and working directory configuration.

    Args:
        api_key: Optional API key for authentication.
        working_dir: Optional working directory for workspace registry.

    Returns:
        Configured APIRouter with workspace endpoints.
    """
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        "",
        response_model=WorkspacesResponse,
        dependencies=[Depends(combined_auth)],
        summary="List all workspaces",
        description="Returns a list of all workspaces that have been registered through document API calls. "
        "Workspaces are automatically registered when document endpoints are called with the "
        "LIGHTRAG-WORKSPACE header.",
    )
    async def list_workspaces():
        """
        List all registered workspaces.

        Returns:
            WorkspacesResponse: A response containing the list of all known workspaces
                               with their first_seen and last_seen timestamps.
        """
        workspace_registry = get_workspace_registry(working_dir=working_dir)
        workspaces = workspace_registry.get_workspaces()
        return WorkspacesResponse(
            workspaces=[
                WorkspaceInfo(
                    name=w["name"],
                    first_seen=w["first_seen"],
                    last_seen=w["last_seen"],
                )
                for w in workspaces
            ]
        )

    return router
