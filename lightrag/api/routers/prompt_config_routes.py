from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

INDEXING_ACTIVATION_WARNING = (
    "Activating a new indexing configuration only affects future indexing work and may "
    "create mixed-schema graph data unless the workspace is rebuilt."
)

PromptConfigGroup = Literal["indexing", "retrieval"]


class PromptVersionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_name: str = Field(min_length=1)
    comment: str = Field(default="")
    payload: dict[str, Any] = Field(default_factory=dict)
    source_version_id: str | None = Field(default=None)


def create_prompt_config_routes(rag, api_key: str | None = None) -> APIRouter:
    router = APIRouter(prefix="/prompt-config", tags=["prompt-config"])

    def _store():
        if not hasattr(rag, "prompt_version_store"):
            raise HTTPException(
                status_code=500, detail="Prompt version store is not available"
            )
        return rag.prompt_version_store

    def _validate_group_type(group_type: str) -> PromptConfigGroup:
        if group_type not in {"indexing", "retrieval"}:
            raise HTTPException(status_code=404, detail="Unknown prompt config group")
        return group_type  # type: ignore[return-value]

    @router.post("/initialize")
    async def initialize_prompt_config(locale: str = "zh") -> dict[str, Any]:
        return _store().initialize(locale=locale)

    @router.get("/groups")
    async def list_groups() -> dict[str, Any]:
        store = _store()
        return {
            "indexing": store.list_versions("indexing"),
            "retrieval": store.list_versions("retrieval"),
        }

    @router.get("/{group_type}/versions")
    async def list_versions(group_type: str) -> dict[str, Any]:
        validated_group = _validate_group_type(group_type)
        return _store().list_versions(validated_group)

    @router.get("/{group_type}/versions/{version_id}")
    async def get_version(group_type: str, version_id: str) -> dict[str, Any]:
        validated_group = _validate_group_type(group_type)
        try:
            return _store().get_version(validated_group, version_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/{group_type}/versions")
    async def create_version(
        group_type: str, request: PromptVersionCreateRequest
    ) -> dict[str, Any]:
        validated_group = _validate_group_type(group_type)
        try:
            return _store().create_version(
                validated_group,
                request.payload,
                request.version_name,
                request.comment,
                request.source_version_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/{group_type}/versions/{version_id}/activate")
    async def activate_version(group_type: str, version_id: str) -> dict[str, Any]:
        validated_group = _validate_group_type(group_type)
        try:
            active_version = _store().activate_version(validated_group, version_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return {
            "group_type": validated_group,
            "active_version_id": version_id,
            "active_version": active_version,
            "warning": (
                INDEXING_ACTIVATION_WARNING if validated_group == "indexing" else None
            ),
        }

    @router.delete("/{group_type}/versions/{version_id}")
    async def delete_version(group_type: str, version_id: str) -> dict[str, Any]:
        validated_group = _validate_group_type(group_type)
        try:
            _store().delete_version(validated_group, version_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "success", "version_id": version_id}

    @router.get("/{group_type}/versions/{version_id}/diff")
    async def diff_version(
        group_type: str, version_id: str, base_version_id: str | None = None
    ) -> dict[str, Any]:
        validated_group = _validate_group_type(group_type)
        try:
            return _store().diff_versions(validated_group, version_id, base_version_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router
