"""Authenticated runtime observability controls."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.llm.langfuse_tracing import (
    get_langfuse_tracing_status,
    set_langfuse_tracing_enabled,
)


class LangfuseTracingStatus(BaseModel):
    installed: bool
    configured: bool
    enabled: bool
    active: bool


class LangfuseTracingUpdate(BaseModel):
    enabled: bool = Field(
        ...,
        description="Enable or disable Langfuse trace export at runtime.",
    )


def create_observability_routes(api_key: Optional[str] = None) -> APIRouter:
    """Create routes for runtime observability controls."""

    router = APIRouter(
        prefix="/observability",
        tags=["observability"],
    )
    auth_dependency = get_combined_auth_dependency(
        api_key,
        respect_whitelist=False,
    )

    @router.get(
        "/langfuse/tracing",
        response_model=LangfuseTracingStatus,
        dependencies=[Depends(auth_dependency)],
    )
    async def get_langfuse_tracing() -> LangfuseTracingStatus:
        return LangfuseTracingStatus(**get_langfuse_tracing_status())

    @router.put(
        "/langfuse/tracing",
        response_model=LangfuseTracingStatus,
        dependencies=[Depends(auth_dependency)],
    )
    async def update_langfuse_tracing(
        request: LangfuseTracingUpdate,
    ) -> LangfuseTracingStatus:
        current = get_langfuse_tracing_status()

        if not current["installed"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Langfuse observability support is not installed",
            )
        if not current["configured"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Langfuse credentials are not configured",
            )

        try:
            set_langfuse_tracing_enabled(request.enabled)
        except RuntimeError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Langfuse runtime tracing state is unavailable",
            ) from None

        return LangfuseTracingStatus(**get_langfuse_tracing_status())

    return router
