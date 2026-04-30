from __future__ import annotations

from decimal import Decimal
from typing import Any

from lightrag_enterprise.model_gateway.catalog import ModelCatalogFilter
from lightrag_enterprise.model_gateway.openrouter import OpenRouterCatalogClient
from lightrag_enterprise.model_gateway.policy import (
    ModelPolicy,
    ModelRoutingContext,
    PolicyModelRouter,
)
from lightrag_enterprise.system import ACTIVITY_MODEL_MANAGE
from lightrag_enterprise.system.runtime import get_access_service, require_principal


def create_enterprise_admin_router(client: OpenRouterCatalogClient | None = None):
    """Create an optional FastAPI router for enterprise model catalog operations."""

    from fastapi import APIRouter, Depends, HTTPException, Query

    router = APIRouter(prefix="/enterprise", tags=["enterprise"])
    catalog_client = client or OpenRouterCatalogClient.from_env()

    def require_model_admin(principal=Depends(require_principal)):
        decision = get_access_service().require(
            principal,
            activity=ACTIVITY_MODEL_MANAGE,
        )
        if not decision.allowed or not principal.is_master_global:
            raise HTTPException(status_code=403, detail="MASTER model administration required.")
        return principal

    @router.post("/model-catalog/sync")
    async def sync_catalog(principal=Depends(require_model_admin)) -> dict[str, Any]:
        catalog = await catalog_client.fetch_catalog(force=True, account_scoped=True)
        return catalog.to_dict()

    @router.get("/model-catalog")
    async def get_catalog(
        provider: str | None = None,
        family: str | None = None,
        max_input_price: str | None = Query(default=None),
        min_context_window: int | None = None,
        requires_tools: bool | None = None,
        requires_structured_output: bool | None = None,
        principal=Depends(require_model_admin),
    ) -> dict[str, Any]:
        catalog = await catalog_client.fetch_catalog(account_scoped=True)
        entries = catalog.filter(
            ModelCatalogFilter(
                provider=provider,
                family=family,
                max_input_price=Decimal(max_input_price)
                if max_input_price is not None
                else None,
                min_context_window=min_context_window,
                requires_tools=requires_tools,
                requires_structured_output=requires_structured_output,
            )
        )
        return {
            "source": catalog.source,
            "synced_at": catalog.synced_at.isoformat(),
            "count": len(entries),
            "data": [entry.to_dict() for entry in entries],
        }

    @router.post("/model-route")
    async def route_model(payload: dict[str, Any], principal=Depends(require_model_admin)) -> dict[str, Any]:
        try:
            catalog = await catalog_client.fetch_catalog(account_scoped=True)
            router_ = PolicyModelRouter(catalog, ModelPolicy())
            decision = router_.route(ModelRoutingContext(**payload))
            return {
                "allowed": decision.allowed,
                "reason": decision.reason,
                "profile": decision.profile,
                "fallback_chain": decision.fallback_chain,
                "model": decision.model.to_dict() if decision.model else None,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
