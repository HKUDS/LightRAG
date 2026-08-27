"""Public, read-only UI customization endpoints (workspace-entry PRD §8.5).

The welcome page appears BEFORE login, so this surface is unauthenticated by
design and strictly scoped: it serves exactly the content and assets the
active bundle's manifest references — never a filesystem path, the bundle
root, or any other server configuration. It must never grow into an
arbitrary-file-read endpoint (route/response-field audits cover it).

Caching model (§8.6):
- ``GET /ui/customization`` is ``Cache-Control: no-store`` with NO ETag and
  no conditional requests — the response is a small per-page-load fetch that
  only changes across server restarts; an ETag would require a canonical
  serialization stable across workers/platforms for no real saving.
- Asset URLs embed the content hash and are served immutable for a year;
  changing bytes change the URL.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from lightrag.api.ui_customization import (
    UICustomizationSnapshot,
    locale_direction,
    normalize_locale,
)

ASSET_CACHE_CONTROL = "public, max-age=31536000, immutable"
# Assets are static images; the restrictive CSP keeps an SVG served here
# inert even if opened as a top-level document.
ASSET_CSP = "default-src 'none'; style-src 'unsafe-inline'; sandbox"


def create_ui_customization_routes(
    snapshot: UICustomizationSnapshot | None,
    webui_title: str | None,
    webui_description: str | None,
) -> APIRouter:
    router = APIRouter(prefix="/ui/customization", tags=["ui-customization"])

    brand_base = {
        # Deployment-level, non-localized strings owned by the existing
        # server configuration; the manifest cannot override them.
        "title": webui_title or "LightRAG",
        "description": webui_description or "Simple and Fast RAG",
    }

    @router.get("")
    async def get_ui_customization(
        request: Request,
        locale: str = Query(default=""),
    ):
        # No bundle is the NORMAL deployment, expressed as 200 + a boolean —
        # never 404 (would blur with rejected unknown assets) nor 503 (this
        # is a stable, correct end state, not a recovering subsystem).
        if snapshot is None:
            return JSONResponse(
                {"customized": False, "brand": dict(brand_base)},
                headers={"Cache-Control": "no-store"},
            )

        try:
            requested = (
                normalize_locale(locale) if locale.strip() else snapshot.default_locale
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        resolved, fallback_used = snapshot.resolve_locale(requested)
        content = snapshot.locales[resolved]

        logo_url = None
        logo_alt = None
        if content.logo_asset_id is not None:
            asset = next(
                (
                    a
                    for (digest, asset_id), a in snapshot.assets.items()
                    if asset_id == content.logo_asset_id
                ),
                None,
            )
            if asset is not None:
                root = request.scope.get("root_path", "")
                logo_url = (
                    f"{root}/ui/customization/assets/{asset.sha256}/{asset.asset_id}"
                )
        if content.logo_asset_id is not None:
            logo_alt = content.logo_alt

        return JSONResponse(
            {
                "customized": True,
                "requested_locale": requested,
                "locale": resolved,
                "fallback_used": fallback_used,
                "direction": locale_direction(resolved),
                "brand": {
                    **brand_base,
                    "logo_url": logo_url,
                    "logo_alt": logo_alt,
                },
                "welcome": {"format": "markdown", "content": content.welcome},
                "query_empty": {"format": "markdown", "content": content.query_empty},
            },
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/assets/{asset_hash}/{asset_id}")
    async def get_ui_customization_asset(asset_hash: str, asset_id: str):
        # Without a bundle there is no legal asset_hash at all; unknown
        # (hash, id) pairs are rejected identically.
        asset = snapshot.assets.get((asset_hash, asset_id)) if snapshot else None
        if asset is None:
            raise HTTPException(status_code=404, detail="Unknown asset")
        return Response(
            content=asset.content,
            media_type=asset.mime,
            headers={
                "Cache-Control": ASSET_CACHE_CONTROL,
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": ASSET_CSP,
            },
        )

    return router
