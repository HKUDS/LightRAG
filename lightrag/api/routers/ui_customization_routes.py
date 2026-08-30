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


def _markdown_or_none(content: str | None) -> dict[str, str] | None:
    """An optional template as the same ``{format, content}`` shape, or null.

    Null rather than an empty-content object, because "this locale declares
    none" is a state the client acts on (no login blurb, no agreement link) --
    not a template that happens to have no text, which the loader rejects
    outright for these fields.
    """
    return None if content is None else {"format": "markdown", "content": content}


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

    # asset_id -> asset, built once. The snapshot keys assets by
    # (asset_hash, asset_id) because a URL must match BOTH parts, but a
    # locale references its logo by asset_id alone; asset_ids are unique
    # within a bundle (BRAND_LOGO_ASSET_ID plus one "logo-<locale>" per
    # locale), so this index is exact and saves a per-request scan.
    assets_by_id = (
        {asset.asset_id: asset for asset in snapshot.assets.values()}
        if snapshot is not None
        else {}
    )

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

        # URL and alt text are ONE decision: a response carrying an alt text
        # for a logo it could not name would describe an image the page never
        # renders. Resolved in a single branch so the two cannot diverge.
        logo_url = None
        logo_alt = None
        asset = (
            assets_by_id.get(content.logo_asset_id)
            if content.logo_asset_id is not None
            else None
        )
        if asset is not None:
            root = request.scope.get("root_path", "")
            logo_url = f"{root}/ui/customization/assets/{asset.sha256}/{asset.asset_id}"
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
                "login": _markdown_or_none(content.login),
                "agreements": _markdown_or_none(content.agreements),
                # The login page's consent gate, decided HERE and obeyed by
                # the frontend rather than re-derived there: one authority
                # over what turns a login-blocking control on. Its two inputs
                # are in the same response purely so the page can render the
                # blurb and the agreement dialog -- not so a client can
                # recompute the flag from them.
                "consent_required": content.consent_required,
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
