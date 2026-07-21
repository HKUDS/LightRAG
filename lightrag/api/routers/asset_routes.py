"""HTTP delivery of workspace-scoped multimodal assets."""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from lightrag.api.assets import WorkspaceAssetCatalog
from lightrag.api.utils_api import get_combined_auth_dependency


def create_asset_routes(document_manager: Any, api_key: Optional[str] = None) -> APIRouter:
    router = APIRouter(tags=["assets"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/assets/{asset_id}", dependencies=[Depends(combined_auth)])
    async def download_asset(
        asset_id: str,
        variant: Literal["original", "thumbnail"] = Query("original"),
        download: bool = Query(False),
    ):
        """Return an asset only when it belongs to the selected workspace."""
        record = WorkspaceAssetCatalog(document_manager).get(asset_id)
        if record is None:
            # Do not disclose whether an ID exists in another workspace.
            raise HTTPException(status_code=404, detail="Asset not found")

        path = WorkspaceAssetCatalog.variant_path(record, variant)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")
        try:
            payload: bytes | None = None
            response_media_type = record.mime_type
            # Generate a bounded thumbnail in memory when no parser-provided
            # thumbnail exists. Pillow is part of the API image extra, but the
            # fallback keeps the endpoint functional in minimal installs.
            if variant == "thumbnail" and path == record.path:
                try:
                    from PIL import Image

                    image = Image.open(path)
                    image.thumbnail((1600, 1600))
                    output = BytesIO()
                    fmt = "JPEG" if record.mime_type in {"image/jpeg", "image/jpg"} else "PNG"
                    image.convert("RGB" if fmt == "JPEG" else image.mode).save(output, format=fmt, optimize=True)
                    payload = output.getvalue()
                    response_media_type = "image/jpeg" if fmt == "JPEG" else "image/png"
                except Exception:
                    payload = None
            file_bytes = payload if payload is not None else path.read_bytes()
            digest = hashlib.sha256(file_bytes).hexdigest()
        except OSError as exc:
            raise HTTPException(status_code=404, detail="Asset not found") from exc

        headers = {
            "ETag": f'"sha256-{digest}"',
            "Cache-Control": "private, max-age=3600",
            "Accept-Ranges": "bytes",
        }
        disposition = "attachment" if download else "inline"
        if payload is not None:
            headers["Content-Disposition"] = f'{disposition}; filename="{record.filename}"'
            return Response(content=payload, media_type=response_media_type, headers=headers)
        return FileResponse(
            path,
            media_type=response_media_type,
            filename=record.filename,
            content_disposition_type=disposition,
            headers=headers,
        )

    return router


__all__ = ["create_asset_routes"]
