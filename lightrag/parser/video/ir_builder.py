"""Translate video extraction records into the common LightRAG sidecar IR."""

from __future__ import annotations

from typing import Any

from lightrag.sidecar.ir import AssetSpec, IRBlock, IRDoc, IRDrawing


def build_video_ir(
    blocks: list[dict[str, Any]],
    *,
    document_name: str,
    asset_dir_name: str,
    metadata: dict[str, Any],
) -> IRDoc:
    assets = [
        AssetSpec(
            ref=str(block["asset_ref"]),
            suggested_name=str(block["asset_ref"]),
            source=None,
        )
        for block in blocks
        if block.get("asset_ref")
    ]
    seen_assets: set[str] = set()
    unique_assets: list[AssetSpec] = []
    for asset in assets:
        if asset.ref not in seen_assets:
            seen_assets.add(asset.ref)
            unique_assets.append(asset)

    ir_blocks: list[IRBlock] = []
    for block in blocks:
        drawing: IRDrawing | None = None
        asset_ref = block.get("asset_ref")
        if asset_ref:
            drawing = IRDrawing(
                placeholder_key=str(block["placeholder_key"]),
                asset_ref=str(asset_ref),
                fmt="jpeg",
                caption=str(block.get("caption") or ""),
                self_ref=str(block.get("manifest_ref") or ""),
                extras={
                    "media_type": "video_contact_sheet",
                    "scene_id": block.get("scene_id"),
                    "start_ms": block.get("start_ms"),
                    "end_ms": block.get("end_ms"),
                    "frame_timestamps_ms": block.get("frame_timestamps_ms", []),
                    "manifest_ref": block.get("manifest_ref"),
                },
            )
        ir_blocks.append(
            IRBlock(
                content_template=str(block["content"]),
                heading=str(block.get("heading") or ""),
                level=int(block.get("level", 0)),
                parent_headings=list(block.get("parent_headings") or []),
                session_type="body",
                drawings=[drawing] if drawing else [],
            )
        )

    return IRDoc(
        document_name=document_name,
        document_format="video",
        doc_title=str(metadata.get("title") or document_name),
        split_option={
            "strategy": "video_scene",
            "asset_dir": asset_dir_name,
            "manifest": metadata.get("manifest_name"),
        },
        blocks=ir_blocks,
        assets=unique_assets,
    )
