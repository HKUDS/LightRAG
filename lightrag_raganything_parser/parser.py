"""RAG-Anything parser adapter for LightRAG.

This plugin uses RAG-Anything only as a document extractor. It writes the
returned MinerU-style ``content_list`` into a LightRAG raw bundle, then reuses
LightRAG's existing MinerU IR builder and sidecar pipeline.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightrag.parser.external._base import ExternalParserBase
from lightrag.parser.external._common import compute_size_and_hash, env_json
from lightrag.parser.external._manifest import (
    Manifest,
    ManifestFile,
    load_manifest,
    write_manifest,
)

from lightrag_raganything_parser import ENGINE_NAME


CONTENT_LIST_FILENAME = "content_list.json"
RAW_DIR_SUFFIX = ".raganything_raw"
DEFAULT_RAGANYTHING_PARSER = "mineru"
DEFAULT_RAGANYTHING_PARSE_METHOD = "auto"
ASSET_FIELDS = ("img_path", "image_path", "table_img_path", "equation_img_path")


class RAGAnythingParser(ExternalParserBase):
    engine_name = ENGINE_NAME
    raw_dir_suffix = RAW_DIR_SUFFIX
    force_reparse_env = "LIGHTRAG_FORCE_REPARSE_RAGANYTHING"

    def is_bundle_valid(self, raw_dir: Path, source_path: Path) -> bool:
        manifest = load_manifest(raw_dir, expected_engine=ENGINE_NAME)
        content_list_path = raw_dir / CONTENT_LIST_FILENAME
        if manifest is None or not content_list_path.is_file():
            return False

        src_size, src_hash = compute_size_and_hash(source_path)
        if manifest.source_size_bytes != src_size:
            return False
        if manifest.source_content_hash != src_hash:
            return False
        if manifest.options_signature != _options_signature():
            return False

        critical_path = raw_dir / manifest.critical_file.path
        if not critical_path.is_file():
            return False
        crit_size, crit_hash = compute_size_and_hash(critical_path)
        return (
            manifest.critical_file.size == crit_size
            and manifest.critical_file.sha256 == crit_hash
        )

    async def download_into(
        self, raw_dir: Path, source_path: Path, *, upload_name: str
    ) -> None:
        _add_raganything_path()
        from raganything.parser import get_parser

        parser_name = _parser_name()
        parse_method = _parse_method()
        output_dir = raw_dir / "_raganything_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        parser = get_parser(parser_name)
        raw_result = await asyncio.to_thread(
            parser.parse_document,
            str(source_path),
            method=parse_method,
            output_dir=str(output_dir),
            lang=_lang(),
            **_parse_kwargs(),
        )
        if inspect.isawaitable(raw_result):
            raw_result = await raw_result

        content_list = _normalize_content_list(raw_result)
        if not isinstance(content_list, list) or not content_list:
            raise ValueError(
                f"RAG-Anything parser {parser_name!r} produced no content blocks"
            )

        normalized = _copy_referenced_assets(content_list, raw_dir, output_dir)
        content_list_path = raw_dir / CONTENT_LIST_FILENAME
        content_list_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_success_manifest(raw_dir, source_path, upload_name, parser_name)

    def build_ir(self, raw_dir: Path, document_name: str):
        from lightrag.parser.external.mineru import MinerUIRBuilder

        return MinerUIRBuilder().normalize_from_workdir(
            raw_dir, document_name=document_name
        )


def _add_raganything_path() -> None:
    configured = os.getenv("RAGANYTHING_PATH", "").strip()
    if not configured:
        return
    path = str(Path(configured).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _parser_name() -> str:
    return os.getenv("RAGANYTHING_PARSER", DEFAULT_RAGANYTHING_PARSER).strip() or (
        DEFAULT_RAGANYTHING_PARSER
    )


def _parse_method() -> str:
    return (
        os.getenv("RAGANYTHING_PARSE_METHOD", DEFAULT_RAGANYTHING_PARSE_METHOD).strip()
        or DEFAULT_RAGANYTHING_PARSE_METHOD
    )


def _lang() -> str | None:
    return os.getenv("RAGANYTHING_LANG", "").strip() or None


def _parse_kwargs() -> dict[str, Any]:
    payload = env_json("RAGANYTHING_PARSE_KWARGS", {})
    if not isinstance(payload, dict):
        return {}
    return payload


def _normalize_content_list(result: Any) -> list[Any]:
    if isinstance(result, list):
        return result
    if isinstance(result, tuple) and result:
        return _normalize_content_list(result[0])
    if isinstance(result, dict):
        content_list = result.get("content_list")
        if isinstance(content_list, list):
            return content_list
    raise TypeError(
        "RAG-Anything parser returned unsupported result shape; expected "
        "content_list list, tuple/list, or {'content_list': [...]}"
    )


def _options_signature() -> str:
    payload = {
        "parser": _parser_name(),
        "parse_method": _parse_method(),
        "lang": _lang(),
        "kwargs": _parse_kwargs(),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _copy_referenced_assets(
    content_list: list[Any], raw_dir: Path, output_dir: Path
) -> list[Any]:
    assets_dir = raw_dir / "images"
    normalized: list[Any] = []
    for item in content_list:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        next_item = dict(item)
        for field in ASSET_FIELDS:
            value = next_item.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            src = _resolve_asset_source(value, output_dir)
            if not src or not src.is_file():
                continue
            assets_dir.mkdir(parents=True, exist_ok=True)
            dest = _unique_asset_dest(assets_dir, src.name)
            shutil.copy2(src, dest)
            next_item[field] = f"images/{dest.name}"
        normalized.append(next_item)
    return normalized


def _resolve_asset_source(value: str, output_dir: Path) -> Path | None:
    if value.startswith(("http://", "https://")):
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (output_dir / candidate).resolve()


def _unique_asset_dest(directory: Path, name: str) -> Path:
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        next_candidate = directory / f"{stem}-{index}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        index += 1


def _write_success_manifest(
    raw_dir: Path, source_path: Path, upload_name: str, parser_name: str
) -> None:
    content_list_path = raw_dir / CONTENT_LIST_FILENAME
    source_size, source_hash = compute_size_and_hash(source_path)
    crit_size, crit_hash = compute_size_and_hash(content_list_path)
    files: list[ManifestFile] = []
    total_size = crit_size
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(raw_dir).as_posix()
        if rel in {CONTENT_LIST_FILENAME, "_manifest.json"}:
            continue
        size = path.stat().st_size
        total_size += size
        files.append(ManifestFile(path=rel, size=size))

    write_manifest(
        raw_dir,
        Manifest(
            engine=ENGINE_NAME,
            source_content_hash=source_hash,
            source_size_bytes=source_size,
            source_filename_at_parse=upload_name,
            critical_file=ManifestFile(
                path=CONTENT_LIST_FILENAME,
                size=crit_size,
                sha256=crit_hash,
            ),
            files=files,
            total_size_bytes=total_size,
            api_mode="raganything",
            options_signature=_options_signature(),
            downloaded_at=datetime.now(timezone.utc).isoformat(),
            extras={"raganything_parser": parser_name},
        ),
    )


__all__ = ["RAGAnythingParser"]
