"""PaddleOCR-VL raw bundle downloader."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from lightrag.parser.external._common import (
    compute_size_and_hash,
    env_int,
    raise_for_status_with_detail,
)
from lightrag.parser.external._manifest import Manifest, ManifestFile, write_manifest
from lightrag.parser.external.paddleocr_vl.cache import (
    CONTENT_LIST_FILENAME,
    DEFAULT_PADDLEOCR_VL_API_MODE,
    DEFAULT_PADDLEOCR_VL_ENGINE_VERSION,
    DEFAULT_PADDLEOCR_VL_OFFICIAL_ENDPOINT,
    MANIFEST_ENGINE,
    PaddleOCRVLParserOptions,
    VALID_PADDLEOCR_VL_API_MODES,
)

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None

DEFAULT_POLL_INTERVAL_SECONDS = 5
DEFAULT_MAX_POLLS = 600


class PaddleOCRVLRawClient:
    """Submit a source file to PaddleOCR-VL and preserve its raw JSON output."""

    def __init__(self, *, overrides: "Mapping[str, Any] | None" = None) -> None:
        self._overrides = overrides or {}
        self.api_mode = (
            os.getenv("PADDLEOCR_VL_API_MODE", DEFAULT_PADDLEOCR_VL_API_MODE)
            .strip()
            .lower()
            or DEFAULT_PADDLEOCR_VL_API_MODE
        )
        if self.api_mode not in VALID_PADDLEOCR_VL_API_MODES:
            allowed = ", ".join(sorted(VALID_PADDLEOCR_VL_API_MODES))
            raise ValueError(
                f"PADDLEOCR_VL_API_MODE must be one of {allowed}, got {self.api_mode!r}"
            )
        self.official_endpoint = (
            os.getenv(
                "PADDLEOCR_VL_OFFICIAL_ENDPOINT",
                os.getenv(
                    "PADDLEOCR_VL_ENDPOINT", DEFAULT_PADDLEOCR_VL_OFFICIAL_ENDPOINT
                ),
            )
            .strip()
            .rstrip("/")
            or DEFAULT_PADDLEOCR_VL_OFFICIAL_ENDPOINT
        )

        self.local_endpoint = (
            os.getenv("PADDLEOCR_VL_LOCAL_ENDPOINT", "").strip().rstrip("/")
        )

        self.api_token = os.getenv("PADDLEOCR_VL_API_TOKEN", "").strip()
        if self.api_mode == "official":
            if not self.api_token:
                raise ValueError(
                    "PADDLEOCR_VL_API_TOKEN is required when "
                    "PADDLEOCR_VL_API_MODE=official"
                )
            self.endpoint = self.official_endpoint
        elif self.api_mode == "local":
            if not self.local_endpoint:
                raise ValueError(
                    "PADDLEOCR_VL_LOCAL_ENDPOINT is required when "
                    "PADDLEOCR_VL_API_MODE=local"
                )
            self.endpoint = self.local_endpoint
            raise NotImplementedError("PaddleOCR-VL local mode is not implemented yet")
        options = PaddleOCRVLParserOptions.from_env(
            api_mode=self.api_mode, overrides=self._overrides
        )
        self._parser_options = options
        self.request_payload = options.request_payload()
        self.poll_interval = env_int(
            "PADDLEOCR_VL_POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL_SECONDS
        )
        self.max_polls = env_int("PADDLEOCR_VL_MAX_POLLS", DEFAULT_MAX_POLLS)
        self.engine_version = (
            os.getenv(
                "PADDLEOCR_VL_ENGINE_VERSION", DEFAULT_PADDLEOCR_VL_ENGINE_VERSION
            ).strip()
            or DEFAULT_PADDLEOCR_VL_ENGINE_VERSION
        )

    async def download_into(
        self,
        raw_dir: Path,
        source_file_path: Path,
        *,
        upload_name: str | None = None,
    ) -> Manifest:
        if httpx is None:
            raise RuntimeError(
                "httpx is required for PaddleOCR-VL parsing but is not installed"
            )
        raw_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            Path(upload_name or source_file_path.name).name or source_file_path.name
        )

        timeout = httpx.Timeout(120.0, connect=30.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                job_id = await self._submit_official(client, source_file_path, filename)
                json_url = await self._poll_official_until_done(client, job_id)
                pages = await self._download_json_result(client, json_url)
                await self._download_referenced_images(client, pages, raw_dir)
        except httpx.RequestError as exc:
            raise RuntimeError(
                "PaddleOCR-VL backend request failed "
                f"(endpoint={self.endpoint}): {type(exc).__name__}: {exc}"
            ) from exc

        (raw_dir / CONTENT_LIST_FILENAME).write_text(
            json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self._write_manifest(raw_dir, source_file_path, job_id, filename)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"bearer {self.api_token}"}

    async def _submit_official(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
        filename: str,
    ) -> str:
        headers = self._headers()
        data = dict(self.request_payload)
        data["optionalPayload"] = json.dumps(data.get("optionalPayload", {}))
        with source_file_path.open("rb") as fh:
            resp = await client.post(
                self.official_endpoint,
                headers=headers,
                data=data,
                files={"file": (filename, fh, "application/octet-stream")},
            )
        raise_for_status_with_detail(resp, f"PaddleOCR-VL upload for {filename!r}")
        payload = resp.json() if resp.text else {}
        job_id = str(_get_by_path(payload, "data.jobId") or "").strip()
        if not job_id:
            raise RuntimeError(f"PaddleOCR-VL upload response missing jobId: {payload}")
        return job_id

    async def _poll_official_until_done(
        self, client: "httpx.AsyncClient", job_id: str
    ) -> str:
        url = f"{self.official_endpoint.rstrip('/')}/{job_id}"
        for _ in range(max(self.max_polls, 1)):
            resp = await client.get(url, headers=self._headers())
            raise_for_status_with_detail(resp, f"PaddleOCR-VL job {job_id} poll")
            payload = resp.json() if resp.text else {}
            data = payload.get("data") if isinstance(payload, dict) else {}
            state = str((data or {}).get("state") or "").lower()
            if state == "done":
                json_url = str(_get_by_path(payload, "data.resultUrl.jsonUrl") or "")
                if not json_url:
                    raise RuntimeError(
                        f"PaddleOCR-VL job {job_id} finished without jsonUrl: {payload}"
                    )
                return json_url
            if state == "failed":
                err = (data or {}).get("errorMsg") or (data or {}).get("error") or data
                raise RuntimeError(f"PaddleOCR-VL job {job_id} failed: {err}")
            await asyncio.sleep(max(self.poll_interval, 0))
        raise TimeoutError(f"PaddleOCR-VL job polling timeout: {job_id}")

    async def _download_json_result(
        self, client: "httpx.AsyncClient", json_url: str
    ) -> list[dict[str, Any]]:
        resp = await client.get(json_url)
        raise_for_status_with_detail(resp, "PaddleOCR-VL json result download")
        text = resp.text or resp.content.decode("utf-8")
        pages: list[dict[str, Any]] = []
        stripped = text.strip()
        if not stripped:
            return pages
        if stripped.startswith("["):
            payload = json.loads(stripped)
            if isinstance(payload, list):
                return [p for p in payload if isinstance(p, dict)]
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            result = payload.get("result") if isinstance(payload, dict) else None
            page_items = (
                result.get("layoutParsingResults") if isinstance(result, dict) else None
            )
            if isinstance(page_items, list):
                pages.extend(p for p in page_items if isinstance(p, dict))
        return pages

    async def _download_referenced_images(
        self,
        client: "httpx.AsyncClient",
        pages: list[dict[str, Any]],
        raw_dir: Path,
    ) -> None:
        for page_index, page in enumerate(pages):
            markdown = page.get("markdown") if isinstance(page, dict) else None
            images = markdown.get("images") if isinstance(markdown, dict) else None
            if isinstance(images, dict):
                for rel_path, url in images.items():
                    await self._download_one_image(
                        client, str(url), raw_dir / _safe_relative_path(str(rel_path))
                    )

            output_images = page.get("outputImages") if isinstance(page, dict) else None
            if isinstance(output_images, dict):
                for name, url in output_images.items():
                    suffix = _suffix_from_url(str(url)) or ".jpg"
                    rel = (
                        Path("outputImages")
                        / f"{_safe_name(str(name))}_{page_index}{suffix}"
                    )
                    await self._download_one_image(client, str(url), raw_dir / rel)

    async def _download_one_image(
        self, client: "httpx.AsyncClient", url: str, target: Path
    ) -> None:
        if not url:
            return
        resp = await client.get(url)
        raise_for_status_with_detail(resp, f"PaddleOCR-VL image download {url!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resp.content)

    def _write_manifest(
        self,
        raw_dir: Path,
        source_file_path: Path,
        job_id: str,
        upload_name: str,
    ) -> Manifest:
        source_size, source_hash = compute_size_and_hash(source_file_path)
        result_path = raw_dir / CONTENT_LIST_FILENAME
        result_size, result_hash = compute_size_and_hash(result_path)

        files: list[ManifestFile] = []
        total = result_size
        for path in sorted(raw_dir.rglob("*")):
            if not path.is_file() or path.name == "_manifest.json":
                continue
            rel = path.relative_to(raw_dir).as_posix()
            if rel == CONTENT_LIST_FILENAME:
                continue
            size = path.stat().st_size
            files.append(ManifestFile(rel, size))
            total += size

        manifest = Manifest(
            engine=MANIFEST_ENGINE,
            source_content_hash=source_hash,
            source_size_bytes=source_size,
            source_filename_at_parse=upload_name,
            critical_file=ManifestFile(CONTENT_LIST_FILENAME, result_size, result_hash),
            files=files,
            total_size_bytes=total,
            task_id=job_id,
            api_mode=self.api_mode,
            endpoint_signature=self.endpoint,
            engine_version=self.engine_version,
            options_signature=self._options_signature(),
            downloaded_at=datetime.now(timezone.utc).isoformat(),
        )
        write_manifest(raw_dir, manifest)
        return manifest

    def _options_signature(self) -> str:
        return self._parser_options.signature()


def _get_by_path(payload: Any, path: str) -> Any:
    cur = payload
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _safe_relative_path(path: str) -> Path:
    clean = Path(path.replace("\\", "/"))
    parts = [_safe_name(part) for part in clean.parts if part not in {"", ".", ".."}]
    return Path(*parts) if parts else Path("asset")


def _safe_name(name: str) -> str:
    cleaned = "".join(ch for ch in name if ord(ch) >= 32 and ch not in "/\\").strip()
    return cleaned.strip(".") or "asset"


def _suffix_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix and len(suffix) <= 8 else ""


__all__ = ["PaddleOCRVLRawClient"]
