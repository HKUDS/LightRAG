"""PaddleOCR-VL raw bundle downloader."""

from __future__ import annotations

import asyncio
import base64
import binascii
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
    VALID_PADDLEOCR_VL_API_MODES,
    PaddleOCRVLParserOptions,
)
from lightrag.utils import logger

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None

DEFAULT_POLL_INTERVAL_SECONDS = 5
DEFAULT_MAX_POLLS = 600
# Comma-separated hostname suffixes whose HTTPS URLs are safe to fetch as
# remote assets. PaddleOCR-VL returns presigned BOS (Baidu Object Storage) URLs
# under *.bcebos.com; other hosts are never fetched (SSRF guard). Override with
# PADDLEOCR_VL_ALLOWED_ASSET_HOSTS to admit additional self-hosted domains.
DEFAULT_ALLOWED_ASSET_HOST_SUFFIXES = (".bcebos.com",)


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
        self.allowed_asset_host_suffixes = self._load_allowed_asset_hosts()

    @staticmethod
    def _load_allowed_asset_hosts() -> tuple[str, ...]:
        raw = os.getenv("PADDLEOCR_VL_ALLOWED_ASSET_HOSTS", "").strip()
        if not raw:
            return DEFAULT_ALLOWED_ASSET_HOST_SUFFIXES
        suffixes = tuple(h.strip().lower() for h in raw.split(",") if h.strip())
        return suffixes or DEFAULT_ALLOWED_ASSET_HOST_SUFFIXES

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
                if self.api_mode == "official":
                    task_id, pages = await self._download_official(
                        client, source_file_path, filename
                    )
                else:
                    task_id, pages = await self._download_local(
                        client, source_file_path
                    )
                await self._download_referenced_images(client, pages, raw_dir)
        except httpx.RequestError as exc:
            raise RuntimeError(
                "PaddleOCR-VL backend request failed "
                f"(endpoint={self.endpoint}): {type(exc).__name__}: {exc}"
            ) from exc

        (raw_dir / CONTENT_LIST_FILENAME).write_text(
            json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self._write_manifest(raw_dir, source_file_path, task_id, filename)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"bearer {self.api_token}"}

    async def _download_official(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
        filename: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        job_id = await self._submit_official(client, source_file_path, filename)
        json_url = await self._poll_official_until_done(client, job_id)
        pages = await self._download_json_result(client, json_url)
        return job_id, pages

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
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"PaddleOCR-VL result which starts with '[' was unparseable: {exc.msg}"
                ) from exc
            if isinstance(payload, list):
                return [p for p in payload if isinstance(p, dict)]
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"PaddleOCR-VL json result had an unparseable line: {exc.msg}"
                ) from exc
            result = payload.get("result") if isinstance(payload, dict) else None
            page_items = (
                result.get("layoutParsingResults") if isinstance(result, dict) else None
            )
            if isinstance(page_items, list):
                pages.extend(p for p in page_items if isinstance(p, dict))
        return pages

    async def _download_local(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
    ) -> tuple[str, list[dict[str, Any]]]:
        resp = await client.post(
            f"{self.local_endpoint}/layout-parsing",
            json=self._local_request_payload(source_file_path),
        )
        raise_for_status_with_detail(resp, "PaddleOCR-VL local layout parsing")
        payload = resp.json() if resp.text else {}
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"PaddleOCR-VL local layout parsing returned non-object payload: "
                f"{payload!r}"
            )
        error_code = payload.get("errorCode", 0)
        if error_code not in (0, "0", None):
            raise RuntimeError(
                "PaddleOCR-VL local layout parsing failed: "
                f"errorCode={error_code} errorMsg={payload.get('errorMsg')!r}"
            )
        result = payload.get("result")
        pages = result.get("layoutParsingResults") if isinstance(result, dict) else None
        if not isinstance(pages, list):
            raise RuntimeError(
                "PaddleOCR-VL local layout parsing response missing "
                f"result.layoutParsingResults: {payload}"
            )
        task_id = str(payload.get("logId") or "local-layout-parsing")
        return task_id, [p for p in pages if isinstance(p, dict)]

    def _local_request_payload(self, source_file_path: Path) -> dict[str, Any]:
        # The local /layout-parsing API takes a JSON body, so the whole file is
        # base64-encoded inline; this reads the entire source into memory (local
        # mode is intended for single-page docs/images, not large multi-page PDFs).
        payload = dict(self._parser_options.optional_payload.request_payload())
        payload["file"] = base64.b64encode(source_file_path.read_bytes()).decode(
            "ascii"
        )
        payload["fileType"] = _local_file_type(source_file_path)
        return payload

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
                # markdown.images are referenced from the rendered document body
                # (their paths appear in the parsing result and in the IR), so a
                # missing/undecodable one breaks downstream rendering — materialize
                # them as mandatory (errors propagate).
                for rel_path, image_value in images.items():
                    await self._materialize_one_image(
                        client,
                        str(image_value),
                        raw_dir / _safe_relative_path(str(rel_path)),
                        mandatory=True,
                    )

            output_images = page.get("outputImages") if isinstance(page, dict) else None
            if isinstance(output_images, dict):
                # outputImages are diagnostic layout renderings; a missing one does
                # not affect the parsed document, so failures are soft-skipped.
                for name, image_value in output_images.items():
                    value = str(image_value)
                    suffix = _suffix_from_url(value) or ".jpg"
                    rel = (
                        Path("outputImages")
                        / f"{_safe_name(str(name))}_{page_index}{suffix}"
                    )
                    await self._materialize_one_image(
                        client, value, raw_dir / rel, mandatory=False
                    )

    async def _materialize_one_image(
        self,
        client: "httpx.AsyncClient",
        value: str,
        target: Path,
        *,
        mandatory: bool = False,
    ) -> None:
        if _looks_like_http_url(value):
            if self._is_allowed_asset_url(value):
                await self._download_one_image(client, value, target)
                return
            if mandatory:
                raise RuntimeError(
                    f"PaddleOCR-VL markdown image URL is not an allowed asset host "
                    f"(suffixes={list(self.allowed_asset_host_suffixes)}): {value!r}"
                )
            logger.warning("[paddleocr_vl] skipping non-allowed asset URL: %r", value)
            return
        image_bytes = _decode_base64_payload(value)
        if image_bytes is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(image_bytes)
            return
        if mandatory:
            raise RuntimeError(
                f"PaddleOCR-VL markdown image could not be decoded "
                f"(neither HTTP nor valid Base64): {value[:80]!r}"
            )
        logger.warning(
            "[paddleocr_vl] skipping undecodable image payload: %r", value[:80]
        )

    async def _download_one_image(
        self, client: "httpx.AsyncClient", url: str, target: Path
    ) -> None:
        if not url:
            return
        resp = await client.get(url)
        raise_for_status_with_detail(resp, f"PaddleOCR-VL image download {url!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resp.content)

    def _is_allowed_asset_url(self, url: str) -> bool:
        """True iff ``url`` is an HTTPS URL on an allowed asset host.

        Host suffixes come from ``self.allowed_asset_host_suffixes`` (default
        ``*.bcebos.com``; overridable via ``PADDLEOCR_VL_ALLOWED_ASSET_HOSTS``).
        Other remote hosts are never fetched (SSRF guard).
        """
        parsed = urlparse(url)
        if parsed.scheme != "https":
            return False
        host = (parsed.hostname or "").rstrip(".").lower()
        return any(host.endswith(s) for s in self.allowed_asset_host_suffixes)

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


def _local_file_type(path: Path) -> int:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "pdf":
        return 0
    # Registry suffix gating ensures every non-PDF reaching this client is an image.
    return 1


def _looks_like_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _decode_base64_payload(value: str) -> bytes | None:
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return None


__all__ = ["PaddleOCRVLRawClient"]
