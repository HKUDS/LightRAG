"""MinerU raw bundle downloader.

Wraps the existing protocol semantics (``MINERU_*`` env vars) and lands the
bundle on disk under ``raw_dir/``. Decides between three handling modes for
``result_url``:

- ``zip`` — response is binary / ``Content-Type: application/zip`` or URL ends
  in ``.zip``. Streamed to ``_bundle.zip`` then extracted; zip deleted.
- ``flat_json`` — response is a JSON content_list. Saved as
  ``content_list.json``; ``img_path`` references are de-duplicated and
  fetched relative to ``result_url``'s base (or
  ``MINERU_IMAGE_URL_TEMPLATE``).
- ``single_json`` — same as flat_json but no image references.

Mode is auto-detected unless ``MINERU_RESULT_MODE`` overrides.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from lightrag.mineru_raw.cache import (
    compute_size_and_hash,
)
from lightrag.mineru_raw.manifest import (
    Manifest,
    ManifestFile,
    write_manifest,
)
from lightrag.utils import logger

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

CONTENT_LIST_FILENAME = "content_list.json"


def _get_by_path(payload: Any, path: str) -> Any:
    """Walk a dotted path through a nested dict; returns None if any segment
    is missing or non-dict."""
    if not path:
        return None
    cur = payload
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _looks_like_zip_url(url: str) -> bool:
    return url.lower().split("?", 1)[0].endswith(".zip")


def _content_type_is_zip(headers: dict[str, str] | Any) -> bool:
    if not headers:
        return False
    ct = ""
    try:
        ct = str(headers.get("content-type") or headers.get("Content-Type") or "")
    except Exception:
        ct = ""
    ct = ct.lower()
    return "zip" in ct or "octet-stream" in ct


def _detect_mode_override() -> str | None:
    raw = os.getenv("MINERU_RESULT_MODE", "auto").strip().lower()
    if raw in {"zip", "flat_json", "single_json"}:
        return raw
    return None  # "auto" or unset


class MinerURawClient:
    """Downloads MinerU bundles into ``raw_dir``.

    Construct once per call (cheap). Reads ``MINERU_*`` env vars at
    construction time. Methods are async and use a single shared httpx
    client across all calls in :meth:`download_into`.

    The class duplicates a small amount of upload/poll choreography from
    :meth:`_PipelineMixin._call_protocol_parse_service` (~70 lines). This
    is deliberate: bundle handling needs the ``result_url`` *and* the
    ``Content-Type`` of the response, which the original helper does not
    expose. The legacy single-text helper remains in place for Docling.
    """

    def __init__(self) -> None:
        self.endpoint = os.getenv("MINERU_ENDPOINT", "").strip()
        if not self.endpoint:
            raise ValueError("MINERU_ENDPOINT is required for MinerU parsing")
        self.poll_url_template = os.getenv(
            "MINERU_POLL_ENDPOINT",
            self.endpoint + "/{trace_id}",
        )
        self.poll_method = os.getenv("MINERU_POLL_METHOD", "GET").upper()
        self.id_field = os.getenv("MINERU_ID_FIELD", "trace_id")
        self.status_field = os.getenv("MINERU_STATUS_FIELD", "status")
        self.result_url_field = os.getenv("MINERU_RESULT_URL_FIELD", "result_url")
        self.content_field = os.getenv("MINERU_CONTENT_FIELD", "content")
        self.file_field = os.getenv("MINERU_FILE_FIELD", "file")
        self.success_values = {
            x.strip().lower()
            for x in os.getenv(
                "MINERU_SUCCESS_VALUES",
                "done,success,succeeded,completed,finished",
            ).split(",")
            if x.strip()
        }
        self.failed_values = {
            x.strip().lower()
            for x in os.getenv("MINERU_FAILED_VALUES", "failed,error").split(",")
            if x.strip()
        }
        self.poll_interval = float(os.getenv("MINERU_POLL_INTERVAL_SECONDS", "2"))
        self.max_polls = int(os.getenv("MINERU_MAX_POLLS", "180"))
        self.engine_version = os.getenv("MINERU_ENGINE_VERSION", "").strip()
        self.image_url_template = os.getenv("MINERU_IMAGE_URL_TEMPLATE", "").strip()
        self.result_mode = _detect_mode_override()  # None ⇒ auto

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def download_into(
        self,
        raw_dir: Path,
        source_file_path: Path,
    ) -> Manifest:
        """Download a fresh bundle and write the manifest.

        Pre-condition: caller cleared ``raw_dir`` contents (recommended via
        :func:`clear_dir_contents`). This method does NOT clean the
        directory itself — leaving that to the caller keeps cache miss
        semantics explicit at the parse_mineru entry point.

        Returns the :class:`Manifest` describing the bundle.
        """
        if httpx is None:
            raise RuntimeError(
                "httpx is required for MinerU parsing but not installed"
            )
        raw_dir.mkdir(parents=True, exist_ok=True)

        timeout = httpx.Timeout(120.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            task_id, poll_payload, embedded_content = await self._upload_and_poll(
                client, source_file_path
            )

            result_url = _get_by_path(poll_payload, self.result_url_field)
            result_url_str = str(result_url) if result_url else ""

            if embedded_content is not None and not result_url_str:
                # Pre-modern MinerU deployment that returns content_list
                # directly in the poll payload. No images possible.
                self._write_content_list(raw_dir, embedded_content)
            elif result_url_str:
                await self._download_result(client, result_url_str, raw_dir)
            else:
                raise RuntimeError(
                    f"MinerU returned neither result_url nor embedded content "
                    f"for task {task_id}"
                )

        return self._build_and_write_manifest(raw_dir, source_file_path, task_id)

    # ------------------------------------------------------------------
    # Upload + poll (mirrors _call_protocol_parse_service)
    # ------------------------------------------------------------------

    async def _upload_and_poll(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
    ) -> tuple[str, dict, Any]:
        """Returns ``(task_id, last_poll_payload, embedded_content_or_None)``.

        ``embedded_content`` is populated only when the *upload response*
        directly carries the parsed content (some MinerU dev deployments do
        this for small inputs).
        """
        with source_file_path.open("rb") as f:
            resp = await client.post(
                self.endpoint,
                files={self.file_field: (source_file_path.name, f)},
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"MinerU upload failed: {resp.status_code} {resp.text[:400]}"
            )
        upload_payload = resp.json() if resp.text else {}

        task_id_raw = _get_by_path(upload_payload, self.id_field)
        if not task_id_raw:
            # Embedded mode on upload; treat as terminal "done" with
            # embedded content. Manifest will record an empty task_id.
            embedded = _get_by_path(upload_payload, self.content_field)
            if embedded is None:
                raise RuntimeError(
                    "MinerU upload payload had neither id field "
                    f"({self.id_field!r}) nor content field "
                    f"({self.content_field!r})"
                )
            return "", upload_payload, embedded
        task_id = str(task_id_raw)

        poll_url = self.poll_url_template.format(
            task_id=task_id, trace_id=task_id, id=task_id
        )
        poll_params = {"task_id": task_id, "trace_id": task_id, "id": task_id}

        for _ in range(self.max_polls):
            await asyncio.sleep(self.poll_interval)
            if self.poll_method == "POST":
                poll_resp = await client.post(poll_url, json=poll_params)
            else:
                poll_resp = await client.get(poll_url, params=poll_params)
            poll_payload = poll_resp.json() if poll_resp.text else {}
            status_raw = _get_by_path(poll_payload, self.status_field)
            status_val = str(status_raw).lower() if status_raw is not None else ""
            if status_val in self.success_values:
                # Some MinerU deployments embed the content_list in the
                # poll payload itself rather than at result_url. Forward
                # both to the caller; download_into prefers result_url
                # when present.
                embedded = _get_by_path(poll_payload, self.content_field)
                return task_id, poll_payload, embedded
            if status_val in self.failed_values:
                raise RuntimeError(
                    f"MinerU parse failed for task {task_id}: {poll_payload}"
                )

        raise TimeoutError(f"MinerU parse polling timeout for task: {task_id}")

    # ------------------------------------------------------------------
    # Result download
    # ------------------------------------------------------------------

    async def _download_result(
        self,
        client: "httpx.AsyncClient",
        result_url: str,
        raw_dir: Path,
    ) -> None:
        """Fetch ``result_url`` and lay out its contents inside ``raw_dir``."""
        mode = self.result_mode or self._auto_detect_mode_pre(result_url)

        if mode == "zip":
            await self._download_zip(client, result_url, raw_dir)
            return

        # JSON path (flat_json / single_json indistinguishable until parsed)
        resp = await client.get(result_url)
        resp.raise_for_status()
        text = resp.text

        if self.result_mode is None and _content_type_is_zip(resp.headers):
            # Detected as zip after fetch; re-download as bytes.
            await self._download_zip(client, result_url, raw_dir, resp=resp)
            return

        # Try to extract content_list from JSON (envelope-aware via
        # MINERU_CONTENT_FIELD).
        content_list = self._extract_content_list(text)
        if content_list is None:
            # As a last resort, save the raw text as content_list.json.
            # Adapter will fail gracefully if it isn't a content_list.
            (raw_dir / CONTENT_LIST_FILENAME).write_text(text, encoding="utf-8")
            return

        self._write_content_list(raw_dir, content_list)
        await self._fetch_image_assets(
            client, content_list, base_url=result_url, raw_dir=raw_dir
        )

    def _auto_detect_mode_pre(self, result_url: str) -> str:
        """Pre-fetch mode hint from URL alone."""
        if _looks_like_zip_url(result_url):
            return "zip"
        return "flat_json"  # final decision deferred to response headers

    async def _download_zip(
        self,
        client: "httpx.AsyncClient",
        result_url: str,
        raw_dir: Path,
        resp: Any = None,
    ) -> None:
        """Download (or re-use already-fetched response) and extract."""
        if resp is None or not hasattr(resp, "content"):
            resp = await client.get(result_url)
            resp.raise_for_status()
        buf = io.BytesIO(resp.content)
        with zipfile.ZipFile(buf) as zf:
            # Safe-extract: refuse absolute paths and ``..`` traversal.
            for name in zf.namelist():
                norm = os.path.normpath(name)
                if norm.startswith("..") or os.path.isabs(norm):
                    raise RuntimeError(
                        f"Refusing zip entry with unsafe path: {name!r}"
                    )
            zf.extractall(raw_dir)

        # Normalize: if the zip nested everything under a single top-level
        # dir, hoist its contents up so content_list.json sits at raw_dir
        # root. This matches the common MinerU bundle layout.
        self._maybe_hoist_single_subdir(raw_dir)

    def _maybe_hoist_single_subdir(self, raw_dir: Path) -> None:
        entries = [p for p in raw_dir.iterdir() if p.name != "_manifest.json"]
        if len(entries) != 1 or not entries[0].is_dir():
            return
        sub = entries[0]
        for child in list(sub.iterdir()):
            child.rename(raw_dir / child.name)
        try:
            sub.rmdir()
        except OSError:
            pass

    def _extract_content_list(self, text: str) -> list[dict] | None:
        """Decode ``text`` and dig out the content_list array."""
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        return _find_content_list(payload, self.content_field)

    def _write_content_list(
        self,
        raw_dir: Path,
        content_list: Any,
    ) -> None:
        (raw_dir / CONTENT_LIST_FILENAME).write_text(
            json.dumps(content_list, ensure_ascii=False),
            encoding="utf-8",
        )

    async def _fetch_image_assets(
        self,
        client: "httpx.AsyncClient",
        content_list: list[dict],
        *,
        base_url: str,
        raw_dir: Path,
    ) -> None:
        """Download every ``img_path`` referenced by image / table items.

        Resolution rules:

        - Absolute http(s) URL → fetched as-is.
        - Relative path → resolved against ``base_url`` (the result_url) or
          ``MINERU_IMAGE_URL_TEMPLATE`` if set.

        Missing images are logged but do not abort the parse — the
        sidecar's drawing item will still exist but its asset will be
        flagged via the adapter.
        """
        img_refs: set[str] = set()
        for item in content_list:
            if not isinstance(item, dict):
                continue
            for key in ("img_path", "image", "image_path"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    img_refs.add(val.strip())

        for ref in sorted(img_refs):
            try:
                await self._fetch_one_image(client, ref, base_url, raw_dir)
            except Exception as e:  # pragma: no cover - tolerate per-image
                logger.warning("[mineru_raw] image fetch failed for %s: %s", ref, e)

    async def _fetch_one_image(
        self,
        client: "httpx.AsyncClient",
        ref: str,
        base_url: str,
        raw_dir: Path,
    ) -> None:
        if ref.startswith(("http://", "https://")):
            url = ref
            target_rel = self._image_dest_rel(ref)
        elif self.image_url_template:
            url = self.image_url_template.format(name=ref, path=ref)
            target_rel = self._image_dest_rel(ref)
        else:
            url = urljoin(base_url, ref)
            target_rel = ref
        # Hash-safe: refuse path traversal.
        norm = os.path.normpath(target_rel)
        if norm.startswith("..") or os.path.isabs(norm):
            logger.warning("[mineru_raw] refusing unsafe image path: %s", ref)
            return

        target = raw_dir / norm
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)

        resp = await client.get(url)
        resp.raise_for_status()
        target.write_bytes(resp.content)

    def _image_dest_rel(self, ref: str) -> str:
        """Map an absolute / templated URL back to a deterministic relative
        path inside ``raw_dir``. Default: ``images/<basename>``."""
        parsed = urlparse(ref)
        name = Path(parsed.path).name or "image"
        return f"images/{name}"

    # ------------------------------------------------------------------
    # Manifest construction
    # ------------------------------------------------------------------

    def _build_and_write_manifest(
        self,
        raw_dir: Path,
        source_file_path: Path,
        task_id: str,
    ) -> Manifest:
        source_size, source_hash = compute_size_and_hash(source_file_path)

        # Critical file — required.
        crit_path = raw_dir / CONTENT_LIST_FILENAME
        if not crit_path.is_file():
            raise RuntimeError(
                f"MinerU bundle missing required {CONTENT_LIST_FILENAME} "
                f"after download (raw_dir={raw_dir})"
            )
        crit_size, crit_hash = compute_size_and_hash(crit_path)

        # Other files.
        others: list[ManifestFile] = []
        total = crit_size
        for p in sorted(raw_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.name == "_manifest.json":
                continue
            rel = p.relative_to(raw_dir).as_posix()
            if rel == CONTENT_LIST_FILENAME:
                continue
            size = p.stat().st_size
            others.append(ManifestFile(path=rel, size=size))
            total += size

        manifest = Manifest(
            source_content_hash=source_hash,
            source_size_bytes=source_size,
            source_filename_at_parse=source_file_path.name,
            critical_file=ManifestFile(
                path=CONTENT_LIST_FILENAME,
                size=crit_size,
                sha256=crit_hash,
            ),
            files=others,
            total_size_bytes=total,
            task_id=task_id,
            engine_version=self.engine_version,
            endpoint_signature=self.endpoint,
            downloaded_at=datetime.now(timezone.utc).isoformat(),
        )
        write_manifest(raw_dir, manifest)
        return manifest


def _find_content_list(payload: Any, content_field: str) -> list[dict] | None:
    """Heuristic content_list extractor.

    Tries (in order):

    1. The configured ``MINERU_CONTENT_FIELD`` dotted path if it lands on a
       list of dicts.
    2. Direct ``content_list`` / ``content`` / ``items`` / ``result`` keys.
    3. Recursive descent.
    """
    if isinstance(payload, list):
        if payload and all(isinstance(x, dict) for x in payload):
            return payload
        return None
    if not isinstance(payload, dict):
        return None

    via_field = _get_by_path(payload, content_field)
    candidate = _find_content_list(via_field, content_field)
    if candidate is not None:
        return candidate

    for key in ("content_list", "content", "items", "result"):
        value = payload.get(key)
        candidate = _find_content_list(value, content_field)
        if candidate is not None:
            return candidate

    for value in payload.values():
        candidate = _find_content_list(value, content_field)
        if candidate is not None:
            return candidate
    return None


__all__ = ["MinerURawClient", "CONTENT_LIST_FILENAME"]
