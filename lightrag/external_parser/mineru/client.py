"""MinerU raw bundle downloader.

Supports MinerU's official cloud and self-hosted API protocols and lands the
final parser bundle on disk under ``raw_dir/``:

- ``official`` — MinerU precision API v4: apply for signed upload URL, PUT the
  local file, poll batch results, download ``full_zip_url``.
- ``local`` — self-hosted ``mineru-api`` / ``mineru-router``: submit
  ``POST /tasks``, poll ``GET /tasks/{task_id}``, download
  ``GET /tasks/{task_id}/result``.

Both protocols request a zip result bundle. Archives are extracted under
``raw_dir/`` and normalized so the adapter can read a root-level
``content_list.json``.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from lightrag.external_parser.mineru.cache import (
    compute_size_and_hash,
)
from lightrag.external_parser.mineru.manifest import (
    Manifest,
    ManifestFile,
    write_manifest,
)
from lightrag.utils import logger

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None

CONTENT_LIST_FILENAME = "content_list.json"
DEFAULT_MINERU_API_MODE = "local"
DEFAULT_MINERU_OFFICIAL_ENDPOINT = "https://mineru.net"
VALID_MINERU_API_MODES = {"official", "local"}
OFFICIAL_DONE_STATES = {"done"}
OFFICIAL_FAILED_STATES = {"failed"}
LOCAL_DONE_STATES = {"completed"}
LOCAL_FAILED_STATES = {"failed"}


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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "[mineru_raw] %s=%r is not an integer; using %s", name, raw, default
        )
        return default


def _strip_trailing_slash(url: str) -> str:
    return url.rstrip("/")


def _validate_base_url(
    name: str, endpoint: str, forbidden_segments: tuple[str, ...]
) -> None:
    parsed = urlparse(endpoint)
    path = (parsed.path or "").rstrip("/")
    for segment in forbidden_segments:
        if path.endswith(segment) or f"{segment}/" in path:
            raise ValueError(
                f"{name} must be a base URL, not an API path: {endpoint!r}"
            )


class MinerURawClient:
    """Downloads MinerU bundles into ``raw_dir``.

    Construct once per call (cheap). Reads ``MINERU_*`` env vars at
    construction time. Methods are async and use a single shared httpx
    client across all calls in :meth:`download_into`.

    Implements the MinerU-specific upload + poll + zip download flow
    inline; bundle handling needs the ``result_url`` *and* the
    ``Content-Type`` of the response, which a generic protocol helper
    cannot expose without leaking abstractions.
    """

    def __init__(self) -> None:
        self.api_mode = (
            os.getenv("MINERU_API_MODE", DEFAULT_MINERU_API_MODE).strip().lower()
        )
        if self.api_mode not in VALID_MINERU_API_MODES:
            allowed = ", ".join(sorted(VALID_MINERU_API_MODES))
            raise ValueError(
                f"MINERU_API_MODE must be one of {allowed}, got {self.api_mode!r}"
            )

        self.official_endpoint = _strip_trailing_slash(
            os.getenv(
                "MINERU_OFFICIAL_ENDPOINT", DEFAULT_MINERU_OFFICIAL_ENDPOINT
            ).strip()
            or DEFAULT_MINERU_OFFICIAL_ENDPOINT
        )
        self.local_endpoint = _strip_trailing_slash(
            os.getenv("MINERU_LOCAL_ENDPOINT", "").strip()
        )
        self.api_token = os.getenv("MINERU_API_TOKEN", "").strip()
        if self.api_mode == "official":
            if not self.api_token:
                raise ValueError(
                    "MINERU_API_TOKEN is required when MINERU_API_MODE=official"
                )
            _validate_base_url(
                "MINERU_OFFICIAL_ENDPOINT",
                self.official_endpoint,
                ("/api/v4", "/api/v4/file-urls/batch", "/api/v4/extract/task"),
            )
            self.endpoint = self.official_endpoint
        elif self.api_mode == "local":
            if not self.local_endpoint:
                raise ValueError(
                    "MINERU_LOCAL_ENDPOINT is required when MINERU_API_MODE=local"
                )
            _validate_base_url(
                "MINERU_LOCAL_ENDPOINT",
                self.local_endpoint,
                ("/tasks", "/file_parse", "/health"),
            )
            self.endpoint = self.local_endpoint
        self.poll_interval = float(os.getenv("MINERU_POLL_INTERVAL_SECONDS", "2"))
        self.max_polls = int(os.getenv("MINERU_MAX_POLLS", "180"))
        self.engine_version = os.getenv("MINERU_ENGINE_VERSION", "").strip()

        self.model_version = os.getenv("MINERU_MODEL_VERSION", "vlm").strip() or "vlm"
        self.language = os.getenv("MINERU_LANGUAGE", "ch").strip() or "ch"
        self.enable_table = _env_bool("MINERU_ENABLE_TABLE", True)
        self.enable_formula = _env_bool("MINERU_ENABLE_FORMULA", True)
        self.is_ocr = _env_bool("MINERU_IS_OCR", False)
        self.page_ranges = os.getenv("MINERU_PAGE_RANGES", "").strip()
        self.local_backend = (
            os.getenv("MINERU_LOCAL_BACKEND", "pipeline").strip() or "pipeline"
        )
        self.local_parse_method = (
            os.getenv("MINERU_LOCAL_PARSE_METHOD", "auto").strip() or "auto"
        )
        self.local_image_analysis = _env_bool("MINERU_LOCAL_IMAGE_ANALYSIS", True)
        self.local_start_page_id = _env_int("MINERU_LOCAL_START_PAGE_ID", 0)
        self.local_end_page_id = _env_int("MINERU_LOCAL_END_PAGE_ID", 99999)
        if self.api_mode == "local" and self.page_ranges:
            self.local_start_page_id, self.local_end_page_id = _local_page_bounds(
                self.page_ranges
            )

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
            raise RuntimeError("httpx is required for MinerU parsing but not installed")
        raw_dir.mkdir(parents=True, exist_ok=True)

        timeout = httpx.Timeout(120.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            if self.api_mode == "official":
                task_id = await self._download_official(
                    client, source_file_path, raw_dir
                )
            else:
                task_id = await self._download_local(client, source_file_path, raw_dir)

        self._normalize_raw_bundle(raw_dir, source_file_path)
        return self._build_and_write_manifest(raw_dir, source_file_path, task_id)

    # ------------------------------------------------------------------
    # Upload + poll
    # ------------------------------------------------------------------

    def _official_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }

    def _official_payload(self, source_file_path: Path) -> dict[str, Any]:
        file_entry: dict[str, Any] = {"name": source_file_path.name}
        if self.is_ocr:
            file_entry["is_ocr"] = True
        if self.page_ranges:
            file_entry["page_ranges"] = self.page_ranges
        return {
            "files": [file_entry],
            "model_version": self.model_version,
            "language": self.language,
            "enable_table": self.enable_table,
            "enable_formula": self.enable_formula,
        }

    async def _download_official(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
        raw_dir: Path,
    ) -> str:
        apply_url = f"{self.official_endpoint}/api/v4/file-urls/batch"
        resp = await client.post(
            apply_url,
            headers=self._official_headers(),
            json=self._official_payload(source_file_path),
        )
        resp.raise_for_status()
        payload = resp.json() if resp.text else {}
        self._raise_if_official_error(payload, "MinerU official upload URL request")
        data = payload.get("data") if isinstance(payload, dict) else {}
        batch_id = str((data or {}).get("batch_id") or "")
        file_urls = (data or {}).get("file_urls") or []
        if not batch_id or not isinstance(file_urls, list) or not file_urls:
            raise RuntimeError(
                f"MinerU official upload URL response missing batch_id/file_urls: "
                f"{payload}"
            )

        first_file_url = file_urls[0]
        if isinstance(first_file_url, dict):
            upload_url = str(
                first_file_url.get("url") or first_file_url.get("file_url") or ""
            )
        else:
            upload_url = str(first_file_url)
        if not upload_url:
            raise RuntimeError(
                f"MinerU official upload URL response had an empty upload URL: "
                f"{payload}"
            )
        # Use ``content=bytes`` rather than ``data=file_object``: httpx 0.28+
        # wraps a sync file-like into an ``IteratorByteStream`` (a SyncByteStream),
        # which ``AsyncClient._send_single_request`` rejects with
        # "Attempted to send an sync request with an AsyncClient instance."
        file_bytes = await asyncio.to_thread(source_file_path.read_bytes)
        upload_resp = await client.put(upload_url, content=file_bytes)
        upload_resp.raise_for_status()

        result_url = await self._poll_official_batch(client, batch_id, source_file_path)
        await self._download_zip(client, result_url, raw_dir)
        return batch_id

    async def _poll_official_batch(
        self,
        client: "httpx.AsyncClient",
        batch_id: str,
        source_file_path: Path,
    ) -> str:
        poll_url = f"{self.official_endpoint}/api/v4/extract-results/batch/{batch_id}"
        for _ in range(self.max_polls):
            await asyncio.sleep(self.poll_interval)
            resp = await client.get(poll_url, headers=self._official_headers())
            resp.raise_for_status()
            payload = resp.json() if resp.text else {}
            self._raise_if_official_error(payload, "MinerU official batch poll")
            results = _get_by_path(payload, "data.extract_result")
            if isinstance(results, dict):
                results = [results]
            if not isinstance(results, list):
                continue

            selected = _select_official_extract_result(results, source_file_path.name)
            if selected is None:
                continue
            state = str(selected.get("state") or "").lower()
            if state in OFFICIAL_DONE_STATES:
                full_zip_url = str(selected.get("full_zip_url") or "")
                if not full_zip_url:
                    raise RuntimeError(
                        f"MinerU official batch {batch_id} is done but has no "
                        f"full_zip_url: {selected}"
                    )
                return full_zip_url
            if state in OFFICIAL_FAILED_STATES:
                err = selected.get("err_msg") or selected.get("error") or selected
                raise RuntimeError(
                    f"MinerU official parse failed for batch {batch_id}: {err}"
                )

        raise TimeoutError(f"MinerU official batch polling timeout: {batch_id}")

    def _raise_if_official_error(self, payload: Any, operation: str) -> None:
        if not isinstance(payload, dict):
            raise RuntimeError(f"{operation} returned non-object payload: {payload!r}")
        code = payload.get("code", 0)
        if code not in (0, "0", None):
            raise RuntimeError(
                f"{operation} failed: code={code} msg={payload.get('msg')!r}"
            )

    def _local_form_data(self) -> dict[str, str]:
        return {
            "lang_list": self.language,
            "backend": self.local_backend,
            "parse_method": self.local_parse_method,
            "formula_enable": _bool_form(self.enable_formula),
            "table_enable": _bool_form(self.enable_table),
            "image_analysis": _bool_form(self.local_image_analysis),
            "return_md": "true",
            "return_middle_json": "true",
            "return_model_output": "true",
            "return_content_list": "true",
            "return_images": "true",
            "response_format_zip": "true",
            "return_original_file": "true",
            "start_page_id": str(self.local_start_page_id),
            "end_page_id": str(self.local_end_page_id),
        }

    async def _download_local(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
        raw_dir: Path,
    ) -> str:
        submit_url = f"{self.local_endpoint}/tasks"
        with source_file_path.open("rb") as f:
            resp = await client.post(
                submit_url,
                files={"files": (source_file_path.name, f)},
                data=self._local_form_data(),
            )
        resp.raise_for_status()
        payload = resp.json() if resp.text else {}
        task_id = str(payload.get("task_id") or "")
        if not task_id:
            raise RuntimeError(
                f"MinerU local /tasks response missing task_id: {payload}"
            )

        await self._poll_local_task(client, task_id)
        await self._download_zip(
            client,
            f"{self.local_endpoint}/tasks/{task_id}/result",
            raw_dir,
        )
        return task_id

    async def _poll_local_task(
        self,
        client: "httpx.AsyncClient",
        task_id: str,
    ) -> None:
        poll_url = f"{self.local_endpoint}/tasks/{task_id}"
        for _ in range(self.max_polls):
            await asyncio.sleep(self.poll_interval)
            resp = await client.get(poll_url)
            resp.raise_for_status()
            payload = resp.json() if resp.text else {}
            status = str(payload.get("status") or "").lower()
            if status in LOCAL_DONE_STATES:
                return
            if status in LOCAL_FAILED_STATES:
                err = payload.get("error") or payload.get("message") or payload
                raise RuntimeError(
                    f"MinerU local parse failed for task {task_id}: {err}"
                )

        raise TimeoutError(f"MinerU local task polling timeout: {task_id}")

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
                    raise RuntimeError(f"Refusing zip entry with unsafe path: {name!r}")
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

    def _normalize_raw_bundle(self, raw_dir: Path, source_file_path: Path) -> None:
        """Ensure a downloaded bundle has root-level ``content_list.json``.

        Official and local MinerU zip archives commonly place parser outputs at
        ``<doc>/<parse_method>/<doc>_content_list.json``. The adapter consumes a
        canonical root ``content_list.json`` plus optional root ``images/``.

        After hoisting we delete the nested originals so the manifest does not
        bookkeep two copies (and disk usage doesn't double for big bundles).
        Sibling artifacts of the parse subdir (``*.md``, ``middle.json`` etc.)
        are also hoisted to ``raw_dir`` root for easier diagnostics.
        """
        if (raw_dir / CONTENT_LIST_FILENAME).is_file():
            return

        candidate = _select_content_list_candidate(raw_dir, source_file_path)
        if candidate is None:
            return

        source_dir = candidate.parent
        target_root = raw_dir.resolve()
        # Guard: never hoist from above raw_dir (defensive — candidate already
        # comes from rglob inside raw_dir, but cheap to verify).
        try:
            source_dir.resolve().relative_to(target_root)
        except ValueError:
            shutil.copy2(candidate, raw_dir / CONTENT_LIST_FILENAME)
            return

        # Move the critical file first; then hoist sibling files/dirs that
        # don't already exist at raw_dir root.
        shutil.move(str(candidate), str(raw_dir / CONTENT_LIST_FILENAME))
        for entry in list(source_dir.iterdir()):
            target = raw_dir / entry.name
            if target.exists():
                continue
            shutil.move(str(entry), str(target))

        # Best-effort cleanup of the now-empty parse subtree.
        cursor = source_dir
        while cursor != raw_dir and cursor.is_dir():
            try:
                cursor.rmdir()
            except OSError:
                break
            cursor = cursor.parent

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
            api_mode=self.api_mode,
            engine_version=self.engine_version,
            endpoint_signature=self.endpoint,
            downloaded_at=datetime.now(timezone.utc).isoformat(),
        )
        write_manifest(raw_dir, manifest)
        return manifest


def _find_content_list(payload: Any, content_field: str) -> list[dict] | None:
    """Heuristic content_list extractor.

    Tries (in order):

    1. The provided dotted path if it lands on a list of dicts.
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


def _bool_form(value: bool) -> str:
    return "true" if value else "false"


def _local_page_bounds(page_ranges: str) -> tuple[int, int]:
    raw = page_ranges.strip()
    if not raw:
        return 0, 99999
    if "," in raw:
        raise ValueError(
            "MINERU_PAGE_RANGES with MINERU_API_MODE=local supports only a "
            "single page or simple range such as '1-10'"
        )
    if raw.isdigit():
        page = max(int(raw), 1)
        return page - 1, page - 1
    if "-" in raw:
        left, _, right = raw.partition("-")
        if left.isdigit() and right.isdigit():
            start = max(int(left), 1)
            end = max(int(right), start)
            return start - 1, end - 1
    raise ValueError(
        "MINERU_PAGE_RANGES with MINERU_API_MODE=local must be a single "
        "positive page number or simple range such as '1-10'"
    )


def _select_official_extract_result(
    results: list[Any],
    source_filename: str,
) -> dict[str, Any] | None:
    """Pick the extract_result entry that matches the file we uploaded.

    Invariant: :meth:`MinerURawClient._download_official` always submits a
    single-file batch, so a non-matching ``file_name`` from the API would
    indicate either a server response we don't understand or a future
    multi-file extension. We fall back to ``dict_results[0]`` to remain
    forward-compatible but log a warning so the mismatch is visible.
    """
    dict_results = [item for item in results if isinstance(item, dict)]
    if not dict_results:
        return None
    source_name = Path(source_filename).name
    source_stem = Path(source_filename).stem
    for item in dict_results:
        file_name = str(item.get("file_name") or item.get("name") or "")
        if Path(file_name).name == source_name or Path(file_name).stem == source_stem:
            return item
    logger.warning(
        "[mineru_raw] official extract_result did not contain a match for "
        "%r; falling back to the first entry (%r). This is unexpected for "
        "a single-file batch.",
        source_name,
        str(dict_results[0].get("file_name") or dict_results[0].get("name") or ""),
    )
    return dict_results[0]


def _select_content_list_candidate(
    raw_dir: Path,
    source_file_path: Path,
) -> Path | None:
    source_stem = source_file_path.stem
    candidates: list[tuple[int, int, str, Path]] = []
    for path in raw_dir.rglob("*.json"):
        if not path.is_file():
            continue
        if path.name != CONTENT_LIST_FILENAME and not path.name.endswith(
            "_content_list.json"
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        content_list = _find_content_list(payload, "content")
        if content_list is None:
            continue

        score = 10
        if path.name == CONTENT_LIST_FILENAME:
            score = 0
        elif path.name == f"{source_stem}_content_list.json":
            score = 1
        elif path.stem.endswith("_content_list"):
            score = 2
        depth = len(path.relative_to(raw_dir).parts)
        candidates.append((score, depth, path.as_posix(), path))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


__all__ = ["MinerURawClient", "CONTENT_LIST_FILENAME"]
