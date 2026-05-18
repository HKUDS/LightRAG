"""Docling raw bundle downloader.

Talks to Docling Serve v1 over HTTP:

- ``POST /v1/convert/file/async`` — multipart upload, returns ``task_id``,
- ``GET  /v1/status/poll/{task_id}?wait=5`` — long-poll for terminal state,
- ``GET  /v1/result/{task_id}`` — zip download (only on ``success``).

The zip is extracted safely under ``raw_dir/`` (refusing path traversal /
absolute entries). A success manifest is written atomically at the very
end; mid-run crashes therefore leave the directory in a state the cache
layer marks as invalid (no manifest → miss → re-download).

Pipeline constants (``pipeline``, ``target_type``, ``to_formats``,
``image_export_mode``) are intentionally **not** env-driven — the sidecar
flow depends on them — and are recorded inside the manifest so a future
code change automatically invalidates pre-existing caches.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.external_parser._common import env_bool, env_int
from lightrag.external_parser._zip import safe_extract_zip
from lightrag.external_parser.docling.cache import (
    compute_options_signature,
    current_endpoint_signature,
    snapshot_tunable_env,
)
from lightrag.external_parser.docling.manifest import (
    build_and_write_docling_manifest,
    select_main_json,
)
from lightrag.utils import logger

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None

# ---------------------------------------------------------------------------
# Fixed pipeline constants (NOT env-driven)
# ---------------------------------------------------------------------------

PIPELINE = "standard"
TARGET_TYPE = "zip"
TO_FORMATS: tuple[str, ...] = ("json", "md")
IMAGE_EXPORT_MODE = "referenced"

FIXED_CONSTANTS: dict[str, object] = {
    "pipeline": PIPELINE,
    "target_type": TARGET_TYPE,
    "to_formats": list(TO_FORMATS),
    "image_export_mode": IMAGE_EXPORT_MODE,
}

CONVERT_PATH = "/v1/convert/file/async"
POLL_PATH = "/v1/status/poll/{task_id}"
RESULT_PATH = "/v1/result/{task_id}"

DEFAULT_POLL_WAIT_SECONDS = 5
DEFAULT_MAX_POLLS = 240  # 240 * 5s long-poll ≈ 20 min worst case

# ConversionStatus enum from the docling-serve OpenAPI
SUCCESS_STATES = {"success"}
FAILURE_STATES = {"failure", "partial_success", "skipped"}
IN_PROGRESS_STATES = {"pending", "started"}


class DoclingRawClient:
    """Downloads docling-serve bundles into ``raw_dir``.

    Construct once per parse call (cheap). Reads ``DOCLING_*`` envs at
    ``__init__`` time, so callers can flip env between calls and pick up
    the new values without holding a stale instance.
    """

    def __init__(self) -> None:
        self.endpoint = current_endpoint_signature()
        if not self.endpoint:
            raise ValueError("DOCLING_ENDPOINT is required")
        self.engine_version = os.getenv("DOCLING_ENGINE_VERSION", "").strip()

        self.do_ocr = env_bool("DOCLING_DO_OCR", True)
        self.force_ocr = env_bool("DOCLING_FORCE_OCR", False)
        self.ocr_engine = os.getenv("DOCLING_OCR_ENGINE", "auto").strip() or "auto"
        self.ocr_preset = os.getenv("DOCLING_OCR_PRESET", "auto").strip() or "auto"
        self.ocr_lang_raw = os.getenv("DOCLING_OCR_LANG", "").strip()
        self.do_formula_enrichment = env_bool("DOCLING_DO_FORMULA_ENRICHMENT", False)

        # Poll cadence: docling-serve's ``?wait=N`` is a server-side long-poll
        # window. ``DOCLING_POLL_INTERVAL_SECONDS`` sets that window; the
        # client does NOT add its own sleep between polls. ``DOCLING_MAX_POLLS``
        # bounds the total polling budget — exceeding it raises ``TimeoutError``.
        wait = env_int("DOCLING_POLL_INTERVAL_SECONDS", DEFAULT_POLL_WAIT_SECONDS)
        self.poll_wait_seconds = wait if wait > 0 else DEFAULT_POLL_WAIT_SECONDS
        max_polls = env_int("DOCLING_MAX_POLLS", DEFAULT_MAX_POLLS)
        self.max_poll_attempts = max_polls if max_polls > 0 else DEFAULT_MAX_POLLS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def download_into(
        self,
        raw_dir: Path,
        source_file_path: Path,
    ):
        """Upload, poll, download, extract, and write the manifest.

        Pre-condition: caller cleared ``raw_dir`` (e.g. via
        :func:`lightrag.external_parser.clear_dir_contents`). This method
        does not clean the directory itself — keeping that explicit at the
        ``parse_docling`` entry point.
        """
        if httpx is None:
            raise RuntimeError(
                "httpx is required for Docling parsing but is not installed"
            )
        raw_dir.mkdir(parents=True, exist_ok=True)

        timeout = httpx.Timeout(120.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            task_id = await self._submit(client, source_file_path)
            await self._poll_until_done(client, task_id)
            payload = await self._download_zip_bytes(client, task_id)

        safe_extract_zip(payload, raw_dir)
        # Defensive: confirm the main JSON exists before anyone reads the
        # bundle. ``select_main_json`` raises a clear error if not.
        select_main_json(raw_dir, source_file_path)

        options_signature = compute_options_signature(
            tunable_env=snapshot_tunable_env(),
            fixed_constants=FIXED_CONSTANTS,
        )
        return build_and_write_docling_manifest(
            raw_dir,
            source_file_path=source_file_path,
            task_id=task_id,
            endpoint_signature=self.endpoint,
            engine_version=self.engine_version,
            options_signature=options_signature,
            fixed_constants=FIXED_CONSTANTS,
        )

    # ------------------------------------------------------------------
    # Upload + poll + download
    # ------------------------------------------------------------------

    def _build_multipart_data(self) -> list[tuple[str, str]]:
        """Form fields (everything except the file payload).

        List-valued fields like ``to_formats`` are sent as repeated keys,
        matching docling-serve's pydantic List[Enum] form parsing. ``ocr_lang``
        is omitted entirely when empty so the engine uses its own default.
        """
        data: list[tuple[str, str]] = [
            ("pipeline", PIPELINE),
            ("target_type", TARGET_TYPE),
            ("image_export_mode", IMAGE_EXPORT_MODE),
            ("do_ocr", _bool_form(self.do_ocr)),
            ("force_ocr", _bool_form(self.force_ocr)),
            ("ocr_engine", self.ocr_engine),
            ("ocr_preset", self.ocr_preset),
            ("do_formula_enrichment", _bool_form(self.do_formula_enrichment)),
        ]
        for fmt in TO_FORMATS:
            data.append(("to_formats", fmt))
        if self.ocr_lang_raw:
            for lang in _parse_ocr_lang(self.ocr_lang_raw):
                data.append(("ocr_lang", lang))
        return data

    async def _submit(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
    ) -> str:
        url = f"{self.endpoint}{CONVERT_PATH}"
        file_bytes = await asyncio.to_thread(source_file_path.read_bytes)
        files = {
            "files": (source_file_path.name, file_bytes, "application/octet-stream")
        }
        resp = await client.post(url, data=self._build_multipart_data(), files=files)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Docling upload failed: {resp.status_code} {resp.text[:400]}"
            )
        payload = resp.json() if resp.text else {}
        task_id = str(payload.get("task_id") or payload.get("id") or "").strip()
        if not task_id:
            raise RuntimeError(f"Docling upload response missing task_id: {payload!r}")
        return task_id

    async def _poll_until_done(
        self,
        client: "httpx.AsyncClient",
        task_id: str,
    ) -> None:
        url = f"{self.endpoint}{POLL_PATH.format(task_id=task_id)}"
        params = {"wait": self.poll_wait_seconds}
        for _ in range(self.max_poll_attempts):
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json() if resp.text else {}
            status = str(
                payload.get("task_status") or payload.get("status") or ""
            ).lower()

            if status in SUCCESS_STATES:
                return
            if status in FAILURE_STATES:
                raise RuntimeError(_format_failure(task_id, status, payload))
            if status in IN_PROGRESS_STATES:
                continue
            # Unknown status: keep polling, but surface it so operators notice.
            logger.warning(
                "[docling] unknown task status %r for task %s; continuing to poll",
                status,
                task_id,
            )

        raise TimeoutError(f"Docling task {task_id} polling timeout")

    async def _download_zip_bytes(
        self,
        client: "httpx.AsyncClient",
        task_id: str,
    ) -> bytes:
        url = f"{self.endpoint}{RESULT_PATH.format(task_id=task_id)}"
        resp = await client.get(url)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if "zip" not in ctype.lower():
            raise RuntimeError(
                f"Docling result {task_id} returned non-zip content-type "
                f"{ctype!r}; body prefix={resp.text[:400]!r}"
            )
        return resp.content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bool_form(v: bool) -> str:
    return "true" if v else "false"


def _parse_ocr_lang(raw: str) -> list[str]:
    """Best-effort parser for ``DOCLING_OCR_LANG``.

    Accepts a JSON array (``["en","zh"]``) or a comma-separated list
    (``en,zh``). Returns a list of stripped non-empty strings; empty in →
    empty out.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _format_failure(task_id: str, status: str, payload: Any) -> str:
    if isinstance(payload, dict):
        err = (
            payload.get("error_message")
            or payload.get("error")
            or payload.get("message")
            or "<no error_message>"
        )
    else:
        err = "<no error_message>"
    truncated = json.dumps(payload, ensure_ascii=False)[:400]
    return f"Docling task {task_id} ended in {status}: {err}; payload={truncated}"


__all__ = [
    "DoclingRawClient",
    "CONVERT_PATH",
    "DEFAULT_MAX_POLLS",
    "DEFAULT_POLL_WAIT_SECONDS",
    "FIXED_CONSTANTS",
    "IMAGE_EXPORT_MODE",
    "PIPELINE",
    "POLL_PATH",
    "RESULT_PATH",
    "SUCCESS_STATES",
    "FAILURE_STATES",
    "TARGET_TYPE",
    "TO_FORMATS",
]
