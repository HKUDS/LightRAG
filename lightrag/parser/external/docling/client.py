"""Docling raw bundle downloader.

Talks to Docling Serve v1 over HTTP:

- ``POST /v1/convert/file/async`` — multipart upload, returns ``task_id``,
- ``GET  /v1/status/poll/{task_id}?wait=5`` — long-poll for terminal state,
- ``GET  /v1/result/{task_id}`` — zip download (only on ``success``).

The zip is extracted safely under ``raw_dir/`` (refusing path traversal /
absolute entries). A success manifest is written atomically at the very
end; mid-run crashes therefore leave the directory in a state the cache
layer marks as invalid (no manifest → miss → re-download).

Pipeline constants (``target_type``, ``to_formats``, ``image_export_mode``) are
intentionally **not** env-driven — the sidecar flow depends on them — and are
recorded inside the manifest so a future code change automatically invalidates
pre-existing caches. The processing ``pipeline`` defaults to ``standard`` and
can be switched to ``vlm`` with ``DOCLING_PIPELINE``; the VLM preset is then
named by ``DOCLING_VLM_PRESET`` (for example ``minicpm_ocr``). Under the VLM
pipeline the OCR-only form fields are omitted, because that pipeline renders
pages through the vision model instead of an OCR engine.

Some docling-serve deployments return a JSON ``ConvertDocumentResponse``
envelope from the result endpoint even when the client requests a zip. When
that happens, the client materializes the nested ``document.json_content``
(the actual ``DoclingDocument``) as the main ``<stem>.json`` and writes
``document.md_content`` beside it as ``<stem>.md``. The response envelope and
its ``ExportDocumentResponse`` wrapper are not treated as document content. A
JSON envelope whose own conversion ``status`` is not ``success`` is rejected
rather than materialized, so a partial conversion never becomes a cache hit.
Because the JSON path cannot deliver the ``artifacts/`` files that
``image_export_mode=referenced`` produces, a ``json_content`` that references
external images (rather than embedding them as ``data:`` URIs) is rejected too,
so silent image loss can't hide behind a success manifest.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from lightrag.parser.external._common import (
    download_deadline_seconds,
    download_max_bytes,
    download_timeout,
    env_bool,
    env_int,
    raise_for_status_with_detail,
    stream_capped_get,
)
from lightrag.parser.external._zip import result_bundle_limits, safe_extract_zip
from lightrag.parser.external.docling.cache import (
    OCR_ENV_DEFAULTS,
    compute_options_signature,
    current_endpoint_signature,
    current_vlm_preset,
    resolve_docling_pipeline,
    snapshot_tunable_env,
)
from lightrag.parser.external.docling.manifest import (
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
# Pipeline constants
# ---------------------------------------------------------------------------

PIPELINE = "standard"
VLM_PIPELINE = "vlm"
TARGET_TYPE = "zip"
TO_FORMATS: tuple[str, ...] = ("json", "md")
IMAGE_EXPORT_MODE = "referenced"

FIXED_CONSTANTS: dict[str, object] = {
    "pipeline": PIPELINE,
    "target_type": TARGET_TYPE,
    "to_formats": list(TO_FORMATS),
    "image_export_mode": IMAGE_EXPORT_MODE,
}

# Values that differ from the documented defaults are the ones a VLM user
# actually changed; env.example ships the defaults, so warning on "set at all"
# would fire for every deployment.
_vlm_ignored_warned: set[tuple[str, ...]] = set()
_vlm_preset_ignored_warned = False


def _non_default_ocr_envs() -> list[str]:
    """OCR envs explicitly set to something other than their documented default."""
    configured: list[str] = []
    for name, default in OCR_ENV_DEFAULTS.items():
        raw = os.getenv(name)
        if raw is None:
            continue
        if isinstance(default, bool):
            value: object = env_bool(name, default)
        else:
            value = raw.strip() or default
        if value != default:
            configured.append(name)
    return configured


def fixed_constants_for(pipeline: str) -> dict[str, object]:
    """The fixed-constant block for ``pipeline``.

    ``standard`` reproduces :data:`FIXED_CONSTANTS` byte-for-byte so existing
    cache signatures stay valid; any other pipeline records its own value and
    therefore invalidates those bundles, as intended.
    """
    if pipeline == PIPELINE:
        return FIXED_CONSTANTS
    return {**FIXED_CONSTANTS, "pipeline": pipeline}


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

    def __init__(self, *, overrides: "Mapping[str, Any] | None" = None) -> None:
        self._overrides = overrides or {}
        self.endpoint = current_endpoint_signature()
        if not self.endpoint:
            raise ValueError("DOCLING_ENDPOINT is required")
        self.engine_version = os.getenv("DOCLING_ENGINE_VERSION", "").strip()

        self.pipeline = resolve_docling_pipeline()
        self.vlm_preset = current_vlm_preset()
        self._failure_vlm_preset = (
            self.vlm_preset if self.pipeline == VLM_PIPELINE else None
        )
        if self.pipeline == VLM_PIPELINE:
            if not self.vlm_preset:
                raise ValueError(
                    "DOCLING_PIPELINE=vlm requires DOCLING_VLM_PRESET "
                    "(for example minicpm_ocr)"
                )
            configured = _non_default_ocr_envs()
            if "force_ocr" in self._overrides and bool(
                self._overrides["force_ocr"]
            ) != env_bool("DOCLING_FORCE_OCR", True):
                configured.append("force_ocr (per-file override)")
            key = tuple(configured)
            if configured and key not in _vlm_ignored_warned:
                _vlm_ignored_warned.add(key)
                logger.warning(
                    "[docling] DOCLING_PIPELINE=vlm ignores the standard-pipeline "
                    "OCR settings: %s",
                    ", ".join(configured),
                )
        elif self.vlm_preset:
            global _vlm_preset_ignored_warned
            if not _vlm_preset_ignored_warned:
                _vlm_preset_ignored_warned = True
                logger.warning(
                    "[docling] DOCLING_VLM_PRESET is set but DOCLING_PIPELINE=%s; "
                    "the preset is ignored",
                    self.pipeline,
                )

        self.do_ocr = env_bool("DOCLING_DO_OCR", True)
        self.force_ocr = (
            bool(self._overrides["force_ocr"])
            if "force_ocr" in self._overrides
            else env_bool("DOCLING_FORCE_OCR", True)
        )
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
        *,
        upload_filename: str | None = None,
    ):
        """Upload, poll, download, extract, and write the manifest.

        ``upload_filename`` overrides the multipart filename sent to
        docling-serve (defaults to ``source_file_path.name``). The pipeline
        passes the canonical, hint-stripped document name here so the
        bundle's ``<stem>.json`` ends up canonical too — otherwise a file
        named ``report.[docling].pdf`` would produce ``report.[docling].json``
        inside the bundle, and the adapter (which only knows the canonical
        ``report.pdf``) would not be able to locate it via the preferred
        ``<stem>.json`` lookup.

        Pre-condition: caller cleared ``raw_dir`` (e.g. via
        :func:`lightrag.parser.external.clear_dir_contents`). This method
        does not clean the directory itself — keeping that explicit at the
        ``parse_docling`` entry point.
        """
        if httpx is None:
            raise RuntimeError(
                "httpx is required for Docling parsing but is not installed"
            )
        raw_dir.mkdir(parents=True, exist_ok=True)

        effective_filename = upload_filename or source_file_path.name

        timeout = httpx.Timeout(120.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            task_id = await self._submit(
                client, source_file_path, filename=effective_filename
            )
            await self._poll_until_done(client, task_id)
            await self._download_result_into(
                client, task_id, raw_dir, effective_filename
            )
        # Defensive: confirm the main JSON exists before anyone reads the
        # bundle. Look it up by the *uploaded* filename's stem — that's
        # what docling-serve uses to name the JSON inside the zip.
        select_main_json(raw_dir, Path(effective_filename))

        options_signature = compute_options_signature(
            tunable_env=snapshot_tunable_env(self._overrides),
            fixed_constants=fixed_constants_for(self.pipeline),
        )
        return build_and_write_docling_manifest(
            raw_dir,
            source_file_path=source_file_path,
            task_id=task_id,
            endpoint_signature=self.endpoint,
            engine_version=self.engine_version,
            options_signature=options_signature,
            fixed_constants=fixed_constants_for(self.pipeline),
            recorded_filename=effective_filename,
        )

    # ------------------------------------------------------------------
    # Upload + poll + download
    # ------------------------------------------------------------------

    def _build_multipart_data(self) -> dict[str, str | list[str]]:
        """Form fields (everything except the file payload).

        Returns a ``dict`` (not a list of tuples): httpx ≥ 0.28 short-circuits
        non-``Mapping`` ``data`` into raw-content encoding and ignores
        ``files=`` entirely, producing a sync-only stream that an
        ``AsyncClient`` then rejects. List-valued entries are emitted as
        repeated form keys by ``MultipartStream``, matching docling-serve's
        pydantic ``List[Enum]`` form parsing. ``ocr_lang`` is omitted entirely
        when empty so the engine uses its own default.
        """
        data: dict[str, str | list[str]] = {
            "pipeline": self.pipeline,
            "target_type": TARGET_TYPE,
            "image_export_mode": IMAGE_EXPORT_MODE,
            "to_formats": list(TO_FORMATS),
        }
        if self.pipeline == VLM_PIPELINE:
            # The VLM pipeline takes the page through the model, so the OCR
            # engine/preset/language fields and formula enrichment do not
            # apply; sending them makes older servers reject the request.
            data["vlm_pipeline_preset"] = self.vlm_preset
            return data
        data.update(
            {
                "do_ocr": _bool_form(self.do_ocr),
                "force_ocr": _bool_form(self.force_ocr),
                "ocr_engine": self.ocr_engine,
                "ocr_preset": self.ocr_preset,
                "do_formula_enrichment": _bool_form(self.do_formula_enrichment),
            }
        )
        if self.ocr_lang_raw:
            langs = _parse_ocr_lang(self.ocr_lang_raw)
            if langs:
                data["ocr_lang"] = langs
        return data

    async def _submit(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
        *,
        filename: str,
    ) -> str:
        url = f"{self.endpoint}{CONVERT_PATH}"
        # Hand httpx a file object so its MultipartStream reads the body in
        # chunks instead of materializing the whole PDF/PPTX in worker memory.
        # With ``max_parallel_parse_docling > 1`` a per-doc bytes copy can
        # OOM the worker before docling-serve ever sees the request.
        with source_file_path.open("rb") as fh:
            files = {"files": (filename, fh, "application/octet-stream")}
            resp = await client.post(
                url, data=self._build_multipart_data(), files=files
            )
        raise_for_status_with_detail(resp, f"Docling upload for {filename!r}")
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
        encoded_task_id = quote(task_id, safe="")
        url = f"{self.endpoint}{POLL_PATH.format(task_id=encoded_task_id)}"
        params = {"wait": self.poll_wait_seconds}
        for _ in range(self.max_poll_attempts):
            iteration_started = time.monotonic()
            resp = await client.get(url, params=params)
            raise_for_status_with_detail(resp, f"Docling task {task_id} poll")
            payload = resp.json() if resp.text else {}
            status = str(
                payload.get("task_status") or payload.get("status") or ""
            ).lower()

            if status in SUCCESS_STATES:
                return
            if status in FAILURE_STATES:
                raise RuntimeError(
                    _format_failure(
                        task_id,
                        status,
                        payload,
                        vlm_preset=self._failure_vlm_preset,
                    )
                )
            if status not in IN_PROGRESS_STATES:
                # Unknown status: keep polling, but surface it so operators notice.
                logger.warning(
                    "[docling] unknown task status %r for task %s; continuing to poll",
                    status,
                    task_id,
                )

            # The intended cadence is one poll per ``poll_wait_seconds`` — the
            # design relies on docling-serve's ``?wait=N`` long-polling for
            # that. Some deployments return immediately instead, which would
            # burn through ``max_poll_attempts`` in milliseconds and fail
            # with a spurious timeout. Cap each iteration at the configured
            # interval ourselves so the total budget holds either way.
            elapsed = time.monotonic() - iteration_started
            remaining = self.poll_wait_seconds - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)

        raise TimeoutError(f"Docling task {task_id} polling timeout")

    async def _download_result_into(
        self,
        client: "httpx.AsyncClient",
        task_id: str,
        raw_dir: Path,
        upload_filename: str,
    ) -> None:
        encoded_task_id = quote(task_id, safe="")
        url = f"{self.endpoint}{RESULT_PATH.format(task_id=encoded_task_id)}"
        max_entries, max_total_bytes = result_bundle_limits()
        # Stream-capped, not client.get(): a plain get() fully buffers the
        # response before returning regardless of size, so a compromised or
        # misbehaving docling-serve deployment could hold an arbitrarily
        # large body in memory before safe_extract_zip's checks below ever
        # run. Streaming lets an oversized response be rejected mid-
        # download. Wrapped in a wall-clock deadline too — the per-read
        # httpx.Timeout(120.0, ...) set around this client only bounds a
        # single socket operation, so a peer trickling one byte per
        # interval can reset it indefinitely; the deadline bounds the
        # whole download regardless of how it stalls.
        try:
            async with download_timeout(download_deadline_seconds()):
                resp, body = await stream_capped_get(
                    client,
                    url,
                    max_bytes=download_max_bytes(),
                    operation=f"Docling result {task_id} download",
                )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Docling result {task_id} download exceeded its wall-clock deadline"
            ) from exc
        raise_for_status_with_detail(
            resp, f"Docling result {task_id} download", body=body
        )
        ctype = resp.headers.get("content-type", "")
        if "zip" not in ctype.lower():
            if _is_json_result(body, ctype):
                _materialize_json_result(
                    body,
                    task_id,
                    raw_dir,
                    upload_filename,
                    vlm_preset=self._failure_vlm_preset,
                )
                return
            raise RuntimeError(
                f"Docling result {task_id} returned non-zip content-type "
                f"{ctype!r}; body prefix={body[:400]!r}"
            )
        safe_extract_zip(
            body,
            raw_dir,
            max_entries=max_entries,
            max_total_bytes=max_total_bytes,
        )


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


def _is_json_result(body: bytes, content_type: str) -> bool:
    ctype = content_type.lower()
    if "json" in ctype:
        return True
    return body.lstrip().startswith((b"{", b"["))


def _result_envelope_status(payload: Any) -> str:
    """Conversion status carried by a JSON result envelope, lower-cased.

    Mirrors ``_poll_until_done``'s field lookup (``task_status`` first, then
    ``status``). Returns ``""`` when neither is present — e.g. a bare
    ``DoclingDocument`` payload — so the caller can skip the status gate.
    """
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("task_status") or payload.get("status") or "").lower()


def _materialize_json_result(
    body: bytes,
    task_id: str,
    raw_dir: Path,
    upload_filename: str,
    *,
    vlm_preset: str | None = None,
) -> None:
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Docling result returned JSON content-type but the body is not valid JSON"
        ) from exc

    # A JSON ``ConvertDocumentResponse`` envelope carries its own conversion
    # ``status``. ``_poll_until_done`` only checked the async *task* status, and
    # the manifest is written after this returns — so honouring the envelope
    # status here is what stops a ``partial_success`` / ``failure`` result from
    # being cached as a successful bundle. (Bare DoclingDocument payloads have
    # no status field; those skip the check.)
    status = _result_envelope_status(payload)
    if status and status not in SUCCESS_STATES:
        raise RuntimeError(
            _format_failure(task_id, status, payload, vlm_preset=vlm_preset)
        )

    document, markdown = _extract_document_payload(payload)

    # ``image_export_mode=referenced`` makes docling reference picture bytes by
    # relative ``artifacts/...`` path. Those files ship inside the zip, never
    # inside the JSON envelope — so the JSON path, which writes only
    # ``<stem>.json`` (+ optional markdown), cannot deliver them. The IR builder
    # would then silently drop every such picture yet the bundle would still be
    # cached as a success. Fail loudly instead so image loss can't hide behind a
    # cache hit. Embedded (``data:``) and remote (``http(s)://``) images carry
    # their own bytes and are unaffected.
    referenced = _referenced_image_uris(document)
    if referenced:
        sample = ", ".join(referenced[:5])
        raise RuntimeError(
            f"Docling JSON result references external image artifacts the JSON "
            f"result path cannot deliver (only <stem>.json/<stem>.md are "
            f"written): {sample}. Configure docling-serve to return a zip bundle "
            f"or embed images as data URIs for this deployment."
        )

    stem = Path(upload_filename).stem
    (raw_dir / f"{stem}.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if markdown is not None:
        (raw_dir / f"{stem}.md").write_text(str(markdown), encoding="utf-8")


def _extract_document_payload(payload: Any) -> tuple[dict[str, Any], Any | None]:
    if not isinstance(payload, dict):
        raise RuntimeError("Docling JSON result is not an object")

    nested = payload.get("document")
    if isinstance(nested, dict):
        # ``ConvertDocumentResponse.document`` is an ``ExportDocumentResponse``
        # wrapper (``filename`` / ``md_content`` / ``json_content`` / ...); the
        # actual ``DoclingDocument`` lives under ``json_content``. Storing the
        # wrapper as ``<stem>.json`` would leave the root without
        # ``body`` / ``texts`` / ``tables``, and the IR builder would produce
        # zero blocks. Extract ``json_content`` and keep ``md_content`` as the
        # markdown twin.
        json_content = nested.get("json_content")
        if isinstance(json_content, dict):
            return json_content, nested.get("md_content")
        # Defensive: some deployments may place the ``DoclingDocument`` directly
        # under ``document`` with no ``json_content`` wrapper.
        if _looks_like_docling_document(nested):
            return nested, nested.get("md_content")
        keys = ", ".join(sorted(str(k) for k in nested)[:20])
        raise RuntimeError(
            f"Docling result 'document' has no DoclingDocument (json_content); "
            f"keys=[{keys}]"
        )

    # No envelope: the endpoint returned the DoclingDocument at the top level.
    if _looks_like_docling_document(payload):
        return payload, payload.get("md_content")

    keys = ", ".join(sorted(str(k) for k in payload.keys())[:20])
    raise RuntimeError(
        f"Docling JSON result missing document object; top-level keys=[{keys}]"
    )


def _looks_like_docling_document(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("schema_name", "body", "texts", "tables"))


def _referenced_image_uris(document: dict[str, Any]) -> list[str]:
    """Relative picture image URIs in a ``DoclingDocument`` — i.e. neither
    ``data:`` nor ``http(s)://``. These name companion ``artifacts/`` files the
    JSON result path never writes, so their presence means images would be lost.

    Mirrors the IR builder's URI classification (``_build_ir_drawing``): only a
    non-empty, non-data, non-remote ``image.uri`` needs a local file.
    """
    pictures = document.get("pictures")
    if not isinstance(pictures, list):
        return []
    referenced: list[str] = []
    for pic in pictures:
        if not isinstance(pic, dict):
            continue
        image = pic.get("image")
        if not isinstance(image, dict):
            continue
        uri = str(image.get("uri") or "")
        if not uri or uri.startswith(("data:", "http://", "https://")):
            continue
        referenced.append(uri)
    return referenced


def _format_failure(
    task_id: str, status: str, payload: Any, *, vlm_preset: str | None = None
) -> str:
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
    message = f"Docling task {task_id} ended in {status}: {err}; payload={truncated}"
    if vlm_preset:
        # docling-serve accepts the preset at upload and resolves it in the
        # worker, so an unknown or disallowed preset fails the task instead of
        # returning HTTP 422 on submit.
        message += (
            f". If the task failed on the VLM preset {vlm_preset!r}, the serve "
            "host needs a docling release that registers it (minicpm_ocr needs "
            "docling>=2.149), and DOCLING_SERVE_ALLOWED_VLM_PRESETS must include it."
        )
    return message


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
