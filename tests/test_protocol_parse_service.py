"""Tests for the protocol-driven parse service used by Docling and MinerU.

These cover the regressions reported in GitHub issue #2995:
  * Docling-serve expects multipart field name ``files`` (plural).
  * Docling-serve poll responses use ``task_status`` (lowercase enum:
    ``pending|started|success|failure``) — not ``status``.
  * Docling-serve serves results from a separate ``/v1/result/{task_id}``
    path, not from a ``result_url`` payload field.
  * Default poll/result paths must derive from the upload endpoint when
    only ``DOCLING_ENDPOINT`` is configured.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer

import numpy as np


# --------------------------------------------------------------------------
# Mock docling-serve endpoint (matches the real spec)
# --------------------------------------------------------------------------


ResultFactory = Callable[[str], Response]


def _build_app(
    *,
    wrapped_result: bool = False,
    result_factory: ResultFactory | None = None,
) -> tuple[FastAPI, dict[str, dict]]:
    app = FastAPI()
    tasks: dict[str, dict] = {}

    @app.post("/v1/convert/file/async")
    async def upload(files: list[UploadFile] = File(...)):
        body = await files[0].read()
        task_id = f"task-{len(tasks) + 1}"
        tasks[task_id] = {"size": len(body), "polls": 0}
        return {"task_id": task_id, "task_status": "pending"}

    @app.get("/v1/status/poll/{task_id}")
    async def poll(task_id: str):
        if task_id not in tasks:
            raise HTTPException(404, "no such task")
        tasks[task_id]["polls"] += 1
        return {"task_id": task_id, "task_status": "success"}

    if result_factory is not None:
        @app.get("/v1/result/{task_id}")
        async def result(task_id: str):
            return result_factory(task_id)
    elif wrapped_result:
        # Mirror the real docling-serve shape: a JSON wrapper with the parsed
        # body nested under ``document.md_content`` (plus other format slots).
        @app.get("/v1/result/{task_id}")
        async def result(task_id: str):
            return JSONResponse(
                {
                    "document": {
                        "md_content": f"# Parsed {task_id}\n\nbody",
                        "text_content": f"Parsed {task_id} body",
                        "html_content": "",
                        "json_content": {},
                        "doctags_content": "",
                    },
                    "status": "success",
                    "processing_time": 0.01,
                    "timings": {},
                    "errors": [],
                }
            )
    else:
        @app.get("/v1/result/{task_id}")
        async def result(task_id: str):
            return PlainTextResponse(f"PARSED:{task_id}")

    return app, tasks


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ServerThread(threading.Thread):
    def __init__(self, app: FastAPI, port: int) -> None:
        super().__init__(daemon=True)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.port = port

    def run(self) -> None:
        self.server.run()

    def wait_ready(self, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                httpx.get(f"http://127.0.0.1:{self.port}/docs", timeout=0.3)
                return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("mock docling-serve did not start in time")

    def stop(self) -> None:
        self.server.should_exit = True
        self.join(timeout=5.0)


@pytest.fixture
def mock_docling_serve():
    app, tasks = _build_app()
    port = _free_port()
    thread = _ServerThread(app, port)
    thread.start()
    thread.wait_ready()
    try:
        yield {
            "base": f"http://127.0.0.1:{port}",
            "tasks": tasks,
        }
    finally:
        thread.stop()


@pytest.fixture
def mock_docling_serve_wrapped():
    """docling-serve mock whose /v1/result/{task_id} returns the real JSON
    wrapper ({"document": {"md_content": ..., ...}, ...}) — used to verify
    that ``_call_protocol_parse_service`` extracts via ``content_field``
    instead of persisting the raw wrapper as document content."""

    app, tasks = _build_app(wrapped_result=True)
    port = _free_port()
    thread = _ServerThread(app, port)
    thread.start()
    thread.wait_ready()
    try:
        yield {
            "base": f"http://127.0.0.1:{port}",
            "tasks": tasks,
        }
    finally:
        thread.stop()


def _spawn_server(result_factory: ResultFactory) -> tuple[_ServerThread, str]:
    """Spin up a one-off mock server whose /v1/result returns ``result_factory``.

    Caller is responsible for stopping the thread.
    """
    app, _tasks = _build_app(result_factory=result_factory)
    port = _free_port()
    thread = _ServerThread(app, port)
    thread.start()
    thread.wait_ready()
    return thread, f"http://127.0.0.1:{port}"


def _build_protocol(base: str, *, content_field: str) -> dict[str, Any]:
    return {
        "upload_url": f"{base}/v1/convert/file/async",
        "upload_field_name": "files",
        "poll_url_template": f"{base}/v1/status/poll/{{task_id}}",
        "id_field": "task_id",
        "status_field": "task_status",
        "result_url_field": "result_url",
        "result_url_template": f"{base}/v1/result/{{task_id}}",
        "content_field": content_field,
        "success_values": "success",
        "failed_values": "failure",
        "poll_interval_seconds": 0.05,
        "max_polls": 20,
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


async def _embed(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 8), dtype=float)


async def _llm(prompt: str, **_) -> str:
    return ""


class _Tok:
    def encode(self, content: str) -> list[int]:
        return [ord(c) for c in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _new_rag(tmp_path: Path) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"protocol-{tmp_path.name}",
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=512, func=_embed
        ),
        tokenizer=Tokenizer("mock", _Tok()),
    )


# --------------------------------------------------------------------------
# Tests for _call_protocol_parse_service
# --------------------------------------------------------------------------


@pytest.mark.offline
def test_call_protocol_uses_configured_multipart_field(
    tmp_path, mock_docling_serve
):
    """Default ``upload_field_name`` is ``"file"`` — verify the override sends
    the multipart field as ``"files"`` so docling-serve does not return 422."""

    base = mock_docling_serve["base"]

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        sample = tmp_path / "sample.pdf"
        sample.write_bytes(b"%PDF-1.4 fake")

        protocol: dict[str, Any] = {
            "upload_url": f"{base}/v1/convert/file/async",
            "upload_field_name": "files",
            "poll_url_template": f"{base}/v1/status/poll/{{task_id}}",
            "id_field": "task_id",
            "status_field": "task_status",
            "result_url_field": "result_url",
            "result_url_template": f"{base}/v1/result/{{task_id}}",
            "success_values": "success",
            "failed_values": "failure",
            "poll_interval_seconds": 0.05,
            "max_polls": 20,
        }

        out = await rag._call_protocol_parse_service(
            protocol=protocol, file_path=str(sample)
        )
        assert out is not None and out.startswith("PARSED:task-")

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_call_protocol_default_field_name_rejected_by_docling(
    tmp_path, mock_docling_serve
):
    """Without overriding upload_field_name, the call sends ``file`` and
    docling-serve returns 422 — the regression from issue #2995."""

    base = mock_docling_serve["base"]

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        sample = tmp_path / "sample.pdf"
        sample.write_bytes(b"%PDF-1.4 fake")

        protocol: dict[str, Any] = {
            "upload_url": f"{base}/v1/convert/file/async",
            # no upload_field_name -> defaults to "file"
            "poll_url_template": f"{base}/v1/status/poll/{{task_id}}",
            "id_field": "task_id",
            "status_field": "task_status",
            "success_values": "success",
            "failed_values": "failure",
            "poll_interval_seconds": 0.05,
            "max_polls": 5,
        }

        with pytest.raises(RuntimeError, match=r"422"):
            await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_call_protocol_falls_back_to_result_url_template(
    tmp_path, mock_docling_serve
):
    """When the poll payload omits ``result_url`` (real docling-serve
    behavior), the result must be fetched from ``result_url_template``."""

    base = mock_docling_serve["base"]

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        sample = tmp_path / "sample.pdf"
        sample.write_bytes(b"%PDF-1.4 fake")

        protocol: dict[str, Any] = {
            "upload_url": f"{base}/v1/convert/file/async",
            "upload_field_name": "files",
            "poll_url_template": f"{base}/v1/status/poll/{{task_id}}",
            "id_field": "task_id",
            "status_field": "task_status",
            "result_url_field": "result_url",  # not present in poll payload
            "result_url_template": f"{base}/v1/result/{{task_id}}",
            "success_values": "success",
            "failed_values": "failure",
            "poll_interval_seconds": 0.05,
            "max_polls": 20,
        }

        out = await rag._call_protocol_parse_service(
            protocol=protocol, file_path=str(sample)
        )
        assert out is not None and out.startswith("PARSED:task-")

        await rag.finalize_storages()

    asyncio.run(_run())


# --------------------------------------------------------------------------
# Tests for parse_docling default protocol construction
# --------------------------------------------------------------------------


@pytest.mark.offline
def test_parse_docling_protocol_defaults_match_docling_serve(
    tmp_path, monkeypatch
):
    """Capture the protocol dict built by ``parse_docling`` when only
    ``DOCLING_ENDPOINT`` is set, and assert it matches the docling-serve
    spec."""

    captured: dict[str, Any] = {}

    async def _capture(self, protocol, file_path):
        captured.update(protocol)
        return "noop"

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        sample = tmp_path / "sample.pdf"
        sample.write_bytes(b"%PDF-1.4 fake")

        monkeypatch.setenv(
            "DOCLING_ENDPOINT", "http://localhost:5001/v1/convert/file/async"
        )
        monkeypatch.setattr(
            type(rag), "_call_protocol_parse_service", _capture, raising=True
        )

        # parse_docling will call _capture (returning "noop"), then attempt to
        # persist; we only care about ``captured``.
        try:
            await rag.parse_docling(
                doc_id="d-1",
                file_path=str(sample),
                content_data={"content": "x"},
            )
        except Exception:
            pass

        assert captured["upload_field_name"] == "files"
        assert captured["status_field"] == "task_status"
        assert (
            captured["poll_url_template"]
            == "http://localhost:5001/v1/status/poll/{task_id}"
        )
        assert (
            captured["result_url_template"]
            == "http://localhost:5001/v1/result/{task_id}"
        )
        assert "failure" in captured["failed_values"]

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_parse_docling_endpoint_with_trailing_slash_still_derives_defaults(
    tmp_path, monkeypatch
):
    """A trailing slash on ``DOCLING_ENDPOINT`` must not defeat the
    suffix-based derivation of poll/result paths."""

    captured: dict[str, Any] = {}

    async def _capture(self, protocol, file_path):
        captured.update(protocol)
        return "noop"

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        sample = tmp_path / "sample.pdf"
        sample.write_bytes(b"%PDF-1.4 fake")

        monkeypatch.setenv(
            "DOCLING_ENDPOINT", "http://localhost:5001/v1/convert/file/async/"
        )
        monkeypatch.setattr(
            type(rag), "_call_protocol_parse_service", _capture, raising=True
        )

        try:
            await rag.parse_docling(
                doc_id="d-2",
                file_path=str(sample),
                content_data={"content": "x"},
            )
        except Exception:
            pass

        # rstrip("/") on the endpoint must drop the trailing slash so the
        # derivation produces spec paths, not endpoint + "/{task_id}".
        assert (
            captured["poll_url_template"]
            == "http://localhost:5001/v1/status/poll/{task_id}"
        )
        assert (
            captured["result_url_template"]
            == "http://localhost:5001/v1/result/{task_id}"
        )
        assert captured["upload_url"] == "http://localhost:5001/v1/convert/file/async"

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_call_protocol_extracts_content_field_from_json_result_wrapper(
    tmp_path, mock_docling_serve_wrapped
):
    """Regression for codex review on PR #3045: docling-serve's
    ``/v1/result/{task_id}`` returns ``{"document": {"md_content": "..."}, ...}``,
    not plain text. ``_call_protocol_parse_service`` must use ``content_field``
    (default ``document.md_content`` for parse_docling) to extract the parsed
    body, otherwise the raw JSON wrapper is persisted as document content."""

    base = mock_docling_serve_wrapped["base"]

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        sample = tmp_path / "sample.pdf"
        sample.write_bytes(b"%PDF-1.4 fake")

        protocol: dict[str, Any] = {
            "upload_url": f"{base}/v1/convert/file/async",
            "upload_field_name": "files",
            "poll_url_template": f"{base}/v1/status/poll/{{task_id}}",
            "id_field": "task_id",
            "status_field": "task_status",
            "result_url_field": "result_url",
            "result_url_template": f"{base}/v1/result/{{task_id}}",
            "content_field": "document.md_content",
            "success_values": "success",
            "failed_values": "failure",
            "poll_interval_seconds": 0.05,
            "max_polls": 20,
        }

        out = await rag._call_protocol_parse_service(
            protocol=protocol, file_path=str(sample)
        )

        assert out is not None
        assert out.startswith("# Parsed task-")
        assert "body" in out
        # The raw JSON wrapper must NOT have leaked through.
        assert '"document"' not in out
        assert "md_content" not in out

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_call_protocol_falls_back_to_raw_text_when_json_path_missing(
    tmp_path, mock_docling_serve_wrapped
):
    """If ``content_field`` does not resolve in the JSON wrapper (e.g. the
    user pointed it at a missing key), the call must fall back to the raw
    body rather than swallow the response. This preserves the historical
    contract for plain-text result endpoints and gives downstream
    ``normalize_parser_result_to_content_list`` a chance to recover."""

    base = mock_docling_serve_wrapped["base"]

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        sample = tmp_path / "sample.pdf"
        sample.write_bytes(b"%PDF-1.4 fake")

        protocol: dict[str, Any] = {
            "upload_url": f"{base}/v1/convert/file/async",
            "upload_field_name": "files",
            "poll_url_template": f"{base}/v1/status/poll/{{task_id}}",
            "id_field": "task_id",
            "status_field": "task_status",
            "result_url_field": "result_url",
            "result_url_template": f"{base}/v1/result/{{task_id}}",
            "content_field": "does.not.exist",
            "success_values": "success",
            "failed_values": "failure",
            "poll_interval_seconds": 0.05,
            "max_polls": 20,
        }

        out = await rag._call_protocol_parse_service(
            protocol=protocol, file_path=str(sample)
        )
        assert out is not None
        assert '"document"' in out  # raw JSON wrapper preserved

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_parse_docling_default_content_field_is_document_md_content(
    tmp_path, monkeypatch
):
    """``parse_docling`` must default ``content_field`` to
    ``document.md_content`` so the wrapper returned by docling-serve's
    ``/v1/result/{task_id}`` is unwrapped automatically."""

    captured: dict[str, Any] = {}

    async def _capture(self, protocol, file_path):
        captured.update(protocol)
        return "noop"

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        sample = tmp_path / "sample.pdf"
        sample.write_bytes(b"%PDF-1.4 fake")

        monkeypatch.setenv(
            "DOCLING_ENDPOINT", "http://localhost:5001/v1/convert/file/async"
        )
        monkeypatch.delenv("DOCLING_CONTENT_FIELD", raising=False)
        monkeypatch.setattr(
            type(rag), "_call_protocol_parse_service", _capture, raising=True
        )

        try:
            await rag.parse_docling(
                doc_id="d-3",
                file_path=str(sample),
                content_data={"content": "x"},
            )
        except Exception:
            pass

        assert captured["content_field"] == "document.md_content"

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
@pytest.mark.parametrize("bad_value", ["", "   ", None])
def test_call_protocol_normalizes_empty_upload_field_name(
    tmp_path, mock_docling_serve, bad_value
):
    """Empty / whitespace-only / missing ``upload_field_name`` must fall back
    to the historical default ``"file"`` rather than producing an invalid
    multipart key. With docling-serve as the upstream this still 422s
    (expected — the default is wrong for docling), but the failure mode is
    explicit instead of a silent malformed request."""

    base = mock_docling_serve["base"]

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        sample = tmp_path / "sample.pdf"
        sample.write_bytes(b"%PDF-1.4 fake")

        protocol: dict[str, Any] = {
            "upload_url": f"{base}/v1/convert/file/async",
            "upload_field_name": bad_value,
            "poll_url_template": f"{base}/v1/status/poll/{{task_id}}",
            "id_field": "task_id",
            "status_field": "task_status",
            "success_values": "success",
            "failed_values": "failure",
            "poll_interval_seconds": 0.05,
            "max_polls": 5,
        }

        # The 422 response from docling-serve must mention the missing
        # ``files`` field — proving the fallback used a real multipart key.
        with pytest.raises(RuntimeError, match=r"422.*files"):
            await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )

        await rag.finalize_storages()

    asyncio.run(_run())


# --------------------------------------------------------------------------
# Exhaustive edge-case coverage for JSON-result extraction
#
# Each test below pins a distinct branch of ``_call_protocol_parse_service``'s
# result-fetch path:
#
#   raw_text = dl.text
#   if content_field:
#       try: parsed = json.loads(raw_text)
#       except (ValueError, TypeError): parsed = None
#       if isinstance(parsed, (dict, list)):
#           extracted = get_by_path(parsed, content_field)
#           if isinstance(extracted, str) and extracted:
#               return extracted
#   return raw_text
# --------------------------------------------------------------------------


@pytest.mark.offline
def test_result_json_text_content_extracted_when_content_field_overridden(tmp_path):
    """Override ``content_field`` to ``document.text_content`` — must extract
    plain text instead of markdown, both fields populated in the wrapper."""

    def factory(task_id: str) -> Response:
        return JSONResponse(
            {
                "document": {
                    "md_content": "# md should be ignored",
                    "text_content": f"plain {task_id} body",
                    "html_content": "<p>html</p>",
                },
                "status": "success",
            }
        )

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            protocol = _build_protocol(base, content_field="document.text_content")
            out = await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )
            assert out == "plain task-1 body"
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()


@pytest.mark.offline
def test_result_json_html_content_extracted_when_content_field_overridden(tmp_path):
    """Same wrapper, ``content_field=document.html_content`` — extracts HTML."""

    def factory(task_id: str) -> Response:
        return JSONResponse(
            {
                "document": {
                    "md_content": "# md",
                    "text_content": "plain",
                    "html_content": f"<h1>html {task_id}</h1>",
                },
                "status": "success",
            }
        )

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            protocol = _build_protocol(base, content_field="document.html_content")
            out = await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )
            assert out == "<h1>html task-1</h1>"
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()


@pytest.mark.offline
def test_result_json_invalid_falls_back_to_raw_text(tmp_path):
    """If the response body is *not* valid JSON, the raw text must be
    returned verbatim — regression guard for the existing plain-text path."""

    def factory(task_id: str) -> Response:
        # Deliberately malformed JSON-looking content.
        return PlainTextResponse(f"{{not json: {task_id}")

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            protocol = _build_protocol(base, content_field="document.md_content")
            out = await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )
            assert out == "{not json: task-1"
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()


@pytest.mark.offline
def test_result_json_empty_string_at_path_falls_back_to_raw(tmp_path):
    """If ``content_field`` resolves to an empty string in the wrapper, the
    extractor must fall back to the raw body (so the user can debug — silent
    empty would mask configuration errors)."""

    def factory(task_id: str) -> Response:
        return JSONResponse(
            {"document": {"md_content": "", "text_content": "fallback"}}
        )

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            protocol = _build_protocol(base, content_field="document.md_content")
            out = await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )
            assert out is not None
            # Raw JSON wrapper preserved so downstream
            # normalize_parser_result_to_content_list can attempt recovery.
            assert isinstance(out, str)
            payload = json.loads(out)
            assert payload["document"]["text_content"] == "fallback"
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()


@pytest.mark.offline
def test_result_json_non_string_at_path_falls_back_to_raw(tmp_path):
    """If ``content_field`` resolves to a dict (e.g.
    ``document.json_content``), the extractor must fall back to the raw
    text rather than ``str()``-coercing the dict."""

    def factory(task_id: str) -> Response:
        return JSONResponse(
            {
                "document": {
                    "md_content": "# md",
                    "json_content": {"sections": [{"text": "x"}]},
                }
            }
        )

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            protocol = _build_protocol(base, content_field="document.json_content")
            out = await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )
            assert isinstance(out, str)
            payload = json.loads(out)
            assert isinstance(payload["document"]["json_content"], dict)
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()


@pytest.mark.offline
def test_result_json_top_level_list_falls_back_to_raw(tmp_path):
    """A top-level JSON list shouldn't be unwrapped via ``get_by_path`` (only
    dict traversal is supported); the raw text must be returned so the
    downstream normalizer can pick it up."""

    def factory(task_id: str) -> Response:
        return JSONResponse(
            [{"type": "text", "text": f"chunk {task_id}"}]
        )

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            protocol = _build_protocol(base, content_field="document.md_content")
            out = await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )
            assert isinstance(out, str)
            payload = json.loads(out)
            assert isinstance(payload, list) and payload[0]["type"] == "text"
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()


@pytest.mark.offline
def test_result_json_top_level_scalar_falls_back_to_raw(tmp_path):
    """A bare JSON-encoded string (e.g. ``"hello"``) should pass through as
    raw text — ``get_by_path`` only walks dicts so the result is unchanged."""

    def factory(task_id: str) -> Response:
        return Response(content='"hello"', media_type="application/json")

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            protocol = _build_protocol(base, content_field="document.md_content")
            out = await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )
            assert out == '"hello"'
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()


@pytest.mark.offline
def test_result_json_empty_content_field_disables_unwrap(tmp_path):
    """If ``content_field`` is empty, the JSON-unwrap branch must be skipped
    entirely (back-compat for callers who don't want any extraction)."""

    def factory(task_id: str) -> Response:
        return JSONResponse(
            {"document": {"md_content": f"md-{task_id}"}}
        )

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            protocol = _build_protocol(base, content_field="")
            out = await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )
            assert isinstance(out, str)
            payload = json.loads(out)
            assert payload["document"]["md_content"] == "md-task-1"
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()


@pytest.mark.offline
def test_result_json_path_with_dots_in_keys_returns_none_safely(tmp_path):
    """``get_by_path`` splits on ``.`` — keys containing literal dots are not
    addressable. Must fall back to raw text rather than crash."""

    def factory(task_id: str) -> Response:
        # The literal key ``"a.b"`` is not reachable via dotted traversal
        # because get_by_path splits ``"a.b"`` into ``["a", "b"]``.
        return JSONResponse({"a.b": "unreachable", "fallback": "x"})

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            protocol = _build_protocol(base, content_field="a.b")
            out = await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )
            assert isinstance(out, str)
            payload = json.loads(out)
            assert payload["a.b"] == "unreachable"
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()


@pytest.mark.offline
def test_result_json_deeply_nested_path_extracts(tmp_path):
    """Sanity check that ``get_by_path`` traverses multi-level dotted paths
    correctly — important if a future docling-serve version nests further."""

    def factory(task_id: str) -> Response:
        return JSONResponse(
            {"a": {"b": {"c": {"d": f"deep-{task_id}"}}}}
        )

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            protocol = _build_protocol(base, content_field="a.b.c.d")
            out = await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )
            assert out == "deep-task-1"
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()


# --------------------------------------------------------------------------
# End-to-end: parse_docling against the wrapped mock must persist *markdown*
# as the document content, not the raw JSON wrapper. This is the bug the
# codex review surfaced.
# --------------------------------------------------------------------------


@pytest.mark.offline
def test_parse_docling_persists_extracted_markdown_not_raw_json_wrapper(
    tmp_path, mock_docling_serve_wrapped, monkeypatch
):
    base = mock_docling_serve_wrapped["base"]

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        sample = tmp_path / "input.pdf"
        sample.write_bytes(b"%PDF-1.4 fake")

        monkeypatch.setenv("DOCLING_ENDPOINT", f"{base}/v1/convert/file/async")
        # Default DOCLING_CONTENT_FIELD (document.md_content) — do not override.
        monkeypatch.delenv("DOCLING_CONTENT_FIELD", raising=False)

        result = await rag.parse_docling(
            doc_id="d-e2e-1",
            file_path=str(sample),
            content_data={"content": "x", "source_path": str(sample)},
        )

        # The persisted ``content`` must be the extracted markdown — NOT the
        # JSON wrapper. Both anchors below come from the mock factory above.
        content = result.get("content", "")
        assert content.startswith("# Parsed task-")
        assert "body" in content
        # Critical: no raw-JSON leakage.
        assert '"document"' not in content
        assert '"md_content"' not in content
        assert '"status": "success"' not in content

        await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.offline
def test_parse_docling_with_text_content_override_persists_plain_text(
    tmp_path, mock_docling_serve_wrapped, monkeypatch
):
    """End-to-end coverage for the documented override:
    ``DOCLING_CONTENT_FIELD=document.text_content`` should land plain-text
    body in ``full_docs``, not markdown."""

    base = mock_docling_serve_wrapped["base"]

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()

        sample = tmp_path / "input.pdf"
        sample.write_bytes(b"%PDF-1.4 fake")

        monkeypatch.setenv("DOCLING_ENDPOINT", f"{base}/v1/convert/file/async")
        monkeypatch.setenv("DOCLING_CONTENT_FIELD", "document.text_content")

        result = await rag.parse_docling(
            doc_id="d-e2e-2",
            file_path=str(sample),
            content_data={"content": "x", "source_path": str(sample)},
        )

        content = result.get("content", "")
        assert content == "Parsed task-1 body"

        await rag.finalize_storages()

    asyncio.run(_run())


# --------------------------------------------------------------------------
# MinerU back-compat: the result-URL JSON-unwrap path is shared with MinerU.
# Verify that:
#   1. Plain-text result (the historical MinerU contract) still passes through.
#   2. JSON result with a top-level ``content`` key extracts cleanly (this is
#      a behavior improvement — previously persisted as raw JSON — but a
#      deliberate one, kept consistent with poll-payload extraction).
# --------------------------------------------------------------------------


@pytest.mark.offline
def test_mineru_plain_text_result_url_unchanged(tmp_path):
    """MinerU deployments that serve the parsed body as plain text from
    ``result_url`` must continue to receive the raw text verbatim."""

    def factory(task_id: str) -> Response:
        return PlainTextResponse(f"MINERU PARSED:{task_id}")

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            # MinerU's historical defaults — content_field=content.
            protocol = _build_protocol(base, content_field="content")
            out = await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )
            assert out == "MINERU PARSED:task-1"
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()


@pytest.mark.offline
def test_mineru_json_with_content_field_extracts(tmp_path):
    """MinerU deployments that wrap the body as ``{"content": "..."}`` from
    ``result_url`` will now have ``content`` extracted (consistent with the
    existing poll-payload extraction). Documents the behavior delta."""

    def factory(task_id: str) -> Response:
        return JSONResponse({"content": f"mineru body {task_id}"})

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            protocol = _build_protocol(base, content_field="content")
            out = await rag._call_protocol_parse_service(
                protocol=protocol, file_path=str(sample)
            )
            assert out == "mineru body task-1"
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()


# --------------------------------------------------------------------------
# Status-code propagation: a 500 from /v1/result must surface as an error,
# not be silently treated as text. Pins the ``raise_for_status()`` contract.
# --------------------------------------------------------------------------


@pytest.mark.offline
def test_result_endpoint_500_raises_http_status_error(tmp_path):
    def factory(task_id: str) -> Response:
        return Response(content="boom", status_code=500)

    thread, base = _spawn_server(factory)
    try:
        async def _run():
            rag = _new_rag(tmp_path)
            await rag.initialize_storages()
            sample = tmp_path / "x.pdf"
            sample.write_bytes(b"%PDF-1.4 fake")
            protocol = _build_protocol(base, content_field="document.md_content")
            with pytest.raises(httpx.HTTPStatusError):
                await rag._call_protocol_parse_service(
                    protocol=protocol, file_path=str(sample)
                )
            await rag.finalize_storages()

        asyncio.run(_run())
    finally:
        thread.stop()
