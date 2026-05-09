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
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer

import numpy as np


# --------------------------------------------------------------------------
# Mock docling-serve endpoint (matches the real spec)
# --------------------------------------------------------------------------


def _build_app() -> tuple[FastAPI, dict[str, dict]]:
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
