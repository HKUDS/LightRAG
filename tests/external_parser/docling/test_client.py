"""Tests for :class:`DoclingRawClient`.

Cover the contract guarantees that protect the sidecar pipeline:

- the fixed pipeline constants (``pipeline=standard`` / ``target_type=zip``
  / ``to_formats=[json,md]`` / ``image_export_mode=referenced``) are sent
  on every upload, regardless of env;
- terminal non-success states (``failure`` / ``partial_success`` /
  ``skipped``) abort the run **before** any result download;
- ``DOCLING_OCR_LANG`` is omitted when empty so docling-serve falls back
  to its own default.

Uses an in-process fake httpx client mirroring ``tests/mineru_raw/test_client.py``
so we don't trip httpx's sync/async stream guard on multipart uploads.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from lightrag.external_parser.docling.client import (
    CONVERT_PATH,
    POLL_PATH,
    RESULT_PATH,
    DoclingRawClient,
)


# ---------------------------------------------------------------------------
# Minimal httpx fake (no MockTransport — avoids the multipart encode path)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.content = content or text.encode("utf-8")
        self.headers = headers or {}

    def json(self) -> Any:
        return json.loads(self.text) if self.text else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Recorder:
    def __init__(
        self,
        *,
        terminal_status: str,
        zip_bytes: bytes,
        task_id: str = "task-abc",
    ) -> None:
        self.terminal_status = terminal_status
        self.zip_bytes = zip_bytes
        self.task_id = task_id

        self.post_calls: list[dict] = []
        self.get_calls: list[dict] = []
        self.result_calls = 0


_CURRENT: dict[str, _Recorder] = {}


class _FakeAsyncClient:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    async def post(
        self,
        url: str,
        files: Any = None,
        data: Any = None,
        json: Any = None,
        headers: Any = None,
    ) -> _FakeResponse:
        recorder = _CURRENT["recorder"]
        recorder.post_calls.append(
            {"url": url, "files": files, "data": data, "json": json}
        )
        if CONVERT_PATH in url:
            return _FakeResponse(
                status_code=200,
                text=json_dump({"task_id": recorder.task_id}),
            )
        raise AssertionError(f"unexpected POST {url}")

    async def get(
        self, url: str, params: Any = None, headers: Any = None
    ) -> _FakeResponse:
        recorder = _CURRENT["recorder"]
        recorder.get_calls.append({"url": url, "params": params})
        if POLL_PATH.format(task_id=recorder.task_id) in url:
            payload: dict[str, Any] = {
                "task_id": recorder.task_id,
                "task_status": recorder.terminal_status,
            }
            if recorder.terminal_status != "success":
                payload["error_message"] = "synthetic-failure"
            return _FakeResponse(status_code=200, text=json_dump(payload))
        if RESULT_PATH.format(task_id=recorder.task_id) in url:
            recorder.result_calls += 1
            return _FakeResponse(
                status_code=200,
                content=recorder.zip_bytes,
                headers={"content-type": "application/zip"},
            )
        raise AssertionError(f"unexpected GET {url}")


def json_dump(payload: Any) -> str:
    return json.dumps(payload)


def _fake_zip_with_main_json(stem: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{stem}.json", b'{"schema_name": "DoclingDocument"}')
        zf.writestr(f"{stem}.md", b"# hello")
    return buf.getvalue()


def _install_fake_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``httpx.AsyncClient`` and ``httpx.Timeout`` references in
    the docling client module with no-arg fakes."""
    monkeypatch.setattr(
        "lightrag.external_parser.docling.client.httpx.AsyncClient",
        _FakeAsyncClient,
    )
    monkeypatch.setattr(
        "lightrag.external_parser.docling.client.httpx.Timeout",
        lambda *a, **kw: None,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "demo.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


@pytest.fixture(autouse=True)
def docling_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")
    for name in (
        "DOCLING_DO_OCR",
        "DOCLING_FORCE_OCR",
        "DOCLING_OCR_ENGINE",
        "DOCLING_OCR_PRESET",
        "DOCLING_OCR_LANG",
        "DOCLING_DO_FORMULA_ENRICHMENT",
        "DOCLING_ENGINE_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_docling_client_sends_fixed_constants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pdf: Path,
) -> None:
    recorder = _Recorder(
        terminal_status="success",
        zip_bytes=_fake_zip_with_main_json("demo"),
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    raw_dir = tmp_path / "demo.docling_raw"
    manifest = await DoclingRawClient().download_into(raw_dir, source_pdf)

    assert len(recorder.post_calls) == 1
    data = recorder.post_calls[0]["data"]
    field_map: dict[str, list[str]] = {}
    for name, value in data:
        field_map.setdefault(name, []).append(value)

    assert field_map["pipeline"] == ["standard"]
    assert field_map["target_type"] == ["zip"]
    assert field_map["image_export_mode"] == ["referenced"]
    assert sorted(field_map["to_formats"]) == ["json", "md"]

    files = recorder.post_calls[0]["files"]
    assert "files" in files
    name, blob, ctype = files["files"]
    assert name == "demo.pdf"
    assert blob.startswith(b"%PDF-1.4")
    assert ctype == "application/octet-stream"

    assert manifest.task_id == recorder.task_id
    assert manifest.engine == "docling"
    assert manifest.extras["fixed_constants"]["pipeline"] == "standard"
    assert manifest.endpoint_signature == "http://docling.test"


async def test_docling_client_partial_success_aborts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pdf: Path,
) -> None:
    recorder = _Recorder(
        terminal_status="partial_success",
        zip_bytes=_fake_zip_with_main_json("demo"),
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        await DoclingRawClient().download_into(
            tmp_path / "demo.docling_raw", source_pdf
        )
    msg = str(excinfo.value)
    assert recorder.task_id in msg
    assert "partial_success" in msg
    assert "synthetic-failure" in msg
    assert recorder.result_calls == 0


async def test_docling_client_failure_aborts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pdf: Path,
) -> None:
    recorder = _Recorder(
        terminal_status="failure",
        zip_bytes=_fake_zip_with_main_json("demo"),
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    with pytest.raises(RuntimeError):
        await DoclingRawClient().download_into(
            tmp_path / "demo.docling_raw", source_pdf
        )
    assert recorder.result_calls == 0


async def test_docling_client_skipped_aborts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pdf: Path,
) -> None:
    recorder = _Recorder(
        terminal_status="skipped",
        zip_bytes=_fake_zip_with_main_json("demo"),
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    with pytest.raises(RuntimeError):
        await DoclingRawClient().download_into(
            tmp_path / "demo.docling_raw", source_pdf
        )
    assert recorder.result_calls == 0


async def test_docling_client_ocr_lang_omitted_when_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pdf: Path,
) -> None:
    recorder = _Recorder(
        terminal_status="success",
        zip_bytes=_fake_zip_with_main_json("demo"),
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    await DoclingRawClient().download_into(tmp_path / "demo.docling_raw", source_pdf)

    data = recorder.post_calls[0]["data"]
    names = [name for name, _ in data]
    assert "ocr_lang" not in names


async def test_docling_client_ocr_lang_sent_when_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pdf: Path,
) -> None:
    monkeypatch.setenv("DOCLING_OCR_LANG", '["en","zh"]')
    recorder = _Recorder(
        terminal_status="success",
        zip_bytes=_fake_zip_with_main_json("demo"),
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    await DoclingRawClient().download_into(tmp_path / "demo.docling_raw", source_pdf)

    data = recorder.post_calls[0]["data"]
    langs = [v for name, v in data if name == "ocr_lang"]
    assert langs == ["en", "zh"]


async def test_docling_client_ocr_lang_csv_form(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pdf: Path,
) -> None:
    """CSV fallback when value isn't valid JSON."""
    monkeypatch.setenv("DOCLING_OCR_LANG", "en, fr")
    recorder = _Recorder(
        terminal_status="success",
        zip_bytes=_fake_zip_with_main_json("demo"),
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    await DoclingRawClient().download_into(tmp_path / "demo.docling_raw", source_pdf)

    data = recorder.post_calls[0]["data"]
    langs = [v for name, v in data if name == "ocr_lang"]
    assert langs == ["en", "fr"]


async def test_docling_client_rejects_missing_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCLING_ENDPOINT", "")
    with pytest.raises(ValueError, match="DOCLING_ENDPOINT"):
        DoclingRawClient()


async def test_docling_client_strips_parser_hint_from_upload_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression: a hinted source (``report.[docling].pdf``) used to cause
    # docling-serve to name its bundle JSON ``report.[docling].json``, which
    # the adapter (looking for ``report.json``) could not locate. The
    # pipeline now passes the canonical name as ``upload_filename`` so the
    # bundle is canonical-stem from the start.
    hinted = tmp_path / "report.[docling].pdf"
    hinted.write_bytes(b"%PDF-1.4 fake")
    # The fake zip mimics docling-serve responding with the *canonical* stem,
    # which is what would happen once we send the canonical filename.
    recorder = _Recorder(
        terminal_status="success",
        zip_bytes=_fake_zip_with_main_json("report"),
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    raw_dir = tmp_path / "report.docling_raw"
    manifest = await DoclingRawClient().download_into(
        raw_dir, hinted, upload_filename="report.pdf"
    )

    name, _blob, _ctype = recorder.post_calls[0]["files"]["files"]
    assert name == "report.pdf"
    assert manifest.source_filename_at_parse == "report.pdf"
    assert manifest.critical_file.path == "report.json"
    assert (raw_dir / "report.json").is_file()


async def test_docling_client_default_upload_filename_falls_back_to_source_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source_pdf: Path
) -> None:
    # Back-compat guard: callers that don't pass ``upload_filename`` (any
    # path other than the production pipeline) keep the legacy behavior of
    # using the on-disk source filename.
    recorder = _Recorder(
        terminal_status="success",
        zip_bytes=_fake_zip_with_main_json("demo"),
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    await DoclingRawClient().download_into(tmp_path / "demo.docling_raw", source_pdf)

    name, _blob, _ctype = recorder.post_calls[0]["files"]["files"]
    assert name == "demo.pdf"
