"""Tests for :class:`DoclingRawClient`.

Cover the contract guarantees that protect the sidecar pipeline:

- the fixed pipeline constants (``pipeline=standard`` / ``target_type=zip``
  / ``to_formats=[json,md]`` / ``image_export_mode=referenced``) are sent
  on every upload, regardless of env;
- terminal non-success states (``failure`` / ``partial_success`` /
  ``skipped``) abort the run **before** any result download;
- ``DOCLING_OCR_LANG`` is omitted when empty so docling-serve falls back
  to its own default.

Uses an in-process fake httpx client mirroring ``tests/parser/external/mineru/test_client.py``
so we don't trip httpx's sync/async stream guard on multipart uploads.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from lightrag.parser.external.docling.client import (
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
        submit_status_code: int = 200,
        submit_text: str | None = None,
        poll_status_code: int = 200,
        poll_text: str | None = None,
        result_status_code: int = 200,
        result_text: str | None = None,
    ) -> None:
        self.terminal_status = terminal_status
        self.zip_bytes = zip_bytes
        self.task_id = task_id
        self.submit_status_code = submit_status_code
        self.submit_text = submit_text
        self.poll_status_code = poll_status_code
        self.poll_text = poll_text
        self.result_status_code = result_status_code
        self.result_text = result_text

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
        # Production passes a file handle inside a `with` block — by the time
        # tests inspect `post_calls` it's already closed. Drain the stream
        # here so assertions can keep reading the payload as bytes.
        snapshot_files = files
        if files and "files" in files:
            name, payload, ctype = files["files"]
            if hasattr(payload, "read"):
                payload = payload.read()
            snapshot_files = {"files": (name, payload, ctype)}
        recorder.post_calls.append(
            {"url": url, "files": snapshot_files, "data": data, "json": json}
        )
        if CONVERT_PATH in url:
            if recorder.submit_status_code != 200:
                return _FakeResponse(
                    status_code=recorder.submit_status_code,
                    text=recorder.submit_text or "",
                )
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
            if recorder.poll_status_code != 200:
                return _FakeResponse(
                    status_code=recorder.poll_status_code,
                    text=recorder.poll_text or "",
                )
            payload: dict[str, Any] = {
                "task_id": recorder.task_id,
                "task_status": recorder.terminal_status,
            }
            if recorder.terminal_status != "success":
                payload["error_message"] = "synthetic-failure"
            return _FakeResponse(status_code=200, text=json_dump(payload))
        if RESULT_PATH.format(task_id=recorder.task_id) in url:
            recorder.result_calls += 1
            if recorder.result_status_code != 200:
                return _FakeResponse(
                    status_code=recorder.result_status_code,
                    text=recorder.result_text or "",
                )
            return _FakeResponse(
                status_code=200,
                content=recorder.zip_bytes,
                headers={"content-type": "application/zip"},
            )
        raise AssertionError(f"unexpected GET {url}")


def json_dump(payload: Any) -> str:
    return json.dumps(payload)


def _form_pairs(data: Any) -> list[tuple[str, str]]:
    """Normalize httpx form data into repeated ``(name, value)`` pairs.

    Production passes a mapping so httpx 0.28 keeps multipart ``files=`` on
    the async path. List values in that mapping represent repeated form keys.
    Older tests used tuple lists directly; accepting both keeps assertions
    focused on the wire contract instead of the container type.
    """
    if isinstance(data, dict):
        pairs: list[tuple[str, str]] = []
        for name, value in data.items():
            values = value if isinstance(value, list) else [value]
            pairs.extend((str(name), str(v)) for v in values)
        return pairs
    return [(str(name), str(value)) for name, value in data]


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
        "lightrag.parser.external.docling.client.httpx.AsyncClient",
        _FakeAsyncClient,
    )
    monkeypatch.setattr(
        "lightrag.parser.external.docling.client.httpx.Timeout",
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
    for name, value in _form_pairs(data):
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


async def test_docling_client_upload_http_error_preserves_response_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pdf: Path,
) -> None:
    recorder = _Recorder(
        terminal_status="success",
        zip_bytes=_fake_zip_with_main_json("demo"),
        submit_status_code=400,
        submit_text=json_dump({"detail": "unsupported file type"}),
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        await DoclingRawClient().download_into(
            tmp_path / "demo.docling_raw", source_pdf
        )

    message = str(excinfo.value)
    assert "Docling upload for 'demo.pdf'" in message
    assert "HTTP 400" in message
    assert "unsupported file type" in message


async def test_docling_client_poll_http_error_preserves_response_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pdf: Path,
) -> None:
    recorder = _Recorder(
        terminal_status="success",
        zip_bytes=_fake_zip_with_main_json("demo"),
        poll_status_code=503,
        poll_text=json_dump({"message": "queue unavailable"}),
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        await DoclingRawClient().download_into(
            tmp_path / "demo.docling_raw", source_pdf
        )

    message = str(excinfo.value)
    assert "Docling task task-abc poll" in message
    assert "HTTP 503" in message
    assert "queue unavailable" in message


async def test_docling_client_result_redirect_treated_as_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pdf: Path,
) -> None:
    # docling-serve fronted by a misconfigured proxy could emit a 302 to a
    # CDN that httpx (default ``follow_redirects=False``) won't follow.
    # Without the explicit non-2xx guard the redirect body would fall into
    # the zip-decoder and surface as a cryptic "bad zip" error.
    recorder = _Recorder(
        terminal_status="success",
        zip_bytes=_fake_zip_with_main_json("demo"),
        result_status_code=302,
        result_text="",
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        await DoclingRawClient().download_into(
            tmp_path / "demo.docling_raw", source_pdf
        )

    message = str(excinfo.value)
    assert "Docling result task-abc download" in message
    assert "HTTP 302" in message


async def test_docling_client_result_http_error_preserves_response_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pdf: Path,
) -> None:
    recorder = _Recorder(
        terminal_status="success",
        zip_bytes=_fake_zip_with_main_json("demo"),
        result_status_code=500,
        result_text="zip artifact missing",
    )
    _CURRENT["recorder"] = recorder
    _install_fake_httpx(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        await DoclingRawClient().download_into(
            tmp_path / "demo.docling_raw", source_pdf
        )

    message = str(excinfo.value)
    assert "Docling result task-abc download" in message
    assert "HTTP 500" in message
    assert "zip artifact missing" in message


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
    names = [name for name, _ in _form_pairs(data)]
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
    langs = [v for name, v in _form_pairs(data) if name == "ocr_lang"]
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
    langs = [v for name, v in _form_pairs(data) if name == "ocr_lang"]
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
