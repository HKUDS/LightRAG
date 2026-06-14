"""``MinerURawClient.download_into`` integration tests.

Uses an in-process fake httpx client so the upload / poll / result fetch
choreography is exercised end-to-end without a live MinerU server. After
the call, the raw dir contains:

- ``content_list.json``
- ``images/`` for any ``img_path`` references
- ``_manifest.json`` whose hashes match the on-disk bytes
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from lightrag.parser.external.mineru import is_bundle_valid
from lightrag.parser.external.mineru.client import MinerURawClient


# ---------------------------------------------------------------------------
# Minimal httpx mock framework
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


class _FakeAsyncClient:
    """Routes calls through a per-test dispatcher."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.posts: list[dict] = []
        self.gets: list[str] = []

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    async def post(
        self,
        url: str,
        content: Any = None,
        files: Any = None,
        json: Any = None,
        data: Any = None,
        headers: Any = None,
    ) -> _FakeResponse:
        self.posts.append(
            {
                "url": url,
                "content": content,
                "files": files,
                "json": json,
                "data": data,
                "headers": headers,
            }
        )
        return _CURRENT.dispatcher.post(
            url, content=content, files=files, json=json, data=data, headers=headers
        )

    async def put(
        self,
        url: str,
        data: Any = None,
        content: Any = None,
        headers: Any = None,
    ) -> _FakeResponse:
        return _CURRENT.dispatcher.put(url, data=data, content=content, headers=headers)

    async def get(
        self, url: str, params: Any = None, headers: Any = None
    ) -> _FakeResponse:
        self.gets.append(url)
        return _CURRENT.dispatcher.get(url, params=params, headers=headers)


class _Dispatcher:
    def post(self, url: str, **_: Any) -> _FakeResponse:  # pragma: no cover
        raise NotImplementedError

    def get(self, url: str, **_: Any) -> _FakeResponse:  # pragma: no cover
        raise NotImplementedError

    def put(self, url: str, **_: Any) -> _FakeResponse:  # pragma: no cover
        raise NotImplementedError


class _CURRENT:  # set per-test via monkeypatch
    dispatcher: _Dispatcher | None = None


async def _collect_async_bytes(stream: Any) -> bytes:
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Common monkeypatch helpers
# ---------------------------------------------------------------------------


class _FakeRequestError(Exception):
    """Stand-in for ``httpx.RequestError`` (transport-level base)."""


class _FakeConnectError(_FakeRequestError):
    """Stand-in for ``httpx.ConnectError``."""


@pytest.fixture
def fake_httpx(monkeypatch: pytest.MonkeyPatch) -> type:
    import lightrag.parser.external.mineru.client as mod

    fake = type(
        "FakeHttpx",
        (),
        {
            "AsyncClient": _FakeAsyncClient,
            "Timeout": lambda *a, **k: None,
            "RequestError": _FakeRequestError,
            "ConnectError": _FakeConnectError,
        },
    )
    monkeypatch.setattr(mod, "httpx", fake)

    async def _instant_sleep(_t: float) -> None:
        return None

    # MinerURawClient uses asyncio.sleep directly; patch via module ref.
    import asyncio

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    return fake


def _nested_mineru_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "demo/auto/demo_content_list.json",
            json.dumps(
                [
                    {"type": "text", "text": "nested"},
                    {"type": "image", "img_path": "images/img_001.png"},
                ],
                ensure_ascii=False,
            ),
        )
        zf.writestr("demo/auto/images/img_001.png", b"\x89PNGnested")
        zf.writestr("demo/auto/demo.md", "# Nested\n")
    return buf.getvalue()


def _flat_mineru_zip() -> bytes:
    """Zip whose root already contains the canonical layout — normalization
    should be a no-op."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "content_list.json",
            json.dumps(
                [
                    {"type": "text", "text": "flat"},
                    {"type": "image", "img_path": "images/img_002.png"},
                ],
                ensure_ascii=False,
            ),
        )
        zf.writestr("images/img_002.png", b"\x89PNGflat")
    return buf.getvalue()


def _multi_doc_mineru_zip() -> bytes:
    """Zip carrying two parse subtrees; only the entry matching the source
    stem should be picked as the canonical content_list."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "other/auto/other_content_list.json",
            json.dumps([{"type": "text", "text": "other"}], ensure_ascii=False),
        )
        zf.writestr(
            "demo/auto/demo_content_list.json",
            json.dumps(
                [
                    {"type": "text", "text": "the right one"},
                    {"type": "image", "img_path": "images/img_001.png"},
                ],
                ensure_ascii=False,
            ),
        )
        zf.writestr("demo/auto/images/img_001.png", b"\x89PNGmulti")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# official mode: signed upload + batch poll + full_zip_url
# ---------------------------------------------------------------------------


class _OfficialDispatcher(_Dispatcher):
    def __init__(self) -> None:
        self.polls = 0
        self.uploaded = False
        self.apply_payload: dict[str, Any] | None = None
        self.upload_content: Any = None
        self.upload_headers: dict[str, str] | None = None

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        if url == "https://mineru.net/api/v4/file-urls/batch":
            headers = kwargs.get("headers") or {}
            assert headers["Authorization"] == "Bearer token-123"
            self.apply_payload = kwargs.get("json")
            return _FakeResponse(
                text=json.dumps(
                    {
                        "code": 0,
                        "msg": "ok",
                        "data": {
                            "batch_id": "B-1",
                            "file_urls": ["https://upload.example/demo.pdf"],
                        },
                    }
                )
            )
        raise AssertionError(f"unexpected POST {url}")

    def put(self, url: str, **kwargs: Any) -> _FakeResponse:
        if url == "https://upload.example/demo.pdf":
            self.upload_content = kwargs.get("content")
            self.upload_headers = kwargs.get("headers")
            assert not isinstance(self.upload_content, bytes)
            assert hasattr(self.upload_content, "__aiter__")
            self.uploaded = True
            return _FakeResponse(status_code=200)
        raise AssertionError(f"unexpected PUT {url}")

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        if url == "https://mineru.net/api/v4/extract-results/batch/B-1":
            headers = kwargs.get("headers") or {}
            assert headers["Authorization"] == "Bearer token-123"
            self.polls += 1
            state = "running" if self.polls == 1 else "done"
            result = {
                "file_name": "demo.pdf",
                "state": state,
            }
            if state == "done":
                result["full_zip_url"] = "https://download.example/full.zip"
            return _FakeResponse(
                text=json.dumps({"code": 0, "data": {"extract_result": [result]}})
            )
        if url == "https://download.example/full.zip":
            return _FakeResponse(
                content=_nested_mineru_zip(),
                headers={"Content-Type": "application/zip"},
            )
        raise AssertionError(f"unexpected GET {url}")


@pytest.mark.offline
async def test_client_official_mode_round_trip(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.setenv("MINERU_API_TOKEN", "token-123")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("MINERU_MAX_POLLS", "5")

    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDFBYTES" * 200)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    dispatcher = _OfficialDispatcher()
    _CURRENT.dispatcher = dispatcher
    manifest = await MinerURawClient().download_into(raw, src)

    assert dispatcher.uploaded is True
    assert dispatcher.upload_headers == {"Content-Length": str(src.stat().st_size)}
    assert await _collect_async_bytes(dispatcher.upload_content) == src.read_bytes()
    assert dispatcher.apply_payload
    assert dispatcher.apply_payload["files"][0]["name"] == "demo.pdf"
    assert dispatcher.apply_payload["model_version"] == "vlm"
    assert manifest.task_id == "B-1"
    assert manifest.api_mode == "official"
    assert manifest.endpoint_signature == "https://mineru.net"
    assert (raw / "content_list.json").is_file()
    assert (raw / "images" / "img_001.png").read_bytes() == b"\x89PNGnested"
    assert is_bundle_valid(raw, src) is True


@pytest.mark.offline
async def test_client_official_upload_name_overrides_source_basename(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.setenv("MINERU_API_TOKEN", "token-123")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")

    src = tmp_path / "demo.[mineru-iet].pdf"
    src.write_bytes(b"PDFBYTES" * 200)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    dispatcher = _OfficialDispatcher()
    _CURRENT.dispatcher = dispatcher
    manifest = await MinerURawClient().download_into(
        raw,
        src,
        upload_name="demo.pdf",
    )

    assert dispatcher.apply_payload
    assert dispatcher.apply_payload["files"][0]["name"] == "demo.pdf"
    assert manifest.source_filename_at_parse == "demo.pdf"


# ---------------------------------------------------------------------------
# local mode: /tasks + /tasks/{id} + /tasks/{id}/result
# ---------------------------------------------------------------------------


class _LocalDispatcher(_Dispatcher):
    def __init__(self) -> None:
        self.content: Any = None
        self.form_data: dict[str, Any] | None = None
        self.files: Any = None
        self.headers: dict[str, str] | None = None
        self.upload_filename: str | None = None
        self.upload_payload: bytes | None = None
        self.upload_content_type: str | None = None

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        if url == "http://127.0.0.1:8000/tasks":
            self.content = kwargs.get("content")
            self.form_data = kwargs.get("data")
            self.files = kwargs.get("files")
            self.headers = kwargs.get("headers")
            assert self.content is None
            assert self.files and "files" in self.files
            name, payload, ctype = self.files["files"]
            assert hasattr(payload, "read")
            assert not isinstance(payload, bytes)
            self.upload_filename = name
            self.upload_payload = payload.read()
            self.upload_content_type = ctype
            return _FakeResponse(text=json.dumps({"task_id": "L-1"}))
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url: str, **_: Any) -> _FakeResponse:
        if url == "http://127.0.0.1:8000/tasks/L-1":
            return _FakeResponse(
                text=json.dumps({"task_id": "L-1", "status": "completed"})
            )
        if url == "http://127.0.0.1:8000/tasks/L-1/result":
            return _FakeResponse(
                content=_nested_mineru_zip(),
                headers={"Content-Type": "application/zip"},
            )
        raise AssertionError(f"unexpected GET {url}")


@pytest.mark.offline
async def test_client_local_mode_round_trip(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")

    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDFBYTES" * 200)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    dispatcher = _LocalDispatcher()
    _CURRENT.dispatcher = dispatcher
    manifest = await MinerURawClient().download_into(raw, src)

    assert dispatcher.headers is None
    assert dispatcher.form_data
    assert dispatcher.form_data["backend"] == "hybrid-auto-engine"
    assert dispatcher.form_data["parse_method"] == "auto"
    assert dispatcher.form_data["image_analysis"] == "false"
    assert dispatcher.form_data["response_format_zip"] == "true"
    assert dispatcher.form_data["return_content_list"] == "true"
    assert dispatcher.form_data["return_images"] == "true"
    assert dispatcher.upload_filename == "demo.pdf"
    assert dispatcher.upload_content_type == "application/octet-stream"
    assert dispatcher.upload_payload == src.read_bytes()
    assert manifest.task_id == "L-1"
    assert manifest.api_mode == "local"
    assert manifest.endpoint_signature == "http://127.0.0.1:8000"
    assert manifest.options_signature.startswith("sha256:")
    assert (raw / "content_list.json").is_file()
    assert (raw / "images" / "img_001.png").read_bytes() == b"\x89PNGnested"


@pytest.mark.offline
async def test_client_local_upload_name_overrides_multipart_filename(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")

    src = tmp_path / "demo.[mineru-R!].pdf"
    src.write_bytes(b"PDFBYTES" * 200)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    dispatcher = _LocalDispatcher()
    _CURRENT.dispatcher = dispatcher
    manifest = await MinerURawClient().download_into(
        raw,
        src,
        upload_name="demo.pdf",
    )

    assert dispatcher.content is None
    assert dispatcher.upload_filename == "demo.pdf"
    assert dispatcher.upload_payload == src.read_bytes()
    assert manifest.source_filename_at_parse == "demo.pdf"


class _OfficialBadRequestDispatcher(_Dispatcher):
    def post(self, url: str, **_: Any) -> _FakeResponse:
        if url == "https://mineru.net/api/v4/file-urls/batch":
            return _FakeResponse(
                status_code=401,
                text=json.dumps({"code": 401, "msg": "invalid api token"}),
            )
        raise AssertionError(f"unexpected POST {url}")


@pytest.mark.offline
async def test_client_official_bad_request_preserves_response_body(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.setenv("MINERU_API_TOKEN", "token-123")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")

    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDFBYTES" * 200)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _OfficialBadRequestDispatcher()
    with pytest.raises(RuntimeError) as exc_info:
        await MinerURawClient().download_into(raw, src)

    message = str(exc_info.value)
    assert "MinerU official upload URL request" in message
    assert "HTTP 401" in message
    assert "invalid api token" in message


class _OfficialFailedDispatcher(_OfficialDispatcher):
    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        if url == "https://mineru.net/api/v4/extract-results/batch/B-1":
            headers = kwargs.get("headers") or {}
            assert headers["Authorization"] == "Bearer token-123"
            return _FakeResponse(
                text=json.dumps(
                    {
                        "code": 0,
                        "data": {
                            "extract_result": [
                                {
                                    "file_name": "demo.pdf",
                                    "state": "failed",
                                    "err_msg": "bad pdf",
                                }
                            ]
                        },
                    }
                )
            )
        raise AssertionError(f"unexpected GET {url}")


@pytest.mark.offline
async def test_client_official_failed_state_raises(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.setenv("MINERU_API_TOKEN", "token-123")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")

    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDFBYTES" * 200)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _OfficialFailedDispatcher()
    with pytest.raises(RuntimeError, match="bad pdf"):
        await MinerURawClient().download_into(raw, src)


class _LocalFailedDispatcher(_Dispatcher):
    def post(self, url: str, **_: Any) -> _FakeResponse:
        if url == "http://127.0.0.1:8000/tasks":
            return _FakeResponse(text=json.dumps({"task_id": "L-bad"}))
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url: str, **_: Any) -> _FakeResponse:
        if url == "http://127.0.0.1:8000/tasks/L-bad":
            return _FakeResponse(
                text=json.dumps(
                    {"task_id": "L-bad", "status": "failed", "error": "bad pdf"}
                )
            )
        raise AssertionError(f"unexpected GET {url}")


@pytest.mark.offline
async def test_client_local_failed_state_raises(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")

    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDFBYTES" * 200)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _LocalFailedDispatcher()
    with pytest.raises(RuntimeError, match="bad pdf"):
        await MinerURawClient().download_into(raw, src)


class _LocalRedirectDispatcher(_Dispatcher):
    def post(self, url: str, **_: Any) -> _FakeResponse:
        if url == "http://127.0.0.1:8000/tasks":
            # Proxy/CDN misconfig: redirect with httpx default
            # ``follow_redirects=False`` would otherwise fall through and
            # break with a confusing "missing task_id" downstream.
            return _FakeResponse(
                status_code=302,
                headers={"Location": "http://alt.example/tasks"},
            )
        raise AssertionError(f"unexpected POST {url}")


@pytest.mark.offline
async def test_client_local_redirect_treated_as_error(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000")

    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDFBYTES" * 200)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _LocalRedirectDispatcher()
    with pytest.raises(RuntimeError) as exc_info:
        await MinerURawClient().download_into(raw, src)

    message = str(exc_info.value)
    assert "MinerU local task submission" in message
    assert "HTTP 302" in message


class _LocalBadRequestDispatcher(_Dispatcher):
    def post(self, url: str, **_: Any) -> _FakeResponse:
        if url == "http://127.0.0.1:8000/tasks":
            return _FakeResponse(
                status_code=400,
                text=json.dumps(
                    {
                        "detail": "unsupported file type: .xlsx extension does not match payload"
                    }
                ),
            )
        raise AssertionError(f"unexpected POST {url}")


@pytest.mark.offline
async def test_client_local_bad_request_preserves_response_body(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000")

    src = tmp_path / "demo.xlsx"
    src.write_bytes(b"not-really-xlsx")
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _LocalBadRequestDispatcher()
    with pytest.raises(RuntimeError) as exc_info:
        await MinerURawClient().download_into(raw, src)

    message = str(exc_info.value)
    assert "MinerU local task submission" in message
    assert "HTTP 400" in message
    assert "unsupported file type" in message
    assert "demo.xlsx" in message


class _LocalConnectErrorDispatcher(_Dispatcher):
    """Backend is down: every request fails at the transport layer."""

    def __init__(self, message: str) -> None:
        self.message = message

    def post(self, url: str, **_: Any) -> _FakeResponse:
        raise _FakeConnectError(self.message)


@pytest.mark.offline
async def test_client_connection_error_is_wrapped_with_engine_context(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport-level failure (e.g. backend restart) must surface as a
    RuntimeError that names MinerU + the endpoint, so the doc_status error_msg
    is attributable instead of an opaque ``All connection attempts failed``."""
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000")

    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDFBYTES" * 200)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _LocalConnectErrorDispatcher("All connection attempts failed")
    with pytest.raises(RuntimeError) as exc_info:
        await MinerURawClient().download_into(raw, src)

    message = str(exc_info.value)
    assert "MinerU local backend request failed" in message
    assert "http://127.0.0.1:8000" in message
    assert "_FakeConnectError" in message
    assert "All connection attempts failed" in message


@pytest.mark.offline
async def test_client_connection_error_with_empty_message_stays_non_empty(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some transport errors stringify to an empty message; the wrapped
    RuntimeError must still carry the exception class name and context."""
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000")

    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDFBYTES" * 200)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _LocalConnectErrorDispatcher("")
    with pytest.raises(RuntimeError) as exc_info:
        await MinerURawClient().download_into(raw, src)

    message = str(exc_info.value)
    assert message.strip()
    assert "MinerU local backend request failed" in message
    assert "_FakeConnectError" in message


@pytest.mark.offline
def test_client_mode_specific_endpoint_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="MINERU_API_TOKEN"):
        MinerURawClient()

    monkeypatch.setenv("MINERU_API_TOKEN", "x")
    monkeypatch.setenv("MINERU_OFFICIAL_ENDPOINT", "https://mineru.net/api/v4")
    with pytest.raises(ValueError, match="MINERU_OFFICIAL_ENDPOINT"):
        MinerURawClient()

    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.delenv("MINERU_LOCAL_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="MINERU_LOCAL_ENDPOINT"):
        MinerURawClient()

    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000/tasks")
    with pytest.raises(ValueError, match="MINERU_LOCAL_ENDPOINT"):
        MinerURawClient()

    monkeypatch.setenv("MINERU_API_MODE", "custom")
    with pytest.raises(ValueError, match="MINERU_API_MODE"):
        MinerURawClient()


# ---------------------------------------------------------------------------
# Manifest is *atomic*: presence implies fully written
# ---------------------------------------------------------------------------


@pytest.mark.offline
async def test_client_manifest_written_atomically(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")

    src = tmp_path / "demo.pdf"
    src.write_bytes(b"X" * 16)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _LocalDispatcher()
    await MinerURawClient().download_into(raw, src)

    # No leftover .tmp marker; only the final _manifest.json should exist.
    leftovers = list(raw.glob("_manifest*"))
    assert leftovers == [raw / "_manifest.json"]


# ---------------------------------------------------------------------------
# Bundle normalization: flat-zip fast path / multi-doc disambiguation
# ---------------------------------------------------------------------------


class _LocalFlatZipDispatcher(_Dispatcher):
    def post(self, url: str, **_: Any) -> _FakeResponse:
        if url == "http://127.0.0.1:8000/tasks":
            return _FakeResponse(text=json.dumps({"task_id": "L-flat"}))
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url: str, **_: Any) -> _FakeResponse:
        if url == "http://127.0.0.1:8000/tasks/L-flat":
            return _FakeResponse(
                text=json.dumps({"task_id": "L-flat", "status": "completed"})
            )
        if url == "http://127.0.0.1:8000/tasks/L-flat/result":
            return _FakeResponse(
                content=_flat_mineru_zip(),
                headers={"Content-Type": "application/zip"},
            )
        raise AssertionError(f"unexpected GET {url}")


@pytest.mark.offline
async def test_client_flat_zip_normalize_is_noop(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zip whose root already has content_list.json + images/ stays flat.
    The manifest must only record the two real files, not a duplicate."""
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")

    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDF" * 50)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _LocalFlatZipDispatcher()
    manifest = await MinerURawClient().download_into(raw, src)

    assert (raw / "content_list.json").is_file()
    assert (raw / "images" / "img_002.png").read_bytes() == b"\x89PNGflat"
    # Only one image listed in manifest files; no nested duplicate.
    file_paths = sorted(f.path for f in manifest.files)
    assert file_paths == ["images/img_002.png"]


class _LocalMultiDocDispatcher(_Dispatcher):
    def post(self, url: str, **_: Any) -> _FakeResponse:
        if url == "http://127.0.0.1:8000/tasks":
            return _FakeResponse(text=json.dumps({"task_id": "L-multi"}))
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url: str, **_: Any) -> _FakeResponse:
        if url == "http://127.0.0.1:8000/tasks/L-multi":
            return _FakeResponse(
                text=json.dumps({"task_id": "L-multi", "status": "completed"})
            )
        if url == "http://127.0.0.1:8000/tasks/L-multi/result":
            return _FakeResponse(
                content=_multi_doc_mineru_zip(),
                headers={"Content-Type": "application/zip"},
            )
        raise AssertionError(f"unexpected GET {url}")


@pytest.mark.offline
async def test_client_multi_doc_zip_picks_source_stem(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two parse subtrees in the zip: the one whose stem matches the source
    file wins, and the rival's content_list.json must NOT bleed into root."""
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://127.0.0.1:8000")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")

    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDF" * 50)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _LocalMultiDocDispatcher()
    await MinerURawClient().download_into(raw, src)

    content_list = json.loads((raw / "content_list.json").read_text())
    assert content_list[0]["text"] == "the right one"
    # Hoist removes the demo subtree; the unrelated 'other' subtree is left
    # untouched (still nested, no false root content_list).
    assert (raw / "images" / "img_001.png").read_bytes() == b"\x89PNGmulti"
    assert not (raw / "demo").exists()
    assert (raw / "other" / "auto" / "other_content_list.json").is_file()


# ---------------------------------------------------------------------------
# Official mode: multiple non-terminal poll rounds before "done"
# ---------------------------------------------------------------------------


class _OfficialSlowDispatcher(_OfficialDispatcher):
    """Returns pending → running → done across three polls."""

    def get(self, url: str, **_: Any) -> _FakeResponse:
        if url == "https://mineru.net/api/v4/extract-results/batch/B-1":
            self.polls += 1
            if self.polls == 1:
                state = "pending"
            elif self.polls == 2:
                state = "running"
            else:
                state = "done"
            result: dict[str, Any] = {"file_name": "demo.pdf", "state": state}
            if state == "done":
                result["full_zip_url"] = "https://download.example/full.zip"
            return _FakeResponse(
                text=json.dumps({"code": 0, "data": {"extract_result": [result]}})
            )
        if url == "https://download.example/full.zip":
            return _FakeResponse(
                content=_nested_mineru_zip(),
                headers={"Content-Type": "application/zip"},
            )
        raise AssertionError(f"unexpected GET {url}")


@pytest.mark.offline
async def test_client_official_polls_through_non_terminal_states(
    tmp_path: Path,
    fake_httpx: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.setenv("MINERU_API_TOKEN", "token-123")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("MINERU_MAX_POLLS", "5")

    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDF" * 50)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    dispatcher = _OfficialSlowDispatcher()
    _CURRENT.dispatcher = dispatcher
    await MinerURawClient().download_into(raw, src)
    assert dispatcher.polls == 3
    assert (raw / "content_list.json").is_file()
