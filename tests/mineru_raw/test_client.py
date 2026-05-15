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

from lightrag.mineru_raw import is_bundle_valid
from lightrag.mineru_raw.client import MinerURawClient


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
        files: Any = None,
        json: Any = None,
        data: Any = None,
        headers: Any = None,
    ) -> _FakeResponse:
        self.posts.append(
            {"url": url, "files": files, "json": json, "data": data, "headers": headers}
        )
        return _CURRENT.dispatcher.post(
            url, files=files, json=json, data=data, headers=headers
        )

    async def put(
        self, url: str, data: Any = None, headers: Any = None
    ) -> _FakeResponse:
        return _CURRENT.dispatcher.put(url, data=data, headers=headers)

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


# ---------------------------------------------------------------------------
# Common monkeypatch helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_httpx(monkeypatch: pytest.MonkeyPatch) -> type:
    import lightrag.mineru_raw.client as mod

    fake = type(
        "FakeHttpx",
        (),
        {
            "AsyncClient": _FakeAsyncClient,
            "Timeout": lambda *a, **k: None,
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

    def put(self, url: str, **_: Any) -> _FakeResponse:
        if url == "https://upload.example/demo.pdf":
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
    assert dispatcher.apply_payload
    assert dispatcher.apply_payload["files"][0]["name"] == "demo.pdf"
    assert dispatcher.apply_payload["model_version"] == "vlm"
    assert manifest.task_id == "B-1"
    assert manifest.api_mode == "official"
    assert manifest.endpoint_signature == "https://mineru.net"
    assert (raw / "content_list.json").is_file()
    assert (raw / "images" / "img_001.png").read_bytes() == b"\x89PNGnested"
    assert is_bundle_valid(raw, src) is True


# ---------------------------------------------------------------------------
# local mode: /tasks + /tasks/{id} + /tasks/{id}/result
# ---------------------------------------------------------------------------


class _LocalDispatcher(_Dispatcher):
    def __init__(self) -> None:
        self.form_data: dict[str, Any] | None = None

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        if url == "http://127.0.0.1:8000/tasks":
            self.form_data = kwargs.get("data")
            assert kwargs.get("files")
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

    assert dispatcher.form_data
    assert dispatcher.form_data["response_format_zip"] == "true"
    assert dispatcher.form_data["return_content_list"] == "true"
    assert dispatcher.form_data["return_images"] == "true"
    assert manifest.task_id == "L-1"
    assert manifest.api_mode == "local"
    assert manifest.endpoint_signature == "http://127.0.0.1:8000"
    assert (raw / "content_list.json").is_file()
    assert (raw / "images" / "img_001.png").read_bytes() == b"\x89PNGnested"


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
