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
        self, url: str, files: Any = None, json: Any = None
    ) -> _FakeResponse:
        self.posts.append({"url": url, "files": files, "json": json})
        return _CURRENT.dispatcher.post(url, files=files, json=json)

    async def get(self, url: str, params: Any = None) -> _FakeResponse:
        self.gets.append(url)
        return _CURRENT.dispatcher.get(url, params=params)


class _Dispatcher:
    def post(self, url: str, **_: Any) -> _FakeResponse:  # pragma: no cover
        raise NotImplementedError

    def get(self, url: str, **_: Any) -> _FakeResponse:  # pragma: no cover
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


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_ENDPOINT", "http://mineru.example/api/v1/task")
    monkeypatch.setenv(
        "MINERU_POLL_ENDPOINT", "http://mineru.example/api/v1/task/{task_id}"
    )
    monkeypatch.setenv("MINERU_POLL_METHOD", "GET")
    monkeypatch.setenv("MINERU_ID_FIELD", "task_id")
    monkeypatch.setenv("MINERU_STATUS_FIELD", "status")
    monkeypatch.setenv("MINERU_RESULT_URL_FIELD", "result_url")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("MINERU_MAX_POLLS", "5")
    monkeypatch.setenv("MINERU_RESULT_MODE", "auto")


# ---------------------------------------------------------------------------
# flat_json mode: result_url returns content_list JSON; images fetched relative
# ---------------------------------------------------------------------------


class _FlatJSONDispatcher(_Dispatcher):
    def __init__(self) -> None:
        self.polls = 0
        self.image_bytes = b"\x89PNGfakeimg"

    def post(self, url: str, **_: Any) -> _FakeResponse:
        if url.endswith("/task"):
            return _FakeResponse(
                text=json.dumps({"task_id": "T-1", "status": "queued"})
            )
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url: str, **_: Any) -> _FakeResponse:
        if "/task/T-1" in url and "result" not in url:
            self.polls += 1
            if self.polls < 2:
                return _FakeResponse(
                    text=json.dumps({"task_id": "T-1", "status": "running"})
                )
            return _FakeResponse(
                text=json.dumps(
                    {
                        "task_id": "T-1",
                        "status": "done",
                        "result_url": "http://mineru.example/results/T-1/content_list.json",
                    }
                )
            )
        if url.endswith("/content_list.json"):
            return _FakeResponse(
                text=json.dumps(
                    [
                        {"type": "text", "text": "hi"},
                        {"type": "image", "img_path": "images/img_001.png"},
                    ],
                    ensure_ascii=False,
                ),
                headers={"Content-Type": "application/json"},
            )
        if url.endswith("/images/img_001.png"):
            return _FakeResponse(content=self.image_bytes)
        raise AssertionError(f"unexpected GET {url}")


@pytest.mark.offline
async def test_client_flat_json_round_trip(
    tmp_path: Path,
    fake_httpx: type,
    env: None,
) -> None:
    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDFBYTES" * 200)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _FlatJSONDispatcher()
    client = MinerURawClient()
    manifest = await client.download_into(raw, src)

    # Critical file present + manifest valid post-download.
    assert (raw / "content_list.json").is_file()
    assert (raw / "_manifest.json").is_file()
    assert (raw / "images" / "img_001.png").read_bytes() == b"\x89PNGfakeimg"
    assert manifest.task_id == "T-1"
    assert manifest.critical_file.path == "content_list.json"
    assert any(f.path == "images/img_001.png" for f in manifest.files)

    # The full validator agrees the bundle is intact.
    assert is_bundle_valid(raw, src) is True


# ---------------------------------------------------------------------------
# zip mode: result_url is a zip; client extracts under raw_dir
# ---------------------------------------------------------------------------


class _ZipDispatcher(_Dispatcher):
    def __init__(self) -> None:
        self.polls = 0
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "content_list.json",
                json.dumps([{"type": "text", "text": "hi"}]),
            )
            zf.writestr("images/img_001.png", b"\x89PNGfakezip")
            zf.writestr("full.md", "# Title\n")
        self.zip_bytes = buf.getvalue()

    def post(self, url: str, **_: Any) -> _FakeResponse:
        if url.endswith("/task"):
            return _FakeResponse(
                text=json.dumps({"task_id": "T-2", "status": "queued"})
            )
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url: str, **_: Any) -> _FakeResponse:
        if "/task/T-2" in url:
            self.polls += 1
            return _FakeResponse(
                text=json.dumps(
                    {
                        "task_id": "T-2",
                        "status": "done",
                        "result_url": "http://mineru.example/results/T-2/bundle.zip",
                    }
                )
            )
        if url.endswith("/bundle.zip"):
            return _FakeResponse(
                content=self.zip_bytes,
                headers={"Content-Type": "application/zip"},
            )
        raise AssertionError(f"unexpected GET {url}")


@pytest.mark.offline
async def test_client_zip_mode_extracts_bundle(
    tmp_path: Path,
    fake_httpx: type,
    env: None,
) -> None:
    src = tmp_path / "demo.pdf"
    src.write_bytes(b"PDFBYTES" * 200)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _ZipDispatcher()
    client = MinerURawClient()
    manifest = await client.download_into(raw, src)

    assert (raw / "content_list.json").is_file()
    assert (raw / "images" / "img_001.png").read_bytes() == b"\x89PNGfakezip"
    assert (raw / "full.md").is_file()
    assert manifest.task_id == "T-2"
    paths = sorted(f.path for f in manifest.files)
    assert "full.md" in paths
    assert "images/img_001.png" in paths


# ---------------------------------------------------------------------------
# Manifest is *atomic*: presence implies fully written
# ---------------------------------------------------------------------------


@pytest.mark.offline
async def test_client_manifest_written_atomically(
    tmp_path: Path,
    fake_httpx: type,
    env: None,
) -> None:
    src = tmp_path / "demo.pdf"
    src.write_bytes(b"X" * 16)
    raw = tmp_path / "demo.mineru_raw"
    raw.mkdir()

    _CURRENT.dispatcher = _FlatJSONDispatcher()
    await MinerURawClient().download_into(raw, src)

    # No leftover .tmp marker; only the final _manifest.json should exist.
    leftovers = list(raw.glob("_manifest*"))
    assert leftovers == [raw / "_manifest.json"]
