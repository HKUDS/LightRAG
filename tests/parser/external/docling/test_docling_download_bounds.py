"""Regression tests for DoclingRawClient._download_result_into's
download-safety bounds (issue #3610, remaining scope after commit
f33fe54 already added safe_extract_zip's declared-size/entry-count caps).

Mirrors tests/parser/external/mineru/test_mineru_download_bounds.py --
same two behaviours, same client-side gap, same fix (shared
_common.py helpers), different engine.

Calls _download_result_into directly against a real httpx.AsyncClient
backed by httpx.MockTransport, rather than through the full
submit/poll/download round trip.
"""

import asyncio
import io
import time
import zipfile

import httpx
import pytest

from lightrag.parser.external.docling.client import DoclingRawClient

pytestmark = pytest.mark.offline


def _small_valid_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("content_list.json", "[]")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _docling_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")


@pytest.mark.asyncio
async def test_download_result_deadline_cuts_off_a_slow_peer(tmp_path, monkeypatch):
    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.3)
        return httpx.Response(
            200, content=_small_valid_zip(), headers={"content-type": "application/zip"}
        )

    monkeypatch.setenv("PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT", "0.05")
    transport = httpx.MockTransport(slow_handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    client_obj = DoclingRawClient()

    start = time.monotonic()
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="wall-clock deadline"):
            await client_obj._download_result_into(
                client, "task-1", raw_dir, "demo.pdf"
            )
    elapsed = time.monotonic() - start

    assert elapsed < 0.2, (
        f"expected the deadline to cut the call off well before the "
        f"mocked 0.3s response, took {elapsed:.3f}s instead"
    )


@pytest.mark.asyncio
async def test_download_result_disabled_deadline_lets_a_slow_peer_finish(
    tmp_path, monkeypatch
):
    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(
            200, content=_small_valid_zip(), headers={"content-type": "application/zip"}
        )

    monkeypatch.setenv("PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT", "0")
    transport = httpx.MockTransport(slow_handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    client_obj = DoclingRawClient()

    async with httpx.AsyncClient(transport=transport) as client:
        await client_obj._download_result_into(client, "task-1", raw_dir, "demo.pdf")

    assert (raw_dir / "content_list.json").is_file()


@pytest.mark.asyncio
async def test_download_result_external_cancellation_is_not_a_timeout(
    tmp_path, monkeypatch
):
    """asyncio.timeout() must distinguish its own deadline firing from an
    unrelated external cancellation. Explicitly proves all five contract
    points: CancelledError escapes; it is not converted to TimeoutError or
    any parser-specific RuntimeError; no archive content is extracted; the
    response stream closes; the client context closes.

    Headers arrive and one chunk is delivered before the body hangs --
    matching a peer that responds but stalls mid-transfer, not one that
    never responds at all -- so a real response object exists to check
    for proper closure, not just the client."""
    captured: dict = {}

    async def hanging_body():
        yield b"partial-chunk-before-the-stall"
        await asyncio.Event().wait()  # never set: stalls forever
        yield b"unreachable"  # pragma: no cover

    async def hanging_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=hanging_body(), headers={"content-type": "application/zip"}
        )

    # Disabled deadline: isolates external cancellation from the deadline
    # contract covered by test_download_result_deadline_cuts_off_a_slow_peer
    # -- these are different contracts and must not be conflated.
    monkeypatch.setenv("PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT", "0")
    transport = httpx.MockTransport(hanging_handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    client_obj = DoclingRawClient()
    client = httpx.AsyncClient(transport=transport)
    real_stream = client.stream

    class _CapturingStreamCM:
        """Wraps httpx's stream context manager to capture the entered
        response. A plain instance-attribute override of __aenter__ on
        the real context manager wouldn't be honoured by `async with`,
        which looks up dunder methods on the type, not the instance."""

        def __init__(self, inner):
            self._inner = inner

        async def __aenter__(self):
            response = await self._inner.__aenter__()
            captured["response"] = response
            return response

        async def __aexit__(self, *exc):
            return await self._inner.__aexit__(*exc)

    def _spy_stream(*args, **kwargs):
        return _CapturingStreamCM(real_stream(*args, **kwargs))

    client.stream = _spy_stream

    async def run() -> None:
        async with client:
            await client_obj._download_result_into(
                client, "task-1", raw_dir, "demo.pdf"
            )

    task = asyncio.create_task(run())
    await asyncio.sleep(0.05)  # let it receive headers + first chunk, then stall
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    # 1. CancelledError specifically escapes.
    assert exc_info.type is asyncio.CancelledError

    # 2. Not converted to TimeoutError or our deadline's RuntimeError:
    # if _download_result_into's `except TimeoutError` had mis-caught
    # this cancellation, pytest.raises(CancelledError) above would
    # already have failed with a type mismatch -- this is the proof.

    # 3. No archive content extracted.
    assert not (raw_dir / "content_list.json").exists(), (
        "expected no extraction -- the body is only validated and "
        "extracted after being fully consumed, which never happened"
    )

    # 4. The response stream closed.
    response = captured.get("response")
    assert response is not None, "expected the stream to have been entered"
    assert response.is_closed, (
        "expected the response stream's async-with block to close it on cancellation"
    )

    # 5. The client context closed.
    assert client.is_closed, (
        "expected the client's async-with block to close it on "
        "cancellation, same as any other unwound exception"
    )


@pytest.mark.asyncio
async def test_download_result_rejects_oversized_response_before_full_buffer(
    tmp_path, monkeypatch
):
    oversized_zip = _small_valid_zip() + b"\x00" * 10_000

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=oversized_zip, headers={"content-type": "application/zip"}
        )

    monkeypatch.setenv("PARSER_RESULT_BUNDLE_DOWNLOAD_MAX_BYTES", "100")
    transport = httpx.MockTransport(handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    client_obj = DoclingRawClient()

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="exceeds 100 bytes"):
            await client_obj._download_result_into(
                client, "task-1", raw_dir, "demo.pdf"
            )


@pytest.mark.asyncio
async def test_download_result_wire_cap_is_independent_of_uncompressed_cap(
    tmp_path, monkeypatch
):
    """PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES (uncompressed, checked by
    safe_extract_zip against the zip's declared size) and
    PARSER_RESULT_BUNDLE_DOWNLOAD_MAX_BYTES (compressed wire bytes, checked
    by stream_capped_get) must be independently configurable. Setting the
    uncompressed cap far below the response's actual wire size must NOT
    trip the streaming cap -- the failure must come from safe_extract_zip's
    later, distinctly-worded check, proving the streaming step never even
    consulted PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES."""
    small_zip = _small_valid_zip()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=small_zip, headers={"content-type": "application/zip"}
        )

    monkeypatch.setenv("PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES", "1")
    transport = httpx.MockTransport(handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    client_obj = DoclingRawClient()

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="uncompressed size"):
            await client_obj._download_result_into(
                client, "task-1", raw_dir, "demo.pdf"
            )


@pytest.mark.asyncio
async def test_download_result_declared_size_lie_still_caught_when_raw_is_small(
    tmp_path, monkeypatch
):
    """Defense-in-depth: a small-on-the-wire zip that lies about its own
    declared uncompressed size must still be caught by safe_extract_zip's
    independent check, distinct from the new streaming cap."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("content_list.json", "[]")
        info = zf.infolist()[-1]
        info.file_size = 10 * 1024 * 1024 * 1024  # lies: declares 10 GiB
    lying_zip = buf.getvalue()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=lying_zip, headers={"content-type": "application/zip"}
        )

    monkeypatch.setenv("PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES", str(1024 * 1024))
    transport = httpx.MockTransport(handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    client_obj = DoclingRawClient()

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="uncompressed size"):
            await client_obj._download_result_into(
                client, "task-1", raw_dir, "demo.pdf"
            )


class _GenuineStream(httpx.AsyncByteStream):
    """A real streamed byte source with nothing preloaded.

    ``httpx.Response(..., content=b"...")`` pre-populates ``_content`` and
    marks the response already-read (``is_stream_consumed=True``) at
    *construction* time -- before any client or transport touches it. A
    test built on that never exercises a genuinely unread response and
    cannot catch a regression in how stream_capped_get's manual
    ``aiter_bytes()`` consumption interacts with httpx's read-tracking.
    This stream starts unread and only becomes "read" through actual
    iteration, matching what a real network response looks like.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type,chunks,expected_detail",
    [
        (
            "text/plain",
            [b"disk full", b": no space for temp file"],
            "disk full: no space for temp file",
        ),
        (
            "application/json",
            [b'{"error": "', b'disk full"}'],
            '{"error": "disk full"}',
        ),
    ],
    ids=["text-body", "json-body"],
)
async def test_download_result_http_error_detail_survives_genuine_streaming(
    tmp_path, content_type, chunks, expected_detail
):
    """A non-2xx response's body text must reach the raised error message
    after being read via stream_capped_get's manual
    response.aiter_bytes() loop, for a genuinely unread stream (see
    _GenuineStream). Without stream_capped_get's collected body being
    threaded into raise_for_status_with_detail, this response is never
    marked read in a way httpx accepts for .text/.json(), so accessing
    either raises httpx.ResponseNotRead -- which is itself a RuntimeError
    subclass (via httpx.StreamError) and would otherwise silently
    masquerade as this function's own intended error. The strict
    ``type(...) is RuntimeError`` check below is what actually catches
    that regression; a bare ``pytest.raises(RuntimeError)`` would not,
    since ResponseNotRead IS a RuntimeError."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            stream=_GenuineStream(chunks),
            headers={"content-type": content_type},
        )

    transport = httpx.MockTransport(handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    client_obj = DoclingRawClient()

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError) as excinfo:
            await client_obj._download_result_into(
                client, "task-1", raw_dir, "demo.pdf"
            )

    assert type(excinfo.value) is RuntimeError, (
        f"expected our own RuntimeError, got {type(excinfo.value).__name__} "
        f"(httpx.ResponseNotRead escaping disguised as RuntimeError would "
        f"also pass a bare pytest.raises(RuntimeError) check)"
    )
    message = str(excinfo.value)
    assert "Docling result task-1 download failed: HTTP 500" in message
    assert expected_detail in message
    assert not (raw_dir / "content_list.json").exists()
