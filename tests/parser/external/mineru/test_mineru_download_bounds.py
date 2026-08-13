"""Regression tests for MinerURawClient._download_zip's download-safety
bounds (issue #3610, remaining scope after commit f33fe54 already added
safe_extract_zip's declared-size/entry-count caps).

Two behaviours added on top of that:
- an overall wall-clock deadline around the download, independent of the
  per-read httpx timeout;
- a streaming byte cap that rejects an oversized response mid-download
  instead of buffering it in full first.

Calls _download_zip directly against a real httpx.AsyncClient backed by
httpx.MockTransport, rather than through the full submit/poll/download
round trip -- these tests are about the download step's own safety
bounds, not the MinerU protocol choreography already covered elsewhere
in this directory.
"""

import asyncio
import io
import os
import time
import zipfile

import httpx
import pytest

from lightrag.parser.external.mineru.client import MinerURawClient

pytestmark = pytest.mark.offline


def _small_valid_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("content_list.json", "[]")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _mineru_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.setenv("MINERU_API_TOKEN", "token-123")


@pytest.mark.asyncio
async def test_download_zip_deadline_cuts_off_a_slow_peer(tmp_path):
    """A response that never arrives within the configured deadline must be
    rejected well before httpx's own (much larger) per-read timeout."""

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.3)
        return httpx.Response(200, content=_small_valid_zip())

    os.environ["PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT"] = "0.05"
    try:
        transport = httpx.MockTransport(slow_handler)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        client_obj = MinerURawClient()

        start = time.monotonic()
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(RuntimeError, match="wall-clock deadline"):
                await client_obj._download_zip(client, "https://x/result.zip", raw_dir)
        elapsed = time.monotonic() - start
    finally:
        os.environ.pop("PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT", None)

    assert elapsed < 0.2, (
        f"expected the deadline to cut the call off well before the "
        f"mocked 0.3s response, took {elapsed:.3f}s instead"
    )


@pytest.mark.asyncio
async def test_download_zip_disabled_deadline_lets_a_slow_peer_finish(tmp_path):
    """Control case: a non-positive deadline disables the gate, matching
    the zip-bomb guards' 'non-positive = unlimited' convention -- a slow
    but eventually-successful response must still complete normally."""

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, content=_small_valid_zip())

    os.environ["PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT"] = "0"
    try:
        transport = httpx.MockTransport(slow_handler)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        client_obj = MinerURawClient()

        async with httpx.AsyncClient(transport=transport) as client:
            await client_obj._download_zip(client, "https://x/result.zip", raw_dir)
    finally:
        os.environ.pop("PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT", None)

    assert (raw_dir / "content_list.json").is_file()


@pytest.mark.asyncio
async def test_download_zip_rejects_oversized_response_before_full_buffer(tmp_path):
    """A response larger than the configured budget is rejected by the
    streaming cap, not merely by safe_extract_zip after full buffering."""
    oversized_zip = _small_valid_zip() + b"\x00" * 10_000

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized_zip)

    os.environ["PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES"] = "100"
    try:
        transport = httpx.MockTransport(handler)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        client_obj = MinerURawClient()

        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(RuntimeError, match="exceeds 100 bytes"):
                await client_obj._download_zip(client, "https://x/result.zip", raw_dir)
    finally:
        os.environ.pop("PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES", None)


@pytest.mark.asyncio
async def test_download_zip_external_cancellation_is_not_a_timeout(tmp_path):
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
        return httpx.Response(200, content=hanging_body())

    # Disabled deadline: isolates external cancellation from the deadline
    # contract covered by test_download_zip_deadline_cuts_off_a_slow_peer --
    # these are different contracts and must not be conflated.
    os.environ["PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT"] = "0"
    try:
        transport = httpx.MockTransport(hanging_handler)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        client_obj = MinerURawClient()
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
                await client_obj._download_zip(client, "https://x/result.zip", raw_dir)

        task = asyncio.create_task(run())
        await asyncio.sleep(0.05)  # let it receive headers + first chunk, then stall
        task.cancel()

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        # 1. CancelledError specifically escapes -- pytest.raises already
        # enforces this, but assert the type explicitly per the review
        # requirement rather than relying on it implicitly.
        assert exc_info.type is asyncio.CancelledError

        # 2. Not converted to TimeoutError or our deadline's RuntimeError:
        # if _download_zip's `except TimeoutError` had mis-caught this
        # cancellation, pytest.raises(CancelledError) above would already
        # have failed with a type mismatch -- this is the direct proof.

        # 3. No archive content extracted.
        assert not (raw_dir / "content_list.json").exists(), (
            "expected no extraction -- the body is only validated and "
            "extracted after being fully consumed, which never happened"
        )

        # 4. The response stream closed.
        response = captured.get("response")
        assert response is not None, "expected the stream to have been entered"
        assert response.is_closed, (
            "expected the response stream's async-with block to close it "
            "on cancellation"
        )

        # 5. The client context closed.
        assert client.is_closed, (
            "expected the client's async-with block to close it on "
            "cancellation, same as any other unwound exception"
        )
    finally:
        os.environ.pop("PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT", None)


@pytest.mark.asyncio
async def test_download_zip_declared_size_lie_still_caught_when_raw_is_small(
    tmp_path,
):
    """Defense-in-depth: a zip whose raw bytes fit under the streaming cap
    but whose OWN declared uncompressed size lies about being huge must
    still be caught by safe_extract_zip's independent check."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("content_list.json", "[]")
        info = zf.infolist()[-1]
        info.file_size = 10 * 1024 * 1024 * 1024  # lies: declares 10 GiB
    lying_zip = buf.getvalue()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=lying_zip)

    # Raw bytes are tiny (well under this), so the streaming cap doesn't
    # fire -- only safe_extract_zip's declared-size check should.
    os.environ["PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES"] = str(1024 * 1024)
    try:
        transport = httpx.MockTransport(handler)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        client_obj = MinerURawClient()

        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(RuntimeError, match="uncompressed size"):
                await client_obj._download_zip(client, "https://x/result.zip", raw_dir)
    finally:
        os.environ.pop("PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES", None)
