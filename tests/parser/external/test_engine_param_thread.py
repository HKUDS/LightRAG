"""Proof that engine params thread identically into both external-parser hooks.

The ONLY test exercising ``ExternalParserBase.parse``'s decode + thread: the
per-file engine params decoded from ``content_data['parse_engine']`` must reach
BOTH ``is_bundle_valid`` (cache-hit check) and ``download_into`` (request), or a
cache signature could be computed with different params than the bundle was
parsed with.  Also asserts a malformed stored directive fails the doc loudly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightrag.parser.base import ParseContext
from lightrag.parser.external._base import ExternalParserBase


class _Stop(Exception):
    """Sentinel to halt parse() right after the hook under test runs."""


def _make_ctx(tmp_path: Path, parse_engine: str) -> ParseContext:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4 test")
    ctx = ParseContext(
        rag=SimpleNamespace(),
        doc_id="doc-1",
        file_path=str(source),
        content_data={"parse_engine": parse_engine},
    )
    # Avoid the pipeline-layer source resolution; hand the template a simple
    # resolved-source stand-in pointing at our temp file.
    rs = SimpleNamespace(
        source_path=source, parsed_dir=tmp_path / "parsed", document_name="doc.pdf"
    )
    ctx.resolve = lambda engine_name: rs  # type: ignore[assignment]
    return ctx


class _RecordingParser(ExternalParserBase):
    engine_name = "mineru"  # registered engine that accepts page_range
    raw_dir_suffix = ".raw"
    force_reparse_env = "LIGHTRAG_TEST_FORCE_REPARSE"

    def __init__(self, *, hit: bool) -> None:
        self._hit = hit
        self.seen: dict[str, object] = {}
        self.download_called = False

    def is_bundle_valid(self, raw_dir, source_path, *, engine_params=None):
        self.seen["is_bundle_valid"] = engine_params
        return self._hit

    async def download_into(
        self, raw_dir, source_path, *, upload_name, engine_params=None
    ):
        self.seen["download_into"] = engine_params
        self.download_called = True
        raise _Stop()

    def build_ir(self, raw_dir, document_name):  # hit path halts here
        raise _Stop()


def test_miss_path_threads_identical_engine_params(tmp_path):
    ctx = _make_ctx(tmp_path, "mineru(page_range=1-3)")
    parser = _RecordingParser(hit=False)
    with pytest.raises(_Stop):
        asyncio.run(parser.parse(ctx))
    assert parser.download_called is True
    # Both hooks received the SAME decoded params dict.
    assert parser.seen["is_bundle_valid"] == {"page_range": "1-3"}
    assert parser.seen["download_into"] == {"page_range": "1-3"}


def test_hit_path_skips_client(tmp_path):
    ctx = _make_ctx(tmp_path, "mineru(page_range=1-3)")
    parser = _RecordingParser(hit=True)
    with pytest.raises(_Stop):  # halts in build_ir
        asyncio.run(parser.parse(ctx))
    assert parser.download_called is False
    assert parser.seen["is_bundle_valid"] == {"page_range": "1-3"}


def test_bare_engine_threads_none(tmp_path):
    ctx = _make_ctx(tmp_path, "mineru")
    parser = _RecordingParser(hit=False)
    with pytest.raises(_Stop):
        asyncio.run(parser.parse(ctx))
    assert parser.seen["is_bundle_valid"] is None
    assert parser.seen["download_into"] is None


def test_malformed_stored_parse_engine_fails_loudly(tmp_path):
    ctx = _make_ctx(tmp_path, "mineru(page_range=")  # unbalanced
    parser = _RecordingParser(hit=False)
    with pytest.raises(ValueError, match="invalid parse_engine"):
        asyncio.run(parser.parse(ctx))
    # Failed before reaching either hook.
    assert "is_bundle_valid" not in parser.seen
