"""G0 wiring tests for the native ``smart_heading`` engine parameter.

Covers the parameter plumbing added ahead of the smart-heading algorithm
itself: ``parse_engine`` decode at parse start, the ``NativeExtractRuntime``
handed to ``extract``, persist-time re-encode, loud failure on malformed
directives, and the markdown warn-and-ignore path.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
from lightrag.parser.base import ParseContext, ParseResult
from lightrag.parser.debug import build_debug_rag
from lightrag.parser.registry import get_parser
from lightrag.parser.routing import decode_parse_engine

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _propagate_lightrag_logs():
    """The ``lightrag`` logger sets propagate=False; caplog needs it on."""
    lg = logging.getLogger("lightrag")
    old = lg.propagate
    lg.propagate = True
    try:
        yield
    finally:
        lg.propagate = old


_STUB_BLOCKS = [
    {
        "uuid": "p1",
        "heading": "Chapter One",
        "content": "# Chapter One\nBody text.",
        "type": "text",
        "parent_headings": [],
        "level": 1,
    }
]


def _stub_extract(
    file_path,
    *,
    drawing_context=None,
    parse_warnings=None,
    parse_metadata=None,
    **_kwargs,
):
    return [dict(b) for b in _STUB_BLOCKS]


def _parse_docx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    parse_engine: Any = "__absent__",
) -> tuple[Any, ParseResult, list[dict[str, Any]]]:
    """Drive ``get_parser("native").parse`` on a stub docx.

    Returns ``(rag, result, extract_calls)`` where ``extract_calls`` records
    the keyword arguments (notably ``runtime``) each ``extract`` call saw.
    """
    from lightrag.parser.docx.parser import NativeDocxParser

    input_dir = tmp_path / "inputs"
    input_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    source_path = input_dir / "doc.docx"
    source_path.write_bytes(b"fake-docx")

    content_data: dict[str, Any] = {
        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
        "content": "",
    }
    if parse_engine != "__absent__":
        content_data["parse_engine"] = parse_engine

    extract_calls: list[dict[str, Any]] = []
    orig_extract = NativeDocxParser.extract

    def _spy_extract(self, source, **kwargs):
        extract_calls.append(dict(kwargs))
        return orig_extract(self, source, **kwargs)

    rag = build_debug_rag()

    with (
        mock.patch.object(NativeDocxParser, "extract", _spy_extract),
        mock.patch(
            "lightrag.parser.docx.parse_document.extract_docx_blocks",
            _stub_extract,
        ),
    ):
        result = asyncio.run(
            get_parser("native").parse(
                ParseContext(rag, "doc-1", str(source_path), content_data)
            )
        )
    return rag, result, extract_calls


def test_decode_smart_heading_param() -> None:
    engine, params, errs = decode_parse_engine("native(smart_heading=true)")
    assert engine == "native"
    assert params == {"smart_heading": True}
    assert errs == []


def test_no_params_persists_bare_engine(tmp_path, monkeypatch) -> None:
    rag, result, calls = _parse_docx(tmp_path, monkeypatch)
    assert rag.full_docs.data["doc-1"]["parse_engine"] == "native"
    assert result.parse_engine == "native"
    assert len(calls) == 1
    runtime = calls[0]["runtime"]
    assert dict(runtime.engine_params) == {}
    assert runtime.llm_invoke is None


def test_none_parse_engine_persists_bare_engine(tmp_path, monkeypatch) -> None:
    rag, _result, _calls = _parse_docx(tmp_path, monkeypatch, parse_engine=None)
    assert rag.full_docs.data["doc-1"]["parse_engine"] == "native"


def test_bare_native_persists_bare_engine(tmp_path, monkeypatch) -> None:
    rag, _result, calls = _parse_docx(tmp_path, monkeypatch, parse_engine="native")
    assert rag.full_docs.data["doc-1"]["parse_engine"] == "native"
    assert dict(calls[0]["runtime"].engine_params) == {}


def test_smart_heading_param_reaches_extract_and_persists(
    tmp_path, monkeypatch
) -> None:
    rag, _result, calls = _parse_docx(
        tmp_path, monkeypatch, parse_engine="native(smart_heading=true)"
    )
    # G0-4: persist re-encodes the directive so the stored value keeps params.
    assert rag.full_docs.data["doc-1"]["parse_engine"] == "native(smart_heading=true)"
    # G0-2: the decoded params reach extract via the runtime.
    assert dict(calls[0]["runtime"].engine_params) == {"smart_heading": True}


def test_malformed_parse_engine_fails_loudly(tmp_path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="invalid parse_engine"):
        _parse_docx(tmp_path, monkeypatch, parse_engine="native(smart_heading=true")


def test_unknown_param_fails_loudly(tmp_path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="invalid parse_engine"):
        _parse_docx(tmp_path, monkeypatch, parse_engine="native(bogus=1)")


def test_wants_llm_bridge_gates_on_smart_heading() -> None:
    from lightrag.parser.docx.parser import NativeDocxParser
    from lightrag.parser.markdown.parser import NativeMarkdownParser

    docx = NativeDocxParser()
    assert docx.wants_llm_bridge({"smart_heading": True}) is True
    assert docx.wants_llm_bridge({"smart_heading": False}) is False
    assert docx.wants_llm_bridge({}) is False
    assert NativeMarkdownParser().wants_llm_bridge({"smart_heading": True}) is False


def test_markdown_warns_and_ignores_params(tmp_path, monkeypatch, caplog) -> None:
    """G0-3: a .md routed with smart_heading parses normally, warns once."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    source_path = input_dir / "notes.md"
    source_path.write_text("# Title\n\nSome body text.\n", encoding="utf-8")

    rag = build_debug_rag()
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        result = asyncio.run(
            get_parser("native").parse(
                ParseContext(
                    rag,
                    "doc-md",
                    str(source_path),
                    {
                        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                        "content": "",
                        "parse_engine": "native(smart_heading=true)",
                    },
                )
            )
        )

    assert (
        result.parse_warnings
        and result.parse_warnings.get("engine_params_ignored") == 1
    )
    assert any("only apply to .docx" in rec.message for rec in caplog.records), (
        "expected a warn-and-ignore log line"
    )
    # Parsing succeeded and the directive is still persisted verbatim.
    assert rag.full_docs.data["doc-md"]["parse_engine"] == "native(smart_heading=true)"
    assert "Title" in rag.full_docs.data["doc-md"]["content"]


def test_markdown_does_not_warn_on_falsy_param(tmp_path, monkeypatch, caplog) -> None:
    """Review: a blanket opt-out (smart_heading=false) turned nothing on, so a
    .md routed through it must NOT emit the warn-and-ignore noise."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    source_path = input_dir / "notes.md"
    source_path.write_text("# Title\n\nSome body text.\n", encoding="utf-8")

    rag = build_debug_rag()
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        result = asyncio.run(
            get_parser("native").parse(
                ParseContext(
                    rag,
                    "doc-md-off",
                    str(source_path),
                    {
                        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                        "content": "",
                        "parse_engine": "native(smart_heading=false)",
                    },
                )
            )
        )

    assert not (result.parse_warnings or {}).get("engine_params_ignored")
    assert not any("only apply to .docx" in rec.message for rec in caplog.records)
    assert "Title" in rag.full_docs.data["doc-md-off"]["content"]


def test_i4_cache_disabled_surfaces_parse_warning(tmp_path, monkeypatch) -> None:
    """A15 (§2.3.5 channel a): the I4 determinism waiver reaches
    parse_warnings → doc_status, not just the process log."""

    async def _fake_llm(prompt: str, **_kw) -> str:
        return "{}"

    input_dir = tmp_path / "inputs"
    input_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    source_path = input_dir / "doc.docx"
    source_path.write_bytes(b"fake-docx")

    content_data: dict[str, Any] = {
        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
        "content": "",
        "parse_engine": "native(smart_heading=true)",
    }

    rag = build_debug_rag(extract_llm_func=_fake_llm)
    orig_config = rag._build_global_config
    rag._build_global_config = lambda: {
        **orig_config(),
        "enable_llm_cache_for_entity_extract": False,
    }

    with mock.patch(
        "lightrag.parser.docx.parse_document.extract_docx_blocks",
        _stub_extract,
    ):
        result = asyncio.run(
            get_parser("native").parse(
                ParseContext(rag, "doc-1", str(source_path), content_data)
            )
        )
    assert (result.parse_warnings or {}).get("smart_i4_cache_disabled") == 1
