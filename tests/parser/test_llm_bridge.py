"""Tests for the parser LLM sync bridge, executor lifecycle, and cache-id plumbing."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest import mock

import numpy as np
import pytest

from lightrag.parser.llm_bridge import (
    LLMBridgeCancelled,
    LLMBridgeShutdown,
    SyncLLMBridge,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# SyncLLMBridge unit tests
# ---------------------------------------------------------------------------


async def _call_bridge_in_thread(bridge: SyncLLMBridge, *args, **kwargs):
    return await asyncio.to_thread(bridge, *args, **kwargs)


async def test_bridge_returns_result() -> None:
    loop = asyncio.get_running_loop()

    async def submit(prompt: str, *, system_prompt: str | None = None) -> str:
        return f"echo:{prompt}:{system_prompt}"

    bridge = SyncLLMBridge(loop, submit, poll_interval=0.05)
    result = await _call_bridge_in_thread(bridge, "hello", system_prompt="sys")
    assert result == "echo:hello:sys"


async def test_bridge_polls_through_slow_llm() -> None:
    loop = asyncio.get_running_loop()

    async def submit(prompt: str, *, system_prompt: str | None = None) -> str:
        await asyncio.sleep(0.18)  # several 0.05s poll slices
        return "slow-ok"

    bridge = SyncLLMBridge(loop, submit, poll_interval=0.05)
    assert await _call_bridge_in_thread(bridge, "p") == "slow-ok"


async def test_bridge_cancel_event_aborts_within_poll_interval() -> None:
    loop = asyncio.get_running_loop()
    cancel = threading.Event()
    started = asyncio.Event()

    async def submit(prompt: str, *, system_prompt: str | None = None) -> str:
        started.set()
        await asyncio.Future()  # never completes
        raise AssertionError("unreachable")

    bridge = SyncLLMBridge(loop, submit, cancel_events=(cancel,), poll_interval=0.05)
    task = asyncio.create_task(_call_bridge_in_thread(bridge, "p"))
    await started.wait()
    t0 = time.monotonic()
    cancel.set()
    with pytest.raises(LLMBridgeCancelled):
        await task
    # Exit within a couple of poll slices, not an unbounded wait.
    assert time.monotonic() - t0 < 1.0


async def test_bridge_pre_cancelled_never_submits() -> None:
    loop = asyncio.get_running_loop()
    cancel = threading.Event()
    cancel.set()
    calls: list[str] = []

    async def submit(prompt: str, *, system_prompt: str | None = None) -> str:
        calls.append(prompt)
        return "x"

    bridge = SyncLLMBridge(loop, submit, cancel_events=(cancel,), poll_interval=0.05)
    with pytest.raises(LLMBridgeCancelled):
        await _call_bridge_in_thread(bridge, "p")
    assert calls == []


async def test_bridge_preserves_shutdown_cancellation_source() -> None:
    loop = asyncio.get_running_loop()
    shutdown = threading.Event()
    shutdown.set()

    async def submit(prompt: str, *, system_prompt: str | None = None) -> str:
        raise AssertionError("shutdown must prevent submission")

    bridge = SyncLLMBridge(
        loop,
        submit,
        cancel_events=((shutdown, LLMBridgeShutdown),),
        poll_interval=0.05,
    )
    with pytest.raises(LLMBridgeShutdown):
        await _call_bridge_in_thread(bridge, "p")


async def test_bridge_loop_thread_call_raises() -> None:
    loop = asyncio.get_running_loop()

    async def submit(prompt: str, *, system_prompt: str | None = None) -> str:
        return "x"

    bridge = SyncLLMBridge(loop, submit, poll_interval=0.05)
    with pytest.raises(RuntimeError, match="event-loop thread"):
        bridge("p")  # called on the loop thread → immediate error, no deadlock


# ---------------------------------------------------------------------------
# ParseResult / carry-over cache-id plumbing
# ---------------------------------------------------------------------------


def test_parse_result_to_dict_emits_cache_ids_only_when_present() -> None:
    from lightrag.parser.base import ParseResult

    base = dict(doc_id="d", file_path="f", parse_format="lightrag", content="c")
    without = ParseResult(**base)
    assert "smartheading_llm_cache_ids" not in without.to_dict()

    with_ids = ParseResult(
        **base, smartheading_llm_cache_ids=["default:smartheading:abc"]
    )
    assert with_ids.to_dict()["smartheading_llm_cache_ids"] == [
        "default:smartheading:abc"
    ]


def test_carry_over_whitelist_preserves_smartheading_ids() -> None:
    from lightrag.utils_pipeline import (
        _DOC_STATUS_METADATA_CARRY_OVER_KEYS,
        doc_status_transition_metadata,
    )

    assert "smartheading_llm_cache_ids" in _DOC_STATUS_METADATA_CARRY_OVER_KEYS

    class _Doc:
        metadata = {
            "smartheading_llm_cache_ids": ["default:smartheading:abc"],
            "unrelated": "dropped",
        }

    carried = doc_status_transition_metadata(_Doc())
    assert carried["smartheading_llm_cache_ids"] == ["default:smartheading:abc"]
    assert "unrelated" not in carried


# ---------------------------------------------------------------------------
# End-to-end: bridge reaches extract via the debug rag injection
# ---------------------------------------------------------------------------


def test_injected_llm_reaches_extract_and_is_callable(tmp_path, monkeypatch) -> None:
    from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
    from lightrag.parser.base import ParseContext
    from lightrag.parser.debug import build_debug_rag
    from lightrag.parser.docx.parser import NativeDocxParser
    from lightrag.parser.registry import get_parser

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    source_path = input_dir / "doc.docx"
    source_path.write_bytes(b"fake-docx")

    async def _mock_llm(prompt: str, **kwargs: Any) -> str:
        return f"judged:{prompt[:10]}"

    seen: dict[str, Any] = {}
    orig_extract = NativeDocxParser.extract

    def _spy_extract(self, source, **kwargs):
        runtime = kwargs["runtime"]
        seen["llm_invoke"] = runtime.llm_invoke
        # Call the bridge from the worker thread — the real usage pattern.
        seen["llm_result"] = runtime.llm_invoke("probe prompt")
        return orig_extract(self, source, **kwargs)

    def _stub_blocks(file_path, **_kwargs):
        return [
            {
                "uuid": "p1",
                "heading": "H",
                "content": "# H\nbody",
                "type": "text",
                "parent_headings": [],
                "level": 1,
            }
        ]

    rag = build_debug_rag(extract_llm_func=_mock_llm)
    with (
        mock.patch.object(NativeDocxParser, "extract", _spy_extract),
        mock.patch(
            "lightrag.parser.docx.parse_document.extract_docx_blocks", _stub_blocks
        ),
    ):
        result = asyncio.run(
            get_parser("native").parse(
                ParseContext(
                    rag,
                    "doc-1",
                    str(source_path),
                    {
                        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                        "content": "",
                        "parse_engine": "native(smart_heading=true)",
                    },
                )
            )
        )

    assert seen["llm_invoke"] is not None
    assert seen["llm_result"] == "judged:probe prom"
    # No cache storage on the debug rag → no cache keys minted.
    assert result.smartheading_llm_cache_ids is None


def test_without_injection_bridge_stays_none(tmp_path, monkeypatch) -> None:
    from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
    from lightrag.parser.base import ParseContext
    from lightrag.parser.debug import build_debug_rag
    from lightrag.parser.docx.parser import NativeDocxParser
    from lightrag.parser.registry import get_parser

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    source_path = input_dir / "doc.docx"
    source_path.write_bytes(b"fake-docx")

    seen: dict[str, Any] = {}
    orig_extract = NativeDocxParser.extract

    def _spy_extract(self, source, **kwargs):
        seen["llm_invoke"] = kwargs["runtime"].llm_invoke
        return orig_extract(self, source, **kwargs)

    def _stub_blocks(file_path, **_kwargs):
        return [
            {
                "uuid": "p1",
                "heading": "H",
                "content": "# H",
                "type": "text",
                "parent_headings": [],
                "level": 1,
            }
        ]

    rag = build_debug_rag()
    with (
        mock.patch.object(NativeDocxParser, "extract", _spy_extract),
        mock.patch(
            "lightrag.parser.docx.parse_document.extract_docx_blocks", _stub_blocks
        ),
    ):
        asyncio.run(
            get_parser("native").parse(
                ParseContext(
                    rag,
                    "doc-1",
                    str(source_path),
                    {
                        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                        "content": "",
                        "parse_engine": "native(smart_heading=true)",
                    },
                )
            )
        )

    assert seen["llm_invoke"] is None


# ---------------------------------------------------------------------------
# Per-rag executor lifecycle (G0-6)
# ---------------------------------------------------------------------------


def _new_rag(tmp_path, name: str, max_parallel: int):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

    class _Tok(TokenizerInterface):
        def encode(self, content: str):
            return [ord(c) for c in content]

        def decode(self, tokens):
            return "".join(chr(t) for t in tokens)

    async def _mock_llm(prompt: str, **kwargs: Any) -> str:
        return "{}"

    async def _mock_embed(texts: list[str]) -> np.ndarray:
        return np.random.rand(len(texts), 8)

    work_dir = tmp_path / name
    work_dir.mkdir()
    return LightRAG(
        working_dir=str(work_dir),
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=4096, func=_mock_embed
        ),
        tokenizer=Tokenizer("mock", _Tok()),
        max_parallel_parse_native=max_parallel,
    )


async def test_per_instance_executor_isolation_and_shutdown(tmp_path) -> None:
    rag_a = _new_rag(tmp_path, "a", 2)
    rag_b = _new_rag(tmp_path, "b", 4)

    ex_a = rag_a._get_parse_native_executor()
    ex_b = rag_b._get_parse_native_executor()
    assert ex_a is not ex_b
    assert ex_a._max_workers == 2
    assert ex_b._max_workers == 4
    assert rag_a._get_parse_native_executor() is ex_a  # cached

    # Park a bridge wait on A's executor: it must exit within a poll slice
    # of finalize, not hang the shutdown.
    loop = asyncio.get_running_loop()

    async def _never(prompt: str, *, system_prompt: str | None = None) -> str:
        await asyncio.Future()
        raise AssertionError("unreachable")

    bridge = SyncLLMBridge(
        loop,
        _never,
        cancel_events=(rag_a._parser_shutdown_event,),
        poll_interval=0.05,
    )
    parked = loop.run_in_executor(ex_a, bridge, "p")
    await asyncio.sleep(0.1)

    await rag_a.finalize_storages()
    with pytest.raises(LLMBridgeCancelled):
        await parked
    assert rag_a._parser_executor is None
    # A fresh (unset) event replaced the old one for a later re-init.
    assert not rag_a._parser_shutdown_event.is_set()

    # B is untouched by A's finalize.
    assert rag_b._parser_executor is ex_b
    await rag_b.finalize_storages()

    # No lingering parse-native threads after both shutdowns.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not any(t.name.startswith("parse-native") for t in threading.enumerate()):
            break
        await asyncio.sleep(0.05)
    assert not any(t.name.startswith("parse-native") for t in threading.enumerate())
