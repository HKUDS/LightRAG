"""Offline tests for per-sidecar item concurrency and failure propagation in
``analyze_multimodal``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.constants import MM_ANALYSIS_ITEM_CONCURRENCY_FACTOR
from lightrag.exceptions import MultimodalAnalysisError
from lightrag.utils import EmbeddingFunc, Tokenizer

from .test_pipeline_analyze_multimodal import PNG_BYTES

pytestmark = pytest.mark.offline

VLM_MAX_ASYNC = 1
ITEM_COUNT = 12


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 8)


async def _ok_vlm(prompt, **kwargs):
    await asyncio.sleep(0.005)
    return json.dumps({"name": "fig", "type": "Chart", "description": "ok"})


def _build_rag(tmp_path: Path) -> LightRAG:
    role_configs = {
        spec.name: (
            RoleLLMConfig(func=_ok_vlm, max_async=VLM_MAX_ASYNC)
            if spec.name == "vlm"
            else RoleLLMConfig()
        )
        for spec in ROLES
    }
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"mm-items-{tmp_path.name}",
        llm_model_func=_ok_vlm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=1024, func=_mock_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        vlm_process_enable=True,
        enable_llm_cache_for_entity_extract=True,
        role_llm_configs=role_configs,
    )


async def _shutdown_role_workers(rag: LightRAG) -> None:
    for func in rag.role_llm_funcs.values():
        await rag._shutdown_llm_wrapper(func)


def _write_drawings(tmp_path: Path, count: int) -> tuple[dict, Path]:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    blocks_path = parsed_dir / "doc.blocks.jsonl"
    blocks_path.write_text(
        json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n", encoding="utf-8"
    )
    drawings = {}
    for idx in range(count):
        image = parsed_dir / f"im-{idx:02d}.png"
        image.write_bytes(PNG_BYTES)
        drawings[f"im-{idx:02d}"] = {"caption": f"figure {idx}", "path": str(image)}
    sidecar_path = parsed_dir / "doc.drawings.json"
    sidecar_path.write_text(json.dumps({"drawings": drawings}), encoding="utf-8")
    return {"blocks_path": str(blocks_path)}, sidecar_path


def _pipeline_status() -> tuple[dict, asyncio.Lock]:
    return (
        {
            "busy": True,
            "history_messages": [],
            "latest_message": "",
            "cancellation_requested": False,
        },
        asyncio.Lock(),
    )


async def _analyze(rag: LightRAG, parsed_data: dict) -> dict:
    status, lock = _pipeline_status()
    return await asyncio.wait_for(
        rag.analyze_multimodal(
            doc_id="doc-1",
            file_path="fixture.pdf",
            parsed_data=parsed_data,
            process_options="i",
            pipeline_status=status,
            pipeline_status_lock=lock,
        ),
        timeout=30.0,
    )


@pytest.mark.asyncio
async def test_cache_reads_are_bounded_by_role_concurrency(tmp_path):
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        parsed_data, sidecar_path = _write_drawings(tmp_path, ITEM_COUNT)
        cache = rag.llm_response_cache
        original_get = cache.get_by_id
        in_flight = {"now": 0, "peak": 0}

        async def tracking_get(key):
            in_flight["now"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
            try:
                await asyncio.sleep(0.01)
                return await original_get(key)
            finally:
                in_flight["now"] -= 1

        cache.get_by_id = tracking_get

        result = await _analyze(rag, parsed_data)

        assert result.get("multimodal_processed") is True
        # Equality, not just an upper bound: the cap must limit concurrency,
        # not serialise it. With ITEM_COUNT well above the bound, every slot
        # is taken at once, so a bound accidentally shrunk to 1 (or to one
        # item per role worker) fails here too.
        assert ITEM_COUNT > VLM_MAX_ASYNC * MM_ANALYSIS_ITEM_CONCURRENCY_FACTOR
        assert in_flight["peak"] == (
            VLM_MAX_ASYNC * MM_ANALYSIS_ITEM_CONCURRENCY_FACTOR
        ), in_flight
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert all(
            item["llm_analyze_result"]["status"] == "success"
            for item in payload["drawings"].values()
        )
    finally:
        await _shutdown_role_workers(rag)
        await rag.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_op", ["get_by_id", "upsert"])
async def test_unexpected_cache_error_fails_analysis_with_root_cause(
    tmp_path, failing_op
):
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        parsed_data, sidecar_path = _write_drawings(tmp_path, ITEM_COUNT)
        cache = rag.llm_response_cache
        original = getattr(cache, failing_op)
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("Too many connections")
            await asyncio.sleep(0.01)
            return await original(*args, **kwargs)

        setattr(cache, failing_op, flaky)

        with pytest.raises(MultimodalAnalysisError) as excinfo:
            await _analyze(rag, parsed_data)

        assert "Too many connections" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, RuntimeError)

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        results = {
            item_id: item["llm_analyze_result"]
            for item_id, item in payload["drawings"].items()
        }
        failed = [
            item_id
            for item_id, r in results.items()
            if r["status"] == "failure" and "Too many connections" in r["message"]
        ]
        assert len(failed) == 1
        assert f"drawings/{failed[0]}" in str(excinfo.value)
        cancelled = [
            r["message"]
            for r in results.values()
            if r["status"] == "failure" and r["message"].startswith("cancelled")
        ]
        assert cancelled
        assert all(f"sibling drawings/{failed[0]} failed" in m for m in cancelled)
    finally:
        await _shutdown_role_workers(rag)
        await rag.finalize_storages()


def _write_tables(tmp_path: Path, count: int) -> tuple[dict, Path]:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    blocks_path = parsed_dir / "doc.blocks.jsonl"
    blocks_path.write_text(
        json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n", encoding="utf-8"
    )
    tables = {
        f"tb-{idx:02d}": {
            "caption": f"table {idx}",
            "format": "html",
            "content": f"<table><tr><td>cell {idx}</td></tr></table>",
        }
        for idx in range(count)
    }
    sidecar_path = parsed_dir / "doc.tables.json"
    sidecar_path.write_text(json.dumps({"tables": tables}), encoding="utf-8")
    return {"blocks_path": str(blocks_path)}, sidecar_path


@pytest.mark.asyncio
async def test_unexpected_table_error_names_item_like_its_own_errors(tmp_path):
    """The wrapped error and the sibling markers name a table the way the
    table's own errors do (``table/<id>``, from ``kind``), not by the sidecar's
    plural root key (``tables/<id>``), so one item is never spelled two ways
    in the same document's messages."""
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        parsed_data, sidecar_path = _write_tables(tmp_path, ITEM_COUNT)
        cache = rag.llm_response_cache
        original_get = cache.get_by_id
        calls = {"n": 0}

        async def flaky_get(key):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("Too many connections")
            await asyncio.sleep(0.01)
            return await original_get(key)

        cache.get_by_id = flaky_get

        status, lock = _pipeline_status()
        with pytest.raises(MultimodalAnalysisError) as excinfo:
            await asyncio.wait_for(
                rag.analyze_multimodal(
                    doc_id="doc-1",
                    file_path="fixture.pdf",
                    parsed_data=parsed_data,
                    process_options="t",
                    pipeline_status=status,
                    pipeline_status_lock=lock,
                ),
                timeout=30.0,
            )

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        results = {
            item_id: item["llm_analyze_result"]
            for item_id, item in payload["tables"].items()
        }
        failed = [
            item_id
            for item_id, r in results.items()
            if r["status"] == "failure" and "Too many connections" in r["message"]
        ]
        assert len(failed) == 1, results
        message = str(excinfo.value)
        assert message.startswith(f"table/{failed[0]}: unexpected error: "), message
        assert "RuntimeError" in message
        cancelled = [
            r["message"]
            for r in results.values()
            if r["message"].startswith("cancelled")
        ]
        assert cancelled, results
        assert all(
            m == f"cancelled (sibling table/{failed[0]} failed)" for m in cancelled
        )
        assert not any("tables/" in r["message"] for r in results.values()), results
    finally:
        await _shutdown_role_workers(rag)
        await rag.finalize_storages()
