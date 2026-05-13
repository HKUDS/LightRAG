"""End-to-end offline tests for the VLM analyze_multimodal pipeline.

Covers the three behaviours unique to the unified image_inputs rewrite:

1. ``VLM_PROCESS_ENABLE=False`` short-circuits every multimodal item with a
   warning and never invokes the VLM mock.
2. ``VLM_PROCESS_ENABLE=True`` writes a ``default:analysis:*`` cache entry on
   the first call and serves the second call from cache without calling the
   VLM mock again.
3. The cache entry's ``original_prompt`` carries the ``<vlm_images>`` audit
   block but never embeds the raw base64 payload.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.utils import EmbeddingFunc, Tokenizer


@pytest.fixture
def _propagate_lightrag_logger(monkeypatch):
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


pytestmark = pytest.mark.offline


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8"
    b"\xcf\xc0\x00\x00\x00\x03\x00\x01\x5c\xcc\xd9\x9e\x00\x00\x00\x00"
    b"IEND\xaeB`\x82"
)


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 8)


def _make_vlm_mock(call_log: list[dict]):
    async def vlm_func(prompt, **kwargs):
        call_log.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return json.dumps(
            {
                "name": "fig-1",
                "summary": "short summary",
                "detail_description": "detail",
                "grounded": True,
                "grounding_reason": "visual_evidence",
            }
        )

    return vlm_func


def _build_rag(tmp_path: Path, *, vlm_process_enable: bool, vlm_func) -> LightRAG:
    role_configs = {
        spec.name: RoleLLMConfig(func=vlm_func)
        if spec.name == "vlm"
        else RoleLLMConfig()
        for spec in ROLES
    }
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"vlm-pipeline-{tmp_path.name}",
        llm_model_func=vlm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=1024,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        vlm_process_enable=vlm_process_enable,
        role_llm_configs=role_configs,
    )


def _write_sidecar_fixtures(tmp_path: Path) -> tuple[str, dict, Path]:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    image_path = parsed_dir / "fig1.png"
    image_path.write_bytes(PNG_BYTES)

    blocks_path = parsed_dir / "doc.blocks.jsonl"
    blocks_path.write_text(
        json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n",
        encoding="utf-8",
    )

    sidecar_path = parsed_dir / "doc.drawings.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "drawings": {
                    "dr-001": {
                        "caption": "Figure 1",
                        "path": str(image_path),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    parsed_data = {"blocks_path": str(blocks_path)}
    return "doc-1", parsed_data, sidecar_path


@pytest.mark.asyncio
async def test_vlm_process_enable_false_skips_with_warning(
    tmp_path, caplog, _propagate_lightrag_logger
):
    """Disabled-VLM items must NOT be persisted: otherwise the idempotency
    guard would lock the item out forever once the operator enables VLM."""
    call_log: list[dict] = []
    rag = _build_rag(
        tmp_path, vlm_process_enable=False, vlm_func=_make_vlm_mock(call_log)
    )
    await rag.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_sidecar_fixtures(tmp_path)
        with caplog.at_level("WARNING"):
            await rag.analyze_multimodal(
                doc_id=doc_id,
                file_path="fixture.pdf",
                parsed_data=parsed_data,
                process_options="i",
            )

        # VLM mock must not be invoked when the master switch is off.
        assert call_log == []
        assert any("VLM_PROCESS_ENABLE=false" in rec.message for rec in caplog.records)

        # No llm_analyze_result is persisted — re-running after enabling VLM
        # must be able to re-process this item.
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        item = payload["drawings"]["dr-001"]
        assert "llm_analyze_result" not in item
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_vlm_disabled_then_enabled_reprocesses_item(tmp_path):
    """Re-running analyze_multimodal after flipping VLM_PROCESS_ENABLE from
    false to true must actually call the VLM and persist a real result."""
    call_log: list[dict] = []
    vlm_func = _make_vlm_mock(call_log)

    rag_off = _build_rag(tmp_path, vlm_process_enable=False, vlm_func=vlm_func)
    await rag_off.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_sidecar_fixtures(tmp_path)
        await rag_off.analyze_multimodal(
            doc_id=doc_id,
            file_path="fixture.pdf",
            parsed_data=parsed_data,
            process_options="i",
        )
        assert call_log == []
    finally:
        await rag_off.finalize_storages()

    # Flip the master switch; reuse the same on-disk workspace.
    rag_on = _build_rag(tmp_path, vlm_process_enable=True, vlm_func=vlm_func)
    await rag_on.initialize_storages()
    try:
        await rag_on.analyze_multimodal(
            doc_id=doc_id,
            file_path="fixture.pdf",
            parsed_data=parsed_data,
            process_options="i",
        )
        assert len(call_log) == 1
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        result = payload["drawings"]["dr-001"]["llm_analyze_result"]
        assert result["grounded"] is True
    finally:
        await rag_on.finalize_storages()


@pytest.mark.asyncio
async def test_vlm_cache_hit_on_second_run(tmp_path):
    call_log: list[dict] = []
    rag = _build_rag(
        tmp_path, vlm_process_enable=True, vlm_func=_make_vlm_mock(call_log)
    )
    await rag.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_sidecar_fixtures(tmp_path)

        # First run: should invoke VLM and persist a cache entry.
        await rag.analyze_multimodal(
            doc_id=doc_id,
            file_path="fixture.pdf",
            parsed_data=parsed_data,
            process_options="i",
        )
        assert len(call_log) == 1
        first_call_kwargs = call_log[0]["kwargs"]
        assert first_call_kwargs.get("stream") is False
        # The pipeline must pass image_inputs (not the legacy `messages`).
        assert first_call_kwargs.get("image_inputs") is not None
        assert "messages" not in first_call_kwargs

        # Cache entry exists under default:analysis:*
        await rag.llm_response_cache.index_done_callback()
        cache_file = (
            Path(rag.working_dir) / rag.workspace / "kv_store_llm_response_cache.json"
        )
        cache_blob = json.loads(cache_file.read_text(encoding="utf-8"))
        analysis_keys = [
            k for k in cache_blob.keys() if k.startswith("default:analysis:")
        ]
        assert len(analysis_keys) == 1
        entry = cache_blob[analysis_keys[0]]
        original_prompt = entry["original_prompt"]
        assert "<vlm_images>" in original_prompt
        # Raw base64 must NOT be embedded in the audit block.
        raw_b64 = base64.b64encode(PNG_BYTES).decode("ascii")
        assert raw_b64 not in original_prompt

        # Second run with idempotency-tracked sidecar removed: hit cache.
        await asyncio.sleep(0)
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        payload["drawings"]["dr-001"].pop("llm_analyze_result", None)
        sidecar_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

        await rag.analyze_multimodal(
            doc_id=doc_id,
            file_path="fixture.pdf",
            parsed_data=parsed_data,
            process_options="i",
        )
        # VLM mock must not be invoked again — cache hit.
        assert len(call_log) == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_image_path_resolved_relative_to_sidecar_dir(tmp_path):
    """The native docx parser writes sidecar paths relative to parsed_dir
    (see commit d8efbf7f). The pipeline must resolve them against the
    sidecar's own directory before considering them missing.
    """
    call_log: list[dict] = []
    rag = _build_rag(
        tmp_path, vlm_process_enable=True, vlm_func=_make_vlm_mock(call_log)
    )
    await rag.initialize_storages()
    try:
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()

        # Image lives inside the parsed_dir under a relative subfolder.
        assets_dir = parsed_dir / "doc.blocks.assets"
        assets_dir.mkdir()
        (assets_dir / "image1.png").write_bytes(PNG_BYTES)

        blocks_path = parsed_dir / "doc.blocks.jsonl"
        blocks_path.write_text(
            json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n",
            encoding="utf-8",
        )
        sidecar_path = parsed_dir / "doc.drawings.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "drawings": {
                        "dr-001": {
                            "caption": "Figure 1",
                            # Parsed_dir-relative path, not absolute.
                            "path": "doc.blocks.assets/image1.png",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        await rag.analyze_multimodal(
            doc_id="doc-1",
            file_path="fixture.pdf",
            parsed_data={"blocks_path": str(blocks_path)},
            process_options="i",
        )

        # The VLM was invoked with image_inputs (path resolved successfully)
        # — not short-circuited to missing_image.
        assert len(call_log) == 1
        assert call_log[0]["kwargs"].get("image_inputs") is not None

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        result = payload["drawings"]["dr-001"]["llm_analyze_result"]
        # Grounded result, not the conservative missing_image fallback.
        assert result["grounded"] is True
        assert result["grounding_reason"] != "missing_image"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_unsupported_vector_format_falls_back_with_warning(
    tmp_path, caplog, _propagate_lightrag_logger
):
    """WMF/EMF/SVG cannot be sent to vision providers; the pipeline must
    skip the image with a warning rather than trying to base64-encode and
    failing mid-call.
    """
    call_log: list[dict] = []
    rag = _build_rag(
        tmp_path, vlm_process_enable=True, vlm_func=_make_vlm_mock(call_log)
    )
    await rag.initialize_storages()
    try:
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()

        wmf_path = parsed_dir / "image1.wmf"
        wmf_path.write_bytes(b"WMF\x00fake-content-bytes")

        blocks_path = parsed_dir / "doc.blocks.jsonl"
        blocks_path.write_text(
            json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n",
            encoding="utf-8",
        )
        sidecar_path = parsed_dir / "doc.drawings.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "drawings": {
                        "dr-001": {
                            "caption": "vector diagram",
                            "path": str(wmf_path),
                            "format": "wmf",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        with caplog.at_level("WARNING"):
            await rag.analyze_multimodal(
                doc_id="doc-1",
                file_path="fixture.docx",
                parsed_data={"blocks_path": str(blocks_path)},
                process_options="i",
            )

        # Warning emitted, VLM still called (no visual evidence, no textual
        # evidence for drawings) — and result lands in conservative branch.
        assert any("unsupported image format" in rec.message for rec in caplog.records)
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        result = payload["drawings"]["dr-001"]["llm_analyze_result"]
        assert result["grounded"] is False
        assert result["grounding_reason"] == "missing_image"
    finally:
        await rag.finalize_storages()


def _make_invalid_then_valid_vlm(call_log: list[dict]):
    """Returns garbage on the first call, valid JSON on the second."""

    async def vlm_func(prompt, **kwargs):
        call_log.append({"prompt": prompt, "kwargs": dict(kwargs)})
        if len(call_log) == 1:
            return "this is not JSON at all"
        return json.dumps(
            {
                "name": "fig-1",
                "summary": "recovered",
                "detail_description": "recovered detail",
                "grounded": True,
                "grounding_reason": "visual_evidence",
            }
        )

    return vlm_func


@pytest.mark.asyncio
async def test_invalid_vlm_response_is_not_cached(tmp_path):
    """Invalid VLM responses must NOT poison the analysis cache; the next
    run for the same prompt+image must re-invoke the VLM."""
    call_log: list[dict] = []
    rag = _build_rag(
        tmp_path,
        vlm_process_enable=True,
        vlm_func=_make_invalid_then_valid_vlm(call_log),
    )
    await rag.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_sidecar_fixtures(tmp_path)

        # First run: VLM returns garbage -> conservative writeback, NO cache.
        await rag.analyze_multimodal(
            doc_id=doc_id,
            file_path="fixture.pdf",
            parsed_data=parsed_data,
            process_options="i",
        )
        assert len(call_log) == 1

        await rag.llm_response_cache.index_done_callback()
        cache_file = (
            Path(rag.working_dir) / rag.workspace / "kv_store_llm_response_cache.json"
        )
        if cache_file.exists():
            cache_blob = json.loads(cache_file.read_text(encoding="utf-8"))
            analysis_keys = [
                k for k in cache_blob.keys() if k.startswith("default:analysis:")
            ]
            assert (
                analysis_keys == []
            ), f"invalid VLM response was cached: {analysis_keys}"

        # Clear the conservative writeback so idempotency does not skip us,
        # then re-run. The VLM must be invoked a second time (no cache hit).
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        payload["drawings"]["dr-001"].pop("llm_analyze_result", None)
        sidecar_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

        await rag.analyze_multimodal(
            doc_id=doc_id,
            file_path="fixture.pdf",
            parsed_data=parsed_data,
            process_options="i",
        )
        assert len(call_log) == 2
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        result = payload["drawings"]["dr-001"]["llm_analyze_result"]
        assert result["grounded"] is True
        assert result["summary"] == "recovered"
    finally:
        await rag.finalize_storages()
