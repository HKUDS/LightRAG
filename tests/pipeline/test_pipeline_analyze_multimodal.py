"""End-to-end offline tests for the analyze_multimodal pipeline.

Covers the contract introduced by the LR2 multimodal rewrite:

1. Drawings route to the VLM role; tables/equations route to the EXTRACT
   role. ``VLM_PROCESS_ENABLE=False`` (or a missing VLM role) is a hard
   failure for image-enabled documents.
2. ``llm_analyze_result`` uses the new ``status / message / analyze_time``
   shape with modality-specific required fields
   (``name/type/description`` for images, ``name/description`` for tables,
   ``name/equation/description`` for equations).
3. Each VLM/EXTRACT call carries ``_priority=DEFAULT_MM_ANALYSIS_PRIORITY``
   and ``image_inputs`` (not legacy ``messages``).
4. Analysis cache ids are written back to the sidecar item's
   ``llm_cache_list`` so document deletion can clean them up.
5. Images smaller than ``VLM_MIN_IMAGE_PIXEL`` (32 px) and unsupported
   formats are pre-emptively flagged ``status="skipped"`` without an LLM
   call.
6. Invalid model JSON is a hard failure — the sidecar carries
   ``status="failure"`` and :class:`MultimodalAnalysisError` propagates.
"""

from __future__ import annotations

import base64
import json
import logging
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.exceptions import MultimodalAnalysisError
from lightrag.utils import EmbeddingFunc, Tokenizer


@pytest.fixture
def _propagate_lightrag_logger(monkeypatch):
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


pytestmark = pytest.mark.offline


def _png_bytes(width: int, height: int) -> bytes:
    """Build a minimal but parser-accepted PNG with the given dimensions."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_payload = struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00"
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_payload).to_bytes(4, "big")
    ihdr = struct.pack(">I", len(ihdr_payload)) + b"IHDR" + ihdr_payload + ihdr_crc
    idat_raw = b"\x00" * (width * height * 4 + height)
    idat_compressed = zlib.compress(idat_raw)
    idat_crc = zlib.crc32(b"IDAT" + idat_compressed).to_bytes(4, "big")
    idat = (
        struct.pack(">I", len(idat_compressed)) + b"IDAT" + idat_compressed + idat_crc
    )
    iend = b"\x00\x00\x00\x00IEND\xaeB`\x82"
    return signature + ihdr + idat + iend


PNG_BYTES = _png_bytes(64, 64)
TINY_PNG_BYTES = _png_bytes(8, 8)


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
                "type": "Chart",
                "description": "concise figure description",
            }
        )

    return vlm_func


def _make_extract_mock(call_log: list[dict]):
    async def extract_func(prompt, **kwargs):
        call_log.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return json.dumps(
            {
                "name": "tbl-1",
                "description": "table summary",
            }
        )

    return extract_func


def _build_rag(
    tmp_path: Path,
    *,
    vlm_process_enable: bool = True,
    vlm_func=None,
    extract_func=None,
) -> LightRAG:
    role_configs = {}
    for spec in ROLES:
        if spec.name == "vlm" and vlm_func is not None:
            role_configs[spec.name] = RoleLLMConfig(func=vlm_func)
        elif spec.name == "extract" and extract_func is not None:
            role_configs[spec.name] = RoleLLMConfig(func=extract_func)
        else:
            role_configs[spec.name] = RoleLLMConfig()
    base_func = vlm_func or extract_func
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"vlm-pipeline-{tmp_path.name}",
        llm_model_func=base_func,
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
                    "im-001": {
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
async def test_vlm_process_enable_false_hard_fails_for_images(
    tmp_path, caplog, _propagate_lightrag_logger
):
    """With i opted-in but VLM disabled, analyze_multimodal must hard-fail
    the document rather than silently skipping."""
    call_log: list[dict] = []
    rag = _build_rag(
        tmp_path, vlm_process_enable=False, vlm_func=_make_vlm_mock(call_log)
    )
    await rag.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_sidecar_fixtures(tmp_path)
        with pytest.raises(MultimodalAnalysisError):
            await rag.analyze_multimodal(
                doc_id=doc_id,
                file_path="fixture.pdf",
                parsed_data=parsed_data,
                process_options="i",
            )
        # VLM mock must not be invoked when the master switch is off.
        assert call_log == []
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        item = payload["drawings"]["im-001"]
        assert item["llm_analyze_result"]["status"] == "failure"
        assert "VLM" in item["llm_analyze_result"]["message"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_vlm_disabled_then_enabled_reprocesses_item(tmp_path):
    """After the first run hard-failed under VLM=disabled, flipping the
    switch must re-invoke the VLM and overwrite the persisted failure."""
    call_log: list[dict] = []
    vlm_func = _make_vlm_mock(call_log)

    rag_off = _build_rag(tmp_path, vlm_process_enable=False, vlm_func=vlm_func)
    await rag_off.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_sidecar_fixtures(tmp_path)
        with pytest.raises(MultimodalAnalysisError):
            await rag_off.analyze_multimodal(
                doc_id=doc_id,
                file_path="fixture.pdf",
                parsed_data=parsed_data,
                process_options="i",
            )
        assert call_log == []
    finally:
        await rag_off.finalize_storages()

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
        result = payload["drawings"]["im-001"]["llm_analyze_result"]
        assert result["status"] == "success"
        assert result["type"] == "Chart"
        assert result["description"] == "concise figure description"
        assert "analyze_time" in result
    finally:
        await rag_on.finalize_storages()


@pytest.mark.asyncio
async def test_vlm_call_carries_image_inputs(tmp_path):
    """Sanity check the call kwargs: image_inputs (not legacy `messages`)
    must be present.  The ``_priority`` argument is consumed by the role
    wrapper before reaching the raw model func, so it is not observable on
    the mock — see ``priority_limit_async_func_call`` in lightrag.utils."""
    call_log: list[dict] = []
    rag = _build_rag(
        tmp_path, vlm_process_enable=True, vlm_func=_make_vlm_mock(call_log)
    )
    await rag.initialize_storages()
    try:
        doc_id, parsed_data, _ = _write_sidecar_fixtures(tmp_path)
        await rag.analyze_multimodal(
            doc_id=doc_id,
            file_path="fixture.pdf",
            parsed_data=parsed_data,
            process_options="i",
        )
        assert len(call_log) == 1
        kwargs = call_log[0]["kwargs"]
        assert kwargs.get("stream") is False
        assert kwargs.get("image_inputs") is not None
        assert "messages" not in kwargs
        # _priority is consumed by the wrapper (see lightrag.utils
        # priority_limit_async_func_call); not observable on the raw mock.
        assert "_priority" not in kwargs
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_vlm_cache_hit_on_second_run(tmp_path):
    call_log: list[dict] = []
    rag = _build_rag(
        tmp_path, vlm_process_enable=True, vlm_func=_make_vlm_mock(call_log)
    )
    await rag.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_sidecar_fixtures(tmp_path)

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
        cache_blob = json.loads(cache_file.read_text(encoding="utf-8"))
        analysis_keys = [
            k for k in cache_blob.keys() if k.startswith("default:analysis:")
        ]
        assert len(analysis_keys) == 1
        cache_id = analysis_keys[0]
        entry = cache_blob[cache_id]
        original_prompt = entry["original_prompt"]
        assert "<vlm_images>" in original_prompt
        # Raw base64 must NOT be embedded in the audit block.
        raw_b64 = base64.b64encode(PNG_BYTES).decode("ascii")
        assert raw_b64 not in original_prompt

        # Cache id was written back to the sidecar item for delete-time cleanup.
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert cache_id in payload["drawings"]["im-001"]["llm_cache_list"]

        # Re-run: analyze_multimodal always recomputes for enabled modalities,
        # but the cache key matches so the VLM is not called again.
        await rag.analyze_multimodal(
            doc_id=doc_id,
            file_path="fixture.pdf",
            parsed_data=parsed_data,
            process_options="i",
        )
        assert len(call_log) == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_image_path_resolved_relative_to_sidecar_dir(tmp_path):
    """Sidecar paths are parsed_dir-relative; the pipeline must resolve
    them against the sidecar directory before falling back to skipped."""
    call_log: list[dict] = []
    rag = _build_rag(
        tmp_path, vlm_process_enable=True, vlm_func=_make_vlm_mock(call_log)
    )
    await rag.initialize_storages()
    try:
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()

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
                        "im-001": {
                            "caption": "Figure 1",
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

        assert len(call_log) == 1
        assert call_log[0]["kwargs"].get("image_inputs") is not None

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        result = payload["drawings"]["im-001"]["llm_analyze_result"]
        assert result["status"] == "success"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_unsupported_vector_format_writes_skipped(tmp_path):
    """WMF/EMF/SVG and other non-raster formats short-circuit to
    status=skipped without calling the VLM."""
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
                        "im-001": {
                            "caption": "vector diagram",
                            "path": str(wmf_path),
                            "format": "wmf",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        await rag.analyze_multimodal(
            doc_id="doc-1",
            file_path="fixture.docx",
            parsed_data={"blocks_path": str(blocks_path)},
            process_options="i",
        )

        assert call_log == []
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        result = payload["drawings"]["im-001"]["llm_analyze_result"]
        assert result["status"] == "skipped"
        assert "unsupported image format" in result["message"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_tiny_image_writes_skipped_without_vlm_call(tmp_path):
    """Image smaller than VLM_MIN_IMAGE_PIXEL (32px) is decorative; skip
    without calling VLM."""
    call_log: list[dict] = []
    rag = _build_rag(
        tmp_path, vlm_process_enable=True, vlm_func=_make_vlm_mock(call_log)
    )
    await rag.initialize_storages()
    try:
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        img_path = parsed_dir / "tiny.png"
        img_path.write_bytes(TINY_PNG_BYTES)

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
                        "im-001": {
                            "caption": "tiny icon",
                            "path": str(img_path),
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
        assert call_log == []
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        result = payload["drawings"]["im-001"]["llm_analyze_result"]
        assert result["status"] == "skipped"
        assert "smaller than" in result["message"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_invalid_vlm_response_hard_fails(tmp_path):
    """Invalid model JSON propagates MultimodalAnalysisError and lands a
    status=failure marker on the sidecar so re-runs don't silently
    consume the failure."""
    call_log: list[dict] = []

    async def vlm_func(prompt, **kwargs):
        call_log.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return "not-json"

    rag = _build_rag(tmp_path, vlm_process_enable=True, vlm_func=vlm_func)
    await rag.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_sidecar_fixtures(tmp_path)
        with pytest.raises(MultimodalAnalysisError):
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
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        item = payload["drawings"]["im-001"]
        assert item["llm_analyze_result"]["status"] == "failure"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_table_routes_to_extract_role_not_vlm(tmp_path):
    """Per design §3.1 tables (and equations) must hit the EXTRACT role,
    not VLM. The mocks below assert exactly that."""
    extract_log: list[dict] = []
    vlm_log: list[dict] = []
    rag = _build_rag(
        tmp_path,
        vlm_process_enable=True,
        vlm_func=_make_vlm_mock(vlm_log),
        extract_func=_make_extract_mock(extract_log),
    )
    await rag.initialize_storages()
    try:
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        blocks_path = parsed_dir / "doc.blocks.jsonl"
        blocks_path.write_text(
            json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n",
            encoding="utf-8",
        )
        tables_path = parsed_dir / "doc.tables.json"
        tables_path.write_text(
            json.dumps(
                {
                    "tables": {
                        "tb-001": {
                            "caption": "Table 1",
                            "format": "html",
                            "content": "<table><tr><td>A</td></tr></table>",
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
            process_options="t",
        )
        assert vlm_log == []
        assert len(extract_log) == 1
        kwargs = extract_log[0]["kwargs"]
        assert kwargs.get("response_format") == {"type": "json_object"}
        payload = json.loads(tables_path.read_text(encoding="utf-8"))
        result = payload["tables"]["tb-001"]["llm_analyze_result"]
        assert result["status"] == "success"
        assert "default:analysis:" in payload["tables"]["tb-001"]["llm_cache_list"][0]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_invalid_json_with_trailing_comma_is_repaired(tmp_path):
    """Slightly malformed VLM JSON (trailing comma) must be repaired via
    ``json_repair`` instead of hard-failing the document — mirrors the
    extraction-side repair contract in operate._process_json_extraction_result.
    """
    call_log: list[dict] = []

    async def vlm_func(prompt, **kwargs):
        call_log.append({"prompt": prompt, "kwargs": dict(kwargs)})
        # Trailing comma after "description" — strict json.loads would reject.
        return '{"name": "fig-1", "type": "Chart", ' '"description": "ok",}'

    rag = _build_rag(tmp_path, vlm_process_enable=True, vlm_func=vlm_func)
    await rag.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_sidecar_fixtures(tmp_path)
        await rag.analyze_multimodal(
            doc_id=doc_id,
            file_path="fixture.pdf",
            parsed_data=parsed_data,
            process_options="i",
        )
        assert len(call_log) == 1
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        result = payload["drawings"]["im-001"]["llm_analyze_result"]
        assert result["status"] == "success"
        assert result["name"] == "fig-1"
        assert result["type"] == "Chart"
        assert result["description"] == "ok"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_analyze_worker_marks_doc_failed_on_multimodal_error(tmp_path):
    """When analyze_multimodal raises MultimodalAnalysisError, the worker
    must upsert DocStatus.FAILED with a diagnostic error_msg instead of
    letting the document stay stuck in ANALYZING."""
    import asyncio
    from dataclasses import asdict
    from lightrag.base import DocProcessingStatus, DocStatus
    from lightrag.pipeline import _BatchRunContext

    async def vlm_func(prompt, **kwargs):
        return ""

    rag = _build_rag(tmp_path, vlm_process_enable=True, vlm_func=vlm_func)
    await rag.initialize_storages()
    try:
        doc_id = "doc-fail-1"
        file_path = "demo.pdf"
        status_doc = DocProcessingStatus(
            content_summary="",
            content_length=0,
            file_path=file_path,
            status=DocStatus.PENDING,
            created_at="2026-05-14T00:00:00Z",
            updated_at="2026-05-14T00:00:00Z",
            track_id="t",
            content_hash="h",
        )
        await rag.doc_status.upsert({doc_id: asdict(status_doc)})

        async def _raise_mm_error(**_kwargs):
            from lightrag.exceptions import MultimodalAnalysisError

            raise MultimodalAnalysisError("forced failure for test")

        # Patch instance method so the worker's call site picks the mock.
        rag.analyze_multimodal = _raise_mm_error  # type: ignore[assignment]

        ctx = _BatchRunContext(
            pipeline_status={
                "latest_message": "",
                "history_messages": [],
                "cancellation_requested": False,
            },
            pipeline_status_lock=asyncio.Lock(),
            semaphore=asyncio.Semaphore(1),
            total_files=1,
            q_native=asyncio.Queue(),
            q_mineru=asyncio.Queue(),
            q_docling=asyncio.Queue(),
            q_analyze=asyncio.Queue(),
            q_process=asyncio.Queue(),
        )
        worker = asyncio.create_task(rag._analyze_worker(ctx))
        await ctx.q_analyze.put((doc_id, status_doc, {"content": "body"}))
        await ctx.q_analyze.join()
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        refreshed = await rag.doc_status.get_by_id(doc_id)
        # doc_status backends return either a dict or a dataclass-style obj.
        if not isinstance(refreshed, dict):
            refreshed = asdict(refreshed)
        assert refreshed["status"] == DocStatus.FAILED
        assert "forced failure for test" in (refreshed.get("error_msg") or "")
        # The worker must NOT advance to q_process when the analyze step
        # raises — otherwise process_single_document would run on a
        # half-baked document.
        assert ctx.q_process.empty()
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_analysis_cache_respects_disabled_flag(tmp_path):
    """When enable_llm_cache_for_entity_extract is False, analyze_multimodal
    must NOT persist the analysis response and MUST NOT attach a cache id
    to sidecar item.llm_cache_list — otherwise document deletion would try
    to clean up cache entries that were never written."""
    call_log: list[dict] = []
    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"vlm-pipeline-cache-{tmp_path.name}",
        llm_model_func=_make_vlm_mock(call_log),
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=1024,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        vlm_process_enable=True,
        # Disable the analysis cache (same flag handle_cache uses for mode="default").
        enable_llm_cache_for_entity_extract=False,
        role_llm_configs={
            spec.name: (
                RoleLLMConfig(func=_make_vlm_mock(call_log))
                if spec.name == "vlm"
                else RoleLLMConfig()
            )
            for spec in ROLES
        },
    )
    await rag.initialize_storages()
    try:
        doc_id, parsed_data, sidecar_path = _write_sidecar_fixtures(tmp_path)
        await rag.analyze_multimodal(
            doc_id=doc_id,
            file_path="fixture.pdf",
            parsed_data=parsed_data,
            process_options="i",
        )
        await rag.llm_response_cache.index_done_callback()
        cache_file = (
            Path(rag.working_dir) / rag.workspace / "kv_store_llm_response_cache.json"
        )
        if cache_file.exists():
            cache_blob = json.loads(cache_file.read_text(encoding="utf-8"))
            assert not any(
                k.startswith("default:analysis:") for k in cache_blob.keys()
            ), "analysis cache must not be written when the flag is off"
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        item = payload["drawings"]["im-001"]
        # Analysis still succeeded — only the cache side-effects are gated.
        assert item["llm_analyze_result"]["status"] == "success"
        # No cache id may be attached when nothing was written.
        assert item.get("llm_cache_list", []) == []
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_oversized_table_content_truncated_to_extract_budget(tmp_path):
    """A sidecar table whose ``content`` alone exceeds the EXTRACT input
    cap must be trimmed before reaching the LLM.  The captured prompt
    must (a) fit within ``DEFAULT_MAX_EXTRACT_INPUT_TOKENS``, (b) still
    wrap the trimmed body in a ``<table>`` tag, and (c) include the
    truncation marker so the LLM is told the body is incomplete.
    """
    from lightrag.constants import DEFAULT_MAX_EXTRACT_INPUT_TOKENS

    extract_log: list[dict] = []
    rag = _build_rag(
        tmp_path,
        vlm_process_enable=False,
        extract_func=_make_extract_mock(extract_log),
    )
    await rag.initialize_storages()
    try:
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        blocks_path = parsed_dir / "doc.blocks.jsonl"
        blocks_path.write_text(
            json.dumps({"type": "meta", "doc_id": "doc-big"}) + "\n",
            encoding="utf-8",
        )

        # Build a JSON-format table whose total token count is well above
        # DEFAULT_MAX_EXTRACT_INPUT_TOKENS so the trim path must engage.
        # Each row is small; the row count is the lever.
        rows = [[f"r{i}c0", f"r{i}c1"] for i in range(8000)]
        big_table = '<table id="tb-big" format="json">' + json.dumps(rows) + "</table>"
        original_tokens = len(rag.tokenizer.encode(big_table))
        assert original_tokens > DEFAULT_MAX_EXTRACT_INPUT_TOKENS

        tables_path = parsed_dir / "doc.tables.json"
        tables_path.write_text(
            json.dumps(
                {
                    "tables": {
                        "tb-big": {
                            "caption": "huge table",
                            "format": "json",
                            "content": big_table,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        await rag.analyze_multimodal(
            doc_id="doc-big",
            file_path="fixture.pdf",
            parsed_data={"blocks_path": str(blocks_path)},
            process_options="t",
        )
        assert len(extract_log) == 1
        sent_prompt = extract_log[0]["prompt"]
        prompt_tokens = len(rag.tokenizer.encode(sent_prompt))

        # Whole prompt fits within the EXTRACT input cap.
        assert prompt_tokens <= DEFAULT_MAX_EXTRACT_INPUT_TOKENS, (
            f"prompt of {prompt_tokens} tokens exceeds "
            f"{DEFAULT_MAX_EXTRACT_INPUT_TOKENS}"
        )

        # Truncation marker present, <table> tag still closed inside the
        # CONTENT section.
        assert (
            "<!-- content truncated from " in sent_prompt
            and "head preserved -->" in sent_prompt
        )
        # Head rows preserved, last rows dropped.
        assert "r0c0" in sent_prompt
        assert "r7999c0" not in sent_prompt

        # Analysis still succeeded — trimming is transparent to status.
        payload = json.loads(tables_path.read_text(encoding="utf-8"))
        assert payload["tables"]["tb-big"]["llm_analyze_result"]["status"] == "success"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_max_extract_input_tokens_env_var_lowers_cap_and_logs_warning(
    tmp_path, caplog, _propagate_lightrag_logger, monkeypatch
):
    """``MAX_EXTRACT_INPUT_TOKENS`` env var overrides the compile-time
    default, and any truncation emits a WARNING-level log line so
    operators see when sidecar bodies are being cut for the EXTRACT call.
    """
    # Cap well below the default 20480 so a modest table triggers trimming,
    # but still comfortably above the ~3980-char prompt template frame so
    # `content_budget` stays positive.
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "8000")

    extract_log: list[dict] = []
    rag = _build_rag(
        tmp_path,
        vlm_process_enable=False,
        extract_func=_make_extract_mock(extract_log),
    )
    await rag.initialize_storages()
    try:
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        blocks_path = parsed_dir / "doc.blocks.jsonl"
        blocks_path.write_text(
            json.dumps({"type": "meta", "doc_id": "doc-mid"}) + "\n",
            encoding="utf-8",
        )

        # Table sized comfortably between the env cap (8000) and the
        # compile-time default (20480) — would NOT trim under the default,
        # MUST trim under the env override.
        rows = [[f"r{i}c0", f"r{i}c1"] for i in range(800)]
        mid_table = '<table id="tb-mid" format="json">' + json.dumps(rows) + "</table>"
        original_tokens = len(rag.tokenizer.encode(mid_table))
        assert 8000 < original_tokens < 20480

        tables_path = parsed_dir / "doc.tables.json"
        tables_path.write_text(
            json.dumps(
                {
                    "tables": {
                        "tb-mid": {
                            "caption": "mid table",
                            "format": "json",
                            "content": mid_table,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING, logger="lightrag.pipeline"):
            await rag.analyze_multimodal(
                doc_id="doc-mid",
                file_path="fixture.pdf",
                parsed_data={"blocks_path": str(blocks_path)},
                process_options="t",
            )
        assert len(extract_log) == 1
        sent_prompt = extract_log[0]["prompt"]
        # Env cap honored.
        assert len(rag.tokenizer.encode(sent_prompt)) <= 8000
        assert "<!-- content truncated from " in sent_prompt

        # WARNING-level log line was emitted naming this item.
        warning_records = [
            r
            for r in caplog.records
            if r.levelname == "WARNING"
            and "[analyze_multimodal]" in r.getMessage()
            and "tb-mid" in r.getMessage()
            and "content trimmed" in r.getMessage()
        ]
        assert (
            warning_records
        ), "expected a WARNING-level log line announcing content truncation"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_extract_cap_below_prompt_frame_fails_item_without_llm_call(
    tmp_path, monkeypatch
):
    """When ``MAX_EXTRACT_INPUT_TOKENS`` sits below the prompt template's
    own ``frame_tokens + SAFETY_BUFFER``, ``content_budget`` goes
    non-positive — no content trim can bring the request under the cap
    because the frame itself overflows.  The pipeline must refuse to
    invoke the LLM and fail this item via :class:`MultimodalAnalysisError`,
    so the sidecar records ``status="failure"`` and operators get a clear
    actionable signal (raise the cap).  Guards the P1 regression where
    marker-replacement merely shrank ``content`` while leaving the
    over-cap frame intact."""
    # The table_analysis prompt frame is ~3980 chars; pick a cap small
    # enough that the frame alone overflows.
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "1000")

    extract_log: list[dict] = []
    rag = _build_rag(
        tmp_path,
        vlm_process_enable=False,
        extract_func=_make_extract_mock(extract_log),
    )
    await rag.initialize_storages()
    try:
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        blocks_path = parsed_dir / "doc.blocks.jsonl"
        blocks_path.write_text(
            json.dumps({"type": "meta", "doc_id": "doc-tight"}) + "\n",
            encoding="utf-8",
        )
        tables_path = parsed_dir / "doc.tables.json"
        tables_path.write_text(
            json.dumps(
                {
                    "tables": {
                        "tb-tight": {
                            "format": "json",
                            "content": (
                                '<table id="tb-tight" format="json">'
                                '[["A","B"]]</table>'
                            ),
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(MultimodalAnalysisError) as excinfo:
            await rag.analyze_multimodal(
                doc_id="doc-tight",
                file_path="fixture.pdf",
                parsed_data={"blocks_path": str(blocks_path)},
                process_options="t",
            )

        # Critical: the LLM mock must NOT have been invoked — we refused
        # to send an over-cap prompt rather than letting the provider
        # reject it with context_length_exceeded.
        assert extract_log == [], (
            f"EXTRACT must not be called when frame exceeds cap; "
            f"got {len(extract_log)} call(s)"
        )

        msg = str(excinfo.value)
        assert "table/tb-tight" in msg
        assert "MAX_EXTRACT_INPUT_TOKENS" in msg
        assert "1000" in msg

        # Sidecar records status=failure for the item, so operators can
        # spot it and re-run after raising the cap.
        payload = json.loads(tables_path.read_text(encoding="utf-8"))
        item = payload["tables"]["tb-tight"]
        assert item["llm_analyze_result"]["status"] == "failure"
        assert "MAX_EXTRACT_INPUT_TOKENS" in item["llm_analyze_result"]["message"]
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Table content-format declaration (prompt tells the LLM html vs json).
# ---------------------------------------------------------------------------


def test_table_content_format_label_html_and_json():
    from lightrag.prompt_multimodal import table_content_format_label

    html_label = table_content_format_label("html")
    assert "HTML" in html_label
    assert "rowspan" in html_label and "colspan" in html_label
    assert "<thead>" in html_label

    json_label = table_content_format_label("JSON")  # case-insensitive
    assert "JSON" in json_label
    assert "2-D" in json_label and "row" in json_label


@pytest.mark.parametrize("bad", [None, "", "csv", "markdown"])
def test_table_content_format_label_rejects_unknown(bad):
    from lightrag.prompt_multimodal import table_content_format_label

    with pytest.raises(ValueError):
        table_content_format_label(bad)


async def _capture_table_prompt(tmp_path, *, fmt, content):
    """Run analyze_multimodal on a single table item and return the rendered
    EXTRACT prompt captured by the mock."""
    extract_log: list[dict] = []
    rag = _build_rag(tmp_path, extract_func=_make_extract_mock(extract_log))
    await rag.initialize_storages()
    try:
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        blocks_path = parsed_dir / "doc.blocks.jsonl"
        blocks_path.write_text(
            json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n",
            encoding="utf-8",
        )
        item: dict = {"caption": "Table 1", "content": content}
        if fmt is not None:
            item["format"] = fmt
        (parsed_dir / "doc.tables.json").write_text(
            json.dumps({"tables": {"tb-001": item}}),
            encoding="utf-8",
        )
        await rag.analyze_multimodal(
            doc_id="doc-1",
            file_path="fixture.pdf",
            parsed_data={"blocks_path": str(blocks_path)},
            process_options="t",
        )
        return extract_log
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_table_prompt_declares_html_format(tmp_path):
    """The EXTRACT prompt for an HTML table must state it is HTML."""
    extract_log = await _capture_table_prompt(
        tmp_path,
        fmt="html",
        content="<table><tr><td>A</td></tr></table>",
    )
    assert len(extract_log) == 1
    prompt = extract_log[0]["prompt"]
    assert "in HTML format —" in prompt
    assert "rowspan/colspan" in prompt


@pytest.mark.asyncio
async def test_table_prompt_declares_json_format(tmp_path):
    """The EXTRACT prompt for a JSON table must state it is JSON."""
    extract_log = await _capture_table_prompt(
        tmp_path,
        fmt="json",
        content='[["A","B"],["1","2"]]',
    )
    assert len(extract_log) == 1
    prompt = extract_log[0]["prompt"]
    assert "in JSON format —" in prompt
    assert "2-D array" in prompt


@pytest.mark.asyncio
async def test_table_missing_format_hard_fails(tmp_path):
    """A table item without a valid ``format`` is a corrupt sidecar — the
    worker must raise MultimodalAnalysisError, not guess the format."""
    with pytest.raises(MultimodalAnalysisError) as excinfo:
        await _capture_table_prompt(
            tmp_path,
            fmt=None,
            content="<table><tr><td>A</td></tr></table>",
        )
    assert "tb-001" in str(excinfo.value)
