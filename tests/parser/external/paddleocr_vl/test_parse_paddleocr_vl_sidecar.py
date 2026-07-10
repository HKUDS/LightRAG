"""End-to-end parser test for PaddleOCR-VL sidecar output."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import lightrag.parser.external.paddleocr_vl.cache as cache_mod
from lightrag import LightRAG
from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG
from lightrag.parser.base import ParseContext
from lightrag.parser.external import Manifest, ManifestFile, write_manifest
from lightrag.parser.external._common import compute_size_and_hash
from lightrag.parser.registry import get_parser, supported_parser_engines
from lightrag.utils import EmbeddingFunc, Tokenizer

# Access internal test helpers via module object (not in __all__)
current_endpoint_signature = cache_mod.current_endpoint_signature
current_paddleocr_vl_options_signature = (
    cache_mod.current_paddleocr_vl_options_signature
)


async def _parse_via_registry(rag, engine, doc_id, file_path, content_data):
    return (
        await get_parser(engine).parse(
            ParseContext(rag, doc_id, file_path, content_data)
        )
    ).to_dict()


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 32)


async def _mock_llm(prompt: Any, **kwargs: Any) -> str:
    return '{"name":"x","summary":"s","detail_description":"d"}'


def _new_rag(tmp_path: Path) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"test-paddleocr-vl-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        vlm_process_enable=False,
    )


_FAKE_RESULT = [
    {
        "prunedResult": {
            "parsing_res_list": [
                {
                    "block_label": "doc_title",
                    "block_content": "# Demo Paper",
                    "block_bbox": [10, 20, 30, 40],
                },
                {
                    "block_label": "text",
                    "block_content": "Intro body.",
                    "block_bbox": [11, 21, 31, 41],
                },
                {
                    "block_label": "table",
                    "block_content": "<table><tr><td>A</td></tr></table>",
                    "block_bbox": [12, 22, 32, 42],
                },
                {
                    "block_label": "image",
                    "block_content": '<img src="imgs/fig.jpg" />',
                    "block_bbox": [13, 23, 33, 43],
                },
                {
                    "block_label": "display_formula",
                    "block_content": "$$x=1$$",
                    "block_bbox": [14, 24, 34, 44],
                },
            ]
        },
        "markdown": {
            "text": "# Demo Paper",
            "images": {"imgs/fig.jpg": "http://files.test/fig.jpg"},
        },
        "outputImages": {},
    }
]


def _install_fake_download(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import lightrag.parser.external.paddleocr_vl.client as client_mod

    counters: dict[str, Any] = {"calls": 0, "upload_names": []}

    async def _fake_download(
        self,
        raw_dir: Path,
        source_file_path: Path,
        *,
        upload_name: str | None = None,
    ):
        counters["calls"] += 1
        counters["upload_names"].append(upload_name)
        raw_dir.mkdir(parents=True, exist_ok=True)
        result = raw_dir / "content_list.json"
        result.write_text(json.dumps(_FAKE_RESULT), encoding="utf-8")
        asset = raw_dir / "imgs" / "fig.jpg"
        asset.parent.mkdir(exist_ok=True)
        asset.write_bytes(b"\xff\xd8fake")

        source_size, source_hash = compute_size_and_hash(source_file_path)
        result_size, result_hash = compute_size_and_hash(result)
        manifest = Manifest(
            engine="paddleocr_vl",
            source_content_hash=source_hash,
            source_size_bytes=source_size,
            source_filename_at_parse=upload_name or source_file_path.name,
            critical_file=ManifestFile("content_list.json", result_size, result_hash),
            files=[ManifestFile("imgs/fig.jpg", asset.stat().st_size)],
            total_size_bytes=result_size + asset.stat().st_size,
            task_id=f"fake-{counters['calls']}",
            endpoint_signature=current_endpoint_signature(),
            options_signature=current_paddleocr_vl_options_signature(),
        )
        write_manifest(raw_dir, manifest)
        return manifest

    monkeypatch.setattr(
        client_mod.PaddleOCRVLRawClient, "download_into", _fake_download
    )
    return counters


@pytest.mark.offline
def test_registry_parse_emits_paddleocr_vl_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        monkeypatch.setenv("PADDLEOCR_VL_API_TOKEN", "token")
        monkeypatch.setenv(
            "PADDLEOCR_VL_ENDPOINT", "http://paddle.test/api/v2/ocr/jobs"
        )
        counters = _install_fake_download(monkeypatch)

        async def _noop_archive(_p: str) -> None:
            return None

        import lightrag.pipeline as pipeline_module

        monkeypatch.setattr(
            pipeline_module,
            "archive_docx_source_after_full_docs_sync",
            _noop_archive,
        )

        assert "paddleocr_vl" in supported_parser_engines()
        parser = get_parser("paddleocr_vl")
        assert parser is not None

        input_dir = tmp_path / "inputs" / "ws"
        input_dir.mkdir(parents=True)
        src = input_dir / "demo.[paddleocr_vl-iteP].pdf"
        src.write_bytes(b"%PDF fake" * 64)

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            doc_id = "doc-abcdef0123456789abcdef0123456789"
            await rag.doc_status.upsert(
                {
                    doc_id: {
                        "status": "PARSING",
                        "content_summary": "",
                        "content_length": 0,
                        "chunks_count": 0,
                        "chunks_list": [],
                        "created_at": "2026-05-15T00:00:00+00:00",
                        "updated_at": "2026-05-15T00:00:00+00:00",
                        "file_path": src.name,
                        "track_id": "trk",
                        "content_hash": "",
                        "metadata": {},
                    }
                }
            )
            monkeypatch.setattr(
                rag, "_resolve_source_file_for_parser", lambda _p: str(src)
            )

            parsed = await _parse_via_registry(
                rag,
                "paddleocr_vl",
                doc_id=doc_id,
                file_path=src.name,
                content_data={},
            )

            assert counters["upload_names"] == ["demo.pdf"]
            assert parsed["parse_format"] == FULL_DOCS_FORMAT_LIGHTRAG
            parsed_dir = Path(parsed["blocks_path"]).parent
            assert parsed_dir.name == "demo.pdf.parsed"
            assert (parsed_dir.parent / "demo.pdf.paddleocr_vl_raw").is_dir()

            lines = (parsed_dir / "demo.blocks.jsonl").read_text().splitlines()
            meta = json.loads(lines[0])
            rows = [json.loads(line) for line in lines[1:]]
            assert meta["parse_engine"] == "paddleocr_vl"
            assert meta["doc_title"] == "Demo Paper"
            assert meta["table_file"] is True
            assert meta["drawing_file"] is True
            assert meta["equation_file"] is True
            assert meta["asset_dir"] is True
            content = "\n".join(row["content"] for row in rows)
            assert '<table id="tb-' in content
            assert '<drawing id="im-' in content
            assert '<equation id="eq-' in content
            assert (parsed_dir / "demo.blocks.assets" / "fig.jpg").is_file()

            # Second parse reuses the raw cache.
            await _parse_via_registry(
                rag,
                "paddleocr_vl",
                doc_id=doc_id,
                file_path=src.name,
                content_data={},
            )
            assert counters["calls"] == 1
        finally:
            await rag.finalize_storages()

    asyncio.new_event_loop().run_until_complete(_run())
