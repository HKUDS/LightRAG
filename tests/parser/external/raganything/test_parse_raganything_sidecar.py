from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG
from lightrag.parser.base import ParseContext
from lightrag.parser.external._common import raw_dir_for_parsed_dir
from lightrag.parser.registry import get_parser
from lightrag.utils import EmbeddingFunc, Tokenizer
from lightrag.utils_pipeline import parsed_artifact_dir_for


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
        workspace=f"test-raganything-sidecar-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        vlm_process_enable=False,
    )


def _install_fake_raganything(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"count": 0, "requests": []}

    class FakeParser:
        def parse_document(
            self,
            file_path: str,
            *,
            method: str = "auto",
            output_dir: str | None = None,
            lang: str | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            calls["count"] += 1
            calls["requests"].append(
                {
                    "file_path": file_path,
                    "method": method,
                    "output_dir": output_dir,
                    "lang": lang,
                    "kwargs": kwargs,
                }
            )
            assert output_dir is not None
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "fig_001.jpg").write_bytes(b"\xff\xd8\xff\xe0fakeJPEG")
            return {
                "content_list": [
                    {"type": "text", "text": "1 Overview", "text_level": 1},
                    {"type": "text", "text": "Body from RAG-Anything."},
                    {
                        "type": "image",
                        "img_path": "fig_001.jpg",
                        "image_caption": ["RAG-Anything figure"],
                        "page_idx": 0,
                    },
                ]
            }

    def get_parser(parser_type: str) -> FakeParser:
        calls["parser_type"] = parser_type
        return FakeParser()

    raganything_pkg = types.ModuleType("raganything")
    parser_mod = types.ModuleType("raganything.parser")
    parser_mod.get_parser = get_parser
    monkeypatch.setitem(sys.modules, "raganything", raganything_pkg)
    monkeypatch.setitem(sys.modules, "raganything.parser", parser_mod)
    return calls


async def _parse_via_registry(rag, engine, doc_id, file_path, content_data):
    result = await get_parser(engine).parse(
        ParseContext(rag, doc_id, file_path, content_data)
    )
    return result.to_dict()


@pytest.mark.offline
def test_parse_raganything_emits_lightrag_sidecar_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lightrag.parser import registry
    from lightrag_raganything_parser.plugin import register

    async def _run() -> None:
        registry._REGISTRY.pop("raganything", None)
        register()
        calls = _install_fake_raganything(monkeypatch)
        monkeypatch.setenv("RAGANYTHING_PARSER", "mineru")
        monkeypatch.setenv("RAGANYTHING_PARSE_METHOD", "auto")

        async def _noop_archive(_p: str) -> None:
            return None

        import lightrag.pipeline as pipeline_module

        monkeypatch.setattr(
            pipeline_module,
            "archive_docx_source_after_full_docs_sync",
            _noop_archive,
        )

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            source = tmp_path / "demo.pdf"
            source.write_bytes(b"%PDF-1.4 fake")
            doc_id = "doc-raganything-0001"
            content_data = {"source_file": str(source)}
            monkeypatch.setattr(
                rag,
                "_resolve_source_file_for_parser",
                lambda _p, **_kwargs: str(source),
            )

            first = await _parse_via_registry(
                rag, "raganything", doc_id, "demo.pdf", content_data
            )
            second = await _parse_via_registry(
                rag, "raganything", doc_id, "demo.pdf", content_data
            )

            assert first["parse_engine"] == "raganything"
            assert first["parse_format"] == FULL_DOCS_FORMAT_LIGHTRAG
            assert second["parse_stage_skipped"] is True
            assert calls["count"] == 1
            assert calls["parser_type"] == "mineru"

            parsed_dir = parsed_artifact_dir_for("demo.pdf", parent_hint=source.parent)
            raw_dir = raw_dir_for_parsed_dir(parsed_dir, suffix=".raganything_raw")
            content_list_path = raw_dir / "content_list.json"
            assert content_list_path.is_file()
            content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
            assert content_list[1]["text"] == "Body from RAG-Anything."
            assert content_list[2]["img_path"].startswith("images/")
            assert (raw_dir / content_list[2]["img_path"]).is_file()
            assert Path(first["blocks_path"]).is_file()
        finally:
            await rag.finalize_storages()
            registry._REGISTRY.pop("raganything", None)

    asyncio.run(_run())
