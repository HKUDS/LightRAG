"""Integration tests for ``parse_mineru`` with the unified sidecar pipeline.

These tests stub :class:`MinerURawClient.download_into` so no real MinerU
service is contacted; the focus is on:

- happy path: cache miss → download → sidecar emitted with all expected
  files in the spec-compliant locations
- cache hit: a pre-existing valid ``*.mineru_raw/`` + manifest causes
  ``MinerURawClient.download_into`` NOT to be called
- ``LIGHTRAG_FORCE_REPARSE_MINERU=true`` forces a re-download even when
  the manifest is valid
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import (
    FULL_DOCS_FORMAT_LIGHTRAG,
)
from lightrag.parser.external.mineru import compute_size_and_hash
from lightrag.parser.external.mineru.cache import current_mineru_options_signature
from lightrag.parser.external.mineru.manifest import (
    Manifest,
    ManifestFile,
    write_manifest,
)
from lightrag.utils import EmbeddingFunc, Tokenizer


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
        workspace=f"test-mineru-sidecar-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        vlm_process_enable=False,
    )


_FAKE_CONTENT_LIST = [
    {"type": "text", "text": "1 Introduction", "text_level": 1},
    {"type": "text", "text": "Body paragraph."},
    {
        "type": "table",
        "table_body": [["A", "B"], ["1", "2"]],
        "num_rows": 2,
        "num_cols": 2,
        "table_caption": ["Tbl"],
        "page_idx": 0,
        "bbox": [10, 10, 100, 50],
    },
    {
        "type": "image",
        "img_path": "images/img_001.jpg",
        "image_caption": ["Fig 1"],
        "page_idx": 1,
        "bbox": [20, 20, 200, 100],
    },
    {"type": "equation", "text": "$E = mc^2$", "caption": "Eq 1", "page_idx": 1},
]


def _install_fake_download(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace :meth:`MinerURawClient.download_into` with a recorder that
    writes a synthetic bundle (content_list.json + one image + manifest).
    """
    import lightrag.parser.external.mineru.client as client_mod

    counters = {"calls": 0, "upload_names": []}

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
        (raw_dir / "content_list.json").write_text(
            json.dumps(_FAKE_CONTENT_LIST, ensure_ascii=False),
            encoding="utf-8",
        )
        (raw_dir / "images").mkdir(exist_ok=True)
        (raw_dir / "images" / "img_001.jpg").write_bytes(b"\xff\xd8\xff\xe0fakeJPEG")

        src_size, src_hash = compute_size_and_hash(source_file_path)
        crit_size, crit_hash = compute_size_and_hash(raw_dir / "content_list.json")
        files = [
            ManifestFile(
                path="images/img_001.jpg",
                size=(raw_dir / "images" / "img_001.jpg").stat().st_size,
            )
        ]
        manifest = Manifest(
            source_content_hash=src_hash,
            source_size_bytes=src_size,
            source_filename_at_parse=upload_name or source_file_path.name,
            critical_file=ManifestFile(
                path="content_list.json", size=crit_size, sha256=crit_hash
            ),
            files=files,
            total_size_bytes=crit_size + sum(f.size for f in files),
            task_id=f"fake-{counters['calls']}",
            api_mode="local",
            options_signature=current_mineru_options_signature(),
        )
        write_manifest(raw_dir, manifest)
        return manifest

    monkeypatch.setattr(client_mod.MinerURawClient, "download_into", _fake_download)
    return counters


@pytest.mark.offline
def test_parse_mineru_emits_compliant_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: parse_mineru produces *.parsed/ with spec-compliant
    blocks.jsonl + per-modality JSONs + assets dir; *.mineru_raw/ kept."""

    async def _run() -> None:
        monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://mineru.example")
        counters = _install_fake_download(monkeypatch)

        # Don't move the source out from under the cache validator between
        # repeated parse_mineru calls.
        async def _noop_archive(_p: str) -> None:
            return None

        import lightrag.pipeline as pipeline_module

        monkeypatch.setattr(
            pipeline_module,
            "archive_docx_source_after_full_docs_sync",
            _noop_archive,
        )

        input_dir = tmp_path / "inputs" / "ws"
        input_dir.mkdir(parents=True)
        src = input_dir / "demo.pdf"
        src.write_bytes(b"PDFPDF" * 256)

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
                        "file_path": "demo.pdf",
                        "track_id": "trk",
                        "content_hash": "",
                        "metadata": {},
                    }
                }
            )

            monkeypatch.setattr(
                rag,
                "_resolve_source_file_for_parser",
                lambda _p: str(src),
            )

            parsed = await rag.parse_mineru(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 1, "download_into should run once on miss"

            parsed_dir = Path(parsed["blocks_path"]).parent
            assert parsed["parse_format"] == FULL_DOCS_FORMAT_LIGHTRAG
            assert parsed_dir.name == "demo.pdf.parsed"

            # Sidecar files present
            files = {p.name for p in parsed_dir.iterdir() if p.is_file()}
            assert "demo.blocks.jsonl" in files
            assert "demo.tables.json" in files
            assert "demo.drawings.json" in files
            assert "demo.equations.json" in files
            assert (parsed_dir / "demo.blocks.assets").is_dir()
            assert (parsed_dir / "demo.blocks.assets" / "img_001.jpg").is_file()

            # Content of blocks.jsonl
            blocks_raw = (parsed_dir / "demo.blocks.jsonl").read_text()
            lines = blocks_raw.splitlines()
            meta = json.loads(lines[0])
            rows = [json.loads(line) for line in lines[1:]]
            assert meta["parse_engine"] == "mineru"
            assert meta["table_file"] is True
            assert meta["drawing_file"] is True
            assert meta["equation_file"] is True
            assert meta["asset_dir"] is True
            assert meta["doc_title"] == "1 Introduction"
            # bbox_attributes present for mineru (PDF coordinate context)
            assert meta["bbox_attributes"] == {"origin": "LEFTTOP", "max": 1000}

            # Spec fix: <table> placeholder inline, not <cite>
            contents = " ".join(row.get("content", "") for row in rows)
            assert '<table id="tb-' in contents
            assert 'format="json"' in contents
            assert "<cite" not in contents

            # bbox positions present on at least one block
            assert any(
                p.get("type") == "bbox"
                for row in rows
                for p in row.get("positions") or []
            )

            # Drawing path points inside *.blocks.assets/
            drawings = json.loads((parsed_dir / "demo.drawings.json").read_text())[
                "drawings"
            ]
            (drawing_id, drawing_item) = next(iter(drawings.items()))
            assert drawing_id.startswith("im-")
            assert drawing_item["path"] == "demo.blocks.assets/img_001.jpg"
            assert drawing_item["self_ref"] == "content_list.json#/3"

            # Raw bundle preserved next to sidecar
            raw_dir = parsed_dir.parent / "demo.pdf.mineru_raw"
            assert (raw_dir / "_manifest.json").is_file()
            assert (raw_dir / "content_list.json").is_file()
            assert (raw_dir / "images" / "img_001.jpg").is_file()

            # No legacy non-spec image field on tables
            tables = json.loads((parsed_dir / "demo.tables.json").read_text())["tables"]
            (_, table_item) = next(iter(tables.items()))
            assert "image" not in table_item
            assert table_item["self_ref"] == "content_list.json#/2"

            equations = json.loads((parsed_dir / "demo.equations.json").read_text())[
                "equations"
            ]
            (_, equation_item) = next(iter(equations.items()))
            assert equation_item["self_ref"] == "content_list.json#/4"
        finally:
            await rag.finalize_storages()

    asyncio.new_event_loop().run_until_complete(_run())


@pytest.mark.offline
def test_parse_mineru_cache_hit_skips_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-existing valid bundle short-circuits the network call entirely."""

    async def _run() -> None:
        monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://mineru.example")
        counters = _install_fake_download(monkeypatch)

        # Don't move the source out from under the cache validator between
        # repeated parse_mineru calls.
        async def _noop_archive(_p: str) -> None:
            return None

        import lightrag.pipeline as pipeline_module

        monkeypatch.setattr(
            pipeline_module,
            "archive_docx_source_after_full_docs_sync",
            _noop_archive,
        )

        input_dir = tmp_path / "inputs" / "ws"
        input_dir.mkdir(parents=True)
        src = input_dir / "demo.pdf"
        src.write_bytes(b"PDFPDF" * 256)

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
                        "file_path": "demo.pdf",
                        "track_id": "trk",
                        "content_hash": "",
                        "metadata": {},
                    }
                }
            )

            monkeypatch.setattr(
                rag,
                "_resolve_source_file_for_parser",
                lambda _p: str(src),
            )

            # First call: cache miss → download once.
            await rag.parse_mineru(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 1

            # Second call: should hit cache.
            await rag.parse_mineru(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 1, "cache hit must not re-download"

            # Third call with force-reparse: cache invalidated.
            monkeypatch.setenv("LIGHTRAG_FORCE_REPARSE_MINERU", "true")
            await rag.parse_mineru(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 2
        finally:
            await rag.finalize_storages()

    asyncio.new_event_loop().run_until_complete(_run())


@pytest.mark.offline
def test_parse_mineru_upload_name_strips_parser_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MinerU upload name should use the canonical filename, not parser
    hints embedded in the source basename."""

    async def _run() -> None:
        monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://mineru.example")
        counters = _install_fake_download(monkeypatch)

        input_dir = tmp_path / "inputs" / "ws"
        input_dir.mkdir(parents=True)
        src = input_dir / "demo.[mineru-iet].pdf"
        src.write_bytes(b"PDFPDF" * 256)

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
                rag,
                "_resolve_source_file_for_parser",
                lambda _p: str(src),
            )

            parsed = await rag.parse_mineru(
                doc_id=doc_id,
                file_path=src.name,
                content_data={},
            )

            assert counters["upload_names"] == ["demo.pdf"]
            parsed_dir = Path(parsed["blocks_path"]).parent
            assert parsed_dir.name == "demo.pdf.parsed"
            manifest = json.loads(
                (
                    parsed_dir.parent / "demo.pdf.mineru_raw" / "_manifest.json"
                ).read_text(encoding="utf-8")
            )
            assert manifest["source_filename_at_parse"] == "demo.pdf"
        finally:
            await rag.finalize_storages()

    asyncio.new_event_loop().run_until_complete(_run())


@pytest.mark.offline
def test_parse_mineru_cache_invalidates_on_source_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source file content swapped (same/different size) → cache miss."""

    async def _run() -> None:
        monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://mineru.example")
        counters = _install_fake_download(monkeypatch)

        # Don't move the source out from under the cache validator between
        # repeated parse_mineru calls.
        async def _noop_archive(_p: str) -> None:
            return None

        import lightrag.pipeline as pipeline_module

        monkeypatch.setattr(
            pipeline_module,
            "archive_docx_source_after_full_docs_sync",
            _noop_archive,
        )

        input_dir = tmp_path / "inputs" / "ws"
        input_dir.mkdir(parents=True)
        src = input_dir / "demo.pdf"
        src.write_bytes(b"PDFPDF" * 256)

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
                        "file_path": "demo.pdf",
                        "track_id": "trk",
                        "content_hash": "",
                        "metadata": {},
                    }
                }
            )
            monkeypatch.setattr(
                rag,
                "_resolve_source_file_for_parser",
                lambda _p: str(src),
            )

            await rag.parse_mineru(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 1

            # Same length, different bytes → fast-path passes, hash fails.
            data = src.read_bytes()
            src.write_bytes(b"\x00" + data[1:])

            await rag.parse_mineru(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 2
        finally:
            await rag.finalize_storages()

    asyncio.new_event_loop().run_until_complete(_run())
