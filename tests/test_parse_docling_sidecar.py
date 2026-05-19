"""Integration tests for ``parse_docling`` with the unified sidecar pipeline.

Stubs :class:`DoclingRawClient.download_into` so no real docling-serve is
contacted; the focus is on:

- happy path: cache miss → fake bundle written → sidecar emitted with all
  expected files at the spec-compliant locations
- cache hit: a pre-existing valid ``*.docling_raw/`` + manifest causes
  ``DoclingRawClient.download_into`` NOT to be called
- ``LIGHTRAG_FORCE_REPARSE_DOCLING=true`` forces a re-download even when
  the manifest is valid
- source content swap → cache miss
- options_signature change (``DOCLING_OCR_LANG`` toggle) → cache miss
- adapter sees zero blocks → parse fails loudly (no half-baked sidecar)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG
from lightrag.external_parser import (
    Manifest,
    ManifestFile,
    compute_size_and_hash,
    write_manifest,
)
from lightrag.external_parser.docling.cache import (
    compute_options_signature,
    snapshot_tunable_env,
)
from lightrag.external_parser.docling.client import FIXED_CONSTANTS
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
        workspace=f"test-docling-sidecar-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        vlm_process_enable=False,
    )


_FAKE_DOCLING_JSON = {
    "schema_name": "DoclingDocument",
    "version": "1.10.0",
    "origin": {"filename": "demo.pdf", "mimetype": "application/pdf"},
    "body": {
        "self_ref": "#/body",
        "children": [
            {"$ref": "#/texts/0"},
        ],
        "content_layer": "body",
        "label": "unspecified",
    },
    "groups": [],
    "texts": [
        {
            "self_ref": "#/texts/0",
            "label": "section_header",
            "text": "Intro",
            "orig": "Intro",
            "level": 1,
            "content_layer": "body",
            "children": [
                {"$ref": "#/texts/1"},
                {"$ref": "#/tables/0"},
                {"$ref": "#/pictures/0"},
                {"$ref": "#/texts/2"},
            ],
            "prov": [
                {
                    "page_no": 1,
                    "bbox": {
                        "l": 10.0,
                        "t": 100.0,
                        "r": 200.0,
                        "b": 80.0,
                        "coord_origin": "BOTTOMLEFT",
                    },
                    "charspan": [0, 5],
                }
            ],
        },
        {
            "self_ref": "#/texts/1",
            "label": "text",
            "text": "Body paragraph.",
            "orig": "Body paragraph.",
            "content_layer": "body",
            "prov": [
                {
                    "page_no": 1,
                    "bbox": {
                        "l": 10.0,
                        "t": 60.0,
                        "r": 200.0,
                        "b": 40.0,
                        "coord_origin": "BOTTOMLEFT",
                    },
                    "charspan": [0, 15],
                }
            ],
        },
        {
            "self_ref": "#/texts/2",
            "label": "formula",
            "text": "E = mc^2",
            "orig": "E = mc^2",
            "content_layer": "body",
            "prov": [],
        },
    ],
    "tables": [
        {
            "self_ref": "#/tables/0",
            "label": "table",
            "content_layer": "body",
            "data": {
                "num_rows": 2,
                "num_cols": 2,
                "grid": [
                    [{"text": "h1"}, {"text": "h2"}],
                    [{"text": "a"}, {"text": "b"}],
                ],
            },
            "prov": [],
        }
    ],
    "pictures": [
        {
            "self_ref": "#/pictures/0",
            "label": "picture",
            "content_layer": "body",
            "image": {"uri": "artifacts/img_000000.png", "mimetype": "image/png"},
            "prov": [],
        }
    ],
    "key_value_items": [],
    "form_items": [],
    "pages": {"1": {"size": {"width": 612.0, "height": 792.0}, "page_no": 1}},
}


def _install_fake_download(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace ``DoclingRawClient.download_into`` with a recorder that
    writes a synthetic raw bundle and a valid manifest."""
    import lightrag.external_parser.docling.client as client_mod

    counters = {"calls": 0}

    async def _fake_download(self, raw_dir: Path, source_file_path: Path, **_kwargs):
        counters["calls"] += 1
        raw_dir.mkdir(parents=True, exist_ok=True)
        main_json = raw_dir / "demo.json"
        main_json.write_text(json.dumps(_FAKE_DOCLING_JSON), encoding="utf-8")
        (raw_dir / "demo.md").write_text("# fake md", encoding="utf-8")
        art = raw_dir / "artifacts"
        art.mkdir(exist_ok=True)
        (art / "img_000000.png").write_bytes(b"\x89PNG fake")

        src_size, src_hash = compute_size_and_hash(source_file_path)
        crit_size, crit_hash = compute_size_and_hash(main_json)
        others = [
            ManifestFile(path="demo.md", size=(raw_dir / "demo.md").stat().st_size),
            ManifestFile(
                path="artifacts/img_000000.png",
                size=(art / "img_000000.png").stat().st_size,
            ),
        ]
        options_signature = compute_options_signature(
            tunable_env=snapshot_tunable_env(),
            fixed_constants=FIXED_CONSTANTS,
        )
        manifest = Manifest(
            engine="docling",
            source_content_hash=src_hash,
            source_size_bytes=src_size,
            source_filename_at_parse=source_file_path.name,
            critical_file=ManifestFile(
                path="demo.json", size=crit_size, sha256=crit_hash
            ),
            files=others,
            total_size_bytes=crit_size + sum(f.size for f in others),
            task_id=f"fake-{counters['calls']}",
            endpoint_signature="http://docling.test",
            options_signature=options_signature,
            extras={"fixed_constants": dict(FIXED_CONSTANTS)},
            downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        write_manifest(raw_dir, manifest)
        return manifest

    monkeypatch.setattr(client_mod.DoclingRawClient, "download_into", _fake_download)
    return counters


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, rag: LightRAG, src: Path) -> None:
    """Common pipeline-level stubs: avoid moving the source file and pin
    the file resolver to the synthetic path."""

    async def _noop_archive(_p: str) -> None:
        return None

    import lightrag.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "archive_docx_source_after_full_docs_sync",
        _noop_archive,
    )
    monkeypatch.setattr(rag, "_resolve_source_file_for_parser", lambda _p: str(src))


def _seed_doc_status(rag: LightRAG, doc_id: str) -> Any:
    return rag.doc_status.upsert(
        {
            doc_id: {
                "status": "PARSING",
                "content_summary": "",
                "content_length": 0,
                "chunks_count": 0,
                "chunks_list": [],
                "created_at": "2026-05-18T00:00:00+00:00",
                "updated_at": "2026-05-18T00:00:00+00:00",
                "file_path": "demo.pdf",
                "track_id": "trk",
                "content_hash": "",
                "metadata": {},
            }
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_parse_docling_emits_compliant_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")
        counters = _install_fake_download(monkeypatch)

        input_dir = tmp_path / "inputs" / "ws"
        input_dir.mkdir(parents=True)
        src = input_dir / "demo.pdf"
        src.write_bytes(b"PDFPDF" * 256)

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            _stub_pipeline(monkeypatch, rag, src)
            doc_id = "doc-abcdef0123456789abcdef0123456789"
            await _seed_doc_status(rag, doc_id)

            parsed = await rag.parse_docling(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 1

            parsed_dir = Path(parsed["blocks_path"]).parent
            assert parsed["parse_format"] == FULL_DOCS_FORMAT_LIGHTRAG
            assert parsed_dir.name == "demo.pdf.parsed"

            files = {p.name for p in parsed_dir.iterdir() if p.is_file()}
            assert "demo.blocks.jsonl" in files
            assert "demo.tables.json" in files
            assert "demo.drawings.json" in files
            assert "demo.equations.json" in files
            assert (parsed_dir / "demo.blocks.assets").is_dir()
            assert (parsed_dir / "demo.blocks.assets" / "img_000000.png").is_file()

            blocks_raw = (parsed_dir / "demo.blocks.jsonl").read_text()
            lines = blocks_raw.splitlines()
            meta = json.loads(lines[0])
            rows = [json.loads(line) for line in lines[1:]]
            assert meta["parse_engine"] == "docling"
            assert meta["bbox_attributes"] == {"origin": "LEFTBOTTOM"}
            assert "max" not in meta["bbox_attributes"]
            assert "page_sizes" not in meta["bbox_attributes"]
            assert meta["table_file"] is True
            assert meta["drawing_file"] is True
            assert meta["equation_file"] is True
            # No label="title" in the fixture (matches the typical PDF case
            # where docling produces only section_headers) → doc_title falls
            # back to the document stem.
            assert meta["doc_title"] == "demo"

            contents = " ".join(row.get("content", "") for row in rows)
            assert '<table id="tb-' in contents
            assert "<drawing" in contents
            assert "<equation" in contents

            # Raw bundle preserved next to sidecar
            raw_dir = parsed_dir.parent / "demo.pdf.docling_raw"
            assert (raw_dir / "_manifest.json").is_file()
            assert (raw_dir / "demo.json").is_file()
            assert (raw_dir / "demo.md").is_file()
            assert (raw_dir / "artifacts" / "img_000000.png").is_file()

            # Drawing path correctly resolved
            drawings = json.loads((parsed_dir / "demo.drawings.json").read_text())[
                "drawings"
            ]
            (drawing_id, drawing_item) = next(iter(drawings.items()))
            assert drawing_id.startswith("im-")
            assert drawing_item["path"] == "demo.blocks.assets/img_000000.png"

            # Table self_ref propagated
            tables = json.loads((parsed_dir / "demo.tables.json").read_text())["tables"]
            (_, table_item) = next(iter(tables.items()))
            assert table_item.get("self_ref") == "#/tables/0"
        finally:
            await rag.finalize_storages()

    asyncio.new_event_loop().run_until_complete(_run())


@pytest.mark.offline
def test_parse_docling_cache_hit_skips_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")
        counters = _install_fake_download(monkeypatch)

        input_dir = tmp_path / "inputs" / "ws"
        input_dir.mkdir(parents=True)
        src = input_dir / "demo.pdf"
        src.write_bytes(b"PDFPDF" * 256)

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            _stub_pipeline(monkeypatch, rag, src)
            doc_id = "doc-abcdef0123456789abcdef0123456789"
            await _seed_doc_status(rag, doc_id)

            await rag.parse_docling(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 1

            await rag.parse_docling(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 1, "cache hit must not re-download"

            monkeypatch.setenv("LIGHTRAG_FORCE_REPARSE_DOCLING", "true")
            await rag.parse_docling(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 2
        finally:
            await rag.finalize_storages()

    asyncio.new_event_loop().run_until_complete(_run())


@pytest.mark.offline
def test_parse_docling_cache_invalidates_on_source_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")
        counters = _install_fake_download(monkeypatch)

        input_dir = tmp_path / "inputs" / "ws"
        input_dir.mkdir(parents=True)
        src = input_dir / "demo.pdf"
        src.write_bytes(b"PDFPDF" * 256)

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            _stub_pipeline(monkeypatch, rag, src)
            doc_id = "doc-abcdef0123456789abcdef0123456789"
            await _seed_doc_status(rag, doc_id)

            await rag.parse_docling(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 1

            data = src.read_bytes()
            src.write_bytes(b"\x00" + data[1:])

            await rag.parse_docling(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 2
        finally:
            await rag.finalize_storages()

    asyncio.new_event_loop().run_until_complete(_run())


@pytest.mark.offline
def test_parse_docling_options_signature_invalidates_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")
        counters = _install_fake_download(monkeypatch)

        input_dir = tmp_path / "inputs" / "ws"
        input_dir.mkdir(parents=True)
        src = input_dir / "demo.pdf"
        src.write_bytes(b"PDFPDF" * 256)

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            _stub_pipeline(monkeypatch, rag, src)
            doc_id = "doc-abcdef0123456789abcdef0123456789"
            await _seed_doc_status(rag, doc_id)

            await rag.parse_docling(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 1

            # Flip an env var that participates in the options signature
            monkeypatch.setenv("DOCLING_OCR_LANG", "en,zh")
            await rag.parse_docling(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert (
                counters["calls"] == 2
            ), "DOCLING_OCR_LANG change must invalidate the bundle cache"
        finally:
            await rag.finalize_storages()

    asyncio.new_event_loop().run_until_complete(_run())


@pytest.mark.offline
def test_parse_docling_endpoint_signature_invalidates_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")
        counters = _install_fake_download(monkeypatch)

        input_dir = tmp_path / "inputs" / "ws"
        input_dir.mkdir(parents=True)
        src = input_dir / "demo.pdf"
        src.write_bytes(b"PDFPDF" * 256)

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            _stub_pipeline(monkeypatch, rag, src)
            doc_id = "doc-abcdef0123456789abcdef0123456789"
            await _seed_doc_status(rag, doc_id)

            await rag.parse_docling(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert counters["calls"] == 1

            # Pointing at a different docling-serve instance must not silently
            # reuse a bundle that was produced by the previous one.
            monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling-other.test")
            await rag.parse_docling(
                doc_id=doc_id,
                file_path="demo.pdf",
                content_data={},
            )
            assert (
                counters["calls"] == 2
            ), "DOCLING_ENDPOINT change must invalidate the bundle cache"
        finally:
            await rag.finalize_storages()

    asyncio.new_event_loop().run_until_complete(_run())


@pytest.mark.offline
def test_parse_docling_zero_blocks_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the docling bundle yields no body blocks (e.g. everything was
    classified as furniture/background) ``parse_docling`` must fail loudly
    so the document is marked failed — never persist a half-baked sidecar.
    """

    async def _run() -> None:
        monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")

        # Install a fake download that writes a valid bundle whose body has
        # no children — the adapter then produces zero IR blocks.
        import lightrag.external_parser.docling.client as client_mod

        empty_json: dict[str, Any] = {
            "schema_name": "DoclingDocument",
            "version": "1.10.0",
            "origin": {"filename": "demo.pdf", "mimetype": "application/pdf"},
            "body": {
                "self_ref": "#/body",
                "children": [],
                "content_layer": "body",
                "label": "unspecified",
            },
            "groups": [],
            "texts": [],
            "tables": [],
            "pictures": [],
            "key_value_items": [],
            "form_items": [],
            "pages": {},
        }

        async def _fake_download(
            self, raw_dir: Path, source_file_path: Path, **_kwargs
        ):
            raw_dir.mkdir(parents=True, exist_ok=True)
            main_json = raw_dir / "demo.json"
            main_json.write_text(json.dumps(empty_json), encoding="utf-8")
            (raw_dir / "demo.md").write_text("# empty", encoding="utf-8")

            src_size, src_hash = compute_size_and_hash(source_file_path)
            crit_size, crit_hash = compute_size_and_hash(main_json)
            others = [
                ManifestFile(path="demo.md", size=(raw_dir / "demo.md").stat().st_size),
            ]
            options_signature = compute_options_signature(
                tunable_env=snapshot_tunable_env(),
                fixed_constants=FIXED_CONSTANTS,
            )
            manifest = Manifest(
                engine="docling",
                source_content_hash=src_hash,
                source_size_bytes=src_size,
                source_filename_at_parse=source_file_path.name,
                critical_file=ManifestFile(
                    path="demo.json", size=crit_size, sha256=crit_hash
                ),
                files=others,
                total_size_bytes=crit_size + sum(f.size for f in others),
                task_id="fake-empty",
                endpoint_signature="http://docling.test",
                options_signature=options_signature,
                extras={"fixed_constants": dict(FIXED_CONSTANTS)},
                downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            write_manifest(raw_dir, manifest)
            return manifest

        monkeypatch.setattr(
            client_mod.DoclingRawClient, "download_into", _fake_download
        )

        input_dir = tmp_path / "inputs" / "ws"
        input_dir.mkdir(parents=True)
        src = input_dir / "demo.pdf"
        src.write_bytes(b"PDFPDF" * 256)

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            _stub_pipeline(monkeypatch, rag, src)
            doc_id = "doc-abcdef0123456789abcdef0123456789"
            await _seed_doc_status(rag, doc_id)

            with pytest.raises(ValueError, match="zero blocks"):
                await rag.parse_docling(
                    doc_id=doc_id,
                    file_path="demo.pdf",
                    content_data={},
                )

            # Sidecar must NOT have been emitted: ``write_sidecar`` is reached
            # only after the zero-blocks check, so no ``*.blocks.jsonl`` may
            # exist anywhere under the workspace.
            blocks_files = list(tmp_path.rglob("*.blocks.jsonl"))
            assert (
                not blocks_files
            ), f"sidecar emitted despite zero-blocks failure: {blocks_files}"
        finally:
            await rag.finalize_storages()

    asyncio.new_event_loop().run_until_complete(_run())
