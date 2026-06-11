"""End-to-end test: native docx → LightRAG Document → stable cache key.

The original bug this guards against: ``parse_native`` used to write a
runtime-stamped structured parser payload into ``full_docs.content``, so
re-parsing the same docx produced different
chunk-0 content and therefore different LLM cache keys.

After the fix, ``parse_native`` writes ``.blocks.jsonl`` + sidecars and
``full_docs`` is in LIGHTRAG format. ``_load_lightrag_document_content``
skips the ``meta`` line (which contains ``parse_time``) and concatenates
only ``"type": "content"`` rows, so re-parsing must yield byte-identical
``merged_text`` and stable downstream chunk-0 content.
"""

import asyncio
import json
from pathlib import Path

import pytest

from lightrag import LightRAG
from lightrag.constants import (
    FULL_DOCS_FORMAT_PENDING_PARSE,
    PARSED_DIR_NAME,
)
from lightrag.parser.base import ParseContext
from lightrag.parser.registry import get_parser
from lightrag.utils import Tokenizer, TokenizerInterface, compute_args_hash


async def _parse_via_registry(rag, engine, doc_id, file_path, content_data):
    """Drive a parser the way the pipeline worker does (registry dispatch)."""
    result = await get_parser(engine).parse(
        ParseContext(rag, doc_id, file_path, content_data)
    )
    return result.to_dict()


def _block(content, *, heading="", level=0, parent=None, uuid="p1"):
    """Build a synthetic block dict matching extract_docx_blocks output."""
    return {
        "uuid": uuid,
        "uuid_end": uuid,
        "heading": heading,
        "content": content,
        "type": "text",
        "parent_headings": list(parent or []),
        "level": level,
        "table_chunk_role": "none",
    }


class _MiniFullDocs:
    def __init__(self):
        self.data = {}

    async def upsert(self, payload):
        self.data.update(payload)

    async def get_by_id(self, doc_id):
        return self.data.get(doc_id)

    async def index_done_callback(self):
        return None


class _MiniDocStatus:
    async def get_by_id(self, doc_id):
        return None

    async def upsert(self, data):
        return None


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class _MiniRag:
    """Just enough surface for parse_native + parser/docx adapter."""

    _persist_parsed_full_docs = LightRAG._persist_parsed_full_docs

    def __init__(self, working_dir):
        self.working_dir = str(working_dir)
        self.full_docs = _MiniFullDocs()
        self.doc_status = _MiniDocStatus()
        self.tokenizer = Tokenizer(model_name="char", tokenizer=_CharTokenizer())

    def _resolve_source_file_for_parser(self, file_path):
        return file_path


@pytest.mark.offline
def test_native_lightrag_path_produces_stable_merged_text(tmp_path, monkeypatch):
    """Re-parsing the same docx must yield byte-identical merged_text and
    therefore identical chunk_args_hash on chunk-0."""

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))

        source_path = input_dir / "stable.docx"
        source_path.write_bytes(b"fake docx bytes")

        # Stub extract_docx_blocks at the adapter so the upstream DOCX
        # parser is never invoked. The adapter still does all the
        # LightRAG-specific writing — that is what we want under test.
        stable_blocks = [
            _block(
                "Title\nFirst paragraph body.\nSecond paragraph body.",
                heading="Title",
                level=1,
            ),
        ]

        def _stub_extract(file_path, fixlevel=None, drawing_context=None, **kwargs):
            return [dict(b) for b in stable_blocks]

        monkeypatch.setattr(
            "lightrag.parser.docx.parse_document.extract_docx_blocks",
            _stub_extract,
        )

        rag = _MiniRag(tmp_path / "work")

        # ---- First parse ----
        # parse_native archives the source after writing, so re-create it
        # before the second parse for a fair comparison.
        result1 = await _parse_via_registry(
            rag,
            "native",
            "doc-stable",
            str(source_path),
            {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )
        merged1 = result1["content"]
        assert merged1, "first parse produced empty merged_text"

        # ---- Second parse ----
        # Restore the source file (archive moved it), reset the in-memory
        # full_docs row, and remove the parsed_dir so the writer rewrites
        # both meta (with a fresh parse_time) and content lines.
        source_path.write_bytes(b"fake docx bytes")
        rag.full_docs.data.clear()
        parsed_artifact_dir = input_dir / PARSED_DIR_NAME / f"{source_path.name}.parsed"
        if parsed_artifact_dir.exists():
            import shutil

            shutil.rmtree(parsed_artifact_dir)

        result2 = await _parse_via_registry(
            rag,
            "native",
            "doc-stable",
            str(source_path),
            {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )
        merged2 = result2["content"]

        # Core invariant: merged_text byte-identical across runs even
        # though parse_time in the .blocks.jsonl meta line differs.
        assert merged1 == merged2

        # And: a hash computed over a chunk-0 derived from merged_text
        # must also be identical — that is what powers LLM cache hits.
        prompt_template = "EXTRACT_PROMPT::{text}"
        chunk0_a = prompt_template.format(text=merged1[:200])
        chunk0_b = prompt_template.format(text=merged2[:200])
        assert chunk0_a == chunk0_b
        assert compute_args_hash(chunk0_a) == compute_args_hash(chunk0_b)

        # And: full_docs.content uses the {{LRdoc}} marker plus a leading
        # summary derived from merged_text (not the legacy placeholder).
        record = rag.full_docs.data["doc-stable"]
        assert record["parse_format"] == "lightrag"
        assert record["content"].startswith("{{LRdoc}}")
        assert merged1[:40] in record["content"]

    asyncio.run(_run())


@pytest.mark.offline
def test_native_lightrag_path_writes_blocks_jsonl_and_skips_meta_on_load(
    tmp_path, monkeypatch
):
    """Sanity check: ``_load_lightrag_document_content`` must skip the
    meta line (where the runtime ``parse_time`` lives) and only return
    body content. This is what lets re-parsing produce stable text."""

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))

        source_path = input_dir / "skipmeta.docx"
        source_path.write_bytes(b"fake")

        def _stub_extract(file_path, fixlevel=None, drawing_context=None, **kwargs):
            return [_block("the body")]

        monkeypatch.setattr(
            "lightrag.parser.docx.parse_document.extract_docx_blocks",
            _stub_extract,
        )

        rag = _MiniRag(tmp_path / "work")
        result = await _parse_via_registry(
            rag,
            "native",
            "doc-skip",
            str(source_path),
            {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )

        # The .blocks.jsonl on disk DOES contain "parse_time" inside the
        # meta line; the merged_text returned by parse_native MUST NOT.
        blocks_path = result["blocks_path"]
        on_disk = open(blocks_path, "r", encoding="utf-8").read()
        assert "parse_time" in on_disk
        assert "parse_time" not in result["content"]
        assert result["content"].strip() == "the body"

    asyncio.run(_run())


@pytest.mark.offline
def test_native_lightrag_path_leaves_unknown_table_caption_empty(tmp_path, monkeypatch):
    """The native DOCX parser does not infer table titles, so its table
    sidecar must not synthesize captions like ``表1``.
    """

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))

        source_path = input_dir / "table.docx"
        source_path.write_bytes(b"fake")

        def _stub_extract(file_path, fixlevel=None, drawing_context=None, **kwargs):
            return [_block('before\n<table>[["A"]]</table>\nafter')]

        monkeypatch.setattr(
            "lightrag.parser.docx.parse_document.extract_docx_blocks",
            _stub_extract,
        )

        rag = _MiniRag(tmp_path / "work")
        result = await _parse_via_registry(
            rag,
            "native",
            "doc-table",
            str(source_path),
            {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )

        blocks_path = Path(result["blocks_path"])
        lines = blocks_path.read_text(encoding="utf-8").splitlines()
        block = json.loads(lines[1])
        assert "caption=" not in block["content"]
        assert "表1" not in block["content"]

        tables_path = blocks_path.with_suffix("").with_suffix(".tables.json")
        tables = json.loads(tables_path.read_text(encoding="utf-8"))
        table_entry = tables["tables"]["tb-table-0001"]
        assert table_entry["caption"] == ""

        # Surrounding is now backfilled at analyze_multimodal entry, not in
        # parse_native — invoke the same routine directly to mirror that.
        from lightrag.multimodal_context import enrich_sidecars_with_surrounding

        enrich_sidecars_with_surrounding(
            blocks_path=str(blocks_path),
            enabled_modalities={"tables"},
            tokenizer=rag.tokenizer,
        )
        tables = json.loads(tables_path.read_text(encoding="utf-8"))
        table_entry = tables["tables"]["tb-table-0001"]
        assert table_entry["surrounding"] == {
            "leading": "before\n",
            "trailing": "\nafter",
        }

    asyncio.run(_run())


@pytest.mark.offline
def test_analyze_entrypoint_backfills_surrounding_for_all_sidecars(
    tmp_path, monkeypatch
):
    """Surrounding is backfilled at analyze_multimodal entry, covering native
    parse output as well as any other sidecar-producing engine."""

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))

        source_path = input_dir / "all_modalities.docx"
        source_path.write_bytes(b"fake")

        def _stub_extract(file_path, fixlevel=None, drawing_context=None, **kwargs):
            assert drawing_context is not None
            assert drawing_context.export_dir_path is not None
            (drawing_context.export_dir_path / "pic.png").write_bytes(b"PNG")
            return [
                _block(
                    'alpha <drawing id="1" format="png" '
                    'path="all_modalities.blocks.assets/pic.png" /> beta\n'
                    '<table>[["A"]]</table> gamma\n'
                    "<equation>E=mc^2</equation>\n"
                    "delta"
                )
            ]

        monkeypatch.setattr(
            "lightrag.parser.docx.parse_document.extract_docx_blocks",
            _stub_extract,
        )

        rag = _MiniRag(tmp_path / "work")
        result = await _parse_via_registry(
            rag,
            "native",
            "doc-mm",
            str(source_path),
            {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )

        blocks_path = Path(result["blocks_path"])
        base = str(blocks_path)[: -len(".blocks.jsonl")]

        # Parse-time sidecars must NOT contain surrounding — that field is
        # now produced at analyze_multimodal entry.
        for root in ("drawings", "tables", "equations"):
            payload = json.loads(Path(base + f".{root}.json").read_text("utf-8"))
            for item in payload[root].values():
                assert "surrounding" not in item

        # Now invoke the same routine analyze_multimodal calls and verify
        # all modalities get populated.
        from lightrag.multimodal_context import enrich_sidecars_with_surrounding

        enrich_sidecars_with_surrounding(
            blocks_path=str(blocks_path),
            enabled_modalities={"drawings", "tables", "equations"},
            tokenizer=rag.tokenizer,
        )
        for root in ("drawings", "tables", "equations"):
            payload = json.loads(Path(base + f".{root}.json").read_text("utf-8"))
            items = payload[root]
            assert items
            for item in items.values():
                assert "surrounding" in item
                assert set(item["surrounding"]) == {"leading", "trailing"}

    asyncio.run(_run())


@pytest.mark.offline
def test_native_lightrag_path_writes_image_assets_to_blocks_assets_dir(
    tmp_path, monkeypatch
):
    """Native parsing must drop image bytes into ``<base>.blocks.assets/``
    after the adapter creates the parsed dir (which it wipes at the start),
    and the drawings sidecar must reference the rewritten ids.
    """
    from pathlib import Path

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))

        source_path = input_dir / "with_pics.docx"
        source_path.write_bytes(b"fake")

        def _stub_extract(file_path, fixlevel=None, drawing_context=None, **kwargs):
            # The adapter already created the asset dir before calling us;
            # write the fake image bytes there as a side-effect, then return
            # a block whose content references that asset via <drawing .../>.
            assert drawing_context is not None
            assert drawing_context.export_dir_path is not None
            (drawing_context.export_dir_path / "pic.png").write_bytes(b"PNG-BYTES")
            return [
                _block(
                    "intro\n"
                    '<drawing id="1" name="pic" format="png" '
                    'path="with_pics.blocks.assets/pic.png" />\n'
                    "outro",
                    heading="intro",
                    level=1,
                ),
            ]

        monkeypatch.setattr(
            "lightrag.parser.docx.parse_document.extract_docx_blocks",
            _stub_extract,
        )

        rag = _MiniRag(tmp_path / "work")
        result = await _parse_via_registry(
            rag,
            "native",
            "doc-pic",
            str(source_path),
            {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )

        blocks_path = Path(result["blocks_path"])
        parsed_dir = blocks_path.parent
        asset_dir = parsed_dir / "with_pics.blocks.assets"
        # Asset dir must exist alongside .blocks.jsonl and survive the
        # adapter's parsed_dir cleanup step.
        assert asset_dir.is_dir(), (
            f"asset dir not created at {asset_dir}; parsed_dir contents: "
            f"{list(parsed_dir.iterdir())}"
        )
        assert (asset_dir / "pic.png").read_bytes() == b"PNG-BYTES"
        # And drawings.json sidecar should also be there since the block
        # contained a <drawing .../> markup the adapter had to record.
        assert (parsed_dir / "with_pics.drawings.json").is_file()

    asyncio.run(_run())
