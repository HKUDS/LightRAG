"""Shared helpers for query attachment tests (CI-portable tmp_path)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lightrag.utils_pipeline import sidecar_uri_for

DOC_HASH = "a" * 32
SYNTH_DOC_ID = f"doc-{DOC_HASH}"
SYNTH_IM_0001 = f"im-{DOC_HASH}-0001"
SYNTH_IM_0002 = f"im-{DOC_HASH}-0002"
SYNTH_IM_0001_TITLE = "Figure One"
SYNTH_IM_0002_TITLE = "Figure Two"
SYNTH_JPG_1 = "image-001.jpg"
SYNTH_JPG_2 = "image-002.jpg"


def build_synthetic_sidecar(tmp_path: Path) -> dict[str, str | Path]:
    """Minimal parsed sidecar tree: blocks.jsonl, drawings.json, asset jpgs."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    stem = SYNTH_DOC_ID

    blocks_path = parsed_dir / f"{stem}.blocks.jsonl"
    blocks_path.write_text("{}\n", encoding="utf-8")

    assets_dir = parsed_dir / f"{stem}.blocks.assets"
    assets_dir.mkdir()
    (assets_dir / SYNTH_JPG_1).write_bytes(b"\xff\xd8\xff\xd9")
    (assets_dir / SYNTH_JPG_2).write_bytes(b"\xff\xd8\xff\xd9")

    drawings_payload = {
        "drawings": {
            SYNTH_IM_0001: {
                "id": SYNTH_IM_0001,
                "format": "jpg",
                "path": f"assets/{SYNTH_JPG_1}",
                "heading": "Section A",
                "llm_analyze_result": {"name": SYNTH_IM_0001_TITLE},
            },
            SYNTH_IM_0002: {
                "id": SYNTH_IM_0002,
                "format": "jpg",
                "path": f"assets/{SYNTH_JPG_2}",
                "heading": "Section B",
                "llm_analyze_result": {"name": SYNTH_IM_0002_TITLE},
            },
        }
    }
    drawings_path = parsed_dir / f"{stem}.drawings.json"
    drawings_path.write_text(
        json.dumps(drawings_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    sidecar_uri = sidecar_uri_for(parsed_dir)
    return {
        "parsed_dir": parsed_dir,
        "sidecar_uri": sidecar_uri,
        "doc_id": SYNTH_DOC_ID,
        "im_0001": SYNTH_IM_0001,
        "im_0002": SYNTH_IM_0002,
    }


class FakeTextChunks:
    """Minimal text_chunks stub keyed by chunk_id (Index needs sidecar records)."""

    def __init__(self, records: dict[str, dict]):
        self._records = records

    async def get_by_ids(self, ids: list[str]):
        return [self._records[cid] for cid in ids if cid in self._records]


@dataclass(frozen=True)
class GoldenPipelineInputs:
    """Shared golden-path inputs for full aquery pipeline tests."""

    truncated_chunks: list[dict]
    text_chunks_db: FakeTextChunks
    full_docs_db: FakeFullDocs
    expected_collected: list[str]
    expected_whitelist: list[str] | None = None
    expected_attachment_ids: list[str] | None = None

    @property
    def resolved_whitelist(self) -> list[str]:
        if self.expected_whitelist is not None:
            return self.expected_whitelist
        return self.expected_collected

    @property
    def resolved_attachment_ids(self) -> list[str]:
        if self.expected_attachment_ids is not None:
            return self.expected_attachment_ids
        return self.resolved_whitelist


def build_golden_pipeline_inputs(sidecar_uri: str) -> GoldenPipelineInputs:
    """Text chunk (0001) before figure chunk (0002); Collect order is [0002, 0001]."""
    fig_id = f"{SYNTH_DOC_ID}-chunk-fig-001"
    text_id = f"{SYNTH_DOC_ID}-chunk-010"
    truncated_chunks = [
        {
            "reference_id": "1",
            "content": f'See <drawing id="{SYNTH_IM_0001}"/> in body.',
            "chunk_id": text_id,
            "file_path": "doc.md",
        },
        {
            "reference_id": "2",
            "content": "Figure caption only.",
            "chunk_id": fig_id,
            "file_path": "doc.md",
        },
    ]
    chunk_records = {
        fig_id: {
            "chunk_id": fig_id,
            "full_doc_id": SYNTH_DOC_ID,
            "sidecar": {
                "type": "drawing",
                "id": SYNTH_IM_0002,
                "refs": [{"type": "drawing", "id": SYNTH_IM_0002}],
            },
        },
        text_id: {
            "chunk_id": text_id,
            "full_doc_id": SYNTH_DOC_ID,
            "sidecar": {
                "type": "block",
                "id": "blk-001",
                "refs": [{"type": "block", "id": "blk-001"}],
            },
        },
    }
    return GoldenPipelineInputs(
        truncated_chunks=truncated_chunks,
        text_chunks_db=FakeTextChunks(chunk_records),
        full_docs_db=FakeFullDocs(sidecar_uri),
        expected_collected=[SYNTH_IM_0002, SYNTH_IM_0001],
    )


def _figure_text_chunk_records(fig_id: str, text_id: str) -> dict[str, dict]:
    return {
        fig_id: {
            "chunk_id": fig_id,
            "full_doc_id": SYNTH_DOC_ID,
            "sidecar": {
                "type": "drawing",
                "id": SYNTH_IM_0002,
                "refs": [{"type": "drawing", "id": SYNTH_IM_0002}],
            },
        },
        text_id: {
            "chunk_id": text_id,
            "full_doc_id": SYNTH_DOC_ID,
            "sidecar": {
                "type": "block",
                "id": "blk-001",
                "refs": [{"type": "block", "id": "blk-001"}],
            },
        },
    }


def build_cross_bucket_dedupe_pipeline_inputs(
    sidecar_uri: str,
) -> GoldenPipelineInputs:
    """0002 in figure sidecar and text cite; merge keeps figure bucket only."""
    fig_id = f"{SYNTH_DOC_ID}-chunk-fig-001"
    text_id = f"{SYNTH_DOC_ID}-chunk-010"
    truncated_chunks = [
        {
            "reference_id": "1",
            "content": (
                f'Cites <drawing id="{SYNTH_IM_0002}"/> and '
                f'<drawing id="{SYNTH_IM_0001}"/>.'
            ),
            "chunk_id": text_id,
            "file_path": "doc.md",
        },
        {
            "reference_id": "2",
            "content": "Figure caption.",
            "chunk_id": fig_id,
            "file_path": "doc.md",
        },
    ]
    return GoldenPipelineInputs(
        truncated_chunks=truncated_chunks,
        text_chunks_db=FakeTextChunks(_figure_text_chunk_records(fig_id, text_id)),
        full_docs_db=FakeFullDocs(sidecar_uri),
        expected_collected=[SYNTH_IM_0002, SYNTH_IM_0001],
    )


class FakeFullDocs:
    """Minimal full_docs stub for sidecar resolution tests."""

    def __init__(self, sidecar_uri: str, doc_id: str = SYNTH_DOC_ID):
        self._uri = sidecar_uri
        self._doc_id = doc_id

    async def get_by_id(self, doc_id: str):
        if doc_id == self._doc_id:
            return {"sidecar_location": self._uri}
        return None


class CountingFakeFullDocs(FakeFullDocs):
    """``FakeFullDocs`` that counts ``get_by_id`` calls (prefetch batching tests)."""

    def __init__(self, sidecar_uri: str, doc_id: str = SYNTH_DOC_ID):
        super().__init__(sidecar_uri, doc_id=doc_id)
        self.get_by_id_calls = 0

    async def get_by_id(self, doc_id: str):
        self.get_by_id_calls += 1
        return await super().get_by_id(doc_id)


REQUIRED_ATTACHMENT_KEYS = frozenset({"im_id", "doc_id", "type", "format", "path"})


def assert_valid_attachment(item: dict) -> None:
    assert REQUIRED_ATTACHMENT_KEYS <= item.keys()
    assert item["type"] == "drawing"
    assert str(item["im_id"]).startswith("im-")
    assert str(item["doc_id"]).startswith("doc-")


def corrupt_drawings_json(synthetic_sidecar: dict[str, str | Path]) -> Path:
    """Overwrite ``drawings.json`` with invalid JSON (Index/resolve degrade tests).

    Mimics a truncated / half-written file: trailing comma inside an object so
    ``json.loads`` raises ``JSONDecodeError``.
    """
    parsed_dir = Path(synthetic_sidecar["parsed_dir"])
    stem = str(synthetic_sidecar["doc_id"])
    drawings_path = parsed_dir / f"{stem}.drawings.json"
    drawings_path.write_text(
        '{"drawings": {"im-placeholder": {"path": "assets/x.jpg",}}}',
        encoding="utf-8",
    )
    return drawings_path


def remove_sidecar_asset(
    synthetic_sidecar: dict[str, str | Path], jpg_name: str
) -> None:
    """Delete one asset file under the synthetic sidecar tree."""
    parsed_dir = Path(synthetic_sidecar["parsed_dir"])
    stem = str(synthetic_sidecar["doc_id"])
    asset = parsed_dir / f"{stem}.blocks.assets" / jpg_name
    asset.unlink(missing_ok=True)


def write_drawings_json(
    synthetic_sidecar: dict[str, str | Path], drawings: dict[str, dict]
) -> Path:
    """Replace ``drawings.json`` ``drawings`` map (keeps other sidecar files)."""
    parsed_dir = Path(synthetic_sidecar["parsed_dir"])
    stem = str(synthetic_sidecar["doc_id"])
    drawings_path = parsed_dir / f"{stem}.drawings.json"
    drawings_path.write_text(
        json.dumps({"drawings": drawings}, ensure_ascii=False),
        encoding="utf-8",
    )
    return drawings_path
