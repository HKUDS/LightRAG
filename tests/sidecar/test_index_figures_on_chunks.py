"""Tests for ``index_figures_on_chunks`` (whitelist-only metadata)."""

from __future__ import annotations

import pytest

from lightrag.sidecar.query_attachments import (
    apply_index_figures_to_raw_data,
    clear_drawings_index_cache,
    index_figures_on_chunks,
)
from tests.sidecar.conftest_query_attachments import (
    CountingFakeFullDocs,
    FakeFullDocs,
    FakeTextChunks,
    SYNTH_DOC_ID,
    SYNTH_IM_0001,
    SYNTH_IM_0002,
    SYNTH_JPG_1,
    corrupt_drawings_json,
    remove_sidecar_asset,
    write_drawings_json,
)


@pytest.fixture(autouse=True)
def _clear_drawings_cache():
    clear_drawings_index_cache()
    yield
    clear_drawings_index_cache()


@pytest.mark.offline
class TestIndexFiguresOnChunks:
    @pytest.mark.asyncio
    async def test_index_builds_whitelist(self, synthetic_sidecar_uri):
        chunks = [
            {
                "reference_id": "1",
                "content": f'Figure <drawing id="{SYNTH_IM_0001}"/> here.',
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-010",
                "file_path": "doc.md",
            },
            {
                "reference_id": "2",
                "content": "No figures in this chunk.",
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-011",
                "file_path": "doc.md",
            },
            {
                "reference_id": "1",
                "content": f'Also cites <drawing id="{SYNTH_IM_0002}"/>.',
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-012",
                "file_path": "doc.md",
            },
        ]
        result = await index_figures_on_chunks(
            chunks,
            text_chunks_db=None,
            full_docs_db=FakeFullDocs(synthetic_sidecar_uri),
        )
        assert result.whitelist == [SYNTH_IM_0001, SYNTH_IM_0002]

    @pytest.mark.asyncio
    async def test_no_full_docs_empty_whitelist(self):
        chunks = [
            {
                "reference_id": "1",
                "content": f'<drawing id="{SYNTH_IM_0001}"/>',
                "chunk_id": "x",
            }
        ]
        result = await index_figures_on_chunks(chunks, None, None)
        assert result.whitelist == []

    @pytest.mark.asyncio
    async def test_index_whitelist_preserves_text_bucket_scan_order(
        self, synthetic_sidecar_uri
    ):
        """Text-only chunks: whitelist follows collect scan order within text bucket."""
        chunks = [
            {
                "reference_id": "1",
                "content": f'Later seq <drawing id="{SYNTH_IM_0002}"/>',
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-010",
                "file_path": "doc.md",
            },
            {
                "reference_id": "2",
                "content": f'Earlier seq <drawing id="{SYNTH_IM_0001}"/>',
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-011",
                "file_path": "doc.md",
            },
        ]
        result = await index_figures_on_chunks(
            chunks,
            text_chunks_db=None,
            full_docs_db=FakeFullDocs(synthetic_sidecar_uri),
        )
        assert result.whitelist == [SYNTH_IM_0002, SYNTH_IM_0001]

    @pytest.mark.asyncio
    async def test_index_preserves_collect_order_text_before_figure_in_list(
        self, synthetic_sidecar_uri
    ):
        """Index filter skips only; figure im-ids stay before text when list order differs."""
        fig_id = f"{SYNTH_DOC_ID}-chunk-fig-001"
        text_id = f"{SYNTH_DOC_ID}-chunk-010"
        chunks = [
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
        text_chunks = FakeTextChunks(
            {
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
        )
        result = await index_figures_on_chunks(
            chunks,
            text_chunks_db=text_chunks,
            full_docs_db=FakeFullDocs(synthetic_sidecar_uri),
        )
        assert result.whitelist == [SYNTH_IM_0002, SYNTH_IM_0001]

    @pytest.mark.asyncio
    async def test_index_prefetches_full_doc_once_per_doc(self, synthetic_sidecar_uri):
        """Same doc with multiple im-ids should not repeat full_docs KV lookups."""
        chunks = [
            {
                "reference_id": "1",
                "content": f'<drawing id="{SYNTH_IM_0001}"/>',
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-010",
                "file_path": "doc.md",
            },
            {
                "reference_id": "2",
                "content": f'<drawing id="{SYNTH_IM_0002}"/>',
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-011",
                "file_path": "doc.md",
            },
        ]
        full_docs = CountingFakeFullDocs(synthetic_sidecar_uri)
        result = await index_figures_on_chunks(
            chunks,
            text_chunks_db=None,
            full_docs_db=full_docs,
        )
        assert result.whitelist == [SYNTH_IM_0001, SYNTH_IM_0002]
        assert full_docs.get_by_id_calls == 1

    @pytest.mark.asyncio
    async def test_apply_writes_whitelist_only(self, synthetic_sidecar_uri):
        chunks = [
            {
                "reference_id": "1",
                "content": f'<drawing id="{SYNTH_IM_0001}"/>',
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-010",
                "file_path": "doc.md",
            }
        ]
        result = await index_figures_on_chunks(
            chunks,
            text_chunks_db=None,
            full_docs_db=FakeFullDocs(synthetic_sidecar_uri),
        )
        raw = {
            "data": {"chunks": [dict(chunks[0])]},
            "metadata": {},
        }
        apply_index_figures_to_raw_data(raw, result)
        assert "figure_ids" not in raw["data"]["chunks"][0]
        assert raw["metadata"]["drawing_candidate_whitelist"] == [SYNTH_IM_0001]
        assert "figure_label_index" not in raw["metadata"]
        assert "figure_reference_lines" not in raw["metadata"]
        assert "figure_list_str" not in raw["metadata"]

    @pytest.mark.asyncio
    async def test_index_whitelists_entry_without_disk_jpg(
        self, synthetic_sidecar, synthetic_sidecar_uri
    ):
        """Index JSON filter only — missing on-disk asset still enters whitelist."""
        remove_sidecar_asset(synthetic_sidecar, SYNTH_JPG_1)
        chunks = [
            {
                "reference_id": "1",
                "content": f'<drawing id="{SYNTH_IM_0001}"/>',
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-010",
                "file_path": "doc.md",
            }
        ]
        result = await index_figures_on_chunks(
            chunks,
            text_chunks_db=None,
            full_docs_db=FakeFullDocs(synthetic_sidecar_uri),
        )
        assert result.whitelist == [SYNTH_IM_0001]

    @pytest.mark.asyncio
    async def test_index_skips_unknown_im_and_empty_path(
        self, synthetic_sidecar, synthetic_sidecar_uri
    ):
        """Index skips ids missing from drawings.json or with empty path (skip only)."""
        unknown_im = f"im-{SYNTH_DOC_ID.removeprefix('doc-')}-9999"
        write_drawings_json(
            synthetic_sidecar,
            {
                SYNTH_IM_0001: {
                    "id": SYNTH_IM_0001,
                    "format": "jpg",
                    "path": f"assets/{SYNTH_JPG_1}",
                },
                SYNTH_IM_0002: {
                    "id": SYNTH_IM_0002,
                    "format": "jpg",
                    "path": "",
                },
            },
        )
        chunks = [
            {
                "reference_id": "1",
                "content": (
                    f'<drawing id="{unknown_im}"/> '
                    f'<drawing id="{SYNTH_IM_0002}"/> '
                    f'<drawing id="{SYNTH_IM_0001}"/>'
                ),
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-010",
                "file_path": "doc.md",
            }
        ]
        result = await index_figures_on_chunks(
            chunks,
            text_chunks_db=None,
            full_docs_db=FakeFullDocs(synthetic_sidecar_uri),
        )
        assert result.whitelist == [SYNTH_IM_0001]

    @pytest.mark.asyncio
    async def test_index_empty_whitelist_when_drawings_unreadable(
        self, synthetic_sidecar, synthetic_sidecar_uri
    ):
        """Corrupt drawings.json → no indexable entries (Enrich would also fail resolve)."""
        corrupt_drawings_json(synthetic_sidecar)
        chunks = [
            {
                "reference_id": "1",
                "content": f'<drawing id="{SYNTH_IM_0001}"/>',
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-010",
                "file_path": "doc.md",
            }
        ]
        result = await index_figures_on_chunks(
            chunks,
            text_chunks_db=None,
            full_docs_db=FakeFullDocs(synthetic_sidecar_uri),
        )
        assert result.whitelist == []
