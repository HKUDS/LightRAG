"""PR-1 L3 tests: ``enrich_raw_data_attachments`` end-to-end with mocked storage.

Layer 3 wires L1 collection and L2 sidecar resolution together and writes the
final ``raw_data["data"]["attachments"]`` list consumed by HTTP and the WebUI.

Coverage:
- Empty raw_data / no im-ids → stable structure or ``attachments: []``
- Golden single ``<drawing>`` chunk → full attachment record
- Missing ``full_docs`` → fail-soft empty attachments
- Idempotent re-enrich → identical attachment list

Uses ``FakeFullDocs`` for ``full_docs.get_by_id → sidecar_location`` and optional
``FakeTextChunks`` for chunk-level doc mapping. Depends on the GB/T sidecar fixture.
"""

from __future__ import annotations

import pytest

from lightrag.sidecar.query_attachments import (
    clear_drawings_index_cache,
    enrich_raw_data_attachments,
)
from tests.sidecar.conftest_query_attachments import (
    FakeFullDocs,
    GB_T_DOC_ID,
    GB_T_IM_ID_0002,
    GB_T_IM_0002_TITLE,
    GB_T_JPG_SUFFIX,
    assert_valid_attachment,
)

_DOC_HASH = "116da71bde652ed78cf28934d0712aa6"
_IM_0001 = f"im-{_DOC_HASH}-0001"


class FakeTextChunks:
    """Minimal text_chunks stub returning persisted chunk records by chunk_id."""

    def __init__(self, records: dict[str, dict]):
        self._records = records

    async def get_by_ids(self, ids: list[str]):
        return [self._records[cid] for cid in ids if cid in self._records]


@pytest.fixture(autouse=True)
def _clear_drawings_cache():
    """Reset drawings.json cache so L2 disk reads do not leak between tests."""
    clear_drawings_index_cache()
    yield
    clear_drawings_index_cache()


@pytest.mark.offline
class TestEnrichRawDataAttachments:
    """End-to-end enrich: collect → resolve → write ``data.attachments``."""

    @pytest.mark.asyncio
    async def test_enrich_empty_raw_data(self, gb_t_sidecar_uri):
        """Empty input dict is returned unchanged without creating a ``data`` key."""
        raw = {}
        result = await enrich_raw_data_attachments(
            raw, None, FakeFullDocs(gb_t_sidecar_uri)
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_enrich_no_im_ids(self, gb_t_sidecar_uri):
        """Chunks without drawing tags produce ``attachments: []``."""
        raw = {"data": {"entities": [], "chunks": [{"content": "no drawings"}]}}
        result = await enrich_raw_data_attachments(
            raw, None, FakeFullDocs(gb_t_sidecar_uri)
        )
        assert result["data"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_enrich_golden_shape(self, gb_t_sidecar_uri):
        """Golden chunk with GB_T_IM_ID_0002 yields a valid attachment without mutating other fields.

        Verifies the PR-1 contract:
        - at least one attachment with expected jpg path and title
        - ``assert_valid_attachment`` checks required keys
        - references/chunks remain identical (enrich only appends attachments)
        """
        chunk_id = f"{GB_T_DOC_ID}-chunk-017"
        raw = {
            "data": {
                "entities": [],
                "chunks": [
                    {
                        "chunk_id": chunk_id,
                        "content": f'<drawing id="{GB_T_IM_ID_0002}" format="jpg"/>',
                        "file_path": "MinerU_markdown_GB_T_70.3-2023.md",
                    }
                ],
                "references": [{"reference_id": "1", "file_path": "x.md"}],
            }
        }
        # text_chunks supplies chunk-level full_doc_id for im-id → doc_id mapping.
        text_chunks = FakeTextChunks(
            {
                chunk_id: {
                    "chunk_id": chunk_id,
                    "full_doc_id": GB_T_DOC_ID,
                    "content": raw["data"]["chunks"][0]["content"],
                }
            }
        )
        refs_before = list(raw["data"]["references"])

        result = await enrich_raw_data_attachments(
            raw,
            text_chunks,
            FakeFullDocs(gb_t_sidecar_uri),
        )

        attachments = result["data"]["attachments"]
        assert len(attachments) >= 1
        golden = next(a for a in attachments if a["im_id"] == GB_T_IM_ID_0002)
        assert_valid_attachment(golden)
        assert GB_T_JPG_SUFFIX in golden["path"]
        assert golden["title"] == GB_T_IM_0002_TITLE
        assert result["data"]["references"] == refs_before
        assert result["data"]["chunks"] == raw["data"]["chunks"]

    @pytest.mark.asyncio
    async def test_enrich_without_full_docs(self):
        """When full_docs is unavailable, enrich degrades to ``attachments: []`` (fail-soft)."""
        raw = {
            "data": {
                "chunks": [
                    {"content": f'<drawing id="{_IM_0001}"/>', "chunk_id": "c1"}
                ]
            }
        }
        result = await enrich_raw_data_attachments(raw, None, None)
        assert result["data"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_enrich_idempotent(self, gb_t_sidecar_uri):
        """Calling enrich twice on the same raw_data yields an identical attachment list."""
        raw = {
            "data": {
                "chunks": [
                    {
                        "chunk_id": f"{GB_T_DOC_ID}-chunk-1",
                        "content": f'<drawing id="{GB_T_IM_ID_0002}"/>',
                    }
                ]
            }
        }
        full_docs = FakeFullDocs(gb_t_sidecar_uri)
        first = await enrich_raw_data_attachments(raw, None, full_docs)
        second = await enrich_raw_data_attachments(first, None, full_docs)
        assert second["data"]["attachments"] == first["data"]["attachments"]
