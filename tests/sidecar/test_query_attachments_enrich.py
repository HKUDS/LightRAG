"""Layer 3: enrich_raw_data_attachments (resolve-only from metadata whitelist)."""

from __future__ import annotations

import pytest

from lightrag.sidecar.query_attachments import (
    clear_drawings_index_cache,
    drawing_candidate_whitelist_from_raw_data,
    enrich_raw_data_attachments,
    resolve_drawing_attachments,
)
from tests.sidecar.conftest_query_attachments import (
    FakeFullDocs,
    SYNTH_DOC_ID,
    SYNTH_IM_0001,
    SYNTH_IM_0002,
    SYNTH_IM_0002_TITLE,
    SYNTH_JPG_1,
    SYNTH_JPG_2,
    assert_valid_attachment,
    remove_sidecar_asset,
    write_drawings_json,
)


@pytest.fixture(autouse=True)
def _clear_drawings_cache():
    clear_drawings_index_cache()
    yield
    clear_drawings_index_cache()


@pytest.mark.offline
class TestDrawingCandidateWhitelistFromRawData:
    def test_reads_whitelist(self):
        raw = {
            "metadata": {
                "drawing_candidate_whitelist": [SYNTH_IM_0001, SYNTH_IM_0002],
            }
        }
        assert drawing_candidate_whitelist_from_raw_data(raw) == [
            SYNTH_IM_0001,
            SYNTH_IM_0002,
        ]

    def test_empty_without_metadata(self):
        assert drawing_candidate_whitelist_from_raw_data({}) == []
        assert drawing_candidate_whitelist_from_raw_data({"metadata": {}}) == []

    def test_dedupes_and_skips_invalid_ids(self):
        raw = {
            "metadata": {
                "drawing_candidate_whitelist": [
                    SYNTH_IM_0001,
                    SYNTH_IM_0001,
                    "not-an-im-id",
                    SYNTH_IM_0002,
                    "",
                    None,
                ],
            }
        }
        assert drawing_candidate_whitelist_from_raw_data(raw) == [
            SYNTH_IM_0001,
            SYNTH_IM_0002,
        ]

    def test_non_list_whitelist_returns_empty(self):
        raw = {"metadata": {"drawing_candidate_whitelist": SYNTH_IM_0001}}
        assert drawing_candidate_whitelist_from_raw_data(raw) == []


@pytest.mark.offline
class TestEnrichRawDataAttachments:
    @pytest.mark.asyncio
    async def test_enrich_empty_raw_data(self, synthetic_sidecar_uri):
        raw = {}
        result = await enrich_raw_data_attachments(
            raw, FakeFullDocs(synthetic_sidecar_uri)
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_enrich_empty_whitelist(self, synthetic_sidecar_uri):
        raw = {
            "data": {"chunks": [{"content": "no drawings"}]},
            "metadata": {"drawing_candidate_whitelist": []},
        }
        result = await enrich_raw_data_attachments(
            raw, FakeFullDocs(synthetic_sidecar_uri)
        )
        assert result["data"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_enrich_from_whitelist_not_chunk_scan(self, synthetic_sidecar_uri):
        """Enrich uses metadata whitelist; chunk content is not scanned for im-ids."""
        chunk_id = f"{SYNTH_DOC_ID}-chunk-017"
        raw = {
            "data": {
                "chunks": [
                    {
                        "chunk_id": chunk_id,
                        "content": "plain text without drawing tags",
                        "file_path": "doc.md",
                    }
                ],
            },
            "metadata": {
                "drawing_candidate_whitelist": [SYNTH_IM_0002],
            },
        }
        result = await enrich_raw_data_attachments(
            raw,
            FakeFullDocs(synthetic_sidecar_uri),
        )

        attachments = result["data"]["attachments"]
        assert len(attachments) == 1
        golden = attachments[0]
        assert_valid_attachment(golden)
        assert golden["im_id"] == SYNTH_IM_0002
        assert SYNTH_JPG_2 in golden["path"]
        assert golden["title"] == SYNTH_IM_0002_TITLE

    @pytest.mark.asyncio
    async def test_enrich_without_full_docs(self):
        raw = {
            "data": {"chunks": []},
            "metadata": {"drawing_candidate_whitelist": [SYNTH_IM_0001]},
        }
        result = await enrich_raw_data_attachments(raw, None)
        assert result["data"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_enrich_degrades_on_operational_storage_error(self):
        """Enrich degrades to [] for ``_SIDECAR_DEGRADE_ERRORS`` (e.g. ConnectionError)."""

        class BrokenStorageError(ConnectionError):
            pass

        class BrokenFullDocs:
            async def get_by_id(self, doc_id: str):
                raise BrokenStorageError("storage down")

        raw = {
            "data": {"chunks": []},
            "metadata": {"drawing_candidate_whitelist": [SYNTH_IM_0001]},
        }

        result = await enrich_raw_data_attachments(raw, BrokenFullDocs())
        assert result["data"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_enrich_degrades_when_drawings_index_raises_oserror(
        self, synthetic_sidecar_uri, monkeypatch
    ):
        """Prefetch OSError on aload_drawings_index → attachments [] (no query abort)."""

        async def boom(_uri):
            raise OSError("sidecar mount unavailable")

        monkeypatch.setattr(
            "lightrag.sidecar.query_attachments.aload_drawings_index",
            boom,
        )
        raw = {
            "data": {"chunks": []},
            "metadata": {"drawing_candidate_whitelist": [SYNTH_IM_0001]},
        }
        result = await enrich_raw_data_attachments(
            raw,
            FakeFullDocs(synthetic_sidecar_uri),
        )
        assert result["data"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_enrich_propagates_unexpected_exception(self):
        """Unexpected Exception subclasses outside the degrade tuple propagate."""

        class WeirdStorageError(Exception):
            pass

        class BrokenFullDocs:
            async def get_by_id(self, doc_id: str):
                raise WeirdStorageError("unexpected")

        raw = {
            "data": {"chunks": []},
            "metadata": {"drawing_candidate_whitelist": [SYNTH_IM_0001]},
        }

        with pytest.raises(WeirdStorageError):
            await enrich_raw_data_attachments(raw, BrokenFullDocs())

    @pytest.mark.asyncio
    async def test_attachments_order_matches_whitelist_not_im_id_sort(
        self, synthetic_sidecar_uri
    ):
        """attachments[] follows metadata whitelist order (collect order), not sorted im-id."""
        im_to_doc = {SYNTH_IM_0002: SYNTH_DOC_ID, SYNTH_IM_0001: SYNTH_DOC_ID}
        ordered_ids = [SYNTH_IM_0002, SYNTH_IM_0001]
        attachments = await resolve_drawing_attachments(
            ordered_ids,
            im_to_doc,
            FakeFullDocs(synthetic_sidecar_uri),
        )
        assert [a["im_id"] for a in attachments] == ordered_ids

    @pytest.mark.asyncio
    async def test_enrich_partial_success_skips_missing_jpg(
        self, synthetic_sidecar, synthetic_sidecar_uri
    ):
        """Enrich skips unverifiable assets but keeps survivors in whitelist order."""
        remove_sidecar_asset(synthetic_sidecar, SYNTH_JPG_1)
        raw = {
            "data": {"chunks": []},
            "metadata": {
                "drawing_candidate_whitelist": [SYNTH_IM_0001, SYNTH_IM_0002],
            },
        }
        result = await enrich_raw_data_attachments(
            raw,
            FakeFullDocs(synthetic_sidecar_uri),
        )
        attachments = result["data"]["attachments"]
        assert len(attachments) == 1
        assert attachments[0]["im_id"] == SYNTH_IM_0002

    @pytest.mark.asyncio
    async def test_enrich_skips_embedded_nul_path_without_raising(
        self, synthetic_sidecar, synthetic_sidecar_uri
    ):
        """Corrupt drawings.json path with NUL → skip that im-id; query must not fail."""
        write_drawings_json(
            synthetic_sidecar,
            {
                SYNTH_IM_0001: {
                    "id": SYNTH_IM_0001,
                    "format": "jpg",
                    "path": f"assets/{SYNTH_JPG_1}\x00evil",
                },
                SYNTH_IM_0002: {
                    "id": SYNTH_IM_0002,
                    "format": "jpg",
                    "path": f"assets/{SYNTH_JPG_2}",
                    "llm_analyze_result": {"name": SYNTH_IM_0002_TITLE},
                },
            },
        )
        raw = {
            "data": {"chunks": []},
            "metadata": {
                "drawing_candidate_whitelist": [SYNTH_IM_0001, SYNTH_IM_0002],
            },
        }
        result = await enrich_raw_data_attachments(
            raw,
            FakeFullDocs(synthetic_sidecar_uri),
        )
        attachments = result["data"]["attachments"]
        assert [a["im_id"] for a in attachments] == [SYNTH_IM_0002]
