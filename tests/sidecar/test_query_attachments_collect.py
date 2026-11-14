from __future__ import annotations

import pytest

from lightrag.sidecar.query_attachments import (
    _warn_im_id_doc_hash_mismatch,
    collect_im_ids_from_text_chunk,
    collect_im_ids_on_truncated_chunks,
    doc_id_from_chunk_id,
    doc_id_from_im_id,
    im_ids_to_doc_map,
    is_im_id,
)
from tests.sidecar.conftest_query_attachments import (
    SYNTH_DOC_ID,
    SYNTH_IM_0001,
    SYNTH_IM_0002,
)


@pytest.mark.offline
class TestImIdHelpers:
    def test_is_im_id_valid(self):
        assert is_im_id(SYNTH_IM_0001)
        assert is_im_id(SYNTH_IM_0002)

    def test_is_im_id_rejects_garbage(self):
        assert not is_im_id("")
        assert not is_im_id("not-an-id")
        assert not is_im_id("doc-abc-chunk-1")

    def test_doc_id_from_im_id(self):
        assert doc_id_from_im_id(SYNTH_IM_0001) == SYNTH_DOC_ID

    def test_doc_id_from_chunk_id(self):
        assert doc_id_from_chunk_id(f"{SYNTH_DOC_ID}-chunk-017") == SYNTH_DOC_ID


@pytest.mark.offline
class TestCollectFromTextChunk:
    def test_text_chunk_from_drawing_tags(self):
        chunk = {
            "content": (
                f'See <drawing id="{SYNTH_IM_0001}"/> and '
                f'<drawing id="{SYNTH_IM_0002}"/>.'
            ),
            "chunk_id": f"{SYNTH_DOC_ID}-chunk-001",
        }
        assert collect_im_ids_from_text_chunk(chunk) == [
            SYNTH_IM_0001,
            SYNTH_IM_0002,
        ]

    def test_empty_content(self):
        assert collect_im_ids_from_text_chunk({"content": "plain text"}) == []

    def test_content_single_quoted_id(self):
        chunk = {
            "content": f"<drawing id='{SYNTH_IM_0001}' format='jpg'/>",
        }
        assert collect_im_ids_from_text_chunk(chunk) == [SYNTH_IM_0001]


@pytest.mark.offline
class TestCollectOnTruncatedChunks:
    """Collect joins retrieval chunks with KV records (same as production get_by_ids).

    - ``chunks``: truncated retrieval list (``chunk_id`` + ``content``).
    - ``records``: ``text_chunks`` rows keyed by ``chunk_id`` (hold ``sidecar``).
    - Result order is always ``figure_ids + text_ids``, not the list order of
      ``chunks``.
    """

    def _figure_text_records(self, fig_id: str, text_id: str) -> dict[str, dict]:
        """KV map: figure chunk → drawing sidecar (IM_0002); text chunk → block."""
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

    def test_figure_ids_before_text_ids_when_figure_chunk_first(self):
        """Baseline: truncated = [figure, text] → still [figure_im, text_im].

        figure sidecar → SYNTH_IM_0002 (not in content)
        text content   → SYNTH_IM_0001
        expected:        [0002, 0001]
        """
        fig_id = f"{SYNTH_DOC_ID}-chunk-fig-001"
        text_id = f"{SYNTH_DOC_ID}-chunk-010"

        chunks = [
            {
                "chunk_id": fig_id,
                "content": "Figure caption only.",  # im from KV sidecar, not content
            },
            {
                "chunk_id": text_id,
                "content": f'See <drawing id="{SYNTH_IM_0001}"/> in body.',
            },
        ]
        # KV join: fig_id → drawing/0002; text_id → block (im only from content)
        records = self._figure_text_records(fig_id, text_id)

        assert collect_im_ids_on_truncated_chunks(chunks, records) == [
            SYNTH_IM_0002,  # figure bucket
            SYNTH_IM_0001,  # text bucket
        ]

    def test_figure_ids_before_text_ids_when_text_chunk_first(self):
        """Critical: truncated = [text, figure] must NOT follow list order.

        text content   → SYNTH_IM_0001  (appears first in truncated)
        figure sidecar → SYNTH_IM_0002
        expected:        [0002, 0001]  (figure bucket still first)
        """
        fig_id = f"{SYNTH_DOC_ID}-chunk-fig-001"
        text_id = f"{SYNTH_DOC_ID}-chunk-010"

        chunks = [
            {
                "chunk_id": text_id,
                "content": f'See <drawing id="{SYNTH_IM_0001}"/> in body.',
            },
            {
                "chunk_id": fig_id,
                "content": "Figure caption only.",  # im from KV sidecar, not content
            },
        ]
        records = self._figure_text_records(fig_id, text_id)

        assert collect_im_ids_on_truncated_chunks(chunks, records) == [
            SYNTH_IM_0002,  # figure bucket (even though figure chunk is second)
            SYNTH_IM_0001,  # text bucket
        ]

    def test_text_bucket_preserves_scan_order(self):
        """Within the text bucket only, order follows truncated scan order."""
        chunks = [
            {
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-010",
                "content": f'Later <drawing id="{SYNTH_IM_0002}"/>',
            },
            {
                "chunk_id": f"{SYNTH_DOC_ID}-chunk-011",
                "content": f'Earlier <drawing id="{SYNTH_IM_0001}"/>',
            },
        ]
        # No KV records → both treated as text chunks; scan order wins
        assert collect_im_ids_on_truncated_chunks(chunks, {}) == [
            SYNTH_IM_0002,
            SYNTH_IM_0001,
        ]

    def test_dedupes_across_buckets_figure_wins(self):
        """Same im in figure sidecar and text content → figure bucket only."""
        fig_id = f"{SYNTH_DOC_ID}-chunk-fig-001"
        text_id = f"{SYNTH_DOC_ID}-chunk-010"

        chunks = [
            {
                "chunk_id": text_id,
                "content": (
                    f'Cites <drawing id="{SYNTH_IM_0002}"/> and '
                    f'<drawing id="{SYNTH_IM_0001}"/>.'
                ),
            },
            {
                "chunk_id": fig_id,
                "content": "Figure caption.",
            },
        ]
        # fig_id sidecar already carries SYNTH_IM_0002
        records = self._figure_text_records(fig_id, text_id)

        assert collect_im_ids_on_truncated_chunks(chunks, records) == [
            SYNTH_IM_0002,  # figure bucket (text cite of 0002 dropped at merge)
            SYNTH_IM_0001,  # text-only id stays in text bucket
        ]


OTHER_DOC_HASH = "b" * 32
OTHER_DOC_ID = f"doc-{OTHER_DOC_HASH}"
OTHER_IM_0001 = f"im-{OTHER_DOC_HASH}-0001"


@pytest.mark.offline
class TestImIdsToDocMap:
    def test_im_ids_to_doc_map_from_hash_only(self):
        """Doc routing uses embedded im-id hash; no chunk scan required."""
        mapping = im_ids_to_doc_map([SYNTH_IM_0001, SYNTH_IM_0002])
        assert mapping == {
            SYNTH_IM_0001: SYNTH_DOC_ID,
            SYNTH_IM_0002: SYNTH_DOC_ID,
        }

    def test_warns_when_im_id_hash_disagrees_with_chunk_doc(
        self, caplog, propagate_lightrag_logs
    ):
        chunk_id = f"{SYNTH_DOC_ID}-chunk-001"
        chunk = {
            "content": f'<drawing id="{OTHER_IM_0001}"/>',
            "chunk_id": chunk_id,
        }
        record = {"full_doc_id": SYNTH_DOC_ID}
        with caplog.at_level("WARNING", logger="lightrag"):
            ordered = collect_im_ids_on_truncated_chunks(
                [chunk],
                {chunk_id: record},
                on_im_id_found=_warn_im_id_doc_hash_mismatch,
            )
            mapping = im_ids_to_doc_map(ordered)
        assert ordered == [OTHER_IM_0001]
        assert mapping[OTHER_IM_0001] == OTHER_DOC_ID
        assert any(
            "im-id doc hash mismatch" in record.message for record in caplog.records
        )
        assert OTHER_IM_0001 in caplog.text
        assert SYNTH_DOC_ID in caplog.text
