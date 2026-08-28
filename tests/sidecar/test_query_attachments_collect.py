"""PR-1 L1 tests: collect drawing ``im-id`` values from retrieval ``raw_data``.

These tests exercise the first layer of the attachment pipeline—gathering stable
drawing identifiers from retrieval output and mapping each ``im-id`` to its
source ``doc_id``. No disk I/O or HTTP calls are involved.

Coverage:
- Format helpers: ``is_im_id``, ``doc_id_from_im_id``, ``doc_id_from_chunk_id``
- Three collection paths: KG entity names, chunk ``<drawing>`` tags, persisted sidecar metadata
- ``collect_im_ids`` deduplication across all sources
- ``build_im_to_doc_map`` reverse lookup from ``im-id`` to ``doc_id``

All cases use in-memory fixtures and are marked ``@pytest.mark.offline``.
"""

from __future__ import annotations

import pytest

from lightrag.sidecar.query_attachments import (
    build_im_to_doc_map,
    collect_im_ids,
    collect_im_ids_from_chunk_content,
    collect_im_ids_from_entities,
    collect_im_ids_from_sidecar_records,
    doc_id_from_chunk_id,
    doc_id_from_im_id,
    is_im_id,
)
from tests.sidecar.conftest_query_attachments import GB_T_DOC_ID, GB_T_IM_ID_0002

# Shared doc hash with the GB/T fixture so doc_id assertions stay consistent.
_DOC_HASH = "116da71bde652ed78cf28934d0712aa6"
# Locally defined first drawing id in this module (seq=0001).
_IM_0001 = f"im-{_DOC_HASH}-0001"


@pytest.mark.offline
class TestImIdHelpers:
    """Unit tests for im-id / doc_id / chunk_id parsing helpers."""

    def test_is_im_id_valid(self):
        """Accept canonical ids of the form ``im-{32hex}-{4digit}``."""
        assert is_im_id(_IM_0001)
        assert is_im_id(GB_T_IM_ID_0002)

    def test_is_im_id_rejects_garbage(self):
        """Reject empty strings, natural language, and non-im prefixes."""
        assert not is_im_id("")
        assert not is_im_id("图1")
        assert not is_im_id("doc-abc-chunk-1")

    def test_doc_id_from_im_id(self):
        """Derive ``doc-{hash}`` from ``im-{hash}-{seq}``."""
        assert doc_id_from_im_id(_IM_0001) == GB_T_DOC_ID

    def test_doc_id_from_chunk_id(self):
        """Parse owning doc_id from regular and mm-drawing chunk ids."""
        assert (
            doc_id_from_chunk_id(f"{GB_T_DOC_ID}-chunk-017") == GB_T_DOC_ID
        )
        assert (
            doc_id_from_chunk_id(f"{GB_T_DOC_ID}-mm-drawing-000")
            == GB_T_DOC_ID
        )


@pytest.mark.offline
class TestCollectImIds:
    """Integration-style tests for the three im-id collection paths."""

    def test_entity_name_im_id(self):
        """Path 1: collect when a KG entity name is itself an im-id."""
        entities = [{"entity_name": _IM_0001}]
        assert collect_im_ids_from_entities(entities) == {_IM_0001}

    def test_chunk_content_drawing_tags(self):
        """Path 2: parse every ``<drawing id="im-…"/>`` tag in chunk content."""
        content = (
            f'<drawing id="{_IM_0001}" format="jpg"/> '
            f'<drawing id="{GB_T_IM_ID_0002}" format="jpg"/>'
        )
        chunks = [{"content": content}]
        found = collect_im_ids_from_chunk_content(chunks)
        assert found == {_IM_0001, GB_T_IM_ID_0002}

    def test_chunk_content_no_drawing(self):
        """Path 2 edge case: plain text chunks yield an empty set."""
        assert collect_im_ids_from_chunk_content([{"content": "plain text"}]) == set()

    def test_sidecar_id_on_mm_chunk(self):
        """Path 3: read im-ids from persisted chunk sidecar type/id/refs metadata."""
        records = [
            {
                "sidecar": {
                    "type": "drawing",
                    "id": _IM_0001,
                    "refs": [{"type": "drawing", "id": _IM_0001}],
                }
            }
        ]
        assert collect_im_ids_from_sidecar_records(records) == {_IM_0001}

    def test_collect_deduplicates(self):
        """``collect_im_ids`` keeps one copy when the same id appears on multiple paths."""
        raw_data = {
            "data": {
                "entities": [{"entity_name": _IM_0001}],
                "chunks": [
                    {
                        "chunk_id": f"{GB_T_DOC_ID}-chunk-1",
                        "content": f'<drawing id="{_IM_0001}"/>',
                    }
                ],
            }
        }
        # chunk_records_by_id mimics text_chunks.get_by_ids sidecar lookups.
        records = {
            f"{GB_T_DOC_ID}-chunk-1": {
                "sidecar": {
                    "type": "drawing",
                    "id": _IM_0001,
                    "refs": [{"type": "drawing", "id": _IM_0001}],
                }
            }
        }
        assert collect_im_ids(raw_data, records) == {_IM_0001}

    def test_build_im_to_doc_map_from_im_id(self):
        """``build_im_to_doc_map`` can infer doc_id from the im-id hash segment alone."""
        im_ids = {_IM_0001, GB_T_IM_ID_0002}
        mapping = build_im_to_doc_map(im_ids)
        assert mapping[_IM_0001] == GB_T_DOC_ID
        assert mapping[GB_T_IM_ID_0002] == GB_T_DOC_ID
