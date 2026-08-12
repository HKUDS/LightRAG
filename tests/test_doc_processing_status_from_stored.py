"""Coverage for DocProcessingStatus.from_stored (tolerant rehydration).

WHY: doc_status rows are written by producers that evolve independently of
this dataclass — RAG-Anything adds scheme_name/multimodal_content
(HKUDS/RAG-Anything#73), newer LightRAG builds add fields an older installed
release does not declare. ``DocProcessingStatus(**row)`` raises ``TypeError``
on any such field, and every read path treats that as a malformed row and
drops it, so the document silently disappears from listings while its record
is intact. ``from_stored`` ignores undeclared fields — matching the PG
backend, whose explicit-kwargs construction always behaved this way — while
still raising on genuinely malformed rows.
"""

import pytest

from lightrag.base import DocProcessingStatus, DocStatus

pytestmark = pytest.mark.offline


def _valid_row(**extra) -> dict:
    return {
        "content_summary": "Summary",
        "content_length": 100,
        "file_path": "doc.pdf",
        "status": DocStatus.PROCESSED,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        **extra,
    }


def test_undeclared_fields_are_ignored():
    # The RAG-Anything shape from HKUDS/RAG-Anything#73
    row = _valid_row(
        scheme_name="default",
        multimodal_content=[{"type": "image"}],
        field_from_the_future="value",
    )

    status = DocProcessingStatus.from_stored(row)

    assert status.file_path == "doc.pdf"
    assert status.status == DocStatus.PROCESSED
    assert not hasattr(status, "scheme_name")
    assert not hasattr(status, "field_from_the_future")


def test_declared_fields_all_pass_through():
    row = _valid_row(
        track_id="track_1",
        chunks_count=3,
        chunks_list=["c1", "c2", "c3"],
        error_msg=None,
        metadata={"k": "v"},
    )

    status = DocProcessingStatus.from_stored(row)

    assert status.track_id == "track_1"
    assert status.chunks_list == ["c1", "c2", "c3"]
    assert status.metadata == {"k": "v"}


def test_missing_required_field_still_raises_type_error():
    # Tolerance is strictly for extras; a malformed row must keep failing
    # so skip-and-log callers keep skipping it.
    row = _valid_row()
    del row["status"]

    with pytest.raises(TypeError):
        DocProcessingStatus.from_stored(row)


def test_post_init_status_conversion_still_applies():
    # __post_init__ downgrades PROCESSED to PREPROCESSED when multimodal
    # processing is pending; from_stored must not bypass it.
    row = _valid_row(multimodal_processed=False)

    status = DocProcessingStatus.from_stored(row)

    assert status.status == DocStatus.PREPROCESSED


def test_input_row_is_not_mutated():
    row = _valid_row(field_from_the_future="value")
    snapshot = dict(row)

    DocProcessingStatus.from_stored(row)

    assert row == snapshot
