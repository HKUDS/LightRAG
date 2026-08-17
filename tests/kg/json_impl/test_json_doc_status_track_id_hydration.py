"""Coverage for JsonDocStatusStorage.get_docs_by_track_id hydration resilience.

WHY: get_docs_by_track_id must delegate to _doc_processing_status_from_row and
catch both KeyError and TypeError so an unhydratable document row under a track_id
does not crash queries for that track_id with an unhandled TypeError.

track_id groups a whole upload BATCH, so the blast radius of letting one bad row
escape is every healthy sibling document in that batch — and permanently, since
the row stays in the store. These tests pin skip-and-log for each shape a row can
be unusable in — missing required field, and not a mapping at all — and pin that
an unknown field does NOT make a row unusable (it is ignored at construction).
"""

import logging

import pytest

from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import logger as lightrag_logger

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


async def _make_storage(tmp_path) -> JsonDocStatusStorage:
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
        workspace="test",
    )
    await storage.initialize()
    return storage


def _valid_row(track_id: str = "track_123") -> dict:
    return {
        "track_id": track_id,
        "status": "processed",
        "content_summary": "Summary of valid doc",
        "content_length": 100,
        "file_path": "valid.txt",
        "created_at": 1000.0,
        "updated_at": 1000.0,
    }


@pytest.mark.asyncio
async def test_get_docs_by_track_id_handles_unhydratable_row_without_crashing(
    tmp_path,
):
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
        workspace="test",
    )
    await storage.initialize()

    await storage.upsert(
        {
            "valid_doc": {
                "track_id": "track_123",
                "status": "processed",
                "content_summary": "Summary of valid doc",
                "content_length": 100,
                "file_path": "valid.txt",
                "created_at": 1000.0,
                "updated_at": 1000.0,
            },
            "invalid_doc": {
                "track_id": "track_123",
                # missing required fields (status, content_summary, etc.), which causes
                # DocProcessingStatus constructor to raise TypeError
            },
        }
    )

    # Must return valid_doc and skip invalid_doc instead of raising TypeError
    docs = await storage.get_docs_by_track_id("track_123")

    assert "valid_doc" in docs
    assert docs["valid_doc"].status == "processed"
    assert "invalid_doc" not in docs


async def test_get_docs_by_track_id_hydrates_row_carrying_an_unknown_field(tmp_path):
    """The realistic trigger: a version rollback, or an external producer.

    A newer build (or a producer like RAG-Anything) persists a field the
    running DocProcessingStatus does not declare. Construction now ignores
    undeclared fields (``DocProcessingStatus.from_stored``, matching the
    PG backend's long-standing explicit-kwargs behavior), so the row stays
    visible instead of silently vanishing from every listing.
    """
    storage = await _make_storage(tmp_path)
    await storage.upsert(
        {
            "old_doc": _valid_row(),
            "written_by_newer_build": {
                **_valid_row(),
                "field_from_the_future": "value",
            },
        }
    )

    docs = await storage.get_docs_by_track_id("track_123")

    assert set(docs) == {"old_doc", "written_by_newer_build"}
    assert not hasattr(docs["written_by_newer_build"], "field_from_the_future")


async def test_get_docs_by_track_id_logs_and_skips_a_non_mapping_row(tmp_path, caplog):
    """A non-mapping record must be reported, not silently dropped.

    Before the guard, ``v.get("track_id")`` on a non-dict raised AttributeError
    OUTSIDE the try and took the whole listing down. Skipping it is the fix, but
    skipping it *silently* would leave an operator with a short listing and no
    trace of why — so the skip is asserted together with the log line.
    """
    storage = await _make_storage(tmp_path)
    await storage.upsert({"valid_doc": _valid_row()})
    # Bypass upsert: only externally-written / corrupted stores hold non-mappings.
    storage._data["corrupt_doc"] = "this is not a mapping"

    # The lightrag logger sets propagate=False (lightrag/utils.py), so caplog —
    # which attaches to root by default — never sees its records. Flip propagate
    # for the duration of the call, then restore.
    caplog.clear()
    old_propagate = lightrag_logger.propagate
    lightrag_logger.propagate = True
    try:
        with caplog.at_level(logging.ERROR, logger="lightrag"):
            docs = await storage.get_docs_by_track_id("track_123")
    finally:
        lightrag_logger.propagate = old_propagate

    assert "valid_doc" in docs
    assert "corrupt_doc" not in docs
    assert "corrupt_doc" in caplog.text
    assert "is not a mapping" in caplog.text
