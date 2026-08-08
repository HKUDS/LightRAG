"""Coverage for JsonDocStatusStorage.get_docs_by_track_id hydration resilience.

WHY: get_docs_by_track_id must delegate to _doc_processing_status_from_row and
catch both KeyError and TypeError so an unhydratable document row under a track_id
does not crash queries for that track_id with an unhandled TypeError.
"""

import pytest

from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


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
