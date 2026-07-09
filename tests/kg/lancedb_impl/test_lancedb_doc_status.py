"""Unit tests for LanceDBDocStatusStorage."""

import pytest

pytest.importorskip("lancedb", reason="lancedb is required for LanceDB storage tests")

from lightrag.base import DocProcessingStatus, DocStatus  # noqa: E402
from lightrag.kg.lancedb_impl import LanceDBDocStatusStorage  # noqa: E402

pytestmark = pytest.mark.offline


def _make_storage(global_config, workspace="ws"):
    return LanceDBDocStatusStorage(
        namespace="doc_status",
        workspace=workspace,
        global_config=global_config,
        embedding_func=None,
    )


def _doc(
    status=DocStatus.PENDING,
    file_path="doc.txt",
    created_at="2024-01-01T00:00:00+00:00",
    updated_at="2024-01-01T00:00:00+00:00",
    **extra,
):
    return {
        "status": status,
        "content_summary": "summary",
        "content_length": 100,
        "file_path": file_path,
        "created_at": created_at,
        "updated_at": updated_at,
        "metadata": {},
        "error_msg": None,
        **extra,
    }


@pytest.fixture
async def storage(global_config):
    doc_status = _make_storage(global_config)
    await doc_status.initialize()
    yield doc_status
    await doc_status.finalize()


async def test_upsert_and_kv_surface(storage):
    assert await storage.is_empty()
    await storage.upsert({"doc-1": _doc(track_id="t1", content_hash="h1")})
    raw = await storage.get_by_id("doc-1")
    # get_by_id returns the RAW dict, not a DocProcessingStatus.
    assert isinstance(raw, dict)
    assert raw["status"] == "pending"
    assert raw["chunks_list"] == []  # defaulted on upsert
    records = await storage.get_by_ids(["doc-1", "missing"])
    assert records[0]["content_summary"] == "summary"
    assert records[1] is None
    assert await storage.filter_keys({"doc-1", "doc-2"}) == {"doc-2"}
    await storage.delete(["doc-1"])
    assert await storage.get_by_id("doc-1") is None


async def test_get_docs_by_status_returns_dataclasses(storage):
    await storage.upsert(
        {
            "doc-1": _doc(status=DocStatus.PENDING),
            "doc-2": _doc(status=DocStatus.PROCESSED, chunks_count=4),
            "doc-3": _doc(status=DocStatus.FAILED, error_msg="boom"),
        }
    )
    pending = await storage.get_docs_by_status(DocStatus.PENDING)
    assert set(pending) == {"doc-1"}
    assert isinstance(pending["doc-1"], DocProcessingStatus)
    assert pending["doc-1"].status == DocStatus.PENDING

    several = await storage.get_docs_by_statuses(
        [DocStatus.PROCESSED, DocStatus.FAILED]
    )
    assert set(several) == {"doc-2", "doc-3"}
    assert several["doc-2"].chunks_count == 4
    assert several["doc-3"].error_msg == "boom"
    assert await storage.get_docs_by_statuses([]) == {}


async def test_doc_status_defaults_for_missing_fields(storage):
    minimal = {
        "status": DocStatus.PENDING,
        "content_summary": "s",
        "content_length": 1,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        # no file_path / metadata / error_msg / content (deprecated)
        "content": "should be stripped",
    }
    await storage.upsert({"doc-1": minimal})
    docs = await storage.get_docs_by_status(DocStatus.PENDING)
    doc = docs["doc-1"]
    assert doc.file_path == "no-file-path"
    assert doc.metadata == {}
    assert doc.error_msg is None


async def test_missing_file_path_indexed_as_normalized(storage):
    """Regression: the typed file_path column is normalized on write the same
    way the read path normalizes it, so get_doc_by_file_path (a SQL filter on
    the typed column) finds a doc upserted without a file_path.
    """
    await storage.upsert(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "content_summary": "s",
                "content_length": 1,
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
                # no file_path
            }
        }
    )
    found = await storage.get_doc_by_file_path("no-file-path")
    assert found is not None
    assert found["content_summary"] == "s"


async def test_status_counts(storage):
    await storage.upsert(
        {
            "doc-1": _doc(status=DocStatus.PENDING),
            "doc-2": _doc(status=DocStatus.PENDING),
            "doc-3": _doc(status=DocStatus.PROCESSED),
        }
    )
    counts = await storage.get_status_counts()
    assert counts["pending"] == 2
    assert counts["processed"] == 1
    assert counts["failed"] == 0  # all enum values present
    all_counts = await storage.get_all_status_counts()
    assert all_counts["all"] == 3


async def test_get_docs_by_track_id(storage):
    await storage.upsert(
        {
            "doc-1": _doc(track_id="upload-1"),
            "doc-2": _doc(track_id="upload-1"),
            "doc-3": _doc(track_id="upload-2"),
        }
    )
    docs = await storage.get_docs_by_track_id("upload-1")
    assert set(docs) == {"doc-1", "doc-2"}
    assert await storage.get_docs_by_track_id("nope") == {}


async def test_get_docs_paginated(storage):
    await storage.upsert(
        {
            f"doc-{i}": _doc(
                status=DocStatus.PENDING if i % 2 == 0 else DocStatus.PROCESSED,
                updated_at=f"2024-01-{i + 1:02d}T00:00:00+00:00",
                file_path=f"file-{i}.txt",
            )
            for i in range(15)
        }
    )
    page, total = await storage.get_docs_paginated(page=1, page_size=10)
    assert total == 15
    assert len(page) == 10
    # default sort: updated_at desc
    assert page[0][0] == "doc-14"
    assert isinstance(page[0][1], DocProcessingStatus)

    page2, total = await storage.get_docs_paginated(page=2, page_size=10)
    assert len(page2) == 5

    asc, _ = await storage.get_docs_paginated(sort_direction="asc", page_size=10)
    assert asc[0][0] == "doc-0"

    by_id, _ = await storage.get_docs_paginated(
        sort_field="id", sort_direction="asc", page_size=10
    )
    assert by_id[0][0] == "doc-0"
    assert by_id[1][0] == "doc-1"

    filtered, filtered_total = await storage.get_docs_paginated(
        status_filter=DocStatus.PENDING, page_size=10
    )
    assert filtered_total == 8
    assert all(doc.status == DocStatus.PENDING for _, doc in filtered)

    multi, multi_total = await storage.get_docs_paginated(
        status_filters=[DocStatus.PENDING, DocStatus.PROCESSED], page_size=200
    )
    assert multi_total == 15

    # page/page_size clamping
    clamped, _ = await storage.get_docs_paginated(page=0, page_size=1)
    assert len(clamped) == 10  # page_size clamped up to 10


async def test_file_path_and_content_hash_lookups(storage):
    await storage.upsert(
        {
            "doc-1": _doc(file_path="report.pdf", content_hash="hash-1"),
            "doc-2": _doc(file_path="notes.txt", content_hash="hash-2"),
        }
    )
    found = await storage.get_doc_by_file_path("report.pdf")
    assert found["content_hash"] == "hash-1"
    assert await storage.get_doc_by_file_path("nope.pdf") is None

    doc_id, doc = await storage.get_doc_by_file_basename("notes.txt")
    assert doc_id == "doc-2"
    assert doc["file_path"] == "notes.txt"
    assert await storage.get_doc_by_file_basename("") is None
    assert await storage.get_doc_by_file_basename("unknown_source") is None

    doc_id, doc = await storage.get_doc_by_content_hash("hash-1")
    assert doc_id == "doc-1"
    assert await storage.get_doc_by_content_hash("") is None
    assert await storage.get_doc_by_content_hash("nope") is None


async def test_upsert_replaces_record(storage):
    await storage.upsert({"doc-1": _doc(status=DocStatus.PENDING)})
    await storage.upsert(
        {"doc-1": _doc(status=DocStatus.PROCESSED, chunks_count=7, chunks_list=["c1"])}
    )
    raw = await storage.get_by_id("doc-1")
    assert raw["status"] == "processed"
    assert raw["chunks_count"] == 7
    assert raw["chunks_list"] == ["c1"]
    counts = await storage.get_status_counts()
    assert counts["pending"] == 0
    assert counts["processed"] == 1


async def test_drop(storage):
    await storage.upsert({"doc-1": _doc()})
    result = await storage.drop()
    assert result == {"status": "success", "message": "data dropped"}
    assert await storage.is_empty()
    await storage.upsert({"doc-2": _doc()})
    assert await storage.get_by_id("doc-2") is not None
