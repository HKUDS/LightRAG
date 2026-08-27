"""Unit tests verifying that JsonDocStatusStorage read methods return deep copies to prevent internal state mutation of nested status fields."""

from unittest.mock import patch

import pytest
from lightrag.base import DocStatus
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.shared_storage import (
    NamespaceLock,
    initialize_share_data,
    finalize_share_data,
)

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.mark.asyncio
async def test_get_by_id_returns_copy(tmp_path):
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc1": {
                "status": "pending",
                "file_path": "test.pdf",
                "metadata": {"key": "original"},
            }
        }
    )

    doc = await storage.get_by_id("doc1")
    assert doc is not None
    assert doc["status"] == "pending"

    # Mutate returned dictionary top-level and nested fields
    doc["status"] = "modified_externally"
    doc["metadata"]["key"] = "modified_nested"
    doc["chunks_list"] = ["chunk_1"]

    # Verify internal storage was not mutated
    doc_after = await storage.get_by_id_strict("doc1")
    assert doc_after is not None
    assert doc_after["status"] == "pending"
    assert doc_after["metadata"]["key"] == "original"
    assert doc_after["chunks_list"] == []


@pytest.mark.asyncio
async def test_get_doc_by_file_path_returns_copy(tmp_path):
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc2": {
                "status": "pending",
                "file_path": "report.pdf",
                "metadata": {"key": "original"},
            }
        }
    )

    doc = await storage.get_doc_by_file_path("report.pdf")
    assert doc is not None
    assert doc["status"] == "pending"

    # Mutate returned dictionary top-level and nested fields
    doc["status"] = "modified_externally"
    doc["metadata"]["key"] = "modified_nested"

    # Verify internal storage was not mutated
    doc_after = await storage.get_by_id_strict("doc2")
    assert doc_after is not None
    assert doc_after["status"] == "pending"
    assert doc_after["metadata"]["key"] == "original"


@pytest.mark.asyncio
async def test_get_by_ids_returns_deep_copy(tmp_path):
    """
    Locks out the bug where get_by_ids returned shallow copies, allowing external callers
    to mutate nested metadata or chunks_list in internal storage.
    """
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc3": {
                "status": "pending",
                "file_path": "data.pdf",
                "metadata": {"kg_purge": {"phase": "prepared"}},
                "chunks_list": ["c1", "c2"],
            }
        }
    )

    docs = await storage.get_by_ids(["doc3"])
    assert len(docs) == 1 and docs[0] is not None
    docs[0]["metadata"]["kg_purge"]["phase"] = "corrupted"
    docs[0]["chunks_list"].append("c3")

    stored = await storage.get_by_id_strict("doc3")
    assert stored is not None
    assert stored["metadata"]["kg_purge"]["phase"] == "prepared"
    assert stored["chunks_list"] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_get_doc_by_file_basename_returns_copy(tmp_path):
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc4": {
                "status": "pending",
                "file_path": "basename.pdf",
                "metadata": {"key": "original"},
            }
        }
    )

    result = await storage.get_doc_by_file_basename("basename.pdf")
    assert result is not None
    doc_id, doc = result
    assert doc_id == "doc4"

    # Mutate returned dictionary top-level and nested fields
    doc["status"] = "modified_externally"
    doc["metadata"]["key"] = "modified_nested"

    stored = await storage.get_by_id_strict("doc4")
    assert stored is not None
    assert stored["status"] == "pending"
    assert stored["metadata"]["key"] == "original"


@pytest.mark.asyncio
async def test_get_docs_paginated_returns_copy(tmp_path):
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc5": {
                "status": "pending",
                "file_path": "paginated.pdf",
                "content_summary": "summary",
                "content_length": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "metadata": {"kg_purge": {"phase": "prepared"}},
                "chunks_list": ["c1", "c2"],
            }
        }
    )

    docs, total = await storage.get_docs_paginated()
    assert total == 1
    doc_id, doc_status = docs[0]
    assert doc_id == "doc5"

    # Pagination never surfaces chunks_list (mirrors the PG backend's paged
    # CTE, which excludes the column for the same reason: DocStatusResponse
    # doesn't expose it) — asserted here so the field isn't silently
    # deep-copied from storage again by a future change.
    assert doc_status.chunks_list == []

    # Mutate the returned DocProcessingStatus's nested mutable metadata field
    doc_status.metadata["kg_purge"]["phase"] = "corrupted"

    stored = await storage.get_by_id_strict("doc5")
    assert stored is not None
    assert stored["metadata"]["kg_purge"]["phase"] == "prepared"
    assert stored["chunks_list"] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_get_docs_paginated_excludes_rows_missing_required_fields(tmp_path):
    """
    get_docs_paginated's validation pass rejects rows missing a required
    DocProcessingStatus field before sorting/pagination, so such rows are
    dropped from both total_count and every page — not just logged and
    silently miscounted.
    """
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc7": {
                "status": "pending",
                "file_path": "good.pdf",
                "content_summary": "summary",
                "content_length": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
            # Missing content_summary/content_length: upsert() itself does
            # not validate required fields, so a malformed row can reach
            # storage (e.g. from a partial write or external tooling).
            "doc8": {
                "status": "pending",
                "file_path": "malformed.pdf",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
        }
    )

    docs, total = await storage.get_docs_paginated()
    assert total == 1
    assert [doc_id for doc_id, _ in docs] == ["doc7"]


@pytest.mark.asyncio
async def test_get_docs_paginated_hydrates_rows_with_unexpected_fields(tmp_path):
    """
    An unexpected top-level key no longer makes a row unhydratable —
    ``DocProcessingStatus.from_stored`` ignores undeclared fields — so the
    row appears in BOTH total_count and the returned page. The invariant
    this file originally pinned still holds and is what this test asserts:
    the validation pass and the page-hydration step must agree, whatever
    the hydration rules are, or total_count disagrees with the page.
    """
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc9": {
                "status": "pending",
                "file_path": "good.pdf",
                "content_summary": "summary",
                "content_length": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
            "doc10": {
                "status": "pending",
                "file_path": "unexpected_field.pdf",
                "content_summary": "summary",
                "content_length": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "unexpected_field": "x",
            },
        }
    )

    docs, total = await storage.get_docs_paginated()
    assert total == 2
    assert sorted(doc_id for doc_id, _ in docs) == ["doc10", "doc9"]
    hydrated = dict(docs)["doc10"]
    assert not hasattr(hydrated, "unexpected_field")


@pytest.mark.asyncio
async def test_get_docs_paginated_uses_a_single_lock_acquisition(tmp_path):
    """
    Regression test for a snapshot-consistency bug: get_docs_paginated used
    to scan+filter+sort under one lock acquisition, release the lock, then
    re-acquire a second time to hydrate just the requested page. A
    concurrent status update or delete landing in the gap between those two
    acquisitions could mix two snapshots into a single response — e.g. a
    status=pending filter returning an already-updated "processed" document,
    or overcounting a row a concurrent delete then removed.

    Filtering, hydratability validation, sorting, page selection and the
    final page's hydration must all happen inside ONE uninterrupted critical
    section, so nothing else can run in between. Pinned here by counting
    NamespaceLock acquisitions during a single call.
    """
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc11": {
                "status": "pending",
                "file_path": "single_lock.pdf",
                "content_summary": "summary",
                "content_length": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            }
        }
    )

    real_aenter = NamespaceLock.__aenter__
    acquisitions = []

    async def counting_aenter(self):
        acquisitions.append(1)
        return await real_aenter(self)

    with patch.object(NamespaceLock, "__aenter__", counting_aenter):
        docs, total = await storage.get_docs_paginated()

    assert total == 1
    assert len(acquisitions) == 1


@pytest.mark.asyncio
async def test_get_docs_by_statuses_returns_copy(tmp_path):
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc6": {
                "status": "pending",
                "file_path": "bystatus.pdf",
                "content_summary": "summary",
                "content_length": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "metadata": {"kg_purge": {"phase": "prepared"}},
                "chunks_list": ["c1", "c2"],
            }
        }
    )

    result = await storage.get_docs_by_statuses([DocStatus.PENDING])
    doc_status = result["doc6"]

    doc_status.metadata["kg_purge"]["phase"] = "corrupted"
    doc_status.chunks_list.append("c3")

    stored = await storage.get_by_id_strict("doc6")
    assert stored is not None
    assert stored["metadata"]["kg_purge"]["phase"] == "prepared"
    assert stored["chunks_list"] == ["c1", "c2"]
