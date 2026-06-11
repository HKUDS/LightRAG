import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip(
    "qdrant_client",
    reason="qdrant-client is required for Qdrant storage tests",
)

from qdrant_client import models  # noqa: E402

from lightrag.kg.qdrant_impl import QdrantVectorDBStorage  # noqa: E402


def _make_point(point_id: str, content: str) -> models.PointStruct:
    return models.PointStruct(
        id=point_id,
        vector=[0.1, 0.2, 0.3],
        payload={"id": point_id, "content": content},
    )


def test_build_upsert_batches_respects_point_limit():
    points = [_make_point(str(i), "x" * 10) for i in range(5)]

    batches = QdrantVectorDBStorage._build_upsert_batches(
        points, max_payload_bytes=1024 * 1024, max_points_per_batch=2
    )

    assert [len(batch_points) for batch_points, _ in batches] == [2, 2, 1]


def test_build_upsert_batches_exact_payload_boundary_no_split():
    point_a = _make_point("a", "x" * 32)
    point_b = _make_point("b", "y" * 32)

    size_a = QdrantVectorDBStorage._estimate_point_payload_bytes(point_a)
    size_b = QdrantVectorDBStorage._estimate_point_payload_bytes(point_b)
    # JSON array envelope: [] => 2 bytes, and comma between two elements => 1 byte
    exact_limit = 2 + size_a + 1 + size_b

    batches = QdrantVectorDBStorage._build_upsert_batches(
        [point_a, point_b],
        max_payload_bytes=exact_limit,
        max_points_per_batch=128,
    )

    assert len(batches) == 1
    assert len(batches[0][0]) == 2
    assert batches[0][1] == exact_limit


def test_build_upsert_batches_single_oversized_point_gets_own_batch():
    """An oversized point is emitted as its own batch rather than raising.

    Failing here would poison the whole flush; instead the server is left as
    the final arbiter on whether the (conservatively estimated) point fits.
    """
    point = _make_point("oversized", "x" * 64)
    point_size = QdrantVectorDBStorage._estimate_point_payload_bytes(point)
    too_small_limit = point_size - 1

    batches = QdrantVectorDBStorage._build_upsert_batches(
        [point],
        max_payload_bytes=too_small_limit,
        max_points_per_batch=128,
    )

    assert len(batches) == 1
    assert len(batches[0][0]) == 1
    assert batches[0][0][0].id == "oversized"


def test_build_upsert_batches_isolates_oversized_point_between_normal_ones():
    """A mid-stream oversized point lands alone; neighbors are not polluted."""
    small_a = _make_point("a", "x" * 4)
    huge = _make_point("HUGE", "x" * 4096)
    small_b = _make_point("b", "y" * 4)

    # Budget fits a small point but not the huge one.
    budget = QdrantVectorDBStorage._estimate_point_payload_bytes(small_a) + 16

    batches = QdrantVectorDBStorage._build_upsert_batches(
        [small_a, huge, small_b],
        max_payload_bytes=budget,
        max_points_per_batch=128,
    )

    huge_batches = [b for b, _ in batches if any(p.id == "HUGE" for p in b)]
    assert len(huge_batches) == 1 and len(huge_batches[0]) == 1
    # No point is dropped.
    assert sum(len(b) for b, _ in batches) == 3


@pytest.mark.asyncio
async def test_flush_fail_fast_stops_on_first_failed_batch():
    """Flush-time fail-fast: once any batch raises, subsequent batches are skipped.

    Mirrors the pre-deferred-embedding `upsert()` contract: the failure
    bubbles out of `_flush_pending_vector_ops`, and the buffer is preserved
    so the next flush can retry.
    """
    storage = QdrantVectorDBStorage.__new__(QdrantVectorDBStorage)
    storage.workspace = "test_ws"
    storage.namespace = "chunks"
    storage.effective_workspace = "test_ws"
    storage.meta_fields = {"content"}
    storage._max_batch_size = 16
    storage._max_upsert_payload_bytes = 1024 * 1024
    storage._max_upsert_points_per_batch = 2
    storage.final_namespace = "test_collection"
    storage._client = MagicMock()
    storage._pending_vector_docs = {}
    storage._pending_vector_deletes = set()
    storage._flush_lock = asyncio.Lock()

    async def fake_embedding_func(texts, **kwargs):
        return np.array([[float(len(text)), 0.0] for text in texts], dtype=np.float32)

    storage.embedding_func = fake_embedding_func
    storage._client.upsert.side_effect = [None, RuntimeError("batch failed"), None]

    data = {f"chunk-{i}": {"content": f"content-{i}"} for i in range(5)}

    # `upsert` only buffers; the failure surfaces from `_flush_pending_vector_ops`.
    await storage.upsert(data)
    assert len(storage._pending_vector_docs) == 5

    with pytest.raises(RuntimeError, match="batch failed"):
        await storage._flush_pending_vector_ops()

    # 5 items with max 2 points per batch => expected 3 batches, but stop at batch #2 on error.
    assert storage._client.upsert.call_count == 2
    first_call = storage._client.upsert.call_args_list[0]
    second_call = storage._client.upsert.call_args_list[1]
    assert len(first_call.kwargs["points"]) == 2
    assert len(second_call.kwargs["points"]) == 2
    # Buffer is preserved so the next flush can retry.
    assert len(storage._pending_vector_docs) == 5


@pytest.mark.asyncio
async def test_flush_chunks_deletes_by_point_count():
    """Deletes are split into chunks of at most _max_delete_points_per_batch."""
    storage = QdrantVectorDBStorage.__new__(QdrantVectorDBStorage)
    storage.workspace = "test_ws"
    storage.namespace = "chunks"
    storage.effective_workspace = "test_ws"
    storage.meta_fields = {"content"}
    storage._max_batch_size = 16
    storage._max_upsert_payload_bytes = 1024 * 1024
    storage._max_upsert_points_per_batch = 128
    storage._max_delete_points_per_batch = 2
    storage.final_namespace = "test_collection"
    storage._client = MagicMock()
    storage._pending_vector_docs = {}
    storage._pending_vector_deletes = {f"chunk-{i}" for i in range(5)}
    storage._flush_lock = asyncio.Lock()

    await storage._flush_pending_vector_ops()

    # 5 delete ids with max 2 per batch => 3 delete calls.
    assert storage._client.delete.call_count == 3
    chunk_sizes = sorted(
        len(call.kwargs["points_selector"].points)
        for call in storage._client.delete.call_args_list
    )
    assert chunk_sizes == [1, 2, 2]
    # Buffer cleared on success.
    assert len(storage._pending_vector_deletes) == 0
