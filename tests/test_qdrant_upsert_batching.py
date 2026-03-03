from unittest.mock import MagicMock

import numpy as np
import pytest
from qdrant_client import models

from lightrag.kg.qdrant_impl import QdrantVectorDBStorage


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


def test_build_upsert_batches_raises_for_single_oversized_point():
    point = _make_point("oversized", "x" * 64)
    point_size = QdrantVectorDBStorage._estimate_point_payload_bytes(point)
    too_small_limit = point_size + 1

    with pytest.raises(ValueError, match="Single Qdrant point exceeds payload limit"):
        QdrantVectorDBStorage._build_upsert_batches(
            [point],
            max_payload_bytes=too_small_limit,
            max_points_per_batch=128,
        )


@pytest.mark.asyncio
async def test_upsert_fail_fast_stops_on_first_failed_batch():
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

    async def fake_embedding_func(texts, **kwargs):
        return np.array([[float(len(text)), 0.0] for text in texts], dtype=np.float32)

    storage.embedding_func = fake_embedding_func
    storage._client.upsert.side_effect = [None, RuntimeError("batch failed"), None]

    data = {f"chunk-{i}": {"content": f"content-{i}"} for i in range(5)}

    with pytest.raises(RuntimeError, match="batch failed"):
        await storage.upsert(data)

    # 5 items with max 2 points per batch => expected 3 batches, but stop at batch #2 on error.
    assert storage._client.upsert.call_count == 2
    first_call = storage._client.upsert.call_args_list[0]
    second_call = storage._client.upsert.call_args_list[1]
    assert len(first_call.kwargs["points"]) == 2
    assert len(second_call.kwargs["points"]) == 2
