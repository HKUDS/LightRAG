"""Milvus always uses COSINE.

``cosine_better_than_threshold`` is passed as the range-search radius, which
is a similarity lower bound only under COSINE: an L2 index turned it into a
distance upper bound and returned nothing. ``MILVUS_METRIC_TYPE`` /
``metric_type`` are therefore ignored, and a collection indexed with another
metric is re-indexed in place at startup.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import lightrag.kg.milvus_impl as milvus_impl
from lightrag.kg.milvus_impl import MilvusVectorDBStorage

pytestmark = pytest.mark.offline


def _make_storage(metric_kwarg=None):
    kwargs = {"cosine_better_than_threshold": 0.2}
    if metric_kwarg is not None:
        kwargs["metric_type"] = metric_kwarg
    embedding_func = MagicMock()
    embedding_func.embedding_dim = 8
    return MilvusVectorDBStorage(
        namespace="entities",
        workspace="test",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": kwargs,
        },
        embedding_func=embedding_func,
        meta_fields=set(),
    )


@pytest.mark.parametrize(
    "source, value, warns",
    [
        ("kwarg", "L2", True),
        ("env", "ip", True),
        ("kwarg", "cosine", False),
        ("env", None, False),
    ],
)
def test_requested_metric_is_ignored_and_warned_once(
    monkeypatch, lightrag_caplog, source, value, warns
):
    monkeypatch.setattr(milvus_impl, "_IGNORED_METRIC_TYPES_WARNED", set())
    monkeypatch.delenv("MILVUS_METRIC_TYPE", raising=False)
    if source == "env" and value is not None:
        monkeypatch.setenv("MILVUS_METRIC_TYPE", value)
    kwarg = value if source == "kwarg" else None

    storages = [_make_storage(kwarg), _make_storage(kwarg)]

    assert [s.index_config.metric_type for s in storages] == ["COSINE", "COSINE"]
    warnings = [
        r for r in lightrag_caplog.records if "no longer supported" in r.getMessage()
    ]
    assert len(warnings) == (1 if warns else 0)


def _initialize_with_index(storage, metric):
    """Run initialize() against a client whose vector index uses ``metric``."""
    client = MagicMock()
    client.has_collection.return_value = True
    client.list_indexes.return_value = [] if metric is None else ["vector"]
    client.describe_index.return_value = {"metric_type": metric}
    storage._client = client
    with (
        patch("lightrag.kg.milvus_impl.get_data_init_lock", return_value=AsyncMock()),
        patch.object(storage, "_create_collection_if_not_exist"),
    ):
        asyncio.run(storage.initialize())
    return client


_REBUILD_CALLS = {"release_collection", "drop_index", "create_index", "load_collection"}


@pytest.mark.parametrize(
    "metric, rebuilt",
    [("L2", True), ("COSINE", False), (None, False)],
)
def test_initialize_reindexes_a_non_cosine_vector_index(metric, rebuilt):
    storage = _make_storage()

    client = _initialize_with_index(storage, metric)

    calls = [c[0] for c in client.method_calls if c[0] in _REBUILD_CALLS]
    if rebuilt:
        assert calls == [
            "release_collection",
            "drop_index",
            "create_index",
            "load_collection",
        ]
        add_index = client.prepare_index_params.return_value.add_index
        assert add_index.call_args.kwargs["metric_type"] == "COSINE"
    else:
        assert calls == []
    assert storage._initialized is True


def test_failed_reindex_fails_initialize():
    storage = _make_storage()
    client = MagicMock()
    client.has_collection.return_value = True
    client.list_indexes.return_value = ["vector"]
    client.describe_index.return_value = {"metric_type": "L2"}
    client.create_index.side_effect = RuntimeError("index build refused")
    storage._client = client

    with (
        patch("lightrag.kg.milvus_impl.get_data_init_lock", return_value=AsyncMock()),
        patch.object(storage, "_create_collection_if_not_exist"),
        pytest.raises(RuntimeError, match="index build refused"),
    ):
        asyncio.run(storage.initialize())

    client.drop_index.assert_called_once()
    assert storage._initialized is False
