"""OpenSearchGraphStorage rejects attribute names that reshape the index mapping.

Both graph indices are created with ``dynamic: true``, so an unknown attribute
name does not fail the write -- it is *added to the index mapping*. A dotted name
becomes a nested object there rather than a field of that name, and every
distinct name consumes one of the index's ``mapping.total_fields.limit`` slots,
which no per-document write gives back. Restricting names to identifiers keeps
the mapping shaped by the schema rather than by request payloads.

Companion to the entry-layer allowlist in ``utils_graph``
(GHSA-c922-pw4m-4wcv): that stops the reported route from delivering such a
name, this stops any other caller.

The client is a double -- the assertion is that the write is refused *before*
any cluster call, so no live OpenSearch is needed here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from lightrag.kg.opensearch_impl import ClientManager, OpenSearchGraphStorage
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline

UNSAFE_NAMES = [
    pytest.param("source_ids.0", id="dotted-array-element"),
    pytest.param("a.b", id="dotted"),
    pytest.param("$set", id="dollar-prefixed"),
    pytest.param("has space", id="space"),
    pytest.param("", id="empty"),
]


async def _embed(texts):
    return np.random.rand(len(texts), 8)


def _make_client():
    from opensearchpy import AsyncOpenSearch

    client = AsyncMock(spec=AsyncOpenSearch)
    # opensearchpy's @query_params decorator hides coroutine-ness from
    # inspect.iscoroutinefunction, so AsyncMock(spec=...) makes these plain
    # MagicMocks unless forced. Same workaround as the other OpenSearch tests.
    client.index = AsyncMock()
    client.mget = AsyncMock(return_value={"docs": []})
    client.bulk = AsyncMock()
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=True)
    return client


@pytest.fixture
def graph_storage():
    with patch.object(ClientManager, "get_client") as mock_get:
        client = _make_client()
        mock_get.return_value = client
        storage = OpenSearchGraphStorage(
            namespace="test_graph",
            global_config={"embedding_batch_num": 10, "max_graph_nodes": 1000},
            embedding_func=EmbeddingFunc(
                embedding_dim=8, max_token_size=512, func=_embed
            ),
        )
        storage.client = client
        storage._indices_ready = True
        yield storage


def _assert_no_cluster_call(storage):
    storage.client.index.assert_not_awaited()
    storage.client.bulk.assert_not_awaited()


class TestUpsertNode:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", UNSAFE_NAMES)
    async def test_unsafe_name_is_refused(self, graph_storage, name):
        with pytest.raises(ValueError, match="invalid attribute name"):
            await graph_storage.upsert_node("n1", {"entity_id": "n1", name: "x"})

        _assert_no_cluster_call(graph_storage)

    @pytest.mark.asyncio
    async def test_ordinary_names_still_reach_the_cluster(self, graph_storage):
        await graph_storage.upsert_node(
            "n1", {"entity_id": "n1", "description": "d", "source_id": "chunk-1"}
        )

        graph_storage.client.index.assert_awaited_once()
        body = graph_storage.client.index.await_args.kwargs["body"]
        assert body["description"] == "d"
        assert body["source_ids"] == ["chunk-1"]

    @pytest.mark.asyncio
    async def test_id_is_still_dropped_not_rejected(self, graph_storage):
        """``_id`` addresses the document; the upsert paths already drop it.

        It is identifier-shaped, so the name rule does not catch it -- and it
        does not need to, because nothing here would write it into the body.
        """
        await graph_storage.upsert_node(
            "n1", {"entity_id": "n1", "_id": "elsewhere", "description": "d"}
        )

        body = graph_storage.client.index.await_args.kwargs["body"]
        assert "_id" not in body


class TestUpsertEdge:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", UNSAFE_NAMES)
    async def test_unsafe_name_is_refused(self, graph_storage, name):
        with pytest.raises(ValueError, match="invalid attribute name"):
            await graph_storage.upsert_edge("a", "b", {"weight": 1.0, name: "x"})

        # Refused before the placeholder endpoints are materialized, so no
        # orphan node documents are left for an edge that never lands.
        _assert_no_cluster_call(graph_storage)


class TestBatchesValidateBeforeWriting:
    @pytest.mark.asyncio
    async def test_nodes_batch_refuses_whole_batch(self, graph_storage):
        nodes = [
            ("good", {"entity_id": "good", "description": "a"}),
            ("bad", {"entity_id": "bad", "source_ids.0": "x"}),
        ]

        with pytest.raises(ValueError, match="invalid attribute name"):
            await graph_storage.upsert_nodes_batch(nodes)

        _assert_no_cluster_call(graph_storage)

    @pytest.mark.asyncio
    async def test_edges_batch_refuses_before_creating_endpoints(self, graph_storage):
        edges = [
            ("a", "b", {"weight": 1.0}),
            ("c", "d", {"weight": 1.0, "a.b": "x"}),
        ]

        with pytest.raises(ValueError, match="invalid attribute name"):
            await graph_storage.upsert_edges_batch(edges)

        _assert_no_cluster_call(graph_storage)
