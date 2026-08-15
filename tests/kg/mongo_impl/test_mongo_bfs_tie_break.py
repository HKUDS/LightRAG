"""``get_knowledge_graph(label)`` must rank each BFS level by degree, then id.

``_bidirectional_bfs_nodes`` admitted a level in the order ``find`` returned the
documents, which is natural order -- not the order of the ``$in`` list, and not
any property of the graph. The ``max_nodes`` cutoff normally lands inside a band
of equal-degree leaves, so re-ingesting the same corpus in a different order
returned a different subgraph at the same cap. See issue #3612.
"""

from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip(
    "pymongo",
    reason="pymongo is required for Mongo storage tests",
)

from lightrag.kg.mongo_impl import MongoGraphStorage  # noqa: E402


pytestmark = pytest.mark.offline


# A three-leaf star whose leaves are deliberately not equal: Z outranks both on
# degree, and X/Y are a genuine tie that only the label can break. The $in list
# is built in discovery order (X, Y, Z), the reverse of the ranking, so an
# unranked level cannot produce the expected answer by accident.
_EDGES = [
    {"source_node_id": "A", "target_node_id": "X"},
    {"source_node_id": "A", "target_node_id": "Y"},
    {"source_node_id": "A", "target_node_id": "Z"},
    {"source_node_id": "Z", "target_node_id": "P"},
    {"source_node_id": "Z", "target_node_id": "Q"},
    {"source_node_id": "X", "target_node_id": "P"},
    {"source_node_id": "Y", "target_node_id": "Q"},
]

_REAL_IDS = {"A", "X", "Y", "Z", "P", "Q"}


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def __aiter__(self):
        self._iter = iter(self._docs)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _node_find(query, projection=None):
    return _AsyncCursor(
        [
            {"_id": nid, "entity_type": "person"}
            for nid in query["_id"]["$in"]
            if nid in _REAL_IDS
        ]
    )


def _edge_find(query):
    if "$or" in query:
        ids = set(query["$or"][0]["source_node_id"]["$in"])
        return _AsyncCursor(
            [
                e
                for e in _EDGES
                if e["source_node_id"] in ids or e["target_node_id"] in ids
            ]
        )
    ids = set(query["$and"][0]["source_node_id"]["$in"])
    return _AsyncCursor(
        [e for e in _EDGES if e["source_node_id"] in ids and e["target_node_id"] in ids]
    )


async def _edge_aggregate(pipeline, **kwargs):
    match = pipeline[0]["$match"]
    field = "source_node_id" if "source_node_id" in match else "target_node_id"
    ids = set(match[field]["$in"])
    counts = Counter(e[field] for e in _EDGES if e[field] in ids)
    return _AsyncCursor(
        [{"_id": key, "degree": count} for key, count in counts.items()]
    )


def _make_storage():
    storage = MongoGraphStorage.__new__(MongoGraphStorage)
    storage.workspace = "test"
    storage.global_config = {"max_graph_nodes": 1000}
    storage._collection_name = "test_nodes"
    storage._edge_collection_name = "test_edges"
    storage.collection = SimpleNamespace(find=Mock(side_effect=_node_find))
    storage.edge_collection = SimpleNamespace(
        find=Mock(side_effect=_edge_find),
        aggregate=AsyncMock(side_effect=_edge_aggregate),
    )
    return storage


@pytest.mark.asyncio
async def test_bfs_level_admits_by_degree_then_id():
    """One slot short of the whole level: the highest-degree leaf takes the
    first, and the label breaks the tie for the second."""
    storage = _make_storage()

    result = await storage.get_knowledge_subgraph_bidirectional_bfs(
        "A", 0, max_depth=1, max_nodes=3
    )

    assert sorted(node.id for node in result.nodes) == ["A", "X", "Z"]
    assert result.is_truncated is True


@pytest.mark.asyncio
async def test_bfs_level_that_fits_keeps_every_node():
    """The rule only decides a cutoff. With room for the whole level nothing is
    ranked away, and nothing is reported truncated."""
    storage = _make_storage()

    result = await storage.get_knowledge_subgraph_bidirectional_bfs(
        "A", 0, max_depth=1, max_nodes=10
    )

    assert sorted(node.id for node in result.nodes) == ["A", "X", "Y", "Z"]
    assert result.is_truncated is False
