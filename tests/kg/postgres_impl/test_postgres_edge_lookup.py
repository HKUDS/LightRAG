"""PGGraphStorage edge lookup retains the Cypher merge semantics."""

import json
from unittest.mock import AsyncMock

import pytest

from lightrag.kg.postgres_impl import PGGraphStorage

pytestmark = pytest.mark.offline


@pytest.fixture
def graph():
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.graph_name = "test_graph"
    storage._query = AsyncMock()
    return storage


def edge(source, target, weight, *, reverse=False):
    return {
        "source": source,
        "target": target,
        "reverse": reverse,
        "edge_properties": json.dumps({"weight": weight}),
    }


async def test_sql_merges_directions_and_preserves_input_keys(graph):
    quoted_id = 'За\\пись "A"'
    graph._query.side_effect = [
        [
            edge(quoted_id, "B", 1),
            edge(quoted_id, "B", 2, reverse=True),
            edge("B", quoted_id, 1, reverse=True),
        ],
        [],
    ]

    result = await graph.get_edges_batch(
        [
            {"src": quoted_id, "tgt": "B"},
            {"src": "B", "tgt": quoted_id},
            {"src": quoted_id, "tgt": "B"},
            {"src": "", "tgt": "B"},
            {"src": "B", "tgt": "missing"},
        ],
        batch_size=2,
    )

    assert result == {(quoted_id, "B"): {"weight": 2}, ("B", quoted_id): {"weight": 1}}
    first, second = graph._query.await_args_list
    assert '"test_graph"."_ag_label_edge"' in first.args[0]
    assert first.kwargs["params"] == {
        "sources": [quoted_id, "B"],
        "targets": ["B", quoted_id],
    }
    assert second.kwargs["params"] == {"sources": ["B"], "targets": ["missing"]}


async def test_parallel_edges_fall_back_only_for_the_affected_batch(graph):
    graph._query.side_effect = [
        [edge("A", "B", 1), edge("A", "B", 2)],
        [edge("A", "B", 3)],
        [edge("A", "B", 4)],
        [edge("B", "C", 5)],
    ]

    result = await graph.get_edges_batch(
        [{"src": "A", "tgt": "B"}, {"src": "B", "tgt": "C"}],
        batch_size=1,
    )

    assert result == {("A", "B"): {"weight": 4}, ("B", "C"): {"weight": 5}}
    first, forward, backward, last = graph._query.await_args_list
    assert first.kwargs["params"] == {"sources": ["A"], "targets": ["B"]}
    assert "MATCH (a)-[r]->(b)" in forward.args[0]
    assert "MATCH (a)<-[r]-(b)" in backward.args[0]
    for call in (forward, backward):
        assert json.loads(call.kwargs["params"]["params"]) == {
            "pairs": [{"src": "A", "tgt": "B"}]
        }
    assert last.kwargs["params"] == {"sources": ["B"], "targets": ["C"]}


async def test_empty_input_avoids_database_calls(graph):
    assert await graph.get_edges_batch([]) == {}
    graph._query.assert_not_awaited()
