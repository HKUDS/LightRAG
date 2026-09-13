"""Every degree path in ``PGGraphStorage`` (Apache AGE) must exclude self-loops.

``BaseGraphStorage.node_degree`` measures connectivity, and a self-loop
connects nothing -- the same reason ``_reject_self_loop_relation`` refuses to
create one. So a self-loop contributes 0.

The SQL text is asserted rather than a returned degree: AGE stores edges as
rows in ``_ag_label_edge`` / ``"DIRECTED"``, and reproducing the row shapes
faithfully enough to count them needs a live server with the extension loaded.
The ``start_id <> end_id`` predicate IS the rule, and what stays checkable
offline is that no degree path was left behind when the rule changed.

``node_degree`` and ``edge_degree`` both delegate to the batch methods on this
backend, so the ``node_degrees_batch`` test below covers all three.

The listing rule is the opposite and is NOT asserted here: ``get_node_edges``
still reports a self-loop, because deletion, rename, merge and document purge
all resolve a node's relation rows through it.
"""

import re

import pytest
from unittest.mock import AsyncMock

from lightrag.kg.postgres_impl import PGGraphStorage


pytestmark = pytest.mark.offline


def _normalize(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


def _make_storage(rows=()):
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
    storage.__post_init__()
    storage._query = AsyncMock(return_value=list(rows))
    return storage


def _issued_sql(storage) -> list[str]:
    return [_normalize(call.args[0]) for call in storage._query.call_args_list]


@pytest.mark.asyncio
async def test_node_degrees_batch_excludes_self_loops():
    """Covers node_degree and edge_degree too: both delegate to the batches."""
    storage = _make_storage([{"node_id": '"Loop"', "out_degree": 0, "in_degree": 0}])

    await storage.node_degrees_batch(["Loop"])

    sql = _issued_sql(storage)
    assert len(sql) == 1, sql
    # Both endpoint arms are filtered: dropping it from either one lets the
    # self-loop back in through the other.
    assert sql[0].count("WHERE d.start_id <> d.end_id") == 2, sql[0]


@pytest.mark.asyncio
async def test_node_degree_reaches_the_filtered_batch_query():
    """The scalar is a one-element batch, so it cannot drift from it."""
    storage = _make_storage([{"node_id": '"Loop"', "out_degree": 0, "in_degree": 0}])

    assert await storage.node_degree("Loop") == 0

    sql = _issued_sql(storage)
    assert len(sql) == 1, sql
    assert sql[0].count("WHERE d.start_id <> d.end_id") == 2, sql[0]


@pytest.mark.asyncio
async def test_get_popular_labels_excludes_self_loops():
    """A self-loop must not lift a node up the entity picker's ranking."""
    storage = _make_storage([{"label": "A"}])

    await storage.get_popular_labels(limit=5)

    sql = _issued_sql(storage)
    assert sql, "get_popular_labels issued no query"
    ranking = sql[0]
    assert ranking.count("WHERE start_id <> end_id") == 2, ranking
    # The tie-break must survive the added predicate.
    assert 'degree DESC, label COLLATE "C" ASC' in ranking, ranking
