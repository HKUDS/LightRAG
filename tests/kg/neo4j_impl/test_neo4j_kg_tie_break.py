"""``get_knowledge_graph('*')`` must break degree ties on the entity label.

The BaseGraphStorage contract ranks by degree descending, ties on the label
ascending, because ``LIMIT max_nodes`` almost always cuts through a band of
equal-degree entities: with no second sort key, which nodes survive is whatever
order the plan happened to produce, so the same graph can answer differently
run to run.

The Cypher is asserted rather than the returned node set: exercising the real
ordering needs a live Neo4j, and the ORDER BY clause is the whole fix.
"""

import re

import pytest

from lightrag.kg.neo4j_impl import Neo4JStorage


pytestmark = pytest.mark.offline


def _normalize(query: str) -> str:
    """Collapse whitespace so multi-line Cypher can be matched as one string."""
    return re.sub(r"\s+", " ", query).strip()


class _FakeResult:
    def __init__(self, record):
        self._record = record
        self.consumed = False

    async def single(self):
        return self._record

    async def consume(self):
        self.consumed = True
        return None


class _FakeSession:
    def __init__(self, results, calls):
        self._results = results
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, parameters=None, **kwargs):
        self._calls.append(query)
        # Keep the last result once the scripted ones are used up, so an
        # unexpected extra query surfaces as a bad record rather than IndexError.
        return self._results.pop(0) if len(self._results) > 1 else self._results[0]


class _FakeDriver:
    def __init__(self, results, calls):
        self._results = results
        self._calls = calls

    def session(self, **kwargs):
        return _FakeSession(self._results, self._calls)


def _make_storage(records):
    calls = []
    storage = Neo4JStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    storage._driver = _FakeDriver([_FakeResult(r) for r in records], calls)
    storage._DATABASE = "neo4j"
    return storage, calls


@pytest.mark.asyncio
async def test_star_mode_ranks_by_degree_then_entity_id():
    storage, calls = _make_storage(
        [
            {"total": 5},
            {"node_info": [], "relationships": []},
        ]
    )

    result = await storage.get_knowledge_graph("*", max_depth=1, max_nodes=2)

    assert len(calls) == 2, calls
    main_query = _normalize(calls[1])
    assert "ORDER BY degree DESC, n.entity_id ASC" in main_query, main_query
    # The tie-break is the point: a bare degree ordering is what allowed the
    # equal-degree band at the cutoff to come back in an arbitrary order.
    assert "ORDER BY degree DESC LIMIT" not in main_query, main_query
    assert result.is_truncated is True
