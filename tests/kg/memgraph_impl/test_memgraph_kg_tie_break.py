"""``get_knowledge_graph('*')`` must break degree ties on the entity label.

Same contract, and same reasoning, as the Neo4j test next to it: ``LIMIT
max_nodes`` cuts through a band of equal-degree entities, and without a second
sort key which of them survive is unconstrained.

The Cypher is asserted rather than the returned node set: exercising the real
ordering needs a live Memgraph, and the ORDER BY clause is the whole fix.
"""

import re

import pytest

from lightrag.kg.memgraph_impl import MemgraphStorage


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
        # Keep the last result once the scripted ones are used up: the storage
        # swallows exceptions from this block, so an IndexError here would show
        # up as a silently empty graph instead of a failed assertion.
        return self._results.pop(0) if len(self._results) > 1 else self._results[0]


class _FakeDriver:
    def __init__(self, results, calls):
        self._results = results
        self._calls = calls

    def session(self, **kwargs):
        return _FakeSession(self._results, self._calls)


def _make_storage(records):
    calls = []
    storage = MemgraphStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    storage._driver = _FakeDriver([_FakeResult(r) for r in records], calls)
    storage._DATABASE = "memgraph"
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
    assert "ORDER BY degree DESC LIMIT" not in main_query, main_query
    assert result.is_truncated is True
