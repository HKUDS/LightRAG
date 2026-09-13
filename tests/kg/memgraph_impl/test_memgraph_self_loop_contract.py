"""Every degree path in ``MemgraphStorage`` must exclude self-loops.

Same contract, and same reasoning, as the Neo4j test next to it:
``BaseGraphStorage.node_degree`` measures connectivity and a self-loop connects
nothing, so it contributes 0.

**Why the Cypher text is asserted rather than a returned degree.** Whether
``(n)-[r]-()`` yields one row or two for a self-loop is a property of the
server, which no offline test here can exercise. The exclusion is written to be
immune to that: it compares the NODES (``m <> n``), so the self-loop is dropped
whichever way the engine matches it. What stays checkable offline -- and what
these tests pin -- is that every degree query still carries the filter, and
that no path was left behind when the rule changed.

``MemgraphStorage`` overrides neither ``node_degrees_batch`` nor
``edge_degrees_batch``, so both reach the graph through ``node_degree`` and are
covered by the scalar test here.
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

    def __aiter__(self):
        records = self._record if isinstance(self._record, list) else [self._record]
        self._iter = iter(records)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

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


def _assert_excludes_self_loops(query: str) -> None:
    """The degree pattern binds the far endpoint and requires it to differ.

    Both halves matter: binding ``(m)`` without the comparison still counts the
    self-loop, and this is what fails on the unfiltered
    ``OPTIONAL MATCH (n)-[r]-()`` it replaced.
    """
    normalized = _normalize(query)
    assert "(n)-[r]-(m)" in normalized, normalized
    assert "WHERE m <> n" in normalized, normalized
    assert "(n)-[r]-()" not in normalized, normalized


@pytest.mark.asyncio
async def test_node_degree_excludes_self_loops():
    """Also covers node_degrees_batch and edge_degrees_batch, which this
    backend does not override -- both reach the graph through here."""
    storage, calls = _make_storage([{"degree": 0}])

    assert await storage.node_degree("Loop") == 0

    assert len(calls) == 1, calls
    _assert_excludes_self_loops(calls[0])


@pytest.mark.asyncio
async def test_get_popular_labels_excludes_self_loops():
    """A self-loop must not lift a node up the entity picker's ranking."""
    storage, calls = _make_storage([[{"label": "A"}]])

    assert await storage.get_popular_labels(limit=5) == ["A"]

    assert len(calls) == 1, calls
    _assert_excludes_self_loops(calls[0])


@pytest.mark.asyncio
async def test_knowledge_graph_star_ranking_excludes_self_loops():
    """The ``*`` path is degree-ranked, so the rule decides which nodes
    survive ``max_nodes`` truncation."""
    storage, calls = _make_storage(
        [
            {"total": 5},
            {"node_info": [], "relationships": []},
        ]
    )

    result = await storage.get_knowledge_graph("*", max_depth=1, max_nodes=2)

    assert len(calls) == 2, calls
    _assert_excludes_self_loops(calls[1])
    # The tie-break this shares with test_memgraph_kg_tie_break must survive
    # the added WHERE: both clauses sit between the same MATCH and LIMIT.
    assert "ORDER BY degree DESC, n.entity_id ASC" in _normalize(calls[1])
    assert result.is_truncated is True
