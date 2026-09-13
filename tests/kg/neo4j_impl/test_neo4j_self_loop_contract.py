"""Every degree path in ``Neo4JStorage`` must exclude self-loops.

``BaseGraphStorage.node_degree`` measures connectivity, and a self-loop
connects nothing -- the same reason ``_reject_self_loop_relation`` refuses to
create one. So a self-loop contributes 0, not the 1-or-2 an undirected match
would otherwise produce.

**Why the Cypher text is asserted rather than a returned degree.** Whether
``(n)-[r]-()`` yields one row or two for a self-loop is a property of the
server, which no offline test here can exercise. The exclusion is written to
be immune to that: it compares the NODES (``m <> n``), so the self-loop is
dropped whichever way the engine matches it. What remains checkable offline --
and what these tests pin -- is that every degree query still carries the
filter, and that no path was left behind when the rule changed. The residual
live-server question shrinks from "how does an undirected match treat a
self-loop?" to "does ``<>`` exclude equal nodes?".

The listing rule is the opposite and is NOT asserted here: ``get_node_edges``
still reports a self-loop, because deletion, rename, merge and document purge
all resolve a node's relation rows through it.
"""

import re

import pytest

from lightrag.kg.neo4j_impl import Neo4JStorage


pytestmark = pytest.mark.offline


def _normalize(query: str) -> str:
    """Collapse whitespace so multi-line Cypher can be matched as one string."""
    return re.sub(r"\s+", " ", query).strip()


class _FakeResult:
    """Supports both read shapes: ``single()`` and async iteration."""

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
    storage = Neo4JStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    storage._driver = _FakeDriver([_FakeResult(r) for r in records], calls)
    storage._DATABASE = "neo4j"
    return storage, calls


def _assert_excludes_self_loops(query: str) -> None:
    """The degree pattern binds the far endpoint and requires it to differ.

    Both halves matter. Binding ``(m)`` without the comparison counts the
    self-loop; comparing without binding cannot be expressed at all. Checking
    them together is what makes this fail on the unfiltered
    ``OPTIONAL MATCH (n)-[r]-()`` this replaced.
    """
    normalized = _normalize(query)
    assert "(n)-[r]-(m)" in normalized, normalized
    assert "WHERE m <> n" in normalized, normalized
    assert "(n)-[r]-()" not in normalized, normalized


@pytest.mark.asyncio
async def test_node_degree_excludes_self_loops():
    storage, calls = _make_storage([{"degree": 0}])

    assert await storage.node_degree("Loop") == 0

    assert len(calls) == 1, calls
    _assert_excludes_self_loops(calls[0])


@pytest.mark.asyncio
async def test_node_degrees_batch_excludes_self_loops():
    """The batch must exclude them the same way the scalar does: a backend
    whose two paths disagree ranks the same node differently depending on
    which one a caller reaches for."""
    storage, calls = _make_storage([[{"entity_id": "Loop", "degree": 0}]])

    assert await storage.node_degrees_batch(["Loop"]) == {"Loop": 0}

    assert len(calls) == 1, calls
    _assert_excludes_self_loops(calls[0])
    # Not a count{} subquery any more: the scalar and the batch now share one
    # provable shape rather than two filters that have to agree by inspection.
    assert "count {" not in _normalize(calls[0])


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
    # The tie-break this shares with test_neo4j_kg_tie_break must survive the
    # added WHERE: both clauses sit between the same MATCH and LIMIT.
    assert "ORDER BY degree DESC, n.entity_id ASC" in _normalize(calls[1])
    assert result.is_truncated is True
