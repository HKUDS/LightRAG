"""A DECLINED graph commit must reach the caller through _persist_graph_updates.

Rule 5 of #3854 says a decline has to surface as a failure, because declining
DISCARDS the in-memory mutation. Two of the three commit paths already did that
(``_commit_graph_or_raise`` for the admin paths that route through it,
``LightRAG._flush_storages``'s ``_flush_one`` for the pipeline).
``_persist_graph_updates`` -- the path ``aedit_relation`` / ``acreate_entity`` /
``acreate_relation`` use -- dropped the return value, so the decline was silent:
the graph mutation gone, the vector and tracking rows of the same operation
durable, and the caller told it succeeded.
"""

from typing import Any

import pytest

from lightrag import utils_graph
from lightrag.exceptions import CommitBookkeepingError

pytestmark = pytest.mark.offline


class _Store:
    """Minimal storage double: records the flush, answers however it is told."""

    def __init__(self, namespace: str, answer: Any = None):
        self.namespace = namespace
        self._answer = answer
        self.flushes = 0

    async def index_done_callback(self):
        self.flushes += 1
        if isinstance(self._answer, BaseException):
            raise self._answer
        return self._answer


@pytest.mark.asyncio
async def test_a_declined_graph_commit_raises():
    """The defect this file exists for: False must not be swallowed."""
    graph = _Store("chunk_entity_relation", answer=False)

    with pytest.raises(RuntimeError, match="in-memory mutation was discarded"):
        await utils_graph._persist_graph_updates(chunk_entity_relation_graph=graph)

    assert graph.flushes == 1


@pytest.mark.asyncio
async def test_the_error_names_the_store_that_declined():
    """An operator reading the 500 has to know which store refused."""
    graph = _Store("chunk_entity_relation", answer=False)

    with pytest.raises(RuntimeError) as excinfo:
        await utils_graph._persist_graph_updates(
            entities_vdb=_Store("entities"),
            chunk_entity_relation_graph=graph,
        )

    assert "chunk_entity_relation" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_backend_returning_none_still_succeeds():
    """Only an explicit False counts.

    ``StorageNameSpace.index_done_callback`` is declared ``-> None``, so every
    backend that simply returns nothing -- which is all of them except
    NetworkX / Nano / FAISS -- must be unaffected. Testing ``is False`` rather
    than falsiness is the whole reason this holds.
    """
    stores = [_Store(f"ns{i}") for i in range(3)]

    await utils_graph._persist_graph_updates(
        entities_vdb=stores[0],
        chunk_entity_relation_graph=stores[1],
        entity_chunks_storage=stores[2],
    )

    assert [s.flushes for s in stores] == [1, 1, 1]


@pytest.mark.asyncio
async def test_a_true_answer_is_not_a_decline():
    """Nano and FAISS return True from a successful commit."""
    await utils_graph._persist_graph_updates(
        entities_vdb=_Store("entities", answer=True),
        chunk_entity_relation_graph=_Store("chunk_entity_relation", answer=True),
    )


@pytest.mark.asyncio
async def test_every_sibling_still_flushes_before_the_raise():
    """Pins the accepted residue, so it stays a decision.

    The siblings are deliberately NOT cancelled: commit_in_storage_io defers
    cancellation, so a cancelled flush may have written anyway and which ones
    did would depend on scheduling. Awaiting them all first makes the outcome
    deterministic -- their rows are durable, the graph object is not, and the
    caller's retry rewrites both under the same deterministic ids.
    """
    siblings = [_Store("entities"), _Store("relationships"), _Store("entity_chunks")]
    graph = _Store("chunk_entity_relation", answer=False)

    with pytest.raises(RuntimeError):
        await utils_graph._persist_graph_updates(
            entities_vdb=siblings[0],
            relationships_vdb=siblings[1],
            chunk_entity_relation_graph=graph,
            entity_chunks_storage=siblings[2],
        )

    assert [s.flushes for s in siblings] == [1, 1, 1]


@pytest.mark.asyncio
async def test_a_publication_failure_is_still_swallowed():
    """CommitBookkeepingError is the OPPOSITE answer and must not start raising.

    It says the write DID land and only its cross-process publication failed.
    Folding it into the decline path would report a durable mutation as one
    that did not happen -- the direction AGENTS.md forbids outright.
    """
    graph = _Store(
        "chunk_entity_relation", answer=CommitBookkeepingError("publish failed")
    )
    sibling = _Store("entities")

    await utils_graph._persist_graph_updates(
        entities_vdb=sibling,
        chunk_entity_relation_graph=graph,
    )

    assert graph.flushes == 1
    assert sibling.flushes == 1


@pytest.mark.asyncio
async def test_a_publication_failure_alongside_a_decline_still_raises():
    """One store's durable-but-unpublished write does not excuse another's decline."""
    with pytest.raises(RuntimeError, match="in-memory mutation was discarded"):
        await utils_graph._persist_graph_updates(
            entities_vdb=_Store("entities", answer=CommitBookkeepingError("boom")),
            chunk_entity_relation_graph=_Store("chunk_entity_relation", answer=False),
        )


@pytest.mark.asyncio
async def test_no_storages_is_still_a_no_op():
    await utils_graph._persist_graph_updates()
