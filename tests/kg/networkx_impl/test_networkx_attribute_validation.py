"""NetworkXStorage rejects an unstorable attribute value before mutating.

This backend mutates ``self._graph`` in ``upsert_*`` and serializes it later in
``index_done_callback``, with no rollback in between. A value GraphML cannot
encode therefore does not fail the one write that carried it -- it stays on the
node and fails *every* later flush, because ``write_nx_graph`` serializes the
whole graph. Reads keep working, so the instance looks healthy while nothing
persists. Validating before the mutation converts that into a failed single
write. See GHSA-c922-pw4m-4wcv.

The entry-layer allowlist in ``utils_graph`` already stops the reported route
from delivering such a value; this is the backend-level guard, so any future
caller (including the Python API) gets the same answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline

UNSTORABLE = [
    pytest.param({"any": ["object"]}, id="dict"),
    pytest.param(["a", "b"], id="list"),
    pytest.param(None, id="none"),
    pytest.param("a\x0bb", id="control-char"),
    pytest.param(float("nan"), id="nan"),
]


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def _embed(texts):
    return np.random.rand(len(texts), 8)


@pytest.fixture
async def storage(tmp_path):
    store = NetworkXStorage(
        namespace="test_graph",
        workspace="ws",
        global_config={"working_dir": str(tmp_path), "embedding_batch_num": 10},
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=512, func=_embed),
    )
    await store.initialize()
    return store


async def _seed(storage):
    """One clean node, committed, so later flushes have something to lose."""
    await storage.upsert_node("clean", {"entity_id": "clean", "description": "ok"})
    assert await storage.index_done_callback() is True


def _graphml(storage) -> str:
    with open(storage._graphml_xml_file, encoding="utf-8") as handle:
        return handle.read()


class TestUpsertNode:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", UNSTORABLE)
    async def test_unstorable_value_is_rejected_before_mutating(self, storage, value):
        await _seed(storage)

        with pytest.raises(ValueError, match="attribute 'poc'"):
            await storage.upsert_node("bad", {"entity_id": "bad", "poc": value})

        # Not merely refused -- never applied. Pre-fix the node was added and
        # every later flush raised from write_graphml.
        assert not await storage.has_node("bad")
        await storage.upsert_node("after", {"entity_id": "after"})
        assert await storage.index_done_callback() is True
        assert "after" in _graphml(storage)

    @pytest.mark.asyncio
    async def test_context_identifies_the_node(self, storage):
        with pytest.raises(ValueError, match=r"\[ws\] node `bad`"):
            await storage.upsert_node("bad", {"entity_id": "bad", "poc": None})

    @pytest.mark.asyncio
    async def test_ordinary_values_still_pass(self, storage):
        await storage.upsert_node(
            "n1",
            {
                "entity_id": "n1",
                "description": "multi\nline\tdescription",
                "created_at": 123,
                "weight": 1.5,
                "flag": True,
            },
        )

        assert (await storage.get_node("n1"))["created_at"] == 123
        assert await storage.index_done_callback() is True

    @pytest.mark.asyncio
    async def test_attribute_names_are_not_restricted(self, storage):
        """Names are inert in GraphML, and rejecting them would break re-ingest.

        A graph written before this validation existed can hold an attribute
        name this backend would now refuse; the ingestion merge re-writes stored
        attributes verbatim, so a name check here would turn that node into a
        permanently failing document for no safety gain.
        """
        await storage.upsert_node("n1", {"entity_id": "n1", "legacy.name": "kept"})

        assert (await storage.get_node("n1"))["legacy.name"] == "kept"
        assert await storage.index_done_callback() is True


class TestUpsertEdge:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", UNSTORABLE)
    async def test_unstorable_value_is_rejected_before_mutating(self, storage, value):
        await _seed(storage)

        with pytest.raises(ValueError, match="attribute 'poc'"):
            await storage.upsert_edge("a", "b", {"weight": 1.0, "poc": value})

        assert not await storage.has_edge("a", "b")
        # add_edge would also have created both endpoints implicitly.
        assert not await storage.has_node("a")
        await storage.upsert_node("after", {"entity_id": "after"})
        assert await storage.index_done_callback() is True
        assert "after" in _graphml(storage)

    @pytest.mark.asyncio
    async def test_context_identifies_the_edge(self, storage):
        with pytest.raises(ValueError, match=r"\[ws\] edge `a`~`b`"):
            await storage.upsert_edge("a", "b", {"poc": None})


class TestBatchesAreAllOrNothing:
    @pytest.mark.asyncio
    async def test_nodes_batch_applies_nothing_when_one_item_is_bad(self, storage):
        """The reason validation is a separate pass over the batch.

        Validating inside the mutation loop would leave every item before the
        offender in the in-memory graph -- a partial write that the next flush
        either persists or, if the offender was already applied, cannot persist
        at all.
        """
        nodes = [
            ("good1", {"entity_id": "good1", "description": "a"}),
            ("bad", {"entity_id": "bad", "poc": {"any": ["object"]}}),
            ("good2", {"entity_id": "good2", "description": "b"}),
        ]

        with pytest.raises(ValueError, match="attribute 'poc'"):
            await storage.upsert_nodes_batch(nodes)

        assert not await storage.has_node("good1")
        assert not await storage.has_node("bad")
        assert not await storage.has_node("good2")

    @pytest.mark.asyncio
    async def test_edges_batch_applies_nothing_when_one_item_is_bad(self, storage):
        edges = [
            ("a", "b", {"weight": 1.0}),
            ("c", "d", {"weight": 1.0, "poc": None}),
        ]

        with pytest.raises(ValueError, match="attribute 'poc'"):
            await storage.upsert_edges_batch(edges)

        assert not await storage.has_edge("a", "b")
        assert not await storage.has_edge("c", "d")

    @pytest.mark.asyncio
    async def test_clean_batches_still_apply(self, storage):
        await storage.upsert_nodes_batch(
            [
                ("n1", {"entity_id": "n1", "description": "a"}),
                ("n2", {"entity_id": "n2", "description": "b"}),
            ]
        )
        await storage.upsert_edges_batch([("n1", "n2", {"weight": 2.0})])

        assert await storage.has_edge("n1", "n2")
        assert await storage.index_done_callback() is True
        assert "n1" in _graphml(storage)
