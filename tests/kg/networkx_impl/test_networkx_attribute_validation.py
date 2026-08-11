"""NetworkXStorage rejects an unencodable attribute name or value before mutating.

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

import networkx as nx
import numpy as np
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline

UNENCODABLE = [
    pytest.param({"any": ["object"]}, id="dict"),
    pytest.param(["a", "b"], id="list"),
    pytest.param(None, id="none"),
    pytest.param("a\x0bb", id="control-char"),
    pytest.param("a\x00b", id="nul"),
]

# Refused by the *caller* contract (the Neo4j driver cannot pack them) but
# round-tripped by GraphML unchanged, so this backend must accept them -- see
# TestPortableButUnencodableValuesAreNotThisBackendsBusiness.
NOT_PORTABLE_BUT_ENCODABLE = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="negative-inf"),
    pytest.param(2**63, id="int64-max-plus-one"),
    pytest.param(10**400, id="huge-int"),
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
    @pytest.mark.parametrize("value", UNENCODABLE)
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
    async def test_a_name_a_portable_rule_would_refuse_is_kept(self, storage):
        """See TestAttributeNames for the full name contract."""
        await storage.upsert_node("n1", {"entity_id": "n1", "legacy.name": "kept"})

        assert (await storage.get_node("n1"))["legacy.name"] == "kept"
        assert await storage.index_done_callback() is True


class TestAttributeNames:
    """Names get the XML rule too -- GraphML writes them into ``attr.name``.

    Checked against networkx, not assumed: a name holding a control character, a
    lone surrogate or U+FFFE makes ``write_graphml`` raise, which is the same
    instance-wide outage an unencodable value causes. And the check is safe on a
    rewrite payload precisely because such a name can never have been persisted
    here -- the write that would have stored it failed.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("bad\x0bname", id="control-char"),
            pytest.param("bad\x00name", id="nul"),
            pytest.param("bad\ud800name", id="lone-surrogate"),
            pytest.param("bad\ufffename", id="non-character"),
        ],
    )
    async def test_unencodable_name_is_rejected_before_mutating(self, storage, name):
        await _seed(storage)

        with pytest.raises(ValueError, match="attribute name"):
            await storage.upsert_node("bad", {"entity_id": "bad", name: "x"})

        assert not await storage.has_node("bad")
        await storage.upsert_node("after", {"entity_id": "after"})
        assert await storage.index_done_callback() is True
        assert "after" in _graphml(storage)

    @pytest.mark.asyncio
    async def test_a_non_string_name_is_a_validation_error(self, storage):
        """networkx would raise TypeError; a 500 for bad input is not useful."""
        with pytest.raises(ValueError, match="must be a string, got int"):
            await storage.upsert_node("n1", {"entity_id": "n1", 1: "x"})

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name",
        ["a.b", "$set", "display-name", "has space", "名前", "", "tab\tname"],
    )
    async def test_encodable_names_are_accepted_and_round_trip(self, storage, name):
        """A portable rule would refuse several of these; GraphML stores them all.

        Refusing them would strand any node whose stored names predate this
        validation, since rewrites spread stored attributes back into the payload.
        """
        await storage.upsert_node("n1", {"entity_id": "n1", name: "kept"})
        assert await storage.index_done_callback() is True

        assert nx.read_graphml(storage._graphml_xml_file).nodes["n1"][name] == "kept"

    @pytest.mark.asyncio
    async def test_edge_names_are_validated_too(self, storage):
        with pytest.raises(ValueError, match="attribute name"):
            await storage.upsert_edge("a", "b", {"weight": 1.0, "bad\x0bname": "x"})

        assert not await storage.has_edge("a", "b")


class TestUpsertEdge:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", UNENCODABLE)
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


class TestPortableButUnencodableValuesAreNotThisBackendsBusiness:
    """The guard is "can GraphML encode this", not the portable contract.

    `NaN`, `inf` and an integer past int64 are all refused by
    `BaseGraphStorage.upsert_node`'s caller contract, because the Neo4j driver
    cannot pack them -- but GraphML round-trips all of them unchanged. So a
    workspace can already hold one, and since every rewrite path spreads a
    fetched object's stored attributes back into the payload, enforcing the
    portable rule *here* would make those objects permanently unmodifiable. The
    portable bounds are enforced where caller input enters instead.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", NOT_PORTABLE_BUT_ENCODABLE)
    async def test_node_accepts_and_round_trips_the_value(self, storage, value):
        await storage.upsert_node("n1", {"entity_id": "n1", "legacy": value})
        assert await storage.index_done_callback() is True

        reread = nx.read_graphml(storage._graphml_xml_file).nodes["n1"]["legacy"]
        # NaN is never equal to itself; compare by identity of behaviour instead.
        assert reread == value or (reread != reread and value != value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", NOT_PORTABLE_BUT_ENCODABLE)
    async def test_a_rewrite_carrying_the_stored_value_still_writes(
        self, storage, value
    ):
        """The dead-end this split removes: an ordinary edit of a legacy object."""
        await storage.upsert_node("n1", {"entity_id": "n1", "legacy": value})
        await storage.index_done_callback()

        stored = await storage.get_node("n1")
        # Exactly what _edit_entity_impl / the extraction rebuild construct.
        await storage.upsert_node("n1", {**stored, "description": "an update"})

        assert (await storage.get_node("n1"))["description"] == "an update"
        assert await storage.index_done_callback() is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", NOT_PORTABLE_BUT_ENCODABLE)
    async def test_edge_rewrite_carrying_the_stored_value_still_writes(
        self, storage, value
    ):
        await storage.upsert_edge("a", "b", {"weight": 1.0, "legacy": value})
        await storage.index_done_callback()

        stored = await storage.get_edge("a", "b")
        await storage.upsert_edge("a", "b", {**stored, "description": "an update"})

        assert await storage.index_done_callback() is True
