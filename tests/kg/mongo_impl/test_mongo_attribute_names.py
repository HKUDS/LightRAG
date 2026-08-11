"""MongoGraphStorage rejects attribute names the server would interpret.

Attribute names reach MongoDB inside a ``$set`` document, where they are field
*paths*, not names. ``{"source_ids.0": x}`` therefore rewrites the first element
of the chunk-attribution array instead of creating a field called
``source_ids.0``, and a leading ``$`` is read as an update operator. ``_id``
addresses the document this class uses as its update filter.

Companion to the entry-layer allowlist in ``utils_graph`` (GHSA-c922-pw4m-4wcv):
that stops the reported route from delivering such a name, this stops any other
caller, including the Python API.

The rule stops at the interpretation hazard. A merely unusual name such as
``display-name`` is stored flat, comes back out of ``get_node``, and is spread
into the next rewrite payload by every edit / rename / merge / rebuild path --
so refusing it would make the entity permanently unmodifiable.
``TestLegacyStoredNamesStillWrite`` pins that.

The collection is a double -- the assertion is that the write is refused
*before* any server call, so no live MongoDB is needed or wanted here.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag.kg.mongo_impl import MongoGraphStorage

pytestmark = pytest.mark.offline

UNSAFE_NAMES = [
    pytest.param("source_ids.0", id="dotted-array-element"),
    pytest.param("a.b", id="dotted"),
    pytest.param("$set", id="update-operator"),
    pytest.param("$where", id="query-operator"),
    pytest.param("", id="empty"),
]

# Names the pre-allowlist edit API accepted and this collection stores flat, so
# they DO come back out of `get_node` and into the next rewrite payload.
LEGACY_STORED_NAMES = [
    pytest.param("display-name", id="dashed"),
    pytest.param("has space", id="space"),
    pytest.param("9legacy", id="leading-digit"),
]


def _make_storage():
    storage = MongoGraphStorage.__new__(MongoGraphStorage)
    storage.workspace = "test"
    storage.namespace = "chunk_entity_relation"
    storage.global_config = {}
    storage._collection_name = "test_nodes"
    storage._edge_collection_name = "test_edges"
    storage._max_upsert_payload_bytes = 1_000_000
    storage._max_upsert_records_per_batch = 100
    storage.collection = SimpleNamespace(update_one=AsyncMock(), bulk_write=AsyncMock())
    storage.edge_collection = SimpleNamespace(
        update_one=AsyncMock(), bulk_write=AsyncMock()
    )
    return storage


def _assert_no_server_call(storage):
    storage.collection.update_one.assert_not_awaited()
    storage.collection.bulk_write.assert_not_awaited()
    storage.edge_collection.update_one.assert_not_awaited()
    storage.edge_collection.bulk_write.assert_not_awaited()


class TestUpsertNode:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", UNSAFE_NAMES)
    async def test_unsafe_name_is_refused(self, name):
        storage = _make_storage()

        with pytest.raises(ValueError, match="invalid attribute name"):
            await storage.upsert_node("n1", {"entity_id": "n1", name: "x"})

        _assert_no_server_call(storage)

    @pytest.mark.asyncio
    async def test_reserved_id_is_refused(self):
        """``_id`` is identifier-shaped, so the generic rule alone lets it past.

        Setting it is not an attribute write -- it is an attempt to move the
        document, which the server refuses as an immutable-field error.
        """
        storage = _make_storage()

        with pytest.raises(ValueError, match="reserved by MongoDB"):
            await storage.upsert_node("n1", {"entity_id": "n1", "_id": "elsewhere"})

        _assert_no_server_call(storage)

    @pytest.mark.asyncio
    async def test_ordinary_names_still_reach_the_server(self):
        storage = _make_storage()

        await storage.upsert_node(
            "n1",
            {
                "entity_id": "n1",
                "description": "d",
                "source_id": "chunk-1",
                "created_at": 1,
            },
        )

        storage.collection.update_one.assert_awaited_once()
        _filter, update = storage.collection.update_one.await_args.args
        assert update["$set"]["description"] == "d"
        # The derived array the dotted-name attack targets.
        assert update["$set"]["source_ids"] == ["chunk-1"]


class TestUpsertEdge:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", UNSAFE_NAMES)
    async def test_unsafe_name_is_refused(self, name):
        storage = _make_storage()

        with pytest.raises(ValueError, match="invalid attribute name"):
            await storage.upsert_edge("a", "b", {"weight": 1.0, name: "x"})

        # Refused before the endpoint-placeholder bulk_write, so no orphan
        # endpoint nodes are created for an edge that never lands.
        _assert_no_server_call(storage)


class TestBatchesValidateBeforeWriting:
    @pytest.mark.asyncio
    async def test_nodes_batch_refuses_whole_batch(self):
        storage = _make_storage()
        nodes = [
            ("good", {"entity_id": "good", "description": "a"}),
            ("bad", {"entity_id": "bad", "source_ids.0": "x"}),
        ]

        with pytest.raises(ValueError, match="invalid attribute name"):
            await storage.upsert_nodes_batch(nodes)

        _assert_no_server_call(storage)

    @pytest.mark.asyncio
    async def test_edges_batch_refuses_before_creating_endpoints(self):
        """The placeholder endpoints go in first, so validation must precede them."""
        storage = _make_storage()
        edges = [
            ("a", "b", {"weight": 1.0}),
            ("c", "d", {"weight": 1.0, "$set": "x"}),
        ]

        with pytest.raises(ValueError, match="invalid attribute name"):
            await storage.upsert_edges_batch(edges)

        _assert_no_server_call(storage)


class TestLegacyStoredNamesStillWrite:
    """Unusual names that predate the field allowlist must keep working.

    A graph edited through the pre-allowlist API can hold ``display-name``; the
    rewrite paths read the stored object and spread every attribute back into
    ``upsert_node``. A name rule wider than the interpretation hazard would turn
    that entity into a permanent 400 even when the request only changes
    ``description``.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", LEGACY_STORED_NAMES)
    async def test_node_rewrite_carrying_a_legacy_name_succeeds(self, name):
        storage = _make_storage()

        await storage.upsert_node(
            "n1", {"entity_id": "n1", name: "legacy", "description": "updated"}
        )

        storage.collection.update_one.assert_awaited_once()
        _filter, update = storage.collection.update_one.await_args.args
        assert update["$set"][name] == "legacy"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", LEGACY_STORED_NAMES)
    async def test_edge_rewrite_carrying_a_legacy_name_succeeds(self, name):
        storage = _make_storage()

        await storage.upsert_edge("a", "b", {"weight": 1.0, name: "legacy"})

        storage.edge_collection.update_one.assert_awaited_once()
        _filter, update = storage.edge_collection.update_one.await_args.args
        assert update["$set"][name] == "legacy"

    @pytest.mark.asyncio
    async def test_a_dollar_sign_inside_the_name_is_accepted(self):
        """Only a *leading* ``$`` is an operator."""
        storage = _make_storage()

        await storage.upsert_node("n1", {"entity_id": "n1", "cost$usd": "5"})

        _filter, update = storage.collection.update_one.await_args.args
        assert update["$set"]["cost$usd"] == "5"


class TestNoValidationToMutationWindow:
    """The edge paths await MongoDB between the check and the use.

    `upsert_edge` materializes both endpoints before writing the edge document,
    and `upsert_edges_batch` does the same for the whole batch. Validating the
    caller's own mapping left a window in which a coroutine holding it could add
    a field path such as `source_ids.0` after the check -- which is the attack
    the name rule exists to stop, since a dotted key in `$set` rewrites an
    element of the chunk-attribution array. Both paths now validate a snapshot
    and build the update from that same snapshot.

    The mutation is injected from inside the endpoint `bulk_write`, which is the
    only place it could ever have happened.
    """

    @pytest.mark.asyncio
    async def test_a_mutation_during_the_endpoint_write_does_not_reach_mongo(self):
        storage = _make_storage()
        payload = {"weight": 1.0, "description": "clean"}

        async def poison_during_bulk_write(*args, **kwargs):
            payload["source_ids.0"] = "attribution rewritten"

        storage.collection.bulk_write = AsyncMock(side_effect=poison_during_bulk_write)

        await storage.upsert_edge("a", "b", payload)

        storage.edge_collection.update_one.assert_awaited_once()
        _filter, update = storage.edge_collection.update_one.await_args.args
        assert "source_ids.0" not in update["$set"]
        assert update["$set"]["description"] == "clean"

    @pytest.mark.asyncio
    async def test_the_batch_path_uses_the_validated_snapshots(self, monkeypatch):
        storage = _make_storage()
        payload = {"weight": 1.0, "description": "clean"}
        captured: list = []

        async def fake_bulk(collection, ops, **kwargs):
            # First call materializes endpoints; poison the mapping then, exactly
            # as an aliasing caller would.
            if not captured:
                payload["source_ids.0"] = "attribution rewritten"
                captured.append("endpoints")
            else:
                captured.append(ops)

        monkeypatch.setattr("lightrag.kg.mongo_impl._run_batched_bulk_write", fake_bulk)

        await storage.upsert_edges_batch([("a", "b", payload)])

        edge_ops = captured[-1]
        assert edge_ops, "edge ops were never written"
        update = edge_ops[0][0]._doc
        assert "source_ids.0" not in update["$set"]
        assert update["$set"]["description"] == "clean"
