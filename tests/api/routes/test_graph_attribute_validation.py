"""Attribute validation for the manual entity/relation mutation APIs.

Regression tests for GHSA-c922-pw4m-4wcv. `updated_data` used to be merged into
the stored object wholesale (`{**node_data, **updated_data}`), so a caller with
one API key could write an attribute no graph backend can store. On the default
NetworkX backend that mutation lands in the in-memory `nx.Graph` *before*
serialization, and there is no rollback -- so every later whole-graph flush by
any caller re-hits the same failure and the instance stops persisting entirely.

`Test*WritePathStaysHealthy` are the fix proofs. Each one runs a hostile payload
through a **real** `NetworkXStorage` and then asserts that an ordinary,
unrelated write still succeeds. On the pre-fix code the rejection does not
happen, the ordinary write raises `TypeError: GraphML does not support ...`, and
the test fails behaviourally. Nothing here imports a symbol added by the fix, so
the module still collects against the pre-fix tree -- otherwise the whole file
would error out on import and prove nothing. The shared validator's own unit
tests live in tests/utils/test_graph_attribute_validator.py.

`TestAllowedFieldsAndTypes` pins the rules, including the two a
plausible-looking fix gets wrong (`None` and control characters are *not*
storable scalars) and the one that over-rejects (a multi-line description is
perfectly legal and must keep working).
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from lightrag import utils_graph
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline

NESTED = {"any": ["object"]}


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


class _NoopLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _patch_graph_lock(monkeypatch):
    monkeypatch.setattr(
        utils_graph, "get_storage_keyed_lock", lambda *a, **k: _NoopLock()
    )


class _KVStorage:
    def __init__(self):
        self.records: dict = {}

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, ids):
        for key in ids:
            self.records.pop(key, None)

    async def index_done_callback(self):
        return None


class _VectorStorage(_KVStorage):
    def __init__(self, global_config):
        super().__init__()
        self.global_config = global_config


class _Fixture:
    """A real NetworkXStorage plus in-memory VDB/KV doubles.

    The graph storage is deliberately the real thing: the defect this file
    guards is that a bad attribute reaches `nx.Graph` and breaks *serialization*
    for every subsequent writer, which a fake graph cannot express.
    """

    def __init__(self, tmp_path):
        self.global_config = {
            "working_dir": str(tmp_path),
            "workspace": "",
            "embedding_batch_num": 10,
            "max_total_tokens": 30000,
        }
        self.graph = NetworkXStorage(
            namespace="chunk_entity_relation",
            workspace="",
            global_config=self.global_config,
            embedding_func=None,
        )
        self.entities_vdb = _VectorStorage(self.global_config)
        self.relationships_vdb = _VectorStorage(self.global_config)
        self.entity_chunks = _KVStorage()
        self.relation_chunks = _KVStorage()

    async def start(self):
        await self.graph.initialize()
        return self

    async def create_entity(self, name, **overrides):
        data = {"description": "d", "entity_type": "t", "source_id": "chunk-1"}
        data.update(overrides)
        return await utils_graph.acreate_entity(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            name,
            data,
            entity_chunks_storage=self.entity_chunks,
            relation_chunks_storage=self.relation_chunks,
        )

    async def edit_entity(self, name, updated_data):
        return await utils_graph.aedit_entity(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            name,
            updated_data,
            entity_chunks_storage=self.entity_chunks,
            relation_chunks_storage=self.relation_chunks,
        )

    async def create_relation(self, source, target, **overrides):
        data = {
            "description": "d",
            "keywords": "k",
            "source_id": "chunk-1",
            "weight": 1.0,
        }
        data.update(overrides)
        return await utils_graph.acreate_relation(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            source,
            target,
            data,
            relation_chunks_storage=self.relation_chunks,
        )

    async def edit_relation(self, source, target, updated_data):
        return await utils_graph.aedit_relation(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            source,
            target,
            updated_data,
            relation_chunks_storage=self.relation_chunks,
        )

    def graphml_text(self):
        with open(self.graph._graphml_xml_file, encoding="utf-8") as handle:
            return handle.read()


@pytest.fixture
async def rag(tmp_path):
    return await _Fixture(tmp_path).start()


# ---------------------------------------------------------------------------
# Fix proofs: a rejected payload must leave the write path usable.
# ---------------------------------------------------------------------------


class TestEntityEditWritePathStaysHealthy:
    @pytest.mark.asyncio
    async def test_unknown_key_with_non_scalar_value_is_refused(self, rag):
        await rag.create_entity("VICTIM")

        with pytest.raises(ValueError, match="Unknown entity field 'poc_nested'"):
            await rag.edit_entity("VICTIM", {"poc_nested": NESTED})

        # The whole point: nothing landed in the in-memory graph, so an
        # unrelated write still reaches disk. Pre-fix this raised TypeError.
        assert "poc_nested" not in await rag.graph.get_node("VICTIM")
        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    async def test_null_on_an_allowed_field_is_refused(self, rag):
        """`None` is the bypass an "allowlist + any scalar" fix leaves open.

        GraphML rejects `NoneType` exactly like it rejects `dict`, so allowing
        it reproduces the original outage through an accepted field name.
        """
        await rag.create_entity("VICTIM")

        with pytest.raises(ValueError, match="'file_path' must be a string"):
            await rag.edit_entity("VICTIM", {"file_path": None})

        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    async def test_control_character_on_an_allowed_field_is_refused(self, rag):
        """The second bypass: a `str` XML cannot encode is still not storable."""
        await rag.create_entity("VICTIM")

        with pytest.raises(ValueError, match=r"must not contain the character U\+000B"):
            await rag.edit_entity("VICTIM", {"file_path": "a\x0bb"})

        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    async def test_wrong_scalar_type_on_source_id_is_refused(self, rag):
        """The durable variant: an `int` here is storable but poisonous.

        It serializes fine, reaches disk and survives a restart -- and then
        every reader that splits `source_id` on GRAPH_FIELD_SEP raises
        `AttributeError`, including the ingestion merge. So a scalar check alone
        is not enough; the field's own type has to be enforced.
        """
        await rag.create_entity("VICTIM")

        with pytest.raises(ValueError, match="'source_id' must be a string, got int"):
            await rag.edit_entity("VICTIM", {"source_id": 123})

        assert (await rag.graph.get_node("VICTIM"))["source_id"] == "chunk-1"
        # Still editable afterwards -- pre-fix this raised AttributeError
        # forever, restart included.
        await rag.edit_entity("VICTIM", {"description": "an ordinary update"})
        assert (await rag.graph.get_node("VICTIM"))["description"] == (
            "an ordinary update"
        )


class TestRelationEditWritePathStaysHealthy:
    @pytest.mark.asyncio
    async def test_unknown_key_with_non_scalar_value_is_refused(self, rag):
        await rag.create_entity("REL_A")
        await rag.create_entity("REL_B")
        await rag.create_relation("REL_A", "REL_B")

        with pytest.raises(ValueError, match="Unknown relation field 'poc_nested'"):
            await rag.edit_relation("REL_A", "REL_B", {"poc_nested": NESTED})

        assert "poc_nested" not in await rag.graph.get_edge("REL_A", "REL_B")
        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()


class TestEntityCreateWritePathStaysHealthy:
    @pytest.mark.asyncio
    async def test_non_scalar_on_a_known_field_is_refused(self, rag):
        """The create path was reported as unaffected. It is not.

        Copying six named fields stops an unknown *key*, but a non-scalar value
        on a known one (`entity_type` here) reached `upsert_node` unexamined and
        produced the identical instance-wide outage. Only `description` was
        incidentally safe, because building the VDB content concatenates it.
        """
        await rag.create_entity("CTRL")

        with pytest.raises(ValueError, match="'entity_type' must be a string"):
            await rag.create_entity("ATTACK", entity_type=NESTED)

        assert not await rag.graph.has_node("ATTACK")
        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    async def test_unknown_key_is_still_ignored_not_rejected(self, rag):
        """Create keeps dropping unknown keys.

        It never wrote them (the named-field copy is a structural allowlist), so
        turning them into a 400 would break working payloads for no gain. Only
        the merging paths need reject-unknown.
        """
        await rag.create_entity("CREATE_EXTRA", poc_nested=NESTED)

        node = await rag.graph.get_node("CREATE_EXTRA")
        assert "poc_nested" not in node
        assert "CREATE_EXTRA" in rag.graphml_text()


# ---------------------------------------------------------------------------
# The rules themselves.
# ---------------------------------------------------------------------------


class TestAllowedFieldsAndTypes:
    @pytest.mark.asyncio
    async def test_multiline_description_is_accepted(self, rag):
        """Guard against over-rejection: tab/newline/CR are valid XML."""
        await rag.create_entity("VICTIM")
        text = "first line\nsecond\tline\rthird"

        await rag.edit_entity("VICTIM", {"description": text})

        assert (await rag.graph.get_node("VICTIM"))["description"] == text
        assert "VICTIM" in rag.graphml_text()

    @pytest.mark.asyncio
    async def test_rename_field_is_still_allowed(self, rag):
        """`entity_name` is the rename target and must stay editable."""
        await rag.create_entity("OLD_NAME")

        await rag.edit_entity("OLD_NAME", {"entity_name": "NEW_NAME"})

        assert await rag.graph.has_node("NEW_NAME")
        assert not await rag.graph.has_node("OLD_NAME")

    @pytest.mark.asyncio
    async def test_entity_id_is_not_an_editable_field(self, rag):
        """It was silently ignored before (overwritten by the rename target).

        A 400 naming the allowed fields is the discoverable behaviour; silently
        accepting a rename request that does nothing is not.
        """
        await rag.create_entity("VICTIM")

        with pytest.raises(ValueError, match="Unknown entity field 'entity_id'"):
            await rag.edit_entity("VICTIM", {"entity_id": "SOMETHING_ELSE"})

    @pytest.mark.asyncio
    async def test_error_message_lists_the_allowed_fields(self, rag):
        await rag.create_entity("VICTIM")

        with pytest.raises(ValueError) as excinfo:
            await rag.edit_entity("VICTIM", {"nope": "x"})

        message = str(excinfo.value)
        for field in ("entity_name", "entity_type", "description", "source_id"):
            assert field in message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "weight, expected",
        [(2, 2.0), (2.5, 2.5), ("2.5", 2.5)],
    )
    async def test_relation_weight_is_normalized_to_float(self, rag, weight, expected):
        """A numeric string is accepted, and stored as a number.

        The create path has always run `weight` through `float()`, so accepting
        a numeric string keeps existing callers working; normalizing here means
        the stored attribute is a float on the edit path too, instead of a
        string that only the VDB payload converted.
        """
        await rag.create_entity("REL_A")
        await rag.create_entity("REL_B")
        await rag.create_relation("REL_A", "REL_B")

        await rag.edit_relation("REL_A", "REL_B", {"weight": weight})

        stored = (await rag.graph.get_edge("REL_A", "REL_B"))["weight"]
        assert isinstance(stored, float)
        assert stored == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "weight", ["nan", "NaN", "inf", "Infinity", "1e999", "-1e999"]
    )
    async def test_relation_weight_rejects_strings_that_coerce_to_non_finite(
        self, rag, weight
    ):
        """The value check has to run on the *coerced* number, not the input.

        A numeric string is a perfectly storable `str`, so validating only what
        the caller sent lets the `float()` conversion manufacture the non-scalar
        the contract forbids. That would be accepted here and rejected later in
        storage -- PGTableGraphStorage's jsonb column refuses the bare `NaN`
        `json.dumps` emits -- turning an intended 400 into a 500.
        """
        await rag.create_entity("REL_A")
        await rag.create_entity("REL_B")
        await rag.create_relation("REL_A", "REL_B")

        with pytest.raises(ValueError, match="'weight' must be a finite number"):
            await rag.edit_relation("REL_A", "REL_B", {"weight": weight})

        assert (await rag.graph.get_edge("REL_A", "REL_B"))["weight"] == 1.0
        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("weight", [10**400, 2**63, -(10**400)])
    async def test_relation_weight_rejects_oversized_integers(self, rag, weight):
        """A huge JSON integer must be a 400, not an uncaught OverflowError.

        `json.loads` yields a Python int of unbounded size, and `float()` raises
        `OverflowError` past ~1e308 -- which is neither `TypeError` nor
        `ValueError`, so it escaped the coercion handler and reached the route as
        a 500. The int64 bound refuses it before the conversion is attempted.
        """
        await rag.create_entity("REL_A")
        await rag.create_entity("REL_B")
        await rag.create_relation("REL_A", "REL_B")

        with pytest.raises(ValueError, match="must be a 64-bit integer"):
            await rag.edit_relation("REL_A", "REL_B", {"weight": weight})

        assert (await rag.graph.get_edge("REL_A", "REL_B"))["weight"] == 1.0
        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("weight", [True, "abc", NESTED, None])
    async def test_relation_weight_rejects_non_numbers(self, rag, weight):
        await rag.create_entity("REL_A")
        await rag.create_entity("REL_B")
        await rag.create_relation("REL_A", "REL_B")

        with pytest.raises(ValueError, match="'weight'"):
            await rag.edit_relation("REL_A", "REL_B", {"weight": weight})

        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    async def test_merge_target_entity_data_is_validated(self, rag):
        """The third copy of the same wholesale merge, in `_merge_entities_impl`.

        Not reachable from the HTTP route (it does not pass the argument), but
        part of the public Python API, so the report's two-site patch would have
        left it open.
        """
        await rag.create_entity("SOURCE")
        await rag.create_entity("TARGET")

        with pytest.raises(ValueError, match="Unknown entity field 'poc_nested'"):
            await utils_graph.amerge_entities(
                rag.graph,
                rag.entities_vdb,
                rag.relationships_vdb,
                ["SOURCE"],
                "TARGET",
                target_entity_data={"poc_nested": NESTED},
                entity_chunks_storage=rag.entity_chunks,
                relation_chunks_storage=rag.relation_chunks,
            )

        assert await rag.graph.has_node("SOURCE")
        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()
