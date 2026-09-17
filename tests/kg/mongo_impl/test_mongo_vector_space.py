"""Embedding-space provenance on the MongoDB vector collection.

The collection name on this backend carries no model information, so an
operator who swaps to a *different model of the same dimension* keeps querying
the previous model's vectors out of the very same collection. The fix is a
marker in the collection's JSON Schema validator ``description`` — metadata, so
no ``$vectorSearch`` and no ``_id`` / ``src_id`` / ``tgt_id`` find can reach it,
and no future full-collection scan owes it an exclusion.

The half that needed the most care is ``drop()``. ``delete_many({})`` removes
documents, not the Atlas search index and not the validator, so a drop that
went straight to ``create_vector_index_if_not_exists()`` re-read the surviving
index definition and raised the same mismatch again — and ``except
PyMongoError`` did not catch that raise. The condition was unrecoverable even
once ``lightrag-rebuild-vdb`` managed to call ``drop()``. The tests below pin
convergence: drop, then attach, and the second attach must not refuse.

Rules pinned here (see ``docs/design/VectorSpaceProvenance.md``):

- a collection created by this backend records the model and dimension;
- attach refuses a foreign model or dimension, with a TYPED exception;
- absent evidence never refuses, and attach is not a backfill;
- a refused instance stays drop-capable, and drop converges;
- a ``collMod`` that is denied degrades to unmarked, never to a refusal.

The Mongo fake is stateful: the collection, its validator and its search index
outlive a storage instance, which is the only way to express "a previous
deployment left this behind".
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "pymongo",
    reason="pymongo is required for Mongo storage tests",
)

from pymongo.errors import OperationFailure  # type: ignore  # noqa: E402

from lightrag.exceptions import VectorSpaceMismatchError  # noqa: E402
from lightrag.kg.mongo_impl import (  # noqa: E402
    ClientManager,
    MongoVectorDBStorage,
    _read_validator_vector_space,
)

pytestmark = pytest.mark.offline

COLLECTION = "test_entities"
INDEX = "vector_knn_index_test_entities"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _mock_lock():
    yield


def _mock_lock_factory(*args, **kwargs):
    return _mock_lock()


@pytest.fixture(autouse=True)
def patch_locks(monkeypatch):
    monkeypatch.delenv("MONGODB_WORKSPACE", raising=False)
    cache: dict[tuple, asyncio.Lock] = {}

    def namespace_lock(namespace, workspace=None, enable_logging=False):
        return cache.setdefault((namespace, workspace or ""), asyncio.Lock())

    with (
        patch(
            "lightrag.kg.mongo_impl.get_data_init_lock",
            side_effect=_mock_lock_factory,
        ),
        patch("lightrag.kg.mongo_impl.get_namespace_lock", side_effect=namespace_lock),
    ):
        yield


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    async def to_list(self, length=None):
        return list(self._docs)


class FakeMongo:
    """One stateful MongoDB: collections, validators and search indexes."""

    def __init__(self):
        self.collections: dict[str, dict] = {}
        self.search_indexes: dict[str, dict] = {}
        self.collmod_error: Exception | None = None
        self.collmod_calls: list[dict] = []
        self.dropped_indexes: list[str] = []

    # -- seeding ------------------------------------------------------------

    def seed_collection(self, name=COLLECTION, *, validator=None, index_dim=None):
        """Leave behind what a previous deployment would have left."""
        self.collections[name] = {"validator": validator}
        if index_dim is not None:
            self.search_indexes[INDEX] = {
                "name": INDEX,
                "status": "READY",
                "queryable": True,
                "latestDefinition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "vector",
                            "numDimensions": index_dim,
                            "similarity": "cosine",
                        }
                    ]
                },
            }

    def validator_of(self, name=COLLECTION):
        return self.collections[name]["validator"]

    def index_dim(self):
        index = self.search_indexes.get(INDEX)
        if index is None:
            return None
        return index["latestDefinition"]["fields"][0]["numDimensions"]

    # -- database -----------------------------------------------------------

    async def list_collection_names(self):
        return list(self.collections)

    async def create_collection(self, name, **kwargs):
        self.collections[name] = {"validator": kwargs.get("validator")}
        return self._collection()

    def get_collection(self, name):
        return self._collection()

    async def list_collections(self, filter=None):
        name = (filter or {}).get("name")
        rows = [
            {
                "name": n,
                "options": {"validator": c["validator"]} if c["validator"] else {},
            }
            for n, c in self.collections.items()
            if name is None or n == name
        ]
        return _AsyncCursor(rows)

    async def command(self, command, name=None, **kwargs):
        assert command == "collMod"
        self.collmod_calls.append(kwargs)
        if self.collmod_error is not None:
            raise self.collmod_error
        self.collections[name]["validator"] = kwargs.get("validator")

    # -- collection ---------------------------------------------------------

    async def list_search_indexes(self):
        return _AsyncCursor(list(self.search_indexes.values()))

    async def create_search_index(self, model):
        document = model.document
        self.search_indexes[document["name"]] = {
            "name": document["name"],
            "status": "READY",
            "queryable": True,
            "latestDefinition": document["definition"],
        }

    async def drop_search_index(self, name):
        self.dropped_indexes.append(name)
        self.search_indexes.pop(name, None)

    def db(self):
        db = MagicMock()
        db.list_collection_names = AsyncMock(side_effect=self.list_collection_names)
        db.create_collection = AsyncMock(side_effect=self.create_collection)
        db.get_collection = MagicMock(side_effect=self.get_collection)
        db.list_collections = AsyncMock(side_effect=self.list_collections)
        db.command = AsyncMock(side_effect=self.command)
        return db

    def _collection(self):
        collection = MagicMock()
        collection.list_search_indexes = AsyncMock(side_effect=self.list_search_indexes)
        collection.create_search_index = AsyncMock(side_effect=self.create_search_index)
        collection.drop_search_index = AsyncMock(side_effect=self.drop_search_index)
        collection.delete_many = AsyncMock(return_value=MagicMock(deleted_count=3))
        collection.bulk_write = AsyncMock()
        collection.find = MagicMock(return_value=_AsyncCursor([]))
        collection.find_one = AsyncMock(return_value=None)
        return collection


class _Embed:
    max_token_size = 512

    def __init__(self, model_name="model-a", embedding_dim=8):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    async def __call__(self, texts, **kwargs):
        return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)


@pytest.fixture
def mongo():
    return FakeMongo()


def _storage(embed, workspace="test", namespace="entities"):
    return MongoVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=embed,
        meta_fields={"content"},
    )


async def _initialize(storage, mongo):
    with patch.object(ClientManager, "get_client", return_value=mongo.db()):
        await storage.initialize()
    return storage


def _marker(mongo):
    return _read_validator_vector_space(mongo.validator_of())


# ---------------------------------------------------------------------------
# The marker is recorded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_created_collection_records_its_embedding_space(mongo):
    await _initialize(_storage(_Embed("bge-m3", 8)), mongo)
    assert _marker(mongo) == ("bge-m3", 8)


@pytest.mark.asyncio
async def test_an_unknown_model_records_no_model(mongo):
    # A recorded None would be indistinguishable from "written before the
    # marker existed", making the never-refuse rule permanent for it.
    await _initialize(_storage(_Embed(None, 8)), mongo)
    assert _marker(mongo) == (None, 8)


@pytest.mark.asyncio
async def test_the_marker_is_not_a_document(mongo):
    """It must never become a row: a row in the ANN index can be recalled."""
    storage = await _initialize(_storage(_Embed("bge-m3", 8)), mongo)
    storage._data.bulk_write.assert_not_called()
    assert "description" in mongo.validator_of()["$jsonSchema"]


# ---------------------------------------------------------------------------
# Attach
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attaching_refuses_a_collection_built_by_another_model(mongo):
    """The defect: same dimension, different model, same collection."""
    await _initialize(_storage(_Embed("bge-m3", 8)), mongo)

    with pytest.raises(VectorSpaceMismatchError) as excinfo:
        await _initialize(_storage(_Embed("e5-large", 8)), mongo)

    message = str(excinfo.value)
    assert "'bge-m3' -> 'e5-large'" in message
    assert "lightrag-rebuild-vdb" in message


@pytest.mark.asyncio
async def test_attaching_accepts_the_same_model(mongo):
    await _initialize(_storage(_Embed("bge-m3", 8)), mongo)
    await _initialize(_storage(_Embed("bge-m3", 8)), mongo)
    assert _marker(mongo) == ("bge-m3", 8)


@pytest.mark.asyncio
async def test_attaching_refuses_a_foreign_dimension(mongo):
    """The index definition is the physical truth, and it outranks the marker."""
    mongo.seed_collection(validator=None, index_dim=16)

    with pytest.raises(VectorSpaceMismatchError, match="16 -> 8"):
        await _initialize(_storage(_Embed("bge-m3", 8)), mongo)


@pytest.mark.asyncio
async def test_a_collection_predating_the_marker_is_still_servable(mongo):
    """Absent evidence never refuses, or every pre-upgrade collection is."""
    mongo.seed_collection(validator=None, index_dim=8)
    await _initialize(_storage(_Embed("bge-m3", 8)), mongo)


@pytest.mark.asyncio
async def test_a_foreign_validator_reads_as_absent_rather_than_refusing(mongo):
    """Someone else's validator is not our marker, and not a mismatch."""
    mongo.seed_collection(
        validator={"$jsonSchema": {"bsonType": "object", "description": "ops notes"}},
        index_dim=8,
    )
    await _initialize(_storage(_Embed("bge-m3", 8)), mongo)


@pytest.mark.asyncio
async def test_attaching_is_not_a_backfill(mongo):
    """An unmarked collection is served, NOT stamped with this model.

    Blind adoption would record the new model's name over the old model's
    vectors for an operator who upgrades and swaps models in one step — a lie
    recorded permanently, after which the gate can never fire. Backfill needs
    evidence, gathered one layer up.
    """
    mongo.seed_collection(validator=None, index_dim=8)
    await _initialize(_storage(_Embed("bge-m3", 8)), mongo)

    assert mongo.validator_of() is None
    assert mongo.collmod_calls == []


# ---------------------------------------------------------------------------
# A refusal must stay recoverable — the part that was unrecoverable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refused_instance_can_still_drop(mongo):
    """initialize() used to take the flush lock AFTER the gate."""
    await _initialize(_storage(_Embed("bge-m3", 8)), mongo)
    storage = _storage(_Embed("e5-large", 8))

    with patch.object(ClientManager, "get_client", return_value=mongo.db()):
        with pytest.raises(VectorSpaceMismatchError):
            await storage.initialize()

        assert storage._flush_lock is not None
        assert storage._data is not None

        assert (await storage.drop())["status"] == "success"


@pytest.mark.asyncio
async def test_drop_converges_after_a_same_dimension_model_change(mongo):
    """drop() then attach; the second attach must not refuse."""
    await _initialize(_storage(_Embed("bge-m3", 8)), mongo)
    storage = _storage(_Embed("e5-large", 8))

    with patch.object(ClientManager, "get_client", return_value=mongo.db()):
        with pytest.raises(VectorSpaceMismatchError):
            await storage.initialize()

        assert (await storage.drop())["status"] == "success"
        assert _marker(mongo) == ("e5-large", 8)
        # Same dimension: the index is perfectly good, and rebuilding it would
        # cost an Atlas index build for nothing.
        assert mongo.dropped_indexes == []

        await storage.initialize()

    await _initialize(_storage(_Embed("e5-large", 8)), mongo)


@pytest.mark.asyncio
async def test_drop_rebuilds_the_search_index_on_a_dimension_change(mongo):
    """The case the old drop() could not recover from.

    delete_many({}) leaves the search index behind, so the recreate step
    re-read a definition built for the previous dimension and raised the very
    mismatch the drop was meant to clear — and PyMongoError did not catch it.
    """
    mongo.seed_collection(validator=None, index_dim=16)
    storage = _storage(_Embed("bge-m3", 8))

    with patch.object(ClientManager, "get_client", return_value=mongo.db()):
        with pytest.raises(VectorSpaceMismatchError):
            await storage.initialize()

        assert (await storage.drop())["status"] == "success"

        assert mongo.dropped_indexes == [INDEX]
        assert mongo.index_dim() == 8
        assert _marker(mongo) == ("bge-m3", 8)

        # Converges: the attach that used to raise forever now succeeds.
        await storage.initialize()


@pytest.mark.asyncio
async def test_drop_without_a_space_change_leaves_the_index_alone(mongo):
    """The ordinary /documents/clear path must not pay an index rebuild."""
    storage = await _initialize(_storage(_Embed("bge-m3", 8)), mongo)

    with patch.object(ClientManager, "get_client", return_value=mongo.db()):
        assert (await storage.drop())["status"] == "success"

    assert mongo.dropped_indexes == []
    assert mongo.index_dim() == 8


# ---------------------------------------------------------------------------
# A denied marker write degrades, it does not refuse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_denied_collmod_cannot_report_a_recovery_that_did_not_happen(mongo):
    """A denied write leaves the PREVIOUS marker, not no marker.

    The dangerous case: the collection is already marked for model A, the
    operator switches to model B, and the role cannot write the validator.
    drop() deletes every vector, the marker still says A, and the next
    initialize() -- which clear_vector_space_refusal() runs immediately -- is
    refused all over again. Reporting success there would tell the tool to
    rebuild into a collection it is about to be refused from, after the
    destructive work is already done.
    """
    await _initialize(_storage(_Embed("bge-m3", 8)), mongo)
    storage = _storage(_Embed("e5-large", 8))

    with patch.object(ClientManager, "get_client", return_value=mongo.db()):
        with pytest.raises(VectorSpaceMismatchError):
            await storage.initialize()

        mongo.collmod_error = OperationFailure("not authorized to execute collMod")
        result = await storage.drop()

        assert result["status"] == "error"
        assert "marker" in result["message"] or "collMod" in result["message"]
        # And the condition it reports is real: the attach still refuses.
        with pytest.raises(VectorSpaceMismatchError):
            await storage.initialize()


@pytest.mark.asyncio
async def test_drop_preserves_an_operator_defined_validator(mongo):
    """An ordinary /documents/clear must not silently disable validation.

    The marker takes over the schema's ``description`` -- that is the field it
    lives in -- but everything else an operator put on the collection is theirs
    and has to survive. Replacing the validator wholesale would drop their
    rules from every future write, the same hazard OpenSearch's ``put_mapping``
    has with ``_meta``.
    """
    operator_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["content"],
            "properties": {"content": {"bsonType": "string"}},
        }
    }
    mongo.seed_collection(validator=operator_schema, index_dim=8)
    storage = await _initialize(_storage(_Embed("bge-m3", 8)), mongo)

    with patch.object(ClientManager, "get_client", return_value=mongo.db()):
        assert (await storage.drop())["status"] == "success"

    schema = mongo.validator_of()["$jsonSchema"]
    assert schema["required"] == ["content"]
    assert schema["properties"] == {"content": {"bsonType": "string"}}
    assert _marker(mongo) == ("bge-m3", 8)


@pytest.mark.asyncio
async def test_a_denied_collmod_on_an_unmarked_collection_still_serves(mongo):
    """A restricted Atlas role must not turn a safeguard into an outage.

    Unmarked is where every collection was before this feature existed: worse
    than marked, far better than a deployment that will not start.
    """
    mongo.seed_collection(validator=None, index_dim=8)
    mongo.collmod_error = OperationFailure("not authorized on admin to execute collMod")
    storage = await _initialize(_storage(_Embed("bge-m3", 8)), mongo)

    with patch.object(ClientManager, "get_client", return_value=mongo.db()):
        result = await storage.drop()

    assert result["status"] == "success"
    assert mongo.validator_of() is None
