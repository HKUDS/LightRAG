"""The configuration container's identity row on the real ``MongoKVStorage``,
against an in-memory collection: created through the strict flush and
read-back, verified on the next bind, and a read failure creates nothing.
See *The anchor and the container identity* in
docs/design/ConfigurationStorageContract.md.
"""

from __future__ import annotations

import pytest
from pymongo.errors import PyMongoError

from lightrag import config_anchor as ca
from lightrag import config_store as cs
from lightrag.exceptions import ConfigurationStorageError
from lightrag.kg.mongo_impl import MongoKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline

CONTAINER = "MongoKVStorage (_lightrag_config)"


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


class _Collection:
    def __init__(self, *, read_error=None):
        self.docs: dict[str, dict] = {}
        self.read_error = read_error
        self.writes = 0

    async def find_one(self, query):
        if self.read_error is not None:
            raise self.read_error
        doc = self.docs.get(query["_id"])
        return None if doc is None else dict(doc)

    async def bulk_write(self, operations, ordered=True):
        for op in operations:
            self.writes += 1
            key = op._filter["_id"]
            doc = dict(self.docs.get(key) or op._doc.get("$setOnInsert", {}))
            doc.update(op._doc["$set"])
            self.docs[key] = doc


def _storage(collection) -> MongoKVStorage:
    storage = cs.create_configuration_storage(
        MongoKVStorage, global_config={}, embedding_func=None
    )
    storage._data = collection
    return storage


async def _bind(storage, tmp_path):
    return await cs.bind_configuration_identity(
        storage,
        working_dir=str(tmp_path),
        backend="MongoKVStorage",
        container=CONTAINER,
    )


async def test_the_identity_is_created_read_back_and_then_verified(tmp_path):
    collection = _Collection()
    storage = _storage(collection)
    created = await _bind(storage, tmp_path)
    assert created.action == "created"
    doc = collection.docs[cs.storage_identity_key()]
    assert doc["value"] == {"uuid": created.storage_uuid}
    assert doc["workspace"] == "_lightrag_server"
    assert ca.read_anchor(str(tmp_path)).backend == "MongoKVStorage"

    writes = collection.writes
    verified = await _bind(storage, tmp_path)
    assert verified.action == "verified"
    assert verified.storage_uuid == created.storage_uuid
    assert collection.writes == writes


async def test_a_read_failure_creates_neither_identity_nor_anchor(tmp_path):
    collection = _Collection(read_error=PyMongoError("connection refused"))
    with pytest.raises(ConfigurationStorageError):
        await _bind(_storage(collection), tmp_path)
    assert collection.docs == {}
    assert ca.read_anchor(str(tmp_path)) is None
