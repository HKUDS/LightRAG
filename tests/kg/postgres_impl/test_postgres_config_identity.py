"""The configuration container's identity row on the real ``PGKVStorage``,
against an in-memory ``LIGHTRAG_CONFIG`` table: created through the strict
flush and read-back, verified on the next bind, and a read failure creates
nothing. See *The anchor and the container identity* in
docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag import config_anchor as ca
from lightrag import config_store as cs
from lightrag.exceptions import ConfigurationStorageError
from lightrag.kg.postgres_impl import SQL_TEMPLATES, PGKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import CONFIG_CONTAINER_TAG

pytestmark = pytest.mark.offline

CONTAINER = "PGKVStorage (_lightrag_config)"


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


class _Table:
    def __init__(self, *, read_error=None):
        self.rows: dict[tuple[str, str], str] = {}
        self.read_error = read_error
        self.writes = 0

    async def query(self, sql, params=None, multirows=False, **kwargs):
        if self.read_error is not None:
            raise self.read_error
        assert sql == SQL_TEMPLATES["get_by_id_config"]
        workspace, key = params
        value = self.rows.get((workspace, key))
        if value is None:
            return None
        return {"id": key, "value": value, "create_time": 1, "update_time": 1}

    async def run_with_retry(self, operation, **kwargs):
        conn = AsyncMock()
        tx = AsyncMock()
        tx.__aenter__.return_value = tx
        tx.__aexit__.return_value = False
        conn.transaction = MagicMock(return_value=tx)
        await operation(conn)
        for call in conn.executemany.call_args_list:
            assert call.args[0] == SQL_TEMPLATES["upsert_config"]
            for workspace, key, payload in call.args[1]:
                self.writes += 1
                self.rows[(workspace, key)] = payload


def _storage(table) -> PGKVStorage:
    storage = cs.create_configuration_storage(
        PGKVStorage, global_config={}, embedding_func=None
    )
    db = MagicMock()
    db.query = AsyncMock(side_effect=table.query)
    db._run_with_retry = AsyncMock(side_effect=table.run_with_retry)
    db.workspace = None
    storage.db = db
    return storage


async def _bind(storage, tmp_path):
    return await cs.bind_configuration_identity(
        storage, working_dir=str(tmp_path), backend="PGKVStorage", container=CONTAINER
    )


async def test_the_identity_is_created_read_back_and_then_verified(tmp_path):
    table = _Table()
    storage = _storage(table)
    created = await _bind(storage, tmp_path)
    assert created.action == "created"
    payload = json.loads(table.rows[(CONFIG_CONTAINER_TAG, cs.storage_identity_key())])
    assert payload["value"] == {"uuid": created.storage_uuid}
    assert ca.read_anchor(str(tmp_path)).backend == "PGKVStorage"

    verified = await _bind(storage, tmp_path)
    assert verified.action == "verified"
    assert verified.storage_uuid == created.storage_uuid
    assert table.writes == 1


async def test_a_read_failure_creates_neither_identity_nor_anchor(tmp_path):
    table = _Table(read_error=ConnectionError("server closed the connection"))
    with pytest.raises(ConfigurationStorageError):
        await _bind(_storage(table), tmp_path)
    assert table.rows == {}
    assert ca.read_anchor(str(tmp_path)) is None
