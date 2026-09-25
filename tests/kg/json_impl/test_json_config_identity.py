"""The configuration container's identity row on the real ``JsonKVStorage``:
created, flushed to ``kv_server_config.json`` and read back, verified by a
fresh process-tree view of the file, and a damaged row refuses rather than
reading as absent. See *The anchor and the container identity* in
docs/design/ConfigurationStorageContract.md.
"""

from __future__ import annotations

import json

import pytest

from lightrag import config_anchor as ca
from lightrag import config_store as cs
from lightrag.exceptions import ConfigurationStorageError
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import CONFIG_CONTAINER_TAG, CONFIG_JSON_FILE_NAME

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


async def _open(tmp_path) -> JsonKVStorage:
    storage = cs.create_configuration_storage(
        JsonKVStorage,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


async def _bind(storage, tmp_path):
    return await cs.bind_configuration_identity(
        storage,
        working_dir=str(tmp_path),
        backend="JsonKVStorage",
        container=f"JsonKVStorage at {tmp_path / CONFIG_CONTAINER_TAG}",
    )


def _file(tmp_path):
    return tmp_path / CONFIG_CONTAINER_TAG / CONFIG_JSON_FILE_NAME


async def test_the_identity_is_durable_and_verified_after_a_restart(tmp_path):
    storage = await _open(tmp_path)
    created = await _bind(storage, tmp_path)
    await storage.finalize()
    on_disk = json.loads(_file(tmp_path).read_text())
    assert on_disk[cs.storage_identity_key()]["value"] == {"uuid": created.storage_uuid}
    # The anchor sits beside the JSON file when config_dir is the default.
    assert ca.read_anchor(str(tmp_path)).storage_uuid == created.storage_uuid

    finalize_share_data()
    initialize_share_data(workers=1)
    storage = await _open(tmp_path)
    verified = await _bind(storage, tmp_path)
    await storage.finalize()
    assert verified.action == "verified"
    assert verified.storage_uuid == created.storage_uuid


async def test_a_damaged_identity_row_refuses_and_is_never_regenerated(tmp_path):
    path = _file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({cs.storage_identity_key(): "not a row"}))
    storage = await _open(tmp_path)
    with pytest.raises(ConfigurationStorageError):
        await _bind(storage, tmp_path)
    await storage.finalize()
    assert json.loads(path.read_text()) == {cs.storage_identity_key(): "not a row"}
    assert ca.read_anchor(str(tmp_path)) is None
