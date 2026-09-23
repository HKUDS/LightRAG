"""A stored payload that is not a row must be named, not crash from inside.

Every read path in ``JsonKVStorage`` normalises the row it hands back
(``create_time``, ``update_time``, ``_id``), so it calls mapping methods on
whatever the file held. A key mapped to a string -- a hand-edited file, a
truncated one repaired by hand, a foreign writer -- used to escape as
``AttributeError: 'str' object has no attribute 'setdefault'`` raised from
inside this class.

That mattered beyond the ugly traceback: ``config_store.read_config_row_strict``
wraps an unrecognised exception from a point read as "could not read
configuration record", which every caller above reads as the STORE not
serving. So one damaged row in ``kv_server_config.json`` presented itself as a
configuration backend outage, and ``lightrag-clear-storage`` refused to clear
the workspace -- the single situation that tool exists for.
"""

import json

import pytest

from lightrag.exceptions import CorruptStorageRecordError
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


async def _storage_over(tmp_path, payload: dict):
    """A storage whose file already holds ``payload``, written behind its back."""
    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / f"kv_store_{NameSpace.KV_STORE_TEXT_CHUNKS}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    storage = JsonKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="ws",
    )
    await storage.initialize()
    return storage


@pytest.mark.asyncio
async def test_get_by_id_names_the_damaged_record(tmp_path):
    storage = await _storage_over(
        tmp_path, {"chunk-1": "garbage", "chunk-2": {"content": "fine"}}
    )

    with pytest.raises(CorruptStorageRecordError) as excinfo:
        await storage.get_by_id("chunk-1")

    message = str(excinfo.value)
    assert "chunk-1" in message, "the damaged key must be named"
    assert "str" in message and "not a mapping" in message
    # The healthy neighbour is unaffected: this is a per-row verdict.
    assert (await storage.get_by_id("chunk-2"))["content"] == "fine"


@pytest.mark.asyncio
async def test_get_by_ids_names_the_damaged_record(tmp_path):
    storage = await _storage_over(tmp_path, {"chunk-1": ["not", "a", "row"]})

    with pytest.raises(CorruptStorageRecordError):
        await storage.get_by_ids(["chunk-1"])


@pytest.mark.asyncio
async def test_iter_rows_raises_rather_than_yielding_a_damaged_record(tmp_path):
    """The base contract requires this stream to raise on a failure. Yielding
    the string would fail further away, with nothing naming the key."""
    storage = await _storage_over(tmp_path, {"chunk-1": 42})

    with pytest.raises(CorruptStorageRecordError):
        async for _row in storage.iter_rows(page_size=10):
            pass


@pytest.mark.asyncio
async def test_a_missing_key_is_still_a_confirmed_absence(tmp_path):
    """The guard must not turn a miss into a refusal: absent stays absent."""
    storage = await _storage_over(tmp_path, {"chunk-1": {"content": "fine"}})

    assert await storage.get_by_id("nope") is None
    assert await storage.get_by_id_strict("nope") is None
