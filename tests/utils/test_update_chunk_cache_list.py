"""``update_chunk_cache_list`` must report whether the reference is durable (#3833).

The return value gates a cache write: ``use_llm_func_with_cache`` attaches a
key to its chunk BEFORE writing the row, and skips the write when the attach
could not be recorded, so a row can never outlive the only reference that
reaches it. The helper therefore has to distinguish "recorded" from "could not
record" while still never raising -- it is called on the ingest hot path, where
an exception would fail a document over bookkeeping.
"""

import pytest

from lightrag.utils import update_chunk_cache_list

pytestmark = pytest.mark.offline


class _FakeKV:
    def __init__(self, rows: dict | None = None):
        self.data = dict(rows or {})
        self.upserts = 0
        self.fail_on_read = False
        self.fail_on_upsert = False

    async def get_by_id(self, key):
        if self.fail_on_read:
            raise RuntimeError("read is down")
        return self.data.get(key)

    async def upsert(self, rows: dict):
        if self.fail_on_upsert:
            raise RuntimeError("write is down")
        self.data.update(rows)
        self.upserts += 1


@pytest.mark.asyncio
async def test_no_keys_to_record_is_durable():
    storage = _FakeKV({"chunk-1": {"content": "c"}})

    assert await update_chunk_cache_list("chunk-1", storage, []) is True
    assert storage.upserts == 0


@pytest.mark.asyncio
async def test_recording_a_new_key_reports_success():
    storage = _FakeKV({"chunk-1": {"content": "c"}})

    assert await update_chunk_cache_list("chunk-1", storage, ["k1"]) is True
    assert storage.data["chunk-1"]["llm_cache_list"] == ["k1"]
    assert storage.upserts == 1


@pytest.mark.asyncio
async def test_an_already_recorded_key_is_durable_without_a_write():
    """The steady state of the re-ingest self-heal must not cost a write."""
    storage = _FakeKV({"chunk-1": {"content": "c", "llm_cache_list": ["k1"]}})

    assert await update_chunk_cache_list("chunk-1", storage, ["k1"]) is True
    assert storage.upserts == 0
    assert storage.data["chunk-1"]["llm_cache_list"] == ["k1"]


@pytest.mark.asyncio
async def test_a_missing_chunk_row_reports_failure():
    """Nothing can carry the reference, so the caller must not write the row."""
    storage = _FakeKV()

    assert await update_chunk_cache_list("chunk-1", storage, ["k1"]) is False
    assert storage.upserts == 0


@pytest.mark.asyncio
async def test_a_failing_read_reports_failure_without_raising():
    storage = _FakeKV({"chunk-1": {"content": "c"}})
    storage.fail_on_read = True

    assert await update_chunk_cache_list("chunk-1", storage, ["k1"]) is False


@pytest.mark.asyncio
async def test_a_failing_write_reports_failure_without_raising():
    storage = _FakeKV({"chunk-1": {"content": "c"}})
    storage.fail_on_upsert = True

    assert await update_chunk_cache_list("chunk-1", storage, ["k1"]) is False
