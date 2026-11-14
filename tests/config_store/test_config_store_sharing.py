"""Several LightRAG instances in one process and the JSON configuration.

The contract above them: **instances in one process have DIFFERENT
workspaces.** The JSON configuration keeps one snapshot per workspace
(``WORKING_DIR/<workspace>/kv_workspace_config.json``), so those instances
open DIFFERENT files -- and the shared-namespace bookkeeping is keyed by the
physical file, so they must not alias into one in-memory namespace either.
Each writes and reads back only its own rows.

Two holders of the SAME snapshot (one workspace) share one in-memory copy;
a hold is per instance, so one leaving never empties it for the other.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from lightrag.config_store import create_configuration_storage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


def _embedding():
    async def _func(texts, **kwargs):
        return np.zeros((len(texts), 4), dtype=np.float32)

    return EmbeddingFunc(embedding_dim=4, func=_func, model_name="bge-m3")


def _config_storage(working_dir, workspace=""):
    return create_configuration_storage(
        JsonKVStorage,
        global_config={
            "working_dir": str(working_dir),
            "workspace": workspace,
            "embedding_batch_num": 1,
        },
        embedding_func=_embedding(),
    )


def _config_file(working_dir, workspace=""):
    base = working_dir / workspace if workspace else working_dir
    return base / "kv_workspace_config.json"


async def test_two_tenants_in_one_process_read_and_write_their_own_rows(tmp_path):
    """Concurrent use by two workspaces, the way the contract intends it:
    two snapshots, two in-memory namespaces, no row crosses over."""
    first = _config_storage(tmp_path, "tenant-a")
    second = _config_storage(tmp_path, "tenant-b")
    assert first._file_name != second._file_name
    await asyncio.gather(first.initialize(), second.initialize())

    await asyncio.gather(
        first.upsert({"tenant-a/embedding_baseline.entities": {"value": {"m": "A"}}}),
        second.upsert({"tenant-b/embedding_baseline.entities": {"value": {"m": "B"}}}),
    )

    mine, theirs = await asyncio.gather(
        first.get_by_id("tenant-a/embedding_baseline.entities"),
        second.get_by_id("tenant-b/embedding_baseline.entities"),
    )
    assert mine["value"] == {"m": "A"}
    assert theirs["value"] == {"m": "B"}

    # Separate snapshots: neither sees the other's row, in memory or on disk.
    assert (await first.get_by_id("tenant-b/embedding_baseline.entities")) is None
    assert (await second.get_by_id("tenant-a/embedding_baseline.entities")) is None

    await asyncio.gather(first.index_done_callback(), second.index_done_callback())
    await asyncio.gather(first.finalize(), second.finalize())

    assert sorted(json.loads(_config_file(tmp_path, "tenant-a").read_text())) == [
        "tenant-a/embedding_baseline.entities"
    ]
    assert sorted(json.loads(_config_file(tmp_path, "tenant-b").read_text())) == [
        "tenant-b/embedding_baseline.entities"
    ]


async def test_one_holder_leaving_does_not_empty_the_snapshot(tmp_path):
    """A hold is per instance; the snapshot outlives any one of its holders."""
    first = _config_storage(tmp_path, "tenant")
    second = _config_storage(tmp_path, "tenant")
    await first.initialize()
    await second.initialize()

    await first.upsert({"tenant-a/embedding_baseline.entities": {"value": {"m": "A"}}})
    await first.index_done_callback()
    await first.finalize()

    still_there = await second.get_by_id("tenant-a/embedding_baseline.entities")
    assert still_there is not None, "the surviving instance lost the snapshot"

    await second.upsert({"tenant-b/embedding_baseline.entities": {"value": {"m": "B"}}})
    await second.index_done_callback()
    await second.finalize()

    persisted = json.loads(_config_file(tmp_path, "tenant").read_text())
    assert sorted(persisted) == [
        "tenant-a/embedding_baseline.entities",
        "tenant-b/embedding_baseline.entities",
    ]


async def test_a_restart_reads_back_both_tenants(tmp_path):
    """Every instance gone, then new ones: each file is the source of truth
    for its own workspace, and only for it."""
    first = _config_storage(tmp_path, "tenant-a")
    second = _config_storage(tmp_path, "tenant-b")
    await first.initialize()
    await second.initialize()
    await first.upsert({"tenant-a/embedding_baseline.entities": {"value": {"m": "A"}}})
    await second.upsert({"tenant-b/embedding_baseline.entities": {"value": {"m": "B"}}})
    await first.index_done_callback()
    await second.index_done_callback()
    await first.finalize()
    await second.finalize()

    restarted_a = _config_storage(tmp_path, "tenant-a")
    restarted_b = _config_storage(tmp_path, "tenant-b")
    await restarted_a.initialize()
    await restarted_b.initialize()
    try:
        assert (
            await restarted_a.get_by_id("tenant-a/embedding_baseline.entities")
        ) is not None
        assert (
            await restarted_b.get_by_id("tenant-b/embedding_baseline.entities")
        ) is not None
        assert (
            await restarted_a.get_by_id("tenant-b/embedding_baseline.entities")
        ) is None
    finally:
        await restarted_a.finalize()
        await restarted_b.finalize()
