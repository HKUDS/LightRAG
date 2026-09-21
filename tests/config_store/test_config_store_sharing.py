"""Two LightRAG instances in one process share the configuration container.

The contract above them: **instances in one process have DIFFERENT
workspaces.** The configuration container is deliberately the opposite --
one fixed reserved workspace for all of them -- so those instances meet
inside it by design, not by accident. That is safe, and this file pins why:
rows are keyed ``<workspace>/<suffix>``, so two tenants write disjoint rows
and each reads back its own.

What is NOT safe is two instances backed by different FILES, which is a
property of ``working_dir``, not of the workspace. That refusal is pinned in
``tests/kg/json_impl/test_json_init_claim.py``; here we pin that it never
fires on the legitimate case.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from lightrag.config_store import create_configuration_storage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import CONFIG_CONTAINER_TAG
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


def _config_storage(working_dir):
    return create_configuration_storage(
        JsonKVStorage,
        global_config={"working_dir": str(working_dir), "embedding_batch_num": 1},
        embedding_func=_embedding(),
    )


def _config_file(working_dir):
    return working_dir / CONFIG_CONTAINER_TAG / "kv_store_config.json"


async def test_two_tenants_in_one_process_read_and_write_their_own_rows(tmp_path):
    """Concurrent use by two workspaces, the way the contract intends it."""
    first = _config_storage(tmp_path)
    second = _config_storage(tmp_path)
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

    # One container, so each also SEES the other's row -- and that is correct:
    # configuration is the server's property, and the key carries the owner.
    assert (await first.get_by_id("tenant-b/embedding_baseline.entities")) is not None

    await first.index_done_callback()
    await asyncio.gather(first.finalize(), second.finalize())

    persisted = json.loads(_config_file(tmp_path).read_text())
    assert sorted(persisted) == [
        "tenant-a/embedding_baseline.entities",
        "tenant-b/embedding_baseline.entities",
    ]


async def test_one_tenant_leaving_does_not_empty_the_container(tmp_path):
    """A hold is per instance; the container outlives any one of them."""
    first = _config_storage(tmp_path)
    second = _config_storage(tmp_path)
    await first.initialize()
    await second.initialize()

    await first.upsert({"tenant-a/embedding_baseline.entities": {"value": {"m": "A"}}})
    await first.index_done_callback()
    await first.finalize()

    still_there = await second.get_by_id("tenant-a/embedding_baseline.entities")
    assert still_there is not None, "the surviving instance lost the container"

    await second.upsert({"tenant-b/embedding_baseline.entities": {"value": {"m": "B"}}})
    await second.index_done_callback()
    await second.finalize()

    persisted = json.loads(_config_file(tmp_path).read_text())
    assert sorted(persisted) == [
        "tenant-a/embedding_baseline.entities",
        "tenant-b/embedding_baseline.entities",
    ]


async def test_a_restart_reads_back_both_tenants(tmp_path):
    """Every instance gone, then a new one: the file is the source of truth."""
    first = _config_storage(tmp_path)
    second = _config_storage(tmp_path)
    await first.initialize()
    await second.initialize()
    await first.upsert({"tenant-a/embedding_baseline.entities": {"value": {"m": "A"}}})
    await second.upsert({"tenant-b/embedding_baseline.entities": {"value": {"m": "B"}}})
    await first.index_done_callback()
    await first.finalize()
    await second.finalize()

    restarted = _config_storage(tmp_path)
    await restarted.initialize()
    try:
        assert (
            await restarted.get_by_id("tenant-a/embedding_baseline.entities")
        ) is not None
        assert (
            await restarted.get_by_id("tenant-b/embedding_baseline.entities")
        ) is not None
    finally:
        await restarted.finalize()
