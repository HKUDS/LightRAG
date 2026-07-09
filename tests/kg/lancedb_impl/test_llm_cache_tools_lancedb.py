"""Offline tests for LanceDB support in the LLM cache maintenance tools.

Unlike the OpenSearch tool tests (which fake the client), these run against a
real embedded LanceDB in a tmp_path — LanceDB is single-process and cheap to
spin up, so the tool branches are exercised end to end through the real
``get_all_keys`` / ``get_by_ids`` / ``delete`` methods.
"""

import pytest

pytest.importorskip("lancedb", reason="lancedb is required for LanceDB tool tests")

from lightrag.kg.lancedb_impl import LanceDBKVStorage  # noqa: E402
from lightrag.namespace import NameSpace  # noqa: E402
from lightrag.tools.clean_llm_query_cache import CleanupStats, CleanupTool  # noqa: E402
from lightrag.tools.migrate_llm_cache import MigrationTool  # noqa: E402
from lightrag.tools.rebuild_vdb import enumerate_kv_keys  # noqa: E402

pytestmark = pytest.mark.offline


async def _make_llm_cache_storage(global_config, embedding_func, workspace="ws"):
    storage = LanceDBKVStorage(
        namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding_func,
    )
    await storage.initialize()
    return storage


class TestRebuildVdbLanceDB:
    async def test_enumerate_kv_keys_lancedb(self, global_config, embedding_func):
        storage = LanceDBKVStorage(
            namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
            workspace="ws",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        await storage.initialize()
        try:
            await storage.upsert(
                {
                    "chunk-1": {"content": "a"},
                    "doc-0001-chunk-0": {"content": "b"},
                }
            )
            keys = await enumerate_kv_keys(storage)
            assert set(keys) == {"chunk-1", "doc-0001-chunk-0"}
        finally:
            await storage.finalize()


class TestCleanupToolLanceDB:
    def test_get_storage_class_lancedb(self):
        assert (
            CleanupTool().get_storage_class("LanceDBKVStorage").__name__
            == "LanceDBKVStorage"
        )

    def test_check_config_ini_for_storage_lancedb(self):
        # Embedded backend needs no config.ini section to be usable.
        assert CleanupTool().check_config_ini_for_storage("LanceDBKVStorage")

    async def test_count_query_caches_lancedb(self, global_config, embedding_func):
        storage = await _make_llm_cache_storage(global_config, embedding_func)
        try:
            await storage.upsert(
                {
                    "mix:query:1": {"content": "a"},
                    "mix:keywords:1": {"content": "b"},
                    "hybrid:query:1": {"content": "c"},
                    "local:keywords:1": {"content": "d"},
                    "default:extract:1": {"content": "ignored"},
                }
            )
            counts = await CleanupTool().count_query_caches(storage, "LanceDBKVStorage")
            assert counts["mix"] == {"query": 1, "keywords": 1}
            assert counts["hybrid"] == {"query": 1, "keywords": 0}
            assert counts["local"] == {"query": 0, "keywords": 1}
            assert counts["global"] == {"query": 0, "keywords": 0}
        finally:
            await storage.finalize()

    @pytest.mark.parametrize(
        ("cleanup_type", "expected_remaining"),
        [
            ("all", {"default:extract:1"}),
            ("query", {"mix:keywords:1", "default:extract:1"}),
            ("keywords", {"mix:query:1", "default:extract:1"}),
        ],
    )
    async def test_delete_query_caches_lancedb(
        self, global_config, embedding_func, cleanup_type, expected_remaining
    ):
        storage = await _make_llm_cache_storage(global_config, embedding_func)
        try:
            await storage.upsert(
                {
                    "mix:query:1": {"content": "a"},
                    "mix:keywords:1": {"content": "b"},
                    "default:extract:1": {"content": "keep"},
                }
            )
            tool = CleanupTool()
            tool.batch_size = 1  # force multi-batch deletion path
            stats = CleanupStats()
            await tool.delete_query_caches(
                storage, "LanceDBKVStorage", cleanup_type, stats
            )
            assert set(await storage.get_all_keys()) == expected_remaining
            deleted = 3 - len(expected_remaining)
            assert stats.successfully_deleted == deleted
        finally:
            await storage.finalize()


class TestMigrationToolLanceDB:
    def test_get_storage_class_lancedb(self):
        assert (
            MigrationTool().get_storage_class("LanceDBKVStorage").__name__
            == "LanceDBKVStorage"
        )

    async def test_count_and_stream_default_caches_lancedb(
        self, global_config, embedding_func
    ):
        storage = await _make_llm_cache_storage(global_config, embedding_func)
        try:
            await storage.upsert(
                {
                    "default:extract:1": {"return": "a"},
                    "default:summary:1": {"return": "b"},
                    "default:extract:2": {"return": "c"},
                    "mix:query:1": {"return": "ignored"},
                }
            )
            tool = MigrationTool()
            count = await tool.count_default_caches(storage, "LanceDBKVStorage")
            assert count == 3

            streamed = {}
            batch_count = 0
            async for batch in tool.stream_default_caches(
                storage, "LanceDBKVStorage", batch_size=2
            ):
                batch_count += 1
                assert len(batch) <= 2
                streamed.update(batch)

            assert set(streamed) == {
                "default:extract:1",
                "default:summary:1",
                "default:extract:2",
            }
            # Synthetic _id is stripped; real payload fields survive.
            assert streamed["default:extract:1"]["return"] == "a"
            assert "_id" not in streamed["default:extract:1"]
            assert batch_count == 2
        finally:
            await storage.finalize()

    async def test_migrate_round_trip_lancedb_target(
        self, global_config, embedding_func
    ):
        """LanceDB as a migration target: streamed batches upsert and persist."""
        source = await _make_llm_cache_storage(
            global_config, embedding_func, workspace="src"
        )
        target = await _make_llm_cache_storage(
            global_config, embedding_func, workspace="dst"
        )
        try:
            await source.upsert(
                {
                    "default:extract:1": {"return": "a"},
                    "default:summary:1": {"return": "b"},
                }
            )
            tool = MigrationTool()
            async for batch in tool.stream_default_caches(
                source, "LanceDBKVStorage", batch_size=1000
            ):
                await target.upsert(batch)
            await target.index_done_callback()

            assert set(await target.get_all_keys()) == {
                "default:extract:1",
                "default:summary:1",
            }
            assert (await target.get_by_id("default:extract:1"))["return"] == "a"
        finally:
            await source.finalize()
            await target.finalize()
