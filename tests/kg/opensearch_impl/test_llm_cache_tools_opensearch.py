"""
Offline tests for OpenSearch support in LLM cache tools.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch tool tests",
)

from lightrag.tools.clean_llm_query_cache import CleanupStats, CleanupTool
from lightrag.tools.migrate_llm_cache import MigrationTool
from lightrag.utils import parse_cache_key

pytestmark = pytest.mark.offline


class FakeOpenSearchStorage:
    def __init__(self, batches, workspace="test-workspace"):
        self._batches = batches
        self.workspace = workspace
        self.deleted_batches = []

    async def _iter_raw_docs(self, batch_size=1000):
        for batch in self._batches:
            yield batch

    async def delete(self, ids):
        self.deleted_batches.append(list(ids))

    async def delete_by_cache_types(self, cache_types):
        """Stand-in for the storage method the tool now delegates to; the real
        one is pinned in test_opensearch_storage.py."""
        self.deleted_cache_types = set(cache_types)
        doomed = [
            hit["_id"]
            for batch in self._batches
            for hit in batch
            if (parsed := parse_cache_key(hit["_id"])) is not None
            and parsed[1] in cache_types
        ]
        await self.delete(doomed)
        return len(doomed)


def _flatten(batches):
    return [item for batch in batches for item in batch]


class TestCleanupToolOpenSearch:
    @pytest.mark.asyncio
    async def test_count_query_caches_opensearch(self):
        tool = CleanupTool()
        storage = FakeOpenSearchStorage(
            [
                [
                    {"_id": "mix:query:1", "_source": {}},
                    {"_id": "mix:keywords:1", "_source": {}},
                    {"_id": "default:extract:1", "_source": {}},
                ],
                [
                    {"_id": "hybrid:query:1", "_source": {}},
                    {"_id": "local:keywords:1", "_source": {}},
                    {"_id": "other:key:1", "_source": {}},
                ],
            ]
        )

        counts = await tool.count_query_caches(storage, "OpenSearchKVStorage")

        assert counts["mix"] == {"query": 1, "keywords": 1}
        assert counts["hybrid"] == {"query": 1, "keywords": 0}
        assert counts["local"] == {"query": 0, "keywords": 1}
        assert counts["global"] == {"query": 0, "keywords": 0}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("cleanup_type", "expected_types", "expected_ids"),
        [
            (
                "all",
                {"query", "keywords"},
                [
                    "mix:query:1",
                    "mix:keywords:1",
                    "global:query:1",
                    "local:keywords:1",
                ],
            ),
            ("query", {"query"}, ["mix:query:1", "global:query:1"]),
            ("keywords", {"keywords"}, ["mix:keywords:1", "local:keywords:1"]),
        ],
    )
    async def test_delete_query_caches_opensearch(
        self, cleanup_type, expected_types, expected_ids
    ):
        """The tool delegates to ``delete_by_cache_types`` -- the one
        implementation the server route shares -- and never names an extract
        type."""
        tool = CleanupTool()
        storage = FakeOpenSearchStorage(
            [
                [
                    {"_id": "mix:query:1", "_source": {}},
                    {"_id": "mix:keywords:1", "_source": {}},
                ],
                [
                    {"_id": "global:query:1", "_source": {}},
                    {"_id": "local:keywords:1", "_source": {}},
                    {"_id": "default:extract:1", "_source": {}},
                ],
            ]
        )
        stats = CleanupStats()

        await tool.delete_query_caches(
            storage, "OpenSearchKVStorage", cleanup_type, stats
        )

        assert storage.deleted_cache_types == expected_types
        assert _flatten(storage.deleted_batches) == expected_ids
        assert stats.successfully_deleted == len(expected_ids)
        assert stats.successful_batches == 1
        assert not stats.errors

    def test_check_config_ini_for_storage_opensearch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.ini").write_text("[opensearch]\nhosts = localhost:9200\n")

        assert CleanupTool().check_config_ini_for_storage("OpenSearchKVStorage")

    def test_get_storage_class_opensearch(self):
        cleanup_cls = CleanupTool().get_storage_class("OpenSearchKVStorage")
        migrate_cls = MigrationTool().get_storage_class("OpenSearchKVStorage")

        assert cleanup_cls.__name__ == "OpenSearchKVStorage"
        assert migrate_cls.__name__ == "OpenSearchKVStorage"


class TestMigrationToolOpenSearch:
    @pytest.mark.asyncio
    async def test_count_and_stream_default_caches_opensearch(self):
        tool = MigrationTool()
        storage = FakeOpenSearchStorage(
            [
                [
                    {"_id": "default:extract:1", "_source": {"return": "a"}},
                    {"_id": "mix:query:1", "_source": {"return": "ignored"}},
                ],
                [
                    {"_id": "default:summary:1", "_source": {"return": "b"}},
                    {"_id": "default:analysis:1", "_source": {"return": "vlm"}},
                    {"_id": "default:extract:2", "_source": {"return": "c"}},
                ],
            ]
        )

        count = await tool.count_default_caches(storage, "OpenSearchKVStorage")
        streamed = [
            batch
            async for batch in tool.stream_default_caches(
                storage, "OpenSearchKVStorage", batch_size=2
            )
        ]

        assert count == 4
        assert streamed == [
            {
                "default:extract:1": {"return": "a"},
                "default:summary:1": {"return": "b"},
            },
            {
                "default:analysis:1": {"return": "vlm"},
                "default:extract:2": {"return": "c"},
            },
        ]

    def test_count_available_storage_types_includes_opensearch(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.ini").write_text("[opensearch]\nhosts = localhost:9200\n")

        with patch.dict("os.environ", {}, clear=True):
            assert MigrationTool().count_available_storage_types() == 2

    @pytest.mark.asyncio
    async def test_setup_storage_returns_effective_workspace(self, monkeypatch):
        tool = MigrationTool()
        fake_storage = SimpleNamespace(workspace="forced-workspace")

        monkeypatch.setattr(tool, "check_env_vars", lambda _: True)
        monkeypatch.setattr(
            tool, "initialize_storage", AsyncMock(return_value=fake_storage)
        )
        monkeypatch.setattr(tool, "count_default_caches", AsyncMock(return_value=3))

        with patch("builtins.input", return_value="5"):
            storage, storage_name, workspace, total_count = await tool.setup_storage(
                "Source", use_streaming=True
            )

        assert storage is fake_storage
        assert storage_name == "OpenSearchKVStorage"
        assert workspace == "forced-workspace"
        assert total_count == 3
