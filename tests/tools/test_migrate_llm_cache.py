"""Regression tests for the LLM cache migration tool."""

import pytest

from lightrag.tools.migrate_llm_cache import MigrationTool

pytestmark = pytest.mark.offline


class _FakeLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeJsonStorage:
    def __init__(self, data):
        self._data = data
        self._storage_lock = _FakeLock()


@pytest.mark.asyncio
async def test_json_migration_includes_multimodal_analysis_caches():
    tool = MigrationTool()
    storage = _FakeJsonStorage(
        {
            "default:extract:extract-hash": {"return": "extract"},
            "default:summary:summary-hash": {"return": "summary"},
            "default:analysis:analysis-hash": {
                "return": '{"name": "diagram"}',
                "cache_type": "analysis",
            },
            "mix:query:query-hash": {"return": "query"},
            "default:unknown:unknown-hash": {"return": "unknown"},
        }
    )

    loaded = await tool.get_default_caches_json(storage)
    counted = await tool.count_default_caches_json(storage)
    streamed = [
        batch async for batch in tool.stream_default_caches_json(storage, batch_size=2)
    ]
    type_counts = await tool.count_cache_types(loaded)

    assert list(loaded) == [
        "default:extract:extract-hash",
        "default:summary:summary-hash",
        "default:analysis:analysis-hash",
    ]
    assert counted == 3
    assert [key for batch in streamed for key in batch] == list(loaded)
    assert type_counts == {"extract": 1, "summary": 1, "analysis": 1}
