"""Regression tests for the smart-heading cache type in the migration tool.

The parse stage writes its LLM judgments into the SAME ``llm_response_cache``
KV namespace as the extraction caches, under ``cache_type="smartheading"``
(``lightrag/parser/native_base.py``), and ``use_llm_func_with_cache`` keys every
row it writes as ``{mode}:{cache_type}:{hash}`` with ``mode="default"``. So the
rows live at ``default:smartheading:<hash>`` -- one of the default-mode cache
types the migration tool exists to move.

The tool's ``DEFAULT_CACHE_TYPES`` omitted that type, so those rows were neither
counted, nor streamed, nor written to the destination: a source holding only
smart-heading rows reported "No records to migrate", and a mixed source left the
smart-heading rows behind -- re-running the parse-time LLM calls the cache was
meant to save, and breaking the re-parse determinism the cache hits provide.
"""

from __future__ import annotations

import pytest

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.tools.migrate_llm_cache import (
    MigrationTool,
    _cache_type_from_key,
    _is_default_cache_key,
)
from lightrag.utils import use_llm_func_with_cache

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    """KV storages never embed; the dataclass only requires the field."""

    embedding_dim = 4

    async def __call__(self, texts):  # pragma: no cover - never invoked
        return [[0.0] * self.embedding_dim for _ in texts]


async def _build_cache_storage(tmp_path, workspace: str) -> JsonKVStorage:
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data()
    storage = JsonKVStorage(
        namespace="llm_response_cache",
        global_config={
            "working_dir": str(tmp_path),
            "enable_llm_cache": True,
            "enable_llm_cache_for_entity_extract": True,
        },
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )
    await storage.initialize()
    return storage


async def _write_smartheading_row(storage, prompt: str) -> str:
    """Write one cache row through the real parse-stage writer path."""

    async def _llm(prompt, **kwargs):
        return "judged"

    keys: list[str] = []
    await use_llm_func_with_cache(
        prompt,
        _llm,
        llm_response_cache=storage,
        cache_type="smartheading",
        cache_keys_collector=keys,
    )
    assert len(keys) == 1
    return keys[0]


async def test_the_parse_stage_writes_a_default_smartheading_row(tmp_path):
    """The premise: the runtime really does produce ``default:smartheading:*``."""
    storage = await _build_cache_storage(tmp_path, "smartheading_key_shape")
    try:
        key = await _write_smartheading_row(storage, "is this a heading?")

        assert key.startswith("default:smartheading:")
        assert _cache_type_from_key(key) == "smartheading"
        assert await storage.get_by_id(key) is not None
    finally:
        await storage.finalize()


async def test_migration_tool_moves_default_smartheading_caches(tmp_path):
    """Counting, streaming and migrating must all see the smart-heading row."""
    storage = await _build_cache_storage(tmp_path, "smartheading_migration")
    try:
        smartheading_key = await _write_smartheading_row(storage, "title?")

        tool = MigrationTool()

        counted = await tool.count_default_caches_json(storage)
        loaded = await tool.get_default_caches_json(storage)
        streamed = [
            {
                key: value
                for batch in [
                    b
                    async for b in tool.stream_default_caches_json(
                        storage, batch_size=2
                    )
                ]
                for key, value in batch.items()
            }
        ][0]
        type_counts = await tool.count_cache_types(loaded)

        assert counted == 1
        assert list(loaded) == [smartheading_key]
        assert list(streamed) == [smartheading_key]
        assert type_counts["smartheading"] == 1
    finally:
        await storage.finalize()


async def test_a_store_of_only_smartheading_rows_is_not_reported_as_empty(tmp_path):
    """A source with smart-heading rows only must not say "No records"."""
    source = await _build_cache_storage(tmp_path, "smartheading_only_source")
    target = await _build_cache_storage(tmp_path, "smartheading_only_target")
    try:
        smartheading_key = await _write_smartheading_row(source, "heading?")

        tool = MigrationTool()
        source_data = await tool.get_default_caches(source, "JsonKVStorage")
        stats = await tool.migrate_caches(source_data, target, "JsonKVStorage")

        assert stats.total_source_records == 1
        assert stats.successful_records == 1

        migrated = await target.get_by_id(smartheading_key)
        assert migrated is not None
        assert migrated["return"] == "judged"
        assert migrated["cache_type"] == "smartheading"
    finally:
        await source.finalize()
        await target.finalize()


def test_every_default_mode_cache_type_the_runtime_writes_is_migrated():
    """``summary``/``smartheading``/``analysis`` are peers: all three migrate.

    ``lightrag/lightrag.py`` groups them as the default-mode types that name no
    owning chunk; ``extract`` completes the set. A type missing here is a type
    the tool silently leaves behind.
    """
    for cache_type in ("extract", "summary", "analysis", "smartheading"):
        assert _is_default_cache_key(f"default:{cache_type}:abc123")
    assert not _is_default_cache_key("mix:query:abc123")
    assert not _is_default_cache_key("default:unknown:abc123")
