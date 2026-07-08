"""Unit tests for LanceDBKVStorage against a real tmp-path LanceDB."""

import asyncio

import pytest

pytest.importorskip("lancedb", reason="lancedb is required for LanceDB storage tests")

from lightrag.kg.lancedb_impl import LanceDBKVStorage  # noqa: E402

pytestmark = pytest.mark.offline


def _make_storage(global_config, embedding_func, namespace="full_docs", workspace="ws"):
    return LanceDBKVStorage(
        namespace=namespace,
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding_func,
    )


async def test_upsert_and_get_by_id(global_config, embedding_func):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        assert await storage.is_empty()
        await storage.upsert({"doc-1": {"content": "hello", "file_path": "a.txt"}})
        record = await storage.get_by_id("doc-1")
        assert record["content"] == "hello"
        assert record["file_path"] == "a.txt"
        assert record["_id"] == "doc-1"
        assert record["create_time"] > 0
        assert record["update_time"] >= record["create_time"]
        assert not await storage.is_empty()
        assert await storage.get_by_id("missing") is None
    finally:
        await storage.finalize()


async def test_get_by_ids_preserves_order_with_none_for_missing(
    global_config, embedding_func
):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert(
            {"a": {"content": "1"}, "b": {"content": "2"}, "c": {"content": "3"}}
        )
        records = await storage.get_by_ids(["c", "missing", "a"])
        assert records[0]["content"] == "3"
        assert records[1] is None
        assert records[2]["content"] == "1"
        assert await storage.get_by_ids([]) == []
    finally:
        await storage.finalize()


async def test_filter_keys_returns_missing_keys(global_config, embedding_func):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert({"a": {"content": "1"}})
        assert await storage.filter_keys({"a", "b", "c"}) == {"b", "c"}
        assert await storage.filter_keys(set()) == set()
    finally:
        await storage.finalize()


async def test_upsert_replaces_record_but_preserves_create_time(
    global_config, embedding_func
):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert({"a": {"content": "v1", "stale_field": True}})
        first = await storage.get_by_id("a")
        await asyncio.sleep(1.1)
        await storage.upsert({"a": {"content": "v2"}})
        second = await storage.get_by_id("a")
        assert second["content"] == "v2"
        # Full-record replace: fields absent from the new record are gone.
        assert "stale_field" not in second
        assert second["create_time"] == first["create_time"]
        assert second["update_time"] > first["update_time"]
    finally:
        await storage.finalize()


async def test_text_chunks_defaults_llm_cache_list(global_config, embedding_func):
    storage = _make_storage(global_config, embedding_func, namespace="text_chunks")
    await storage.initialize()
    try:
        await storage.upsert(
            {
                "chunk-1": {"content": "c", "tokens": 3, "full_doc_id": "doc-1"},
                "chunk-2": {"content": "c2", "llm_cache_list": ["default:extract:x"]},
            }
        )
        first = await storage.get_by_id("chunk-1")
        assert first["llm_cache_list"] == []
        second = await storage.get_by_id("chunk-2")
        assert second["llm_cache_list"] == ["default:extract:x"]
    finally:
        await storage.finalize()


async def test_nested_records_round_trip(global_config, embedding_func):
    storage = _make_storage(global_config, embedding_func, namespace="text_chunks")
    await storage.initialize()
    try:
        record = {
            "content": "chunk text",
            "tokens": 42,
            "chunk_order_index": 3,
            "full_doc_id": "doc-1",
            "heading": {"level": 1, "titles": ["A", "B"]},
            "sidecar": {"images": [{"path": "x.png"}]},
        }
        await storage.upsert({"chunk-1": dict(record)})
        loaded = await storage.get_by_id("chunk-1")
        for key, value in record.items():
            assert loaded[key] == value
    finally:
        await storage.finalize()


async def test_delete_removes_rows_and_ignores_missing(global_config, embedding_func):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert({"a": {"content": "1"}, "b": {"content": "2"}})
        await storage.delete(["a", "not-there"])
        assert await storage.get_by_id("a") is None
        assert await storage.get_by_id("b") is not None
        await storage.delete([])
    finally:
        await storage.finalize()


async def test_ids_with_quotes_are_escaped(global_config, embedding_func):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        tricky = "doc-'; DROP TABLE x; --"
        await storage.upsert({tricky: {"content": "quoted"}})
        record = await storage.get_by_id(tricky)
        assert record["content"] == "quoted"
        assert await storage.filter_keys({tricky}) == set()
        await storage.delete([tricky])
        assert await storage.get_by_id(tricky) is None
    finally:
        await storage.finalize()


async def test_drop_resets_storage(global_config, embedding_func):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert({"a": {"content": "1"}})
        result = await storage.drop()
        assert result == {"status": "success", "message": "data dropped"}
        assert await storage.is_empty()
        # Storage stays usable after drop.
        await storage.upsert({"b": {"content": "2"}})
        assert (await storage.get_by_id("b"))["content"] == "2"
    finally:
        await storage.finalize()


async def test_workspace_isolation(global_config, embedding_func):
    ws1 = _make_storage(global_config, embedding_func, workspace="ws1")
    ws2 = _make_storage(global_config, embedding_func, workspace="ws2")
    await ws1.initialize()
    await ws2.initialize()
    try:
        await ws1.upsert({"a": {"content": "ws1"}})
        assert await ws2.get_by_id("a") is None
        assert (await ws1.get_by_id("a"))["content"] == "ws1"
    finally:
        await ws1.finalize()
        await ws2.finalize()


async def test_lancedb_workspace_env_override(
    global_config, embedding_func, monkeypatch
):
    monkeypatch.setenv("LANCEDB_WORKSPACE", "forced")
    storage = _make_storage(global_config, embedding_func, workspace="ignored")
    assert storage.workspace == "forced"
    assert storage.final_namespace == "forced_full_docs"


async def test_workspaces_differing_only_by_case_stay_isolated(
    global_config, embedding_func
):
    """Regression: lossy table-name folding must not merge distinct workspaces."""
    upper = _make_storage(global_config, embedding_func, workspace="TeamA")
    lower = _make_storage(global_config, embedding_func, workspace="teama")
    assert upper._table_name != lower._table_name
    await upper.initialize()
    await lower.initialize()
    try:
        await upper.upsert({"a": {"content": "upper"}})
        assert await lower.get_by_id("a") is None
        assert (await upper.get_by_id("a"))["content"] == "upper"
    finally:
        await upper.finalize()
        await lower.finalize()


async def test_concurrent_upserts_do_not_duplicate(global_config, embedding_func):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await asyncio.gather(
            *[
                storage.upsert({f"k{i}": {"content": f"v{i}-{worker}"}})
                for worker in range(8)
                for i in range(5)
            ]
        )
        records = await storage.get_by_ids([f"k{i}" for i in range(5)])
        assert all(record is not None for record in records)
        assert await storage.filter_keys({f"k{i}" for i in range(5)}) == set()
        # Exactly one row per key even under concurrent same-key writes.
        assert await storage._table().count_rows() == 5
    finally:
        await storage.finalize()
