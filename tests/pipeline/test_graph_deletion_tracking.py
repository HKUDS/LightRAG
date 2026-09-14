"""Manual graph deletion must remove persisted provenance before a new generation."""

import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG, utils_graph
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import EmbeddingFunc, Tokenizer, make_relation_chunk_key

pytestmark = pytest.mark.offline


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


async def _no_llm(*args, **kwargs):
    raise AssertionError("This lifecycle fixture must not call an LLM")


async def _extract(chunks, *args, **kwargs):
    results = []
    for chunk_id, payload in chunks.items():
        entities = {
            name: [
                {
                    "entity_name": name,
                    "entity_type": "company",
                    "description": f"{name} company",
                    "source_id": chunk_id,
                    "file_path": payload["file_path"],
                    "timestamp": 1,
                }
            ]
            for name in ("Atlas", "Borealis")
        }
        relations = {
            ("Atlas", "Borealis"): [
                {
                    "src_id": "Atlas",
                    "tgt_id": "Borealis",
                    "weight": 1.0,
                    "description": "Atlas cooperates with Borealis",
                    "keywords": "cooperates",
                    "source_id": chunk_id,
                    "file_path": payload["file_path"],
                    "timestamp": 1,
                }
            ]
        }
        results.append((entities, relations))
    return results


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["sdk", "rest"])
@pytest.mark.parametrize("kind", ["entity", "relation", "relation-reversed"])
async def test_delete_removes_tracking_before_reinsert(
    tmp_path, monkeypatch, kind, entrypoint
):
    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"tracking_{uuid4().hex}",
        llm_model_func=_no_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    rag._process_extract_entities = _extract
    try:
        await rag.ainsert(
            "Atlas and Borealis: old harbor", ids=["old"], file_paths=["old.txt"]
        )
        old_chunk = (await rag.doc_status.get_by_id("old"))["chunks_list"][0]
        relation_key = make_relation_chunk_key("Atlas", "Borealis")
        assert (await rag.relation_chunks.get_by_id(relation_key))["chunk_ids"] == [
            old_chunk
        ]
        names = (
            ("Borealis", "Atlas")
            if kind.endswith("reversed")
            else ("Atlas", "Borealis")
        )
        if entrypoint == "rest":
            # Router imports parse argv; do not hand them pytest's options.
            monkeypatch.setattr(sys, "argv", [sys.argv[0]])
            from fastapi import FastAPI
            from httpx import ASGITransport, AsyncClient
            from lightrag.api.routers.graph_routes import create_graph_routes

            app = FastAPI()
            app.include_router(create_graph_routes(rag, api_key="fixture-key"))
            resource = "entity" if kind == "entity" else "relation"
            payload = (
                {"entity_name": "Atlas"}
                if kind == "entity"
                else {"source_entity": names[0], "target_entity": names[1]}
            )
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.request(
                    "DELETE",
                    f"/graph/{resource}/delete",
                    json=payload,
                    headers={"X-API-Key": "fixture-key"},
                )
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "success"
        else:
            result = (
                await rag.adelete_by_entity("Atlas")
                if kind == "entity"
                else await rag.adelete_by_relation(*names)
            )
            assert result.status == "success"
        assert await rag.relation_chunks.get_by_id(relation_key) is None
        assert relation_key not in json.loads(
            Path(rag.relation_chunks._file_name).read_text()
        )
        if kind == "entity":
            assert await rag.entity_chunks.get_by_id("Atlas") is None
            assert "Atlas" not in json.loads(
                Path(rag.entity_chunks._file_name).read_text()
            )
        # Unrelated entity provenance must survive both deletion paths.
        assert (await rag.entity_chunks.get_by_id("Borealis"))["chunk_ids"] == [
            old_chunk
        ]

        await rag.ainsert(
            "Atlas and Borealis: new mountain", ids=["new"], file_paths=["new.txt"]
        )
        new_chunk = (await rag.doc_status.get_by_id("new"))["chunks_list"][0]
        edge = await rag.chunk_entity_relation_graph.get_edge("Atlas", "Borealis")
        assert edge["source_id"].split(GRAPH_FIELD_SEP) == [new_chunk]
        assert (await rag.relation_chunks.get_by_id(relation_key))["chunk_ids"] == [
            new_chunk
        ]
        if kind == "entity":
            node = await rag.chunk_entity_relation_graph.get_node("Atlas")
            assert node["source_id"].split(GRAPH_FIELD_SEP) == [new_chunk]
            assert (await rag.entity_chunks.get_by_id("Atlas"))["chunk_ids"] == [
                new_chunk
            ]
    finally:
        await rag.finalize_storages()


@pytest.fixture
async def creation_rag(tmp_path):
    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"creation_{uuid4().hex}",
        llm_model_func=_no_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    rag._process_extract_entities = _extract
    try:
        yield rag
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["sdk", "helper"])
@pytest.mark.parametrize("kind", ["entity", "relation"])
@pytest.mark.parametrize(
    "source_id,expected",
    [
        (None, []),
        ("", []),
        ("manual_creation", []),
        ("UNKNOWN", []),
        (
            GRAPH_FIELD_SEP.join(
                ["manual_creation", "chunk-new", "UNKNOWN", "", "chunk-new", "chunk-2"]
            ),
            ["chunk-new", "chunk-2"],
        ),
    ],
)
async def test_explicit_creation_replaces_orphan_evidence(
    creation_rag, monkeypatch, kind, source_id, expected, entrypoint
):
    rag = creation_rag
    await rag.ainsert("Atlas and Borealis: old harbor", ids=["old"])
    tracking = rag.entity_chunks if kind == "entity" else rag.relation_chunks
    key = "Atlas" if kind == "entity" else make_relation_chunk_key("Atlas", "Borealis")
    old_row = await tracking.get_by_id(key)
    assert old_row["chunk_ids"]

    # Real JSON storage: a failed cleanup leaves durable orphan evidence after
    # graph deletion. Explicit creation must replace it, even without sources.
    async def fail_cleanup(ids):
        raise OSError("injected tracking cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(tracking, "delete", fail_cleanup)
        result = (
            await rag.adelete_by_entity("Atlas")
            if kind == "entity"
            else await rag.adelete_by_relation("Atlas", "Borealis")
        )
    assert result.status == "fail"
    assert (
        json.loads(Path(tracking._file_name).read_text())[key]["chunk_ids"]
        == old_row["chunk_ids"]
    )

    data = {"description": "Manually recreated"}
    if source_id is not None:
        data["source_id"] = source_id
    if kind == "entity":
        if entrypoint == "sdk":
            await rag.acreate_entity("Atlas", data)
        else:
            await utils_graph.acreate_entity(
                rag.chunk_entity_relation_graph,
                rag.entities_vdb,
                rag.relationships_vdb,
                "Atlas",
                data,
                entity_chunks_storage=rag.entity_chunks,
            )
    else:
        data["weight"] = max(1, len(expected))
        if entrypoint == "sdk":
            await rag.acreate_relation("Atlas", "Borealis", data)
        else:
            await utils_graph.acreate_relation(
                rag.chunk_entity_relation_graph,
                rag.entities_vdb,
                rag.relationships_vdb,
                "Atlas",
                "Borealis",
                data,
                relation_chunks_storage=rag.relation_chunks,
            )

    row = await tracking.get_by_id(key)
    assert row["chunk_ids"] == expected
    assert row["count"] == len(expected)
    persisted = json.loads(Path(tracking._file_name).read_text())[key]
    assert persisted["chunk_ids"] == expected
    assert persisted["count"] == len(expected)


@pytest.mark.asyncio
async def test_manual_entity_then_document_is_deleted_without_rebuild(
    creation_rag, monkeypatch
):
    rag = creation_rag
    await rag.acreate_entity("Atlas", {"description": "Atlas company"})
    row = await rag.entity_chunks.get_by_id("Atlas")
    assert row["chunk_ids"] == []
    assert row["count"] == 0

    await rag.ainsert("Atlas and Borealis: new mountain", ids=["new"])
    chunk = (await rag.doc_status.get_by_id("new"))["chunks_list"][0]
    assert (await rag.entity_chunks.get_by_id("Atlas"))["chunk_ids"] == [chunk]
    assert "Atlas" in (await rag.full_entities.get_by_id("new"))["entity_names"]

    async def unexpected_rebuild(*args, **kwargs):
        raise AssertionError("The document owns all evidence; delete outright")

    monkeypatch.setattr(
        "lightrag.lightrag.rebuild_knowledge_from_chunks", unexpected_rebuild
    )
    result = await rag.adelete_by_doc_id("new")
    assert result.status == "success", result.message
    assert not await rag.chunk_entity_relation_graph.has_node("Atlas")
    assert await rag.entity_chunks.get_by_id("Atlas") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["entity", "relation"])
@pytest.mark.parametrize("fail_migration", [False, True])
async def test_sdk_creation_migrates_legacy_tracking_first(
    creation_rag, monkeypatch, kind, fail_migration
):
    rag = creation_rag
    graph = rag.chunk_entity_relation_graph
    # Model an upgraded working directory: graph provenance exists on disk,
    # but neither tracking namespace has been seeded yet.
    for name in ("LegacyA", "LegacyB"):
        await graph.upsert_node(
            name,
            {"entity_id": name, "description": name, "source_id": "legacy-chunk"},
        )
    await graph.upsert_edge(
        "LegacyA",
        "LegacyB",
        {"description": "legacy", "source_id": "legacy-chunk", "weight": 1.0},
    )
    await graph.index_done_callback()
    assert await rag.entity_chunks.is_empty()
    assert await rag.relation_chunks.is_empty()

    async def create():
        if kind == "entity":
            await rag.acreate_entity("Manual", {"description": "manual"})
        else:
            await rag.acreate_relation(
                "LegacyA", "ManualTarget", {"description": "manual"}
            )

    if kind == "relation":
        await graph.upsert_node(
            "ManualTarget",
            {"entity_id": "ManualTarget", "description": "target", "source_id": ""},
        )
        await graph.index_done_callback()

    if fail_migration:

        async def fail_read():
            raise OSError("legacy graph read unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(graph, "get_all_nodes", fail_read)
            with pytest.raises(OSError, match="legacy graph read unavailable"):
                await create()
        assert not await graph.has_node("Manual")
        assert not await graph.has_edge("LegacyA", "ManualTarget")
        assert await rag.entity_chunks.is_empty()
        assert await rag.relation_chunks.is_empty()

    await create()
    # Reopening the stores and invoking the same migration as server startup
    # must preserve both historical provenance and the new authoritative empty row.
    await rag.finalize_storages()
    reopened = LightRAG(
        working_dir=rag.working_dir,
        workspace=rag.workspace,
        llm_model_func=_no_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
    )
    await reopened.initialize_storages()
    try:
        await reopened.check_and_migrate_data()
        for name in ("LegacyA", "LegacyB"):
            assert (await reopened.entity_chunks.get_by_id(name))["chunk_ids"] == [
                "legacy-chunk"
            ]
        key = make_relation_chunk_key("LegacyA", "LegacyB")
        assert (await reopened.relation_chunks.get_by_id(key))["chunk_ids"] == [
            "legacy-chunk"
        ]
        tracking = (
            reopened.entity_chunks if kind == "entity" else reopened.relation_chunks
        )
        new_key = (
            "Manual"
            if kind == "entity"
            else make_relation_chunk_key("LegacyA", "ManualTarget")
        )
        row = await tracking.get_by_id(new_key)
        assert row["chunk_ids"] == []
        assert row["count"] == 0
        persisted = json.loads(Path(tracking._file_name).read_text())
        assert persisted[new_key]["chunk_ids"] == []
    finally:
        await reopened.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["entity", "relation"])
async def test_invalid_sdk_creation_does_not_start_migration(
    creation_rag, monkeypatch, kind
):
    rag = creation_rag
    for name in ("LegacyA", "LegacyB"):
        await rag.chunk_entity_relation_graph.upsert_node(
            name, {"entity_id": name, "description": name, "source_id": "legacy-chunk"}
        )

    async def unexpected_migration():
        raise AssertionError("Invalid input must be rejected before migration writes")

    monkeypatch.setattr(
        rag, "_migrate_chunk_tracking_before_creation", unexpected_migration
    )
    with pytest.raises(ValueError):
        if kind == "entity":
            await rag.acreate_entity(
                "Manual", {"description": "manual", "source_id": []}
            )
        else:
            await rag.acreate_relation(
                "LegacyA", "LegacyB", {"description": "manual", "weight": -1}
            )
    assert await rag.entity_chunks.is_empty()
    assert await rag.relation_chunks.is_empty()


@pytest.mark.asyncio
async def test_creation_skips_migration_lock_after_success(creation_rag, monkeypatch):
    rag = creation_rag
    await rag.acreate_entity("First", {"description": "first"})
    # No relations were created: is_empty() cannot prove this namespace has
    # already been checked, even though another full graph scan is unnecessary.
    assert await rag.relation_chunks.is_empty()

    def unexpected_lock():
        raise AssertionError("A successful migration check must bypass the lock")

    monkeypatch.setattr(
        "lightrag.storage_migrations.get_data_init_lock", unexpected_lock
    )
    await rag.acreate_entity("Second", {"description": "second"})
    await rag.acreate_relation("First", "Second", {"description": "manual"})
    assert await rag.chunk_entity_relation_graph.has_edge("First", "Second")


@pytest.mark.asyncio
async def test_concurrent_creation_migration_checks_run_once(creation_rag, monkeypatch):
    import asyncio
    from contextlib import asynccontextmanager
    from lightrag import storage_migrations

    rag = creation_rag
    original_lock = storage_migrations.get_data_init_lock
    first_migrating = asyncio.Event()
    second_waiting = asyncio.Event()
    finish_migration = asyncio.Event()
    acquisitions = 0
    migrations = 0

    @asynccontextmanager
    async def observed_lock():
        nonlocal acquisitions
        acquisitions += 1
        if acquisitions == 2:
            second_waiting.set()
        async with original_lock():
            yield

    async def blocked_migration():
        nonlocal migrations
        migrations += 1
        first_migrating.set()
        await finish_migration.wait()

    monkeypatch.setattr(storage_migrations, "get_data_init_lock", observed_lock)
    monkeypatch.setattr(rag, "_migrate_chunk_tracking_storage", blocked_migration)
    first = asyncio.create_task(rag._migrate_chunk_tracking_before_creation())
    second = None
    try:
        await asyncio.wait_for(first_migrating.wait(), timeout=5)
        second = asyncio.create_task(rag._migrate_chunk_tracking_before_creation())
        await asyncio.wait_for(second_waiting.wait(), timeout=5)
        finish_migration.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
        assert migrations == 1
    finally:
        for task in (first, second):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )
