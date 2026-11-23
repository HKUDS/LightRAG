"""Offline contract tests for rebuild-from-snapshot (zero re-embed promotion).

Each test pins one clause of the snapshot-restore rules: exact source/vector
fidelity with zero embedding calls, idempotent re-run, fail-before-mutation
on missing tag/table/dimension/vector, unsupported-backend refusal, and the
interrupted-restore heal-by-rerun path. CLI tests drive the real
``lightrag-rebuild-vdb --from-snapshot`` entry point.
"""

import asyncio
from types import SimpleNamespace

import lancedb
import pyarrow as pa
import pytest

from lightrag import config_store as cs
from lightrag.config_shards import json_config_path
from lightrag.exceptions import StorageCapabilityError
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.lancedb_impl import _schemas, export_snapshot
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.noop_vector_db_impl import NoopVectorDBStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace
from lightrag.tools.rebuild_vdb import main
from lightrag.tools.snapshot_restore import (
    RestoreError,
    open_snapshot,
    restore_workspace_snapshot,
)
from lightrag.utils import EmbeddingFunc, compute_mdhash_id

from tests.kg.lancedb_impl.conftest import (
    CHUNKS,
    DIM,
    DOCS,
    EDGES,
    MODEL_NAME,
    NODES,
    CountingEmbedder,
    entity_content,
    expected_vector,
    relation_content,
    seed_rag,
)

pytestmark = pytest.mark.offline

ENTITY_IDS = {name: compute_mdhash_id(name, prefix="ent-") for name in NODES}
RELATION_IDS = {
    (src, tgt): compute_mdhash_id("".join(sorted((src, tgt))), prefix="rel-")
    for src, tgt, _ in EDGES
}


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def build_target(tmp_path, embedder):
    """Empty INITIALIZED real storages in a fresh working dir (no LightRAG)."""
    ef = EmbeddingFunc(embedding_dim=DIM, func=embedder, model_name=MODEL_NAME)
    gc = {
        "working_dir": str(tmp_path / "restored"), "workspace": "",
        "embedding_batch_num": 8,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        "embedding_func": ef, "kv_storage": "JsonKVStorage",
        "vector_storage": "NanoVectorDBStorage", "graph_storage": "NetworkXStorage",
    }

    def kv(namespace):
        return JsonKVStorage(namespace=namespace, workspace="",
                             global_config=gc, embedding_func=ef)

    def vdb(namespace, meta_fields):
        return NanoVectorDBStorage(namespace=namespace, workspace="",
                                   global_config=gc, embedding_func=ef,
                                   meta_fields=meta_fields)

    rag = SimpleNamespace(
        full_docs=kv(NameSpace.KV_STORE_FULL_DOCS),
        text_chunks=kv(NameSpace.KV_STORE_TEXT_CHUNKS),
        chunk_entity_relation_graph=NetworkXStorage(
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace="", global_config=gc, embedding_func=ef),
        entities_vdb=vdb(NameSpace.VECTOR_STORE_ENTITIES,
                         {"entity_name", "source_id", "content", "file_path"}),
        relationships_vdb=vdb(NameSpace.VECTOR_STORE_RELATIONSHIPS,
                              {"src_id", "tgt_id", "source_id", "content", "file_path"}),
        chunks_vdb=vdb(NameSpace.VECTOR_STORE_CHUNKS,
                       {"full_doc_id", "content", "file_path"}),
        embedding_func=ef,
    )
    for storage in (rag.full_docs, rag.text_chunks, rag.chunk_entity_relation_graph,
                    rag.entities_vdb, rag.relationships_vdb, rag.chunks_vdb):
        await storage.initialize()
    return rag, gc


async def read_back_vdbs(rag):
    """Vectors + meta records of all three vdbs, keyed by id (no wall-clock
    stamps: ``__created_at__`` and the ``created_at`` read-back mirror derived
    from it both reset on every rebuild by design)."""
    out = {}
    for label, vdb, ids in (
        ("entities", rag.entities_vdb, list(ENTITY_IDS.values())),
        ("relationships", rag.relationships_vdb, list(RELATION_IDS.values())),
        ("chunks", rag.chunks_vdb, list(CHUNKS)),
    ):
        records = {r["id"]: {k: v for k, v in r.items()
                             if k not in ("id", "__created_at__", "created_at")}
                   for r in await vdb.get_by_ids(ids)}
        out[label] = {"vectors": await vdb.get_vectors_by_ids(ids), "records": records}
    return out


async def assert_restored_matches_seed(rag):
    """Compare the restored workspace against the independent seed oracle."""
    docs = {r["_id"]: r["content"] async for r in rag.full_docs.iter_rows(page_size=10)}
    assert docs == {k: v["content"] for k, v in DOCS.items()}
    chunks = {r["_id"]: r async for r in rag.text_chunks.iter_rows(page_size=10)}
    assert set(chunks) == set(CHUNKS)
    for cid, source in CHUNKS.items():
        for key in ("content", "full_doc_id", "chunk_order_index", "tokens", "file_path"):
            assert chunks[cid][key] == source[key]
    for name, props in NODES.items():
        assert await rag.chunk_entity_relation_graph.get_node(name) == props
    for src, tgt, props in EDGES:
        assert await rag.chunk_entity_relation_graph.get_edge(src, tgt) == props

    read = await read_back_vdbs(rag)
    assert read["entities"]["vectors"] == {
        eid: expected_vector(entity_content(name)) for name, eid in ENTITY_IDS.items()}
    assert read["relationships"]["vectors"] == {
        RELATION_IDS[(src, tgt)]: expected_vector(relation_content(*sorted((src, tgt)), props))
        for src, tgt, props in EDGES
    }
    assert read["chunks"]["vectors"] == {
        cid: expected_vector(c["content"]) for cid, c in CHUNKS.items()}
    for name, eid in ENTITY_IDS.items():
        assert read["entities"]["records"][eid]["content"] == entity_content(name)
        assert read["entities"]["records"][eid]["entity_name"] == name
    for cid, source in CHUNKS.items():
        assert read["chunks"]["records"][cid]["content"] == source["content"]
        assert read["chunks"]["records"][cid]["full_doc_id"] == source["full_doc_id"]


async def seeded_snapshot(tmp_path):
    """Source workspace + local tagged snapshot; returns (dest, embedder).

    The source is finalized after export: one process may hold only one
    working_dir's file-backed namespace claims at a time, and the restore
    target is a second working_dir (the real flow runs them in separate
    processes)."""
    source, embedder = await seed_rag(tmp_path)
    embedder.armed = True
    dest = str(tmp_path / "lake")
    await export_snapshot(source, dest, "snap", batch_size=2)
    assert embedder.export_calls == 0
    for storage in (source.full_docs, source.text_chunks,
                    source.chunk_entity_relation_graph, source.entities_vdb,
                    source.relationships_vdb, source.chunks_vdb):
        await storage.finalize()
    return dest, embedder


async def test_restore_recovers_sources_and_vectors_with_zero_embed_calls(tmp_path):
    dest, _ = await seeded_snapshot(tmp_path)
    target_embedder = CountingEmbedder()
    target_embedder.armed = True
    target, gc = await build_target(tmp_path, target_embedder)

    tables = open_snapshot(dest, "snap", dim=DIM)
    stats = await restore_workspace_snapshot(tables, target, gc, batch_size=2)

    assert target_embedder.export_calls == 0
    assert {label: s["rebuilt"] for label, s in stats.items()} == {
        "entities": len(NODES), "relationships": len(EDGES), "chunks": len(CHUNKS)}
    await assert_restored_matches_seed(target)


async def test_restore_rerun_is_idempotent(tmp_path):
    dest, _ = await seeded_snapshot(tmp_path)
    target_embedder = CountingEmbedder()
    target_embedder.armed = True
    target, gc = await build_target(tmp_path, target_embedder)

    await restore_workspace_snapshot(open_snapshot(dest, "snap", dim=DIM), target, gc, batch_size=2)
    first = await read_back_vdbs(target)
    await restore_workspace_snapshot(open_snapshot(dest, "snap", dim=DIM), target, gc, batch_size=2)

    assert target_embedder.export_calls == 0
    assert await read_back_vdbs(target) == first


async def test_open_snapshot_missing_tag_fails_loud(tmp_path):
    dest, _ = await seeded_snapshot(tmp_path)
    with pytest.raises(RestoreError, match="absent"):
        open_snapshot(dest, "absent", dim=DIM)
    db = lancedb.connect(dest)
    for name in ("docs", "chunks", "entities", "relations"):
        assert set(db.open_table(name).tags.list()) == {"snap"}


async def test_open_snapshot_missing_table_fails_loud(tmp_path):
    dest, _ = await seeded_snapshot(tmp_path)
    lancedb.connect(dest).drop_table("relations")
    with pytest.raises(RestoreError, match="relations"):
        open_snapshot(dest, "snap", dim=DIM)


async def test_open_snapshot_dimension_mismatch_fails_loud(tmp_path):
    dest, _ = await seeded_snapshot(tmp_path)
    with pytest.raises(RestoreError, match="dimension"):
        open_snapshot(dest, "snap", dim=DIM + 1)


async def test_open_snapshot_null_vector_fails_loud(tmp_path):
    dest = str(tmp_path / "hand-lake")
    db = lancedb.connect(dest)
    schemas = _schemas(DIM, MODEL_NAME)
    vector = expected_vector("x")
    rows = {
        "docs": [{"id": "doc-1", "content": "body"}],
        "chunks": [{"id": "chunk-1", "content": "c", "full_doc_id": "doc-1",
                    "chunk_order_index": 0, "tokens": 1, "file_path": "a.txt",
                    "embedding": vector}],
        "entities": [{"id": "ent-x", "entity_name": "X", "entity_type": "person",
                      "description": "d", "source_id": "chunk-1", "file_path": "a.txt",
                      "embedding": None}],
        "relations": [{"id": "rel-x", "src_id": "X", "tgt_id": "Y", "keywords": "k",
                       "description": "d", "source_id": "chunk-1", "weight": 1.0,
                       "file_path": "a.txt", "embedding": vector}],
    }
    for name, data in rows.items():
        table = db.create_table(name, data=pa.Table.from_pylist(data, schema=schemas[name]))
        table.tags.create("snap", table.version)

    with pytest.raises(RestoreError, match="without a stored vector"):
        open_snapshot(dest, "snap", dim=DIM)


async def test_restore_missing_storages_refused(tmp_path):
    rag = SimpleNamespace(
        full_docs=None, text_chunks=None, chunk_entity_relation_graph=None,
        entities_vdb=None, relationships_vdb=None, chunks_vdb=None,
        embedding_func=None,
    )
    with pytest.raises(RestoreError, match="initialize_storages"):
        await restore_workspace_snapshot({}, rag, {})


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"concurrency": 0}],
                         ids=["batch_size", "concurrency"])
async def test_restore_invalid_parallelism_refused(tmp_path, kwargs):
    dest, _ = await seeded_snapshot(tmp_path)
    target_embedder = CountingEmbedder()
    target, gc = await build_target(tmp_path, target_embedder)
    tables = open_snapshot(dest, "snap", dim=DIM)
    with pytest.raises(RestoreError, match="must be >= 1"):
        await restore_workspace_snapshot(tables, target, gc, **kwargs)
    assert target_embedder.export_calls == 0


async def test_restore_unsupported_backend_refused_before_drop(tmp_path):
    dest, _ = await seeded_snapshot(tmp_path)
    target_embedder = CountingEmbedder()
    target, gc = await build_target(tmp_path, target_embedder)
    await target.chunks_vdb.upsert({"chunk-1": {
        "content": CHUNKS["chunk-1"]["content"], "full_doc_id": "doc-1",
        "file_path": "a.txt"}})
    await target.chunks_vdb.index_done_callback()
    target.entities_vdb = NoopVectorDBStorage(
        namespace=NameSpace.VECTOR_STORE_ENTITIES, workspace="",
        global_config=gc, embedding_func=target.embedding_func)

    tables = open_snapshot(dest, "snap", dim=DIM)
    with pytest.raises(StorageCapabilityError, match="NoopVectorDBStorage"):
        await restore_workspace_snapshot(tables, target, gc)
    assert await target.chunks_vdb.get_vectors_by_ids(["chunk-1"]) == {
        "chunk-1": expected_vector(CHUNKS["chunk-1"]["content"])}


async def test_interrupted_restore_heals_on_rerun(tmp_path, monkeypatch):
    dest, _ = await seeded_snapshot(tmp_path)
    target_embedder = CountingEmbedder()
    target_embedder.armed = True
    target, gc = await build_target(tmp_path, target_embedder)
    original = target.chunks_vdb.index_done_callback
    calls = {"n": 0}

    async def flaky_flush():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated interruption mid-restore")
        await original()

    with monkeypatch.context() as m:
        m.setattr(target.chunks_vdb, "index_done_callback", flaky_flush)
        with pytest.raises(ExceptionGroup) as exc:
            await restore_workspace_snapshot(open_snapshot(dest, "snap", dim=DIM), target, gc)
    assert any("simulated interruption" in str(e) for e in exc.value.exceptions)
    assert calls["n"] >= 1

    await restore_workspace_snapshot(open_snapshot(dest, "snap", dim=DIM), target, gc)
    assert target_embedder.export_calls == 0
    await assert_restored_matches_seed(target)


async def register_workspace(tmp_path):
    """Bind the configuration anchor for the target working_dir.

    The rebuild tool never registers a workspace itself (ConfigurationStorage-
    Contract); a start binds the anchor. This is the minimal registration
    boundary the existing workspace tests use, not a full LightRAG."""
    initialize_share_data()
    working_dir = str(tmp_path / "restored")
    config = cs.create_configuration_storage(
        JsonKVStorage, global_config={"working_dir": working_dir, "workspace": ""},
        embedding_func=None)
    await config.initialize()
    try:
        await cs.bind_configuration_identity(
            config, working_dir=working_dir, backend="JsonKVStorage",
            container=json_config_path(working_dir, ""), workspace="")
    finally:
        await config.finalize()
    finalize_share_data()


def _cli_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "restored"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.setenv("EMBEDDING_MODEL", MODEL_NAME)
    monkeypatch.setenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
    monkeypatch.setenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
    monkeypatch.setenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
    monkeypatch.setattr("builtins.input", lambda *a: "yes")


def test_cli_restores_snapshot_into_fresh_workspace(tmp_path, monkeypatch, capsys):
    """Drive the real CLI; seeding/export run in their own event loop, then
    shared storage is reset so the CLI's fresh loop builds its own locks."""
    dest, _ = asyncio.run(seeded_snapshot(tmp_path))
    finalize_share_data()
    asyncio.run(register_workspace(tmp_path))
    _cli_env(monkeypatch, tmp_path)

    main(["--from-snapshot", dest, "--tag", "snap", "--batch-size", "2", "--concurrency", "1"])

    assert "Snapshot restore embedding calls: 0" in capsys.readouterr().out

    async def read_back():
        initialize_share_data()
        embedder = CountingEmbedder()
        embedder.armed = True
        target, _ = await build_target(tmp_path, embedder)
        await assert_restored_matches_seed(target)
        assert embedder.export_calls == 0
        for storage in (target.full_docs, target.text_chunks,
                        target.chunk_entity_relation_graph, target.entities_vdb,
                        target.relationships_vdb, target.chunks_vdb):
            await storage.finalize()
        finalize_share_data()

    asyncio.run(read_back())


@pytest.mark.parametrize("argv", [
    ["--from-snapshot", "unused"],
    ["--tag", "snap"],
    ["--batch-size", "2"],
], ids=["missing-tag", "tag-without-snapshot", "batch-without-snapshot"])
def test_cli_snapshot_flag_combinations_refused(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_cli_missing_tag_exits_nonzero_without_mutation(tmp_path, monkeypatch, capsys):
    dest, _ = asyncio.run(seeded_snapshot(tmp_path))
    finalize_share_data()
    asyncio.run(register_workspace(tmp_path))
    _cli_env(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc:
        main(["--from-snapshot", dest, "--tag", "absent"])

    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().out

    async def assert_no_data_restored():
        initialize_share_data()
        target, _ = await build_target(tmp_path, CountingEmbedder())
        assert await target.entities_vdb.get_exact_count() == 0
        assert await target.relationships_vdb.get_exact_count() == 0
        assert await target.chunks_vdb.get_exact_count() == 0
        assert [row async for row in target.text_chunks.iter_rows()] == []
        assert [row async for row in target.full_docs.iter_rows()] == []
        for storage in (target.full_docs, target.text_chunks,
                        target.chunk_entity_relation_graph, target.entities_vdb,
                        target.relationships_vdb, target.chunks_vdb):
            await storage.finalize()
        finalize_share_data()

    asyncio.run(assert_no_data_restored())
