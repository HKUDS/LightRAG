"""Offline regressions for complete Nano coverage and reverse inventories."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from nano_vectordb import NanoVectorDB

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.tools.vector_census import audit_vector_census
from lightrag.utils import EmbeddingFunc, compute_mdhash_id, make_relation_vdb_ids

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def shared():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def no_embedding(*args, **kwargs):
    raise AssertionError("Census must never embed")


async def nano(tmp_path, namespace, ids=()):
    workspace = tmp_path / "isolated"
    workspace.mkdir(exist_ok=True)
    path = workspace / f"vdb_{namespace}.json"
    db = NanoVectorDB(2, storage_file=str(path))
    if ids:
        db.upsert(
            [
                {"__id__": key, "__vector__": np.array([1, 1], dtype=np.float32)}
                for key in ids
            ]
        )
    db.save()
    storage = NanoVectorDBStorage(
        namespace=namespace,
        workspace="isolated",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 2,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(embedding_dim=2, func=no_embedding),
    )
    await storage.initialize()
    return storage


async def chunks(tmp_path, data):
    workspace = tmp_path / "isolated"
    workspace.mkdir(exist_ok=True)
    (workspace / "kv_store_text_chunks.json").write_text(json.dumps(data))
    storage = JsonKVStorage(
        namespace="text_chunks",
        workspace="isolated",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


async def test_equal_totals_different_ids_and_legacy_relations(tmp_path):
    entities = await nano(tmp_path, "entities", ["unrelated"])
    relation_ids = make_relation_vdb_ids("A", "B")
    relations = await nano(tmp_path, "relationships", relation_ids)
    vectors = await nano(tmp_path, "chunks", ["c1"])
    source = await chunks(
        tmp_path, {"c1": {"full_doc_id": "doc1"}, "c2": {"full_doc_id": "doc2"}}
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    result = await audit_vector_census(
        [{"id": "A", "source_id": "c1"}],
        [{"source": "A", "target": "B"}, {"source": "B", "target": "A"}],
        entities,
        relations,
        chunks_vdb=vectors,
        text_chunks=source,
    )
    entity = result["targets"]["entities"]
    assert entity["source_total"] == entity["vector_total"] == 1
    assert entity["missing"] == entity["reverse_excess"] == 1
    assert entity["missing_by_document"] == {"doc1": 1}
    rel = result["targets"]["relationships"]
    assert (
        rel["source_total"],
        rel["matched_physical"],
        rel["legacy_duplicates"],
        rel["reverse_excess"],
    ) == (1, 2, 1, 0)
    assert result["targets"]["chunks"]["missing_by_document"] == {"doc2": 1}
    assert result["complete"] and result["coverage_gaps"]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*.json")}


@pytest.mark.parametrize(
    "buffer",
    ["_pending_upserts", "_pending_deletes", "_unsaved_upserts", "_unsaved_deletes"],
)
async def test_buffers_refused_without_flushing(tmp_path, buffer):
    storage = await nano(tmp_path, "entities")
    setattr(storage, buffer, {"pending": object()})
    storage.index_done_callback = AsyncMock(
        side_effect=AssertionError("must not flush")
    )
    result = await audit_vector_census([], [], storage, None)
    assert result["targets"]["entities"] == {
        "status": "unavailable",
        "reason": "pending_changes",
    }
    storage.index_done_callback.assert_not_called()


@pytest.mark.parametrize("rows", [[{}], [{"__id__": "x"}, {"__id__": "x"}]])
async def test_malformed_inventory_is_not_zero(tmp_path, rows):
    storage = await nano(tmp_path, "entities")
    snapshot = await storage.client_storage
    snapshot["data"] = rows
    result = await audit_vector_census([], [], storage, None)
    assert result["targets"]["entities"]["reason"] == "missing_or_duplicate_vector_id"
    assert "vector_total" not in result["targets"]["entities"]


async def test_empty_missing_failed_and_changed_are_distinct(tmp_path, monkeypatch):
    empty = await nano(tmp_path, "entities")
    result = await audit_vector_census([], [], empty, None)
    assert result["targets"]["entities"]["vector_total"] == 0
    before = empty._stat_fingerprint()
    samples = iter([before, ((before[0][0] + 1, before[0][1]),)])
    monkeypatch.setattr(empty, "_stat_fingerprint", lambda: next(samples))
    result = await audit_vector_census([], [], empty, None)
    assert result["targets"]["entities"]["status"] == "inconclusive"
    monkeypatch.undo()
    empty._get_client = AsyncMock(side_effect=OSError("unreadable"))
    result = await audit_vector_census([], [], empty, None)
    assert result["targets"]["entities"]["status"] == "unavailable"
    Path(empty._client_file_name).unlink()
    result = await audit_vector_census([], [], empty, None)
    assert result["targets"]["entities"]["reason"] == "namespace_missing_or_unreadable"


async def test_checked_loader_rejects_corrupt_file(tmp_path):
    storage = await nano(tmp_path, "entities")
    Path(storage._client_file_name).write_text("invalid json")
    with pytest.raises(json.JSONDecodeError):
        storage._build_client()
    # A stale single-process reader is not certified after an out-of-band change.
    result = await audit_vector_census([], [], storage, None)
    assert result["targets"]["entities"]["status"] == "inconclusive"


async def test_unsupported_noop_and_incompatible(tmp_path):
    result = await audit_vector_census(
        [],
        [],
        SimpleNamespace(),
        SimpleNamespace(persists_vectors=False),
        incompatible={"chunks": "wrong model"},
    )
    assert result["targets"]["entities"]["status"] == "unavailable"
    assert result["targets"]["relationships"]["status"] == "not_applicable"
    assert result["targets"]["chunks"]["status"] == "incompatible"


async def test_tracking_precedence_and_incomplete_source_batch(tmp_path):
    storage = await nano(tmp_path, "entities")
    source = await chunks(
        tmp_path, {"c1": {"full_doc_id": "doc1"}, "c2": {"full_doc_id": "doc2"}}
    )
    tracking = SimpleNamespace(
        get_by_ids=AsyncMock(return_value=[{"chunk_ids": ["c2"]}])
    )
    result = await audit_vector_census(
        [{"id": "A", "source_id": "c1"}],
        [],
        storage,
        None,
        text_chunks=source,
        entity_chunks=tracking,
    )
    assert result["targets"]["entities"]["missing_by_document"] == {"doc2": 1}
    tracking.get_by_ids.return_value = [{"chunk_ids": []}]
    result = await audit_vector_census(
        [{"id": "A", "source_id": "c1"}],
        [],
        storage,
        None,
        text_chunks=source,
        entity_chunks=tracking,
    )
    assert result["targets"]["entities"]["unattributed_missing"] == 1
    source.get_by_ids = AsyncMock(return_value=[])
    result = await audit_vector_census([], [], storage, None, text_chunks=source)
    assert result["targets"]["chunks"]["status"] == "unavailable"


async def test_legacy_api_and_census_share_graph_enumeration(tmp_path):
    from lightrag.tools.rebuild_vdb import check_vdb_consistency

    eid = compute_mdhash_id("A", prefix="ent-")
    entities = await nano(tmp_path, "entities", [eid])
    relations = await nano(tmp_path, "relationships")
    cv = await nano(tmp_path, "chunks")
    source = await chunks(tmp_path, {})
    graph = SimpleNamespace(
        get_all_nodes=AsyncMock(return_value=[{"id": "A"}]),
        get_all_edges=AsyncMock(return_value=[]),
    )
    legacy = await check_vdb_consistency(graph, entities, relations)
    assert legacy["consistent"] and "census" not in legacy
    graph.get_all_nodes.reset_mock()
    graph.get_all_edges.reset_mock()
    result = await check_vdb_consistency(
        graph,
        entities,
        relations,
        include_census=True,
        chunks_vdb=cv,
        text_chunks=source,
    )
    assert result["census"]["complete"]
    graph.get_all_nodes.assert_awaited_once()
    graph.get_all_edges.assert_awaited_once()


@pytest.mark.parametrize("mismatch", [False, True])
async def test_check_cli_preserves_legacy_probe_with_census(tmp_path, capsys, mismatch):
    from lightrag.tools.rebuild_vdb import RebuildTool, check_vdb_consistency

    entities = await nano(tmp_path, "entities")
    relationships = SimpleNamespace(get_by_ids=AsyncMock(return_value=[None, None]))
    graph = SimpleNamespace(
        get_all_nodes=AsyncMock(return_value=[{"id": "Missing entity"}]),
        get_all_edges=AsyncMock(return_value=[{"source": "A", "target": "B"}]),
    )
    report = await check_vdb_consistency(
        graph,
        entities,
        relationships,
        include_census=True,
        incompatible={"relationships": "wrong model"} if mismatch else None,
    )
    RebuildTool().print_check_report(report)
    output = capsys.readouterr().out
    assert "entities: source=1, vectors=0, missing=1" in output
    assert "Legacy forward probe" in output
    assert "not a certified census" in output
    assert "Missing entities (first few):" in output
    assert "    - Missing entity" in output
    assert output.index("Vector census") < output.index("Legacy forward probe")
    if mismatch:
        assert "relationships: incompatible (wrong model)" in output
        assert "Embedding space mismatch" in output
        assert "(menu options 2-4) is required, not optional." in output
        relationships.get_by_ids.assert_not_awaited()
    else:
        assert (
            "relationships: unavailable (complete_population_not_verified_for_"
            in output
        )
        assert "Missing relations (first few):" in output
        assert "    - A ~ B" in output
