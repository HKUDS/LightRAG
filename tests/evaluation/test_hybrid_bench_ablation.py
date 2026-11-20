"""Ablation combinations run the COMBINED mix-mode retrieval path with the
two storage dispatches independently controlled, and score the final
context. Pins: dispatch override shape/restoration, kg_query call contract,
final-context scoring, and the CLI wiring (offline, file backends + fakes).
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from lightrag.evaluation.hybrid_bench import arms, cli
from lightrag.evaluation.hybrid_bench.dataset import BenchDataset, DatasetMeta
from lightrag.evaluation.hybrid_bench.entity_truth import derive_entity_truth
from lightrag.evaluation.hybrid_bench.fakes import FakeLLM
from lightrag.evaluation.hybrid_bench.rag_setup import (
    BenchConfig,
    build_llm_func,
    build_rag,
    ingest_dataset,
)
from lightrag.kg.shared_storage import initialize_share_data

pytestmark = pytest.mark.offline

TEXT_A = "optocoupler isolates signals in robot joints"
TEXT_B = "planetary gears transmit large torque"


def _dataset() -> BenchDataset:
    return BenchDataset(
        name="abl",
        meta=DatasetMeta(
            domain="general",
            language="en",
            source="https://example.org",
            license="CC-BY-4.0",
            subset_protocol="test",
            counts={},
        ),
        corpus={"d1": TEXT_A, "d2": TEXT_B},
        queries={"q1": TEXT_A},
        qrels={"q1": {"d1": 1}},
    )


class _CapableVDB:
    """Fake vector storage declaring hybrid capability at class level."""

    supports_hybrid_query = True


class _IncapableVDB:
    """Fake vector storage without hybrid capability (file-backend shape)."""


def _fake_rag(vdb_cls):
    return SimpleNamespace(
        entities_vdb=vdb_cls(),
        chunks_vdb=vdb_cls(),
        relationships_vdb=vdb_cls(),
        chunk_entity_relation_graph=object(),
        text_chunks=object(),
        llm_response_cache=object(),
        _build_global_config=lambda: {},
    )


def _recording_kg_query(calls, raw_data):
    async def fake(
        query,
        graph,
        entities_vdb,
        relationships_vdb,
        text_chunks,
        param,
        global_config,
        hashing_kv=None,
        chunks_vdb=None,
        **kwargs,
    ):
        calls.append(
            {
                "query": query,
                "param": param,
                "entity_flag": getattr(entities_vdb, "supports_hybrid_query", False),
                "chunk_flag": getattr(chunks_vdb, "supports_hybrid_query", False),
                "entity_own_attr": "supports_hybrid_query" in vars(entities_vdb),
                "chunk_own_attr": "supports_hybrid_query" in vars(chunks_vdb),
            }
        )
        return SimpleNamespace(raw_data=raw_data)

    return fake


@pytest.mark.parametrize(
    ("combo", "entity_hybrid", "chunk_hybrid", "entity_flag", "chunk_flag"),
    [
        ("entity_only", True, False, True, False),
        ("chunk_only", False, True, False, True),
        ("both", True, True, True, True),
    ],
)
@pytest.mark.asyncio
async def test_combined_dispatch_overrides_and_final_context_scoring(
    monkeypatch, combo, entity_hybrid, chunk_hybrid, entity_flag, chunk_flag
):
    rag = _fake_rag(_CapableVDB)
    calls = []
    raw_data = {
        "data": {
            "chunks": [{"chunk_id": "chunk-x", "content": "c"}],
            "entities": [{"entity_name": "E1"}],
        }
    }
    monkeypatch.setattr(arms, "kg_query", _recording_kg_query(calls, raw_data))
    hits, entities = await arms.retrieve_combined(
        rag, "query text", entity_hybrid, chunk_hybrid, 7, {"chunk-x": ["d2", "d1"]}
    )
    assert len(calls) == 1
    call = calls[0]
    assert call["entity_flag"] is entity_flag
    assert call["chunk_flag"] is chunk_flag
    # Only a DISABLED dispatch carries an instance override; an enabled one
    # reads the untouched class flag.
    assert call["entity_own_attr"] is (not entity_flag)
    assert call["chunk_own_attr"] is (not chunk_flag)
    param = call["param"]
    assert param.mode == "mix"
    assert param.enable_hybrid is True
    assert param.only_need_context is True
    assert param.enable_rerank is False
    assert param.hl_keywords == ["query text"]
    assert param.ll_keywords == ["query text"]
    assert param.top_k == 7
    assert param.chunk_top_k == 7
    # Overrides are removed on exit: the class flag survives untouched.
    assert "supports_hybrid_query" not in vars(rag.entities_vdb)
    assert "supports_hybrid_query" not in vars(rag.chunks_vdb)
    assert type(rag.entities_vdb).supports_hybrid_query is True
    # Scoring sees the FINAL context, aliases expanded in sorted order.
    assert [hit.doc_ids for hit in hits] == [("d1", "d2")]
    assert entities == ["E1"]


@pytest.mark.asyncio
async def test_incapable_backends_are_never_enabled(monkeypatch):
    rag = _fake_rag(_IncapableVDB)
    calls = []
    monkeypatch.setattr(arms, "kg_query", _recording_kg_query(calls, {"data": {}}))
    hits, entities = await arms.retrieve_combined(rag, "q", True, True, 5, {})
    assert calls[0]["entity_flag"] is False
    assert calls[0]["chunk_flag"] is False
    assert calls[0]["entity_own_attr"] is False
    assert calls[0]["chunk_own_attr"] is False
    assert hits == []
    assert entities == []


@pytest.mark.parametrize(
    "result",
    [None, SimpleNamespace(raw_data=None), SimpleNamespace(raw_data={})],
    ids=["no-result", "no-raw-data", "empty-raw-data"],
)
@pytest.mark.asyncio
async def test_absent_context_scores_empty(monkeypatch, result):
    rag = _fake_rag(_CapableVDB)

    async def fake(*args, **kwargs):
        return result

    monkeypatch.setattr(arms, "kg_query", fake)
    hits, entities = await arms.retrieve_combined(rag, "q", True, True, 5, {})
    assert hits == []
    assert entities == []


async def _build_ingested(tmp_path):
    initialize_share_data(workers=1)
    workspace = f"abl_{uuid4().hex[:8]}"
    config = BenchConfig(
        storage="file",
        working_dir=str(tmp_path / "working"),
        workspace=workspace,
        embedding_backend="fake-hash",
        embedding_dim=32,
        llm_backend="fake",
        extraction_cap=2,
        max_gleaning=0,
        cosine_threshold=0.1,
    )
    dataset = _dataset()
    fake_llm = FakeLLM(
        {
            TEXT_A: [
                {
                    "entity_name": "optocoupler",
                    "entity_type": "concept",
                    "description": "isolates signals",
                }
            ],
            TEXT_B: [
                {
                    "entity_name": "gear",
                    "entity_type": "concept",
                    "description": "transmits torque",
                }
            ],
        }
    )
    rag = await build_rag(config, build_llm_func(config, fake_llm))
    ingest = await ingest_dataset(rag, dataset, config)
    return rag, config, dataset, ingest


@pytest.mark.asyncio
async def test_real_combined_path_returns_final_chunks_and_entities(tmp_path):
    rag, config, dataset, ingest = await _build_ingested(tmp_path)
    try:
        hits, entities = await arms.retrieve_combined(
            rag, TEXT_A, True, True, config.top_k, ingest.chunk_doc_map
        )
        assert hits, "the matching passage must appear in the final context"
        assert hits[0].doc_ids == ("d1",)
        assert "optocoupler" in entities
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_cli_ablation_drives_the_combined_path_per_combination(
    tmp_path, monkeypatch
):
    rag, config, dataset, ingest = await _build_ingested(tmp_path)
    try:
        truth = await derive_entity_truth(
            rag.chunk_entity_relation_graph,
            rag.text_chunks,
            {qid: dataset.positives(qid) for qid in dataset.queries},
            entity_chunks=rag.entity_chunks,
        )
        real = arms.retrieve_combined
        recorded: list[tuple[bool, bool]] = []

        async def recording(rag_, query, entity_hybrid, chunk_hybrid, *rest):
            recorded.append((entity_hybrid, chunk_hybrid))
            return await real(rag_, query, entity_hybrid, chunk_hybrid, *rest)

        monkeypatch.setattr(cli.arms_mod, "retrieve_combined", recording)
        _, _, _, _, b5 = await cli._run_arms(
            rag, dataset, config, ingest, truth, {"ablation"}
        )
        assert recorded == [(True, False), (False, True), (True, True)]
        assert set(b5) == {"entity_only", "chunk_only", "both"}
        for entries in b5.values():
            assert entries["q1"]["recall"] == 1.0
    finally:
        await rag.finalize_storages()
