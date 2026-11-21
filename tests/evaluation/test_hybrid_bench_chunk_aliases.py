"""Content-hash chunk ids keep EVERY corpus alias (duplicate-content docs).

Offline file-backend pins: two documents with identical text share one
content-addressed chunk; the attribution map must keep both corpus ids and
scoring must count a query positive on EITHER alias, never score it as a
miss because a later insert overwrote the earlier one.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from lightrag.evaluation.hybrid_bench import arms, scoring
from lightrag.evaluation.hybrid_bench.dataset import BenchDataset, DatasetMeta
from lightrag.evaluation.hybrid_bench.fakes import FakeLLM
from lightrag.evaluation.hybrid_bench.metrics import recall_at_k
from lightrag.evaluation.hybrid_bench.rag_setup import (
    BenchConfig,
    build_llm_func,
    build_rag,
    ingest_dataset,
)
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

pytestmark = pytest.mark.offline

SHARED_TEXT = "identical passage about optocouplers and signal isolation"
OTHER_TEXT = "a different passage about gears and torque"


def _dataset() -> BenchDataset:
    return BenchDataset(
        name="dup",
        meta=DatasetMeta(
            domain="general",
            language="en",
            source="https://example.org",
            license="CC-BY-4.0",
            subset_protocol="test",
            counts={},
        ),
        corpus={"d1": SHARED_TEXT, "d2": SHARED_TEXT, "d3": OTHER_TEXT},
        queries={"q1": SHARED_TEXT},
        # The positive is d1, the FIRST alias: a last-writer-wins map keeps
        # d2 and would score this query as a miss.
        qrels={"q1": {"d1": 1}},
    )


def _dataset_extracted() -> BenchDataset:
    base = _dataset()
    # Positive d3 (unique text) so cap 1 extracts d3 while the duplicate
    # pair d1/d2 stays in the chunks-only remainder. Extraction chunk ids
    # are doc-scoped (hash of "doc_id:content"), so alias sharing happens
    # between chunks-only documents; this fixture exercises BOTH map
    # construction sites in one ingest.
    return BenchDataset(
        name=base.name,
        meta=base.meta,
        corpus=base.corpus,
        queries=base.queries,
        qrels={"q1": {"d3": 1}},
    )


def _chunk_id(text: str) -> str:
    return compute_mdhash_id(sanitize_text_for_encoding(text), prefix="chunk-")


def _fake_llm() -> FakeLLM:
    spec = {
        "entity_name": "optocoupler",
        "entity_type": "concept",
        "description": "an optocoupler isolates signals",
    }
    other = {
        "entity_name": "gear",
        "entity_type": "concept",
        "description": "a gear transmits torque",
    }
    return FakeLLM({SHARED_TEXT: [spec], OTHER_TEXT: [other]})


async def _build(tmp_path, workspace, cap, llm=None):
    initialize_share_data(workers=1)
    config = BenchConfig(
        storage="file",
        working_dir=str(tmp_path / "working"),
        workspace=workspace,
        embedding_backend="fake-hash",
        embedding_dim=32,
        llm_backend="fake",
        extraction_cap=cap,
        max_gleaning=0,
        cosine_threshold=0.1,
    )
    rag = await build_rag(config, build_llm_func(config, llm or _fake_llm()))
    return rag, config


@pytest.mark.asyncio
async def test_ingest_keeps_all_aliases_of_a_shared_chunk(tmp_path):
    workspace = f"dup_{uuid4().hex[:8]}"
    rag, config = await _build(tmp_path, workspace, cap=0)
    try:
        result = await ingest_dataset(rag, _dataset(), config)
        shared, other = _chunk_id(SHARED_TEXT), _chunk_id(OTHER_TEXT)
        assert result.chunk_count == 2
        assert set(result.chunk_doc_map) == {shared, other}
        assert result.chunk_doc_map[shared] == ["d1", "d2"]
        assert result.chunk_doc_map[other] == ["d3"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_ingest_aliases_across_extraction_and_remainder(tmp_path):
    workspace = f"dup_{uuid4().hex[:8]}"
    rag, config = await _build(tmp_path, workspace, cap=1)
    try:
        result = await ingest_dataset(rag, _dataset_extracted(), config)
        shared = _chunk_id(SHARED_TEXT)
        assert result.chunk_count == 2
        assert result.chunk_doc_map[shared] == ["d1", "d2"]
        extracted = [
            chunk_id
            for chunk_id, aliases in result.chunk_doc_map.items()
            if aliases == ["d3"]
        ]
        assert len(extracted) == 1
        # The extraction path hashes "doc_id:content", not content alone.
        assert extracted[0] != _chunk_id(OTHER_TEXT)
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize(
    ("cap", "dataset_fn"),
    [(0, _dataset), (1, _dataset_extracted)],
    ids=["chunks-only", "one-extracted"],
)
@pytest.mark.asyncio
async def test_resume_rebuilds_every_alias(tmp_path, cap, dataset_fn):
    workspace = f"dup_{uuid4().hex[:8]}"
    rag1, config1 = await _build(tmp_path, workspace, cap)
    try:
        first = await ingest_dataset(rag1, dataset_fn(), config1)
    finally:
        await rag1.finalize_storages()

    llm_calls = []

    async def refuse_llm(prompt, *args, **kwargs):
        llm_calls.append(prompt)
        raise AssertionError("resume must not call the LLM")

    rag2, config2 = await _build(tmp_path, workspace, cap, llm=refuse_llm)
    try:
        second = await ingest_dataset(rag2, dataset_fn(), config2)
        assert second.resumed is True
        assert second.chunk_doc_map == first.chunk_doc_map
        assert llm_calls == []
    finally:
        await rag2.finalize_storages()


@pytest.mark.asyncio
async def test_scoring_counts_a_positive_on_the_first_alias(tmp_path):
    workspace = f"dup_{uuid4().hex[:8]}"
    rag, config = await _build(tmp_path, workspace, cap=0)
    try:
        ingest = await ingest_dataset(rag, _dataset(), config)
        embedding = await arms.embed_query(rag, SHARED_TEXT)
        hits = await arms.retrieve_chunks(
            rag, SHARED_TEXT, "dense", 10, embedding, ingest.chunk_doc_map
        )
        assert hits[0].chunk_id == _chunk_id(SHARED_TEXT)
        assert hits[0].doc_ids == ("d1", "d2")
        docs = arms.doc_id_sequence(hits)
        assert {"d1", "d2"} <= set(docs)
        assert recall_at_k(docs, {"d1": 1}, 10) == 1.0
        scored, _, _ = scoring.score_chunk_arms(_dataset(), {"dense": {"q1": docs}}, 10)
        assert scored["dense"]["q1"]["recall"] == 1.0
    finally:
        await rag.finalize_storages()


def test_resolve_hits_sorts_aliases_and_fails_loud_on_unmapped_chunks():
    rows = [{"chunk_id": "chunk-x", "content": "c"}]
    hits = arms._resolve_hits(rows, {"chunk-x": ["d2", "d1"]})
    assert hits[0].doc_ids == ("d1", "d2")
    with pytest.raises(RuntimeError, match="no corpus attribution"):
        arms._resolve_hits(rows, {})
