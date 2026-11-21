"""Content-duplicate twins in the extraction subset: alias, don't abort.

Offline file-backend pins on the REAL pipeline: the product's enqueue
content-hash dedup demotes the second twin of an identical-text pair to a
``dup-<hash>`` FAILED pointer row and leaves the twin id without a status
row. The harness must accept that demotion ONLY with proof (pointer row +
content-hash match to a PROCESSED original), alias the twin into the
original's chunk attribution so both qrels-positive twins score, and keep
failing loud on every unproven missing status.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from lightrag.base import DocStatus
from lightrag.evaluation.hybrid_bench import arms, scoring
from lightrag.evaluation.hybrid_bench.dataset import BenchDataset, DatasetMeta
from lightrag.evaluation.hybrid_bench.fakes import FakeLLM
from lightrag.evaluation.hybrid_bench.rag_setup import (
    BenchConfig,
    build_llm_func,
    build_rag,
    ingest_dataset,
)
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding
from lightrag.utils_pipeline import compute_text_content_hash

pytestmark = pytest.mark.offline

SHARED_TEXT = "identical passage about optocouplers and signal isolation"
OTHER_TEXT = "a different passage about gears and torque"


def _meta() -> DatasetMeta:
    return DatasetMeta(
        domain="general",
        language="en",
        source="https://example.org",
        license="CC-BY-4.0",
        subset_protocol="test",
        counts={},
    )


def _twin_dataset() -> BenchDataset:
    # Both twins are qrels-positive, so both sit in the mandatory subset;
    # cap 2 extracts exactly the pair and leaves d3 chunks-only.
    return BenchDataset(
        name="twins",
        meta=_meta(),
        corpus={"d1": SHARED_TEXT, "d2": SHARED_TEXT, "d3": OTHER_TEXT},
        queries={"q1": SHARED_TEXT, "q2": SHARED_TEXT},
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
    )


def _unique_dataset(second_text: str = "text 1") -> BenchDataset:
    return BenchDataset(
        name="unique",
        meta=_meta(),
        corpus={"d0": "text 0", "d1": second_text},
        queries={"q1": "query one", "q2": "query two"},
        qrels={"q1": {"d0": 1}, "q2": {"d1": 1}},
    )


def _fake_llm() -> FakeLLM:
    return FakeLLM(
        {
            SHARED_TEXT: [
                {
                    "entity_name": "optocoupler",
                    "entity_type": "concept",
                    "description": "an optocoupler isolates signals",
                }
            ],
            OTHER_TEXT: [
                {
                    "entity_name": "gear",
                    "entity_type": "concept",
                    "description": "a gear transmits torque",
                }
            ],
            "text 0": [
                {
                    "entity_name": "alpha",
                    "entity_type": "concept",
                    "description": "alpha appears in passage d0",
                }
            ],
            "text 1": [
                {
                    "entity_name": "beta",
                    "entity_type": "concept",
                    "description": "beta appears in passage d1",
                }
            ],
        }
    )


def _chunk_id(text: str) -> str:
    return compute_mdhash_id(sanitize_text_for_encoding(text), prefix="chunk-")


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


def _drop_from_ainsert(rag, drop_id: str) -> list[list[str]]:
    """Fault injection: ainsert silently skips one id (the silent-skip defect
    class the fail-loud check guards). Records every batch it receives."""
    real = rag.ainsert
    calls: list[list[str]] = []

    async def dropping(input, ids=None, file_paths=None, **kwargs):
        calls.append(list(ids))
        keep = [i for i, cid in enumerate(ids) if cid != drop_id]
        if not keep:
            return
        await real(
            [input[i] for i in keep],
            ids=[ids[i] for i in keep],
            file_paths=[file_paths[i] for i in keep],
            **kwargs,
        )

    rag.ainsert = dropping
    return calls


async def _statuses(rag):
    return await rag.doc_status.get_docs_by_statuses(
        [DocStatus.PROCESSED, DocStatus.FAILED], strict=True
    )


async def test_batch_duplicate_twin_is_aliased_to_processed_original(tmp_path):
    workspace = f"twin_{uuid4().hex[:8]}"
    rag, config = await _build(tmp_path, workspace, cap=2)
    try:
        result = await ingest_dataset(rag, _twin_dataset(), config)
        assert result.duplicate_aliased_doc_ids == ["d2"]
        assert result.persistently_failed_doc_ids == []
        assert result.recovered_doc_ids == []
        assert result.recovery_rounds_used == 0

        statuses = await _statuses(rag)
        assert statuses["d1"].status == DocStatus.PROCESSED
        assert "d2" not in statuses
        pointers = [
            status
            for status in statuses.values()
            if status.status == DocStatus.FAILED and status.metadata.get("is_duplicate")
        ]
        assert len(pointers) == 1
        assert pointers[0].metadata["duplicate_kind"] == "content_hash"
        assert pointers[0].metadata["original_doc_id"] == "d1"
        assert pointers[0].file_path == "d2"

        winner_chunks = statuses["d1"].chunks_list
        assert len(winner_chunks) == 1
        assert result.chunk_doc_map[winner_chunks[0]] == ["d1", "d2"]
        assert result.chunk_count == 2
        assert result.chunk_doc_map[_chunk_id(OTHER_TEXT)] == ["d3"]
    finally:
        await rag.finalize_storages()


async def test_both_twin_positives_score_on_the_shared_chunk(tmp_path):
    workspace = f"twin_{uuid4().hex[:8]}"
    rag, config = await _build(tmp_path, workspace, cap=2)
    try:
        ingest = await ingest_dataset(rag, _twin_dataset(), config)
        shared_chunks = [
            chunk_id
            for chunk_id, aliases in ingest.chunk_doc_map.items()
            if aliases == ["d1", "d2"]
        ]
        assert len(shared_chunks) == 1
        embedding = await arms.embed_query(rag, SHARED_TEXT)
        hits = await arms.retrieve_chunks(
            rag, SHARED_TEXT, "dense", 10, embedding, ingest.chunk_doc_map
        )
        assert hits[0].chunk_id == shared_chunks[0]
        assert hits[0].doc_ids == ("d1", "d2")
        docs = arms.doc_id_sequence(hits)
        scored, _, _ = scoring.score_chunk_arms(
            _twin_dataset(), {"dense": {"q1": docs, "q2": docs}}, 10
        )
        assert scored["dense"]["q1"]["recall"] == 1.0
        assert scored["dense"]["q2"]["recall"] == 1.0
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize(
    "case",
    ["genuine-silent-skip", "stale-pointer-hash-mismatch"],
    ids=["no-pointer", "hash-mismatch"],
)
async def test_unproven_missing_status_still_fails_loud(tmp_path, case):
    workspace = f"twin_{uuid4().hex[:8]}"
    rag, config = await _build(tmp_path, workspace, cap=2)
    try:
        if case == "stale-pointer-hash-mismatch":
            # Real product demotion first: d1 is d0's content twin, so the
            # pipeline writes a genuine dup pointer naming PROCESSED d0.
            first = await ingest_dataset(rag, _unique_dataset("text 0"), config)
            assert first.duplicate_aliased_doc_ids == ["d1"]
            statuses = await _statuses(rag)
            pointers = [
                status
                for status in statuses.values()
                if status.status == DocStatus.FAILED
                and status.metadata.get("is_duplicate")
            ]
            assert len(pointers) == 1
            assert pointers[0].metadata["original_doc_id"] == "d0"
            # The corpus text now differs from what the pointer claims: the
            # hash proof must reject the stale pointer, not accept it.
            changed = _unique_dataset("a completely different passage")
            assert compute_text_content_hash(
                sanitize_text_for_encoding(changed.corpus["d1"])
            ) != compute_text_content_hash(
                sanitize_text_for_encoding(changed.corpus["d0"])
            )
            dataset = changed
        else:
            dataset = _unique_dataset()
        calls = _drop_from_ainsert(rag, "d1")
        with pytest.raises(RuntimeError, match="ingested document 'd1' has no status"):
            await ingest_dataset(rag, dataset, config)
        assert calls, "the silent-skip fault never fired"
        assert "d1" in calls[0]
        statuses = await _statuses(rag)
        assert "d1" not in statuses
    finally:
        await rag.finalize_storages()


async def test_resume_rebuilds_identical_twin_alias_map(tmp_path):
    workspace = f"twin_{uuid4().hex[:8]}"
    rag1, config1 = await _build(tmp_path, workspace, cap=2)
    try:
        first = await ingest_dataset(rag1, _twin_dataset(), config1)
    finally:
        await rag1.finalize_storages()

    llm_calls = []

    async def refuse_llm(prompt, *args, **kwargs):
        llm_calls.append(prompt)
        raise AssertionError("resume must not call the LLM")

    rag2, config2 = await _build(tmp_path, workspace, cap=2, llm=refuse_llm)
    try:
        second = await ingest_dataset(rag2, _twin_dataset(), config2)
        assert second.resumed is True
        assert second.chunk_doc_map == first.chunk_doc_map
        assert second.duplicate_aliased_doc_ids == ["d2"]
        assert second.persistently_failed_doc_ids == []
        assert llm_calls == []
    finally:
        await rag2.finalize_storages()
