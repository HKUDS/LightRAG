"""Resume contract pins, offline (file backends, real pipeline + stubs).

Each resume test drives the REAL ingest pipeline twice against the SAME
workspace with a fresh LightRAG instance in between (the process-restart
shape of an OOM kill). Extraction is a per-document recording stub; a stub
that must never be called records the call as well as raising. FAILED
documents re-enter only through the sanctioned commit + drive sequence,
observed via a delegating spy on ``commit_manual_retry_request``.
"""

from __future__ import annotations

import json
import re
from uuid import uuid4

import pytest

from lightrag.evaluation.hybrid_bench import cli, rag_setup
from lightrag.evaluation.hybrid_bench.dataset import BenchDataset, DatasetMeta
from lightrag.evaluation.hybrid_bench.rag_setup import (
    BenchConfig,
    build_rag,
    ingest_dataset,
)
from lightrag.kg.shared_storage import initialize_share_data

pytestmark = pytest.mark.offline


def _dataset() -> BenchDataset:
    return BenchDataset(
        name="tiny",
        meta=DatasetMeta(
            domain="general",
            language="en",
            source="https://example.org",
            license="CC-BY-4.0",
            subset_protocol="test",
            counts={},
        ),
        corpus={"d0": "text 0", "d1": "text 1", "d2": "text 2"},
        queries={"q1": "query one", "q2": "query two"},
        qrels={"q1": {"d0": 1}, "q2": {"d1": 1}},
    )


class _StubExtract:
    """Extraction stub recording every document, failing the first N calls."""

    def __init__(self, fail_plan: dict[str, int] | None = None):
        self.fail_plan = dict(fail_plan or {})
        self.doc_calls: list[str] = []

    async def __call__(self, chunks, *args, **kwargs):
        doc_id = next(iter(chunks.values()))["full_doc_id"]
        self.doc_calls.append(doc_id)
        remaining = self.fail_plan.get(doc_id, 0)
        if remaining > 0:
            self.fail_plan[doc_id] = remaining - 1
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


class _RefuseExtract:
    """Extraction fake for a resume that must not re-extract anything."""

    def __init__(self):
        self.doc_calls: list[str] = []

    async def __call__(self, chunks, *args, **kwargs):
        doc_id = next(iter(chunks.values()))["full_doc_id"]
        self.doc_calls.append(doc_id)
        raise AssertionError("resume must not re-extract a completed document")


def _refuse_custom_kg(rag) -> list[str]:
    calls: list[str] = []

    async def refuse(custom_kg, full_doc_id=None):
        calls.append(full_doc_id)
        raise AssertionError("resume must not re-ingest a complete chunk")

    rag.ainsert_custom_kg = refuse
    return calls


def _refuse_ainsert(rag) -> list[list[str]]:
    calls: list[list[str]] = []

    async def refuse(input, ids=None, file_paths=None, **kwargs):
        calls.append(list(ids or []))
        raise AssertionError("fully-ingested resume must not submit documents")

    rag.ainsert = refuse
    return calls


def _record_ainsert(rag) -> list[list[str]]:
    """Delegate to the real ainsert, recording the submitted document ids."""
    real = rag.ainsert
    calls: list[list[str]] = []

    async def recording(input, ids=None, file_paths=None, **kwargs):
        calls.append(list(ids or []))
        return await real(input, ids=ids, file_paths=file_paths, **kwargs)

    rag.ainsert = recording
    return calls


async def _build(tmp_path, workspace, extract, extraction_cap=2, **overrides):
    initialize_share_data(workers=1)
    config = BenchConfig(
        storage="file",
        working_dir=str(tmp_path / "working"),
        workspace=workspace,
        embedding_backend="fake-hash",
        embedding_dim=32,
        extraction_cap=extraction_cap,
        max_gleaning=0,
        **overrides,
    )
    llm_calls = []

    async def refuse_llm(prompt, *args, **kwargs):
        llm_calls.append(prompt)
        raise AssertionError("resume pin must not call the LLM")

    rag = await build_rag(config, refuse_llm)
    rag._process_extract_entities = extract
    return rag, config, llm_calls


@pytest.fixture
def commit_spy(monkeypatch):
    """Record request ids while DELEGATING to the real sanctioned commit."""
    real = rag_setup.commit_manual_retry_request
    calls = []

    async def recording(pipeline_status, lock, ingress, request_id, state):
        calls.append(request_id)
        return await real(pipeline_status, lock, ingress, request_id, state)

    monkeypatch.setattr(rag_setup, "commit_manual_retry_request", recording)
    return calls


async def test_partial_resume_skips_processed_and_recovers_failed(tmp_path, commit_spy):
    workspace = f"resume_{uuid4().hex[:8]}"
    stub1 = _StubExtract({"d1": 1})
    rag1, config, llm1 = await _build(
        tmp_path, workspace, stub1, ingest_recovery_rounds=0
    )
    try:
        with pytest.raises(RuntimeError, match="document 'd1' failed ingestion"):
            await ingest_dataset(rag1, _dataset(), config)
        assert sorted(stub1.doc_calls) == ["d0", "d1"]
    finally:
        await rag1.finalize_storages()

    stub2 = _StubExtract()
    rag2, config2, llm2 = await _build(tmp_path, workspace, stub2)
    custom_kg_calls = _refuse_custom_kg(rag2)
    ainsert_ids = _record_ainsert(rag2)
    try:
        result = await ingest_dataset(rag2, _dataset(), config2)
        # d0 PROCESSED -> never re-extracted; d1 FAILED -> exactly one
        # sanctioned retry; d2 chunk complete -> never re-ingested.
        assert ainsert_ids == [["d1"]]
        assert stub2.doc_calls == ["d1"]
        assert custom_kg_calls == []
        assert len(commit_spy) == 1
        assert result.resumed is True
        assert result.recovery_rounds_used == 1
        assert result.recovered_doc_ids == ["d1"]
        assert result.persistently_failed_doc_ids == []
        assert result.chunk_count == 3
        assert set(result.chunk_doc_map.values()) == {"d0", "d1", "d2"}
        assert llm1 == []
        assert llm2 == []
    finally:
        await rag2.finalize_storages()


async def test_interrupted_pending_rows_are_continued(tmp_path):
    workspace = f"resume_{uuid4().hex[:8]}"
    rag1, config, _ = await _build(tmp_path, workspace, _StubExtract())
    try:
        # Enqueue without processing: the durable state an OOM kill leaves.
        await rag1.apipeline_enqueue_documents(
            ["text 0", "text 1"], ids=["d0", "d1"], file_paths=["d0", "d1"]
        )
    finally:
        await rag1.finalize_storages()

    stub2 = _StubExtract()
    rag2, config2, llm2 = await _build(tmp_path, workspace, stub2)
    try:
        result = await ingest_dataset(rag2, _dataset(), config2)
        assert sorted(stub2.doc_calls) == ["d0", "d1"]
        assert result.chunk_count == 3
        assert result.persistently_failed_doc_ids == []
        assert llm2 == []
    finally:
        await rag2.finalize_storages()


@pytest.mark.parametrize("cap", [0, 2], ids=["chunks-only", "with-extraction"])
async def test_fully_ingested_dataset_skips_straight_to_retrieval(tmp_path, cap):
    workspace = f"resume_{uuid4().hex[:8]}"
    stub1 = _StubExtract()
    rag1, config, _ = await _build(tmp_path, workspace, stub1, extraction_cap=cap)
    try:
        first = await ingest_dataset(rag1, _dataset(), config)
        assert first.resumed is False
        assert first.chunk_count == 3
        assert len(stub1.doc_calls) == cap
    finally:
        await rag1.finalize_storages()

    refuse = _RefuseExtract()
    rag2, config2, llm2 = await _build(tmp_path, workspace, refuse, extraction_cap=cap)
    custom_kg_calls = _refuse_custom_kg(rag2)
    ainsert_calls = _refuse_ainsert(rag2)
    try:
        result = await ingest_dataset(rag2, _dataset(), config2)
        assert refuse.doc_calls == []
        assert custom_kg_calls == []
        assert ainsert_calls == []
        assert result.resumed is True
        assert result.recovery_rounds_used == 0
        assert result.chunk_count == 3
        assert set(result.chunk_doc_map.values()) == {"d0", "d1", "d2"}
        assert result.extracted_doc_ids == first.extracted_doc_ids
        assert result.chunks_only_doc_ids == first.chunks_only_doc_ids
        assert llm2 == []
        progress = json.loads(
            rag_setup._progress_path(config2).read_text(encoding="utf-8")
        )
        assert progress["phase"] == "complete"
    finally:
        await rag2.finalize_storages()


async def test_progress_checkpoint_artifact_records_counts(tmp_path):
    workspace = f"resume_{uuid4().hex[:8]}"
    rag, config, _ = await _build(tmp_path, workspace, _StubExtract())
    try:
        await ingest_dataset(rag, _dataset(), config)
        progress = json.loads(
            rag_setup._progress_path(config).read_text(encoding="utf-8")
        )
        assert progress["dataset"] == "tiny"
        assert progress["workspace"] == workspace
        assert progress["phase"] == "complete"
        assert progress["doc_status_counts"]["processed"] == 2
        assert "updated_utc" in progress
    finally:
        await rag.finalize_storages()


def _cli_argv(*extra) -> list[str]:
    return ["--dataset-dir", "d", "--output-dir", "o", *extra]


def test_stable_workspace_flag_derives_per_dataset_name():
    args = cli.parse_args(_cli_argv("--workspace", "cloud_run1"))
    assert cli._resolve_workspace(args, "miracl-zh") == "cloud_run1_miracl_zh"
    assert args.keep_workspace is True


def test_ephemeral_workspace_keeps_random_suffix():
    args = cli.parse_args(_cli_argv())
    first = cli._resolve_workspace(args, "miracl-zh")
    second = cli._resolve_workspace(args, "miracl-zh")
    assert re.fullmatch(r"hbench_miracl_zh_[0-9a-f]{8}", first)
    assert first != second
    assert args.keep_workspace is False


def test_workspace_prefix_still_honored_without_stable_flag():
    args = cli.parse_args(_cli_argv("--workspace-prefix", "hb2"))
    assert re.fullmatch(
        r"hb2_miracl_zh_[0-9a-f]{8}", cli._resolve_workspace(args, "miracl-zh")
    )
