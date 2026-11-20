"""Ingest execution robustness: configured timeout + sanctioned recovery.

Offline, file backends only. Recovery pins drive the REAL pipeline with a
controllable extraction stub and a recording wrapper that DELEGATES to the
real ``commit_manual_retry_request`` — proving FAILED documents re-enter
through the /documents/reprocess_failed commit + drive sequence (one attempt
per document per round), never through doc_status state surgery. The timeout
pin replaces ingest with a slow fake and lets the REAL ``asyncio.wait_for``
expire at the configured value; the fake outlives every tested timeout, so
timer ordering makes the expiry deterministic without timing assertions.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from lightrag.evaluation.hybrid_bench import cli, rag_setup
from lightrag.evaluation.hybrid_bench.cli import _base_config, parse_args
from lightrag.evaluation.hybrid_bench.dataset import BenchDataset, DatasetMeta
from lightrag.evaluation.hybrid_bench.rag_setup import (
    BenchConfig,
    IngestResult,
    build_rag,
    ingest_dataset,
)
from lightrag.kg.shared_storage import (
    get_pipeline_ingress,
    initialize_share_data,
)

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
        corpus={"d0": "text 0", "d1": "text 1"},
        queries={"q1": "query one", "q2": "query two"},
        qrels={"q1": {"d0": 1}, "q2": {"d1": 1}},
    )


def _cli_argv(*extra) -> list[str]:
    return ["--dataset-dir", "d", "--output-dir", "o", *extra]


@pytest.mark.parametrize(
    ("attr", "default"),
    [("ingest_timeout", 86400), ("ingest_recovery_rounds", 2)],
)
def test_cli_defaults(attr, default):
    args = parse_args(_cli_argv())
    assert getattr(args, attr) == default
    assert getattr(_base_config(args, "ws"), attr) == default


@pytest.mark.parametrize(
    ("flag", "value", "attr"),
    [
        ("--ingest-timeout", "7", "ingest_timeout"),
        ("--ingest-recovery-rounds", "0", "ingest_recovery_rounds"),
        ("--ingest-recovery-rounds", "5", "ingest_recovery_rounds"),
    ],
)
def test_cli_flag_reaches_config(flag, value, attr):
    args = parse_args(_cli_argv(flag, value))
    assert getattr(args, attr) == int(value)
    assert getattr(_base_config(args, "ws"), attr) == int(value)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--ingest-timeout", "0"),
        ("--ingest-timeout", "-1"),
        ("--ingest-recovery-rounds", "-1"),
    ],
)
def test_invalid_flag_values_rejected(flag, value):
    with pytest.raises(SystemExit) as exc:
        parse_args(_cli_argv(flag, value))
    assert exc.value.code == 2


@pytest.mark.parametrize("timeout", [1, 2])
async def test_ingest_timeout_expires_at_configured_value(
    tmp_path, monkeypatch, timeout
):
    initialize_share_data(workers=1)
    args = parse_args(
        _cli_argv(
            "--storage",
            "file",
            "--embedding-backend",
            "fake-hash",
            "--embedding-dim",
            "32",
            "--llm-backend",
            "fake",
            "--extraction-cap",
            "2",
            "--phases",
            "",
            "--working-dir",
            str(tmp_path / "working"),
            "--ingest-timeout",
            str(timeout),
        )
    )
    calls = []

    async def slow_ingest(rag, dataset, config):
        calls.append(config.ingest_timeout)
        await asyncio.sleep(3)
        return IngestResult(
            chunk_doc_map={}, extracted_doc_ids=[], chunks_only_doc_ids=[]
        )

    monkeypatch.setattr(cli, "ingest_dataset", slow_ingest)
    with pytest.raises(TimeoutError):
        await cli.run_dataset(args, tmp_path, _dataset())
    assert calls == [timeout]


class _StubExtract:
    """Extraction stub failing the first N calls per document, counting all."""

    def __init__(self, fail_plan: dict[str, int]):
        self.fail_plan = dict(fail_plan)
        self.calls = 0
        self.failed_calls = 0

    async def __call__(self, chunks, *args, **kwargs):
        self.calls += 1
        doc_id = next(iter(chunks.values()))["full_doc_id"]
        remaining = self.fail_plan.get(doc_id, 0)
        if remaining > 0:
            self.fail_plan[doc_id] = remaining - 1
            self.failed_calls += 1
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


async def _build_rag_with_stub(tmp_path, fail_plan, **overrides):
    initialize_share_data(workers=1)
    config = BenchConfig(
        storage="file",
        working_dir=str(tmp_path / "working"),
        workspace=f"recv_{uuid4().hex[:8]}",
        embedding_backend="fake-hash",
        embedding_dim=32,
        extraction_cap=2,
        max_gleaning=0,
        **overrides,
    )
    llm_calls = []

    async def refuse_llm(prompt, *args, **kwargs):
        llm_calls.append(prompt)
        raise AssertionError("recovery pin must not call the LLM")

    rag = await build_rag(config, refuse_llm)
    stub = _StubExtract(fail_plan)
    rag._process_extract_entities = stub
    return rag, config, stub, llm_calls


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


async def test_recovery_round_recovers_failed_via_sanctioned_path(tmp_path, commit_spy):
    rag, config, stub, llm_calls = await _build_rag_with_stub(
        tmp_path, {"d0": 1, "d1": 1}
    )
    try:
        result = await ingest_dataset(rag, _dataset(), config)
        assert stub.calls == 4
        assert stub.failed_calls == 2
        assert len(commit_spy) == 1
        assert result.recovery_rounds_used == 1
        assert result.recovered_doc_ids == ["d0", "d1"]
        assert result.persistently_failed_doc_ids == []
        assert result.chunk_count == 2
        assert set(result.chunk_doc_map.values()) == {"d0", "d1"}
        ingress = await get_pipeline_ingress(rag.workspace)
        assert ingress.snapshot_manual_retries() == []
        statuses = await rag.doc_status.get_docs_by_statuses(
            [rag_setup.DocStatus.PROCESSED], strict=True
        )
        assert set(statuses) == {"d0", "d1"}
        assert llm_calls == []
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize("rounds", [1, 2])
async def test_persistent_failure_raises_loud_after_bounded_rounds(
    tmp_path, commit_spy, rounds
):
    rag, config, stub, llm_calls = await _build_rag_with_stub(
        tmp_path,
        {"d0": rounds + 1, "d1": rounds + 1},
        ingest_recovery_rounds=rounds,
    )
    try:
        with pytest.raises(RuntimeError, match="document 'd0' failed ingestion") as exc:
            await ingest_dataset(rag, _dataset(), config)
        message = str(exc.value)
        assert f"recovery rounds used: {rounds}" in message
        assert "recovered: []" in message
        assert "persistently failed: ['d0', 'd1']" in message
        assert stub.calls == 2 * (rounds + 1)
        assert stub.failed_calls == 2 * (rounds + 1)
        assert len(commit_spy) == rounds
        assert len(set(commit_spy)) == rounds
        assert llm_calls == []
    finally:
        await rag.finalize_storages()


async def test_mixed_recovery_annotates_recovered_and_persistent(tmp_path, commit_spy):
    rag, config, stub, llm_calls = await _build_rag_with_stub(
        tmp_path, {"d0": 3, "d1": 1}, ingest_recovery_rounds=2
    )
    try:
        with pytest.raises(RuntimeError, match="document 'd0' failed ingestion") as exc:
            await ingest_dataset(rag, _dataset(), config)
        message = str(exc.value)
        assert "recovery rounds used: 2" in message
        assert "recovered: ['d1']" in message
        assert "persistently failed: ['d0']" in message
        assert stub.calls == 5
        assert stub.failed_calls == 4
        assert len(commit_spy) == 2
        assert llm_calls == []
    finally:
        await rag.finalize_storages()


async def test_zero_rounds_keeps_legacy_immediate_fail_loud(tmp_path, commit_spy):
    rag, config, stub, llm_calls = await _build_rag_with_stub(
        tmp_path, {"d0": 1, "d1": 1}, ingest_recovery_rounds=0
    )
    try:
        with pytest.raises(RuntimeError, match="document 'd0' failed ingestion") as exc:
            await ingest_dataset(rag, _dataset(), config)
        assert "recovery" not in str(exc.value)
        assert stub.calls == 2
        assert stub.failed_calls == 2
        assert commit_spy == []
        assert llm_calls == []
    finally:
        await rag.finalize_storages()
