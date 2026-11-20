"""Provider output-moderation failures: chunks-only fallback, honest gap.

Offline file-backend pins on the REAL pipeline with an LLM-level fault: a
FakeLLM wrapper raises the provider's exact BadRequestError 400
``data_inspection_failed`` payload for flagged passages, so the failure
travels the real extraction -> chunk-prefix -> doc_status error_msg path.
The sanctioned recovery rounds run FIRST; only the moderation class falls
back to chunks-only ingestion afterwards (chunk arm intact, NO entities).
Every other permanent failure class still aborts, and the fallback doc's
doc_status row stays FAILED — no state surgery.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from lightrag.base import DocStatus
from lightrag.evaluation.hybrid_bench import arms, scoring
from lightrag.evaluation.hybrid_bench.dataset import BenchDataset, DatasetMeta
from lightrag.evaluation.hybrid_bench.entity_truth import derive_entity_truth
from lightrag.evaluation.hybrid_bench.fakes import FakeLLM, _INPUT_TEXT_RE
from lightrag.evaluation.hybrid_bench.rag_setup import (
    BenchConfig,
    build_llm_func,
    build_rag,
    ingest_dataset,
)
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

pytestmark = pytest.mark.offline

POISON_TEXT = "a passage the provider output inspection flags every time"
CLEAN_TEXT = "a clean passage about gears and torque"

# The exact provider payload shape recorded in the cloud run logs (the
# pipeline adds its chunk/batch prefixes around it into error_msg).
MODERATION_ERROR = (
    "BadRequestError: Error code: 400 - {'error': {'code': "
    "'data_inspection_failed', 'param': None, 'message': 'Output data may "
    "contain inappropriate content.', 'type': 'data_inspection_failed'}, "
    "'id': 'chatcmpl-offline-pin'}"
)


def _dataset(poison_second: bool = False) -> BenchDataset:
    return BenchDataset(
        name="moderation",
        meta=DatasetMeta(
            domain="general",
            language="en",
            source="https://example.org",
            license="CC-BY-4.0",
            subset_protocol="test",
            counts={},
        ),
        corpus={
            "d0": POISON_TEXT,
            "d1": POISON_TEXT if poison_second else CLEAN_TEXT,
        },
        queries={"q1": POISON_TEXT, "q2": CLEAN_TEXT},
        qrels={"q1": {"d0": 1}, "q2": {"d1": 1}},
    )


def _fake_llm() -> FakeLLM:
    return FakeLLM(
        {
            POISON_TEXT: [
                {
                    "entity_name": "flagged",
                    "entity_type": "concept",
                    "description": "a flagged concept",
                }
            ],
            CLEAN_TEXT: [
                {
                    "entity_name": "gear",
                    "entity_type": "concept",
                    "description": "a gear transmits torque",
                }
            ],
        }
    )


class _FaultLLM:
    """FakeLLM wrapper raising a provider-style extraction error for flagged
    inputs. ``fail_plan`` maps passage text to remaining failing calls
    (None = permanent). Records every raise so the test proves the fault
    fired and was consumed; all other calls delegate to the real FakeLLM.
    """

    def __init__(self, delegate, fail_plan, error: str = MODERATION_ERROR):
        self.delegate = delegate
        self.fail_plan = dict(fail_plan)
        self.error = error
        self.fault_calls: list[str] = []

    async def __call__(self, prompt, *args, **kwargs):
        match = _INPUT_TEXT_RE.search(str(prompt))
        if match is not None and match.group(1) in self.fail_plan:
            text = match.group(1)
            remaining = self.fail_plan[text]
            if remaining is None or remaining > 0:
                if remaining is not None:
                    self.fail_plan[text] = remaining - 1
                self.fault_calls.append(text)
                raise RuntimeError(self.error)
        return await self.delegate(prompt, *args, **kwargs)


def _chunk_id(text: str) -> str:
    return compute_mdhash_id(sanitize_text_for_encoding(text), prefix="chunk-")


async def _build(tmp_path, workspace, llm, **overrides):
    initialize_share_data(workers=1)
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
        **overrides,
    )
    rag = await build_rag(config, build_llm_func(config, llm))
    return rag, config


async def _statuses(rag):
    return await rag.doc_status.get_docs_by_statuses(
        [DocStatus.PROCESSED, DocStatus.FAILED], strict=True
    )


@pytest.mark.parametrize("rounds", [0, 2])
async def test_permanent_moderation_falls_back_to_chunks_only(tmp_path, rounds):
    workspace = f"mod_{uuid4().hex[:8]}"
    llm = _FaultLLM(_fake_llm(), {POISON_TEXT: None})
    rag, config = await _build(tmp_path, workspace, llm, ingest_recovery_rounds=rounds)
    try:
        result = await ingest_dataset(rag, _dataset(), config)
        assert result.moderation_fallback_doc_ids == ["d0"]
        assert result.persistently_failed_doc_ids == []
        assert result.recovered_doc_ids == []
        assert result.recovery_rounds_used == rounds
        # initial attempt + one per recovery round; the fault fired on each
        assert len(llm.fault_calls) == rounds + 1

        statuses = await _statuses(rag)
        assert statuses["d0"].status == DocStatus.FAILED
        assert "data_inspection_failed" in statuses["d0"].error_msg
        assert statuses["d1"].status == DocStatus.PROCESSED

        # Both the failed attempt's doc-scoped chunk and the fallback's
        # content-hash chunk carry the corpus attribution.
        fallback_chunk = _chunk_id(POISON_TEXT)
        assert result.chunk_doc_map[fallback_chunk] == ["d0"]
        attempt_chunks = statuses["d0"].chunks_list
        assert len(attempt_chunks) == 1
        assert attempt_chunks[0] != fallback_chunk
        assert result.chunk_doc_map[attempt_chunks[0]] == ["d0"]

        embedding = await arms.embed_query(rag, POISON_TEXT)
        hits = await arms.retrieve_chunks(
            rag, POISON_TEXT, "dense", 10, embedding, result.chunk_doc_map
        )
        assert any(
            hit.chunk_id == fallback_chunk and hit.doc_ids == ("d0",) for hit in hits
        )
        docs = arms.doc_id_sequence(hits)
        scored, _, _ = scoring.score_chunk_arms(_dataset(), {"dense": {"q1": docs}}, 10)
        assert scored["dense"]["q1"]["recall"] == 1.0

        # Honest entity-arm gap: the clean doc contributes truth, the
        # moderation-fallback doc contributes none.
        truth = await derive_entity_truth(
            rag.chunk_entity_relation_graph,
            rag.text_chunks,
            {"q1": {"d0": 1}, "q2": {"d1": 1}},
            entity_chunks=rag.entity_chunks,
        )
        assert truth.per_query.get("q2")
        assert not truth.per_query.get("q1")
        assert "q1" in truth.zero_truth_query_ids
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize(
    "error",
    [
        "extract fail sentinel",
        "BadRequestError: Error code: 400 - {'error': {'code': "
        "'invalid_parameter_error', 'message': 'bad request'}}",
    ],
    ids=["generic", "other-400"],
)
async def test_non_moderation_permanent_failure_still_aborts(tmp_path, error):
    workspace = f"mod_{uuid4().hex[:8]}"
    llm = _FaultLLM(_fake_llm(), {POISON_TEXT: None}, error=error)
    rag, config = await _build(tmp_path, workspace, llm)
    try:
        with pytest.raises(RuntimeError, match="document 'd0' failed ingestion") as exc:
            await ingest_dataset(rag, _dataset(), config)
        message = str(exc.value)
        assert "recovery rounds used: 2" in message
        assert "persistently failed: ['d0']" in message
        assert len(llm.fault_calls) == 3
        statuses = await _statuses(rag)
        assert statuses["d0"].status == DocStatus.FAILED
        # No fallback chunk was written for the non-moderation failure
        # (filter_keys returns the keys ABSENT from storage).
        assert _chunk_id(POISON_TEXT) in await rag.text_chunks.filter_keys(
            {_chunk_id(POISON_TEXT)}
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize("recover_after", [1, 2])
async def test_recovery_rounds_run_first_and_can_recover_moderation(
    tmp_path, recover_after
):
    workspace = f"mod_{uuid4().hex[:8]}"
    llm = _FaultLLM(_fake_llm(), {POISON_TEXT: recover_after})
    rag, config = await _build(tmp_path, workspace, llm)
    try:
        result = await ingest_dataset(rag, _dataset(), config)
        assert result.recovered_doc_ids == ["d0"]
        assert result.moderation_fallback_doc_ids == []
        assert result.persistently_failed_doc_ids == []
        assert result.recovery_rounds_used == recover_after
        assert llm.fail_plan[POISON_TEXT] == 0
        statuses = await _statuses(rag)
        assert statuses["d0"].status == DocStatus.PROCESSED
        # Recovered means FULLY extracted: the poisoned doc's entity exists.
        truth = await derive_entity_truth(
            rag.chunk_entity_relation_graph,
            rag.text_chunks,
            {"q1": {"d0": 1}},
            entity_chunks=rag.entity_chunks,
        )
        assert truth.per_query.get("q1")
    finally:
        await rag.finalize_storages()


async def test_moderation_fallback_resume_is_idempotent(tmp_path):
    workspace = f"mod_{uuid4().hex[:8]}"
    llm1 = _FaultLLM(_fake_llm(), {POISON_TEXT: None})
    rag1, config1 = await _build(tmp_path, workspace, llm1)
    try:
        first = await ingest_dataset(rag1, _dataset(), config1)
    finally:
        await rag1.finalize_storages()

    llm2 = _FaultLLM(_fake_llm(), {POISON_TEXT: None})
    rag2, config2 = await _build(tmp_path, workspace, llm2)
    try:
        second = await ingest_dataset(rag2, _dataset(), config2)
        assert second.resumed is True
        assert second.chunk_doc_map == first.chunk_doc_map
        assert second.chunk_count == first.chunk_count
        assert second.moderation_fallback_doc_ids == ["d0"]
        assert second.persistently_failed_doc_ids == []
        # The FAILED row persists (no surgery), so resume re-runs the bounded
        # rounds (2 attempts; enqueue skips the existing row) and re-applies
        # the idempotent fallback.
        assert second.recovery_rounds_used == 2
        assert len(llm2.fault_calls) == 2
    finally:
        await rag2.finalize_storages()


async def test_twin_of_moderation_fallback_original_still_fails_loud(tmp_path):
    # The twin's original is FAILED (moderation), not PROCESSED: the twin
    # proof chain requires a PROCESSED holder, so the missing status raises
    # even though the original's content sits in the chunk store.
    workspace = f"mod_{uuid4().hex[:8]}"
    llm = _FaultLLM(_fake_llm(), {POISON_TEXT: None})
    rag, config = await _build(tmp_path, workspace, llm)
    try:
        with pytest.raises(RuntimeError, match="ingested document 'd1' has no status"):
            await ingest_dataset(rag, _dataset(poison_second=True), config)
        assert len(llm.fault_calls) == 3
    finally:
        await rag.finalize_storages()
