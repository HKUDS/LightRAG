"""Token-limit truncation must survive as a per-document record (issue #3601).

``pipeline_status.history_messages`` is a bounded ring scoped to the running
batch: by the time an operator notices a short knowledge graph, the truncation
lines that explain it are long gone, and nothing in ``/documents`` says the
document was built from partial model output. So the pipeline stamps the
document-scoped tally into ``doc_status.metadata.llm_truncation``, which is
durable and reaches the API.

The write rules that make that record trustworthy — and which the tests below
pin — are the ones that come from the metadata whitelists in
``utils_pipeline``:

- the key is written at the TERMINAL transition (PROCESSED, and both FAILED
  stage boundaries), because that is the first point at which the attempt's
  count is final;
- it is NOT in ``_DOC_STATUS_METADATA_CARRY_OVER_KEYS``, so a re-run that does
  not truncate clears it instead of resurrecting a stale summary — the whole
  point of retrying with a larger output budget;
- it IS in ``_DOC_STATUS_METADATA_ATTEMPT_KEYS``, so a document reset back to
  PENDING is normalized to directives-only rather than waiting in the queue
  advertising a previous attempt's truncation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import (
    EmbeddingFunc,
    LLM_TRUNCATION_METADATA_KEY,
    Tokenizer,
    TokenizerInterface,
    TruncatedResponse,
)
from lightrag.utils_pipeline import (
    doc_status_reset_metadata,
    doc_status_transition_metadata,
)

pytestmark = pytest.mark.offline


_EXTRACTION_RESULT = "(entity<|#|>ALICE<|#|>person<|#|>An analyst)<|COMPLETE|>"


class _SimpleTokenizerImpl(TokenizerInterface):
    def encode(self, content: str):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.full((len(texts), 32), 0.1, dtype=np.float32)


def _new_rag(tmp_path: Path, llm_func) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"truncation-{tmp_path.name}",
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=32, max_token_size=4096, func=_mock_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
    )


async def _ingest(rag: LightRAG, *, doc_id: str, body: str) -> dict:
    await rag.apipeline_enqueue_documents(
        body, ids=[doc_id], file_paths=f"{doc_id}.txt", track_id=f"track-{doc_id}"
    )
    await rag.apipeline_process_enqueue_documents()
    return await rag.doc_status.get_by_id(doc_id)


async def test_a_truncated_run_records_the_summary_on_the_document(tmp_path):
    """The document still reaches PROCESSED — that is the reported behavior
    this issue does not change — but it now SAYS its graph is partial."""

    async def _truncating_llm(prompt, **kwargs):
        return TruncatedResponse(_EXTRACTION_RESULT)

    rag = _new_rag(tmp_path, _truncating_llm)
    await rag.initialize_storages()
    try:
        status = await _ingest(rag, doc_id="doc-trunc", body="Alice is an analyst.")
    finally:
        await rag.finalize_storages()

    assert status["status"] == DocStatus.PROCESSED
    summary = status["metadata"][LLM_TRUNCATION_METADATA_KEY]
    assert summary["events"] >= 1
    assert summary["affected"] == 1
    assert "initial" in summary["stages"]
    assert summary["samples"] == status["chunks_list"][:1]


async def test_a_clean_run_leaves_no_truncation_record(tmp_path):
    async def _complete_llm(prompt, **kwargs):
        return _EXTRACTION_RESULT

    rag = _new_rag(tmp_path, _complete_llm)
    await rag.initialize_storages()
    try:
        status = await _ingest(rag, doc_id="doc-clean", body="Alice is an analyst.")
    finally:
        await rag.finalize_storages()

    assert status["status"] == DocStatus.PROCESSED
    assert LLM_TRUNCATION_METADATA_KEY not in status["metadata"]


def test_the_summary_is_not_carried_over_to_the_next_attempt():
    """Carry-over would resurrect a stale summary on a re-run that fixed the
    budget: the PROCESSING transition must drop it, and only the terminal
    transition of the NEW attempt may write it back."""
    status_doc = {
        "metadata": {
            LLM_TRUNCATION_METADATA_KEY: {"events": 12, "affected": 12},
            "process_options": "R",
        }
    }

    carried = doc_status_transition_metadata(status_doc)

    assert LLM_TRUNCATION_METADATA_KEY not in carried
    # ...while a genuine long-lived directive still survives.
    assert carried["process_options"] == "R"


def test_a_reset_to_pending_drops_the_summary():
    status_doc = {
        "metadata": {
            LLM_TRUNCATION_METADATA_KEY: {"events": 12, "affected": 12},
            "process_options": "R",
        }
    }

    reset = doc_status_reset_metadata(status_doc)

    assert LLM_TRUNCATION_METADATA_KEY not in reset
    assert reset["process_options"] == "R"


def test_the_summary_marks_a_pending_row_as_carrying_stale_attempt_metadata():
    from lightrag.utils_pipeline import _DOC_STATUS_METADATA_ATTEMPT_KEYS

    assert LLM_TRUNCATION_METADATA_KEY in _DOC_STATUS_METADATA_ATTEMPT_KEYS


async def test_analyze_stage_truncation_reaches_the_processed_row(tmp_path):
    """Codex review (PR #3607): multimodal analysis runs BEFORE the process
    stage creates the document tally, and the metadata key is per-attempt
    (dropped at the PROCESSING transition), so its truncations rode only in
    server logs. They arrive on the analyze→process hand-off dict and must be
    merged into the terminal metadata alongside extraction's own counts."""

    async def _truncating_llm(prompt, **kwargs):
        return TruncatedResponse(_EXTRACTION_RESULT)

    rag = _new_rag(tmp_path, _truncating_llm)
    await rag.initialize_storages()

    original_analyze = rag.analyze_multimodal

    async def _analyze_with_truncation(*args, **kwargs):
        parsed_data = await original_analyze(*args, **kwargs)
        # What analyze_multimodal records when a schema-valid truncated VLM
        # analysis is accepted (covered by its own stage tests).
        parsed_data["llm_truncation"] = {
            "events": 2,
            "affected": 2,
            "stages": {"multimodal": 2},
            "samples": ["drawings/im-001", "table/tb-001"],
        }
        return parsed_data

    rag.analyze_multimodal = _analyze_with_truncation
    try:
        status = await _ingest(rag, doc_id="doc-mm", body="Alice is an analyst.")
    finally:
        await rag.finalize_storages()

    assert status["status"] == DocStatus.PROCESSED
    summary = status["metadata"][LLM_TRUNCATION_METADATA_KEY]
    # Both stages in one record: the analyze hand-off merged with extraction.
    assert summary["stages"]["multimodal"] == 2
    assert summary["stages"]["initial"] >= 1
    assert "drawings/im-001" in summary["samples"]
