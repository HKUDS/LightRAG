"""An empty LLM summary must fail the merge, not replace the description.

When an entity or relation collects enough description fragments, the merge
asks the LLM for a summary and stores it as the description. A summary that is
empty -- a bare ``""``, or ``<think>...</think>`` with no answer after it, which
think-tag removal turns into ``""`` -- used to be stored as is, so every
fragment collected so far was lost without a warning and the document was
still marked processed.
"""

import hashlib
import re
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.exceptions import EmptySummaryResponseError
from lightrag.kg.pipeline_ingress import PipelineIngressMessage
from lightrag.kg.shared_storage import get_pipeline_ingress
from lightrag.operate import _summarize_descriptions
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _summary_config(summary: str) -> dict:
    return {
        "role_llm_funcs": {"extract": AsyncMock(return_value=summary)},
        "addon_params": {},
        "_resolved_summary_language": "English",
        "summary_length_recommended": 100,
        "summary_context_size": 10_000,
        "tokenizer": Tokenizer("char", _CharTokenizer()),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "summary",
    [
        pytest.param("", id="empty"),
        pytest.param(" \n\t ", id="whitespace"),
        pytest.param("<think>reasoning, no answer</think>", id="think-only"),
        pytest.param("\x00\x07", id="control-chars-only"),
    ],
)
async def test_empty_summary_raises(summary):
    with pytest.raises(EmptySummaryResponseError, match="Entity:Alpha Corp"):
        await _summarize_descriptions(
            "Entity",
            "Alpha Corp",
            ["first fact", "second fact"],
            _summary_config(summary),
            llm_response_cache=None,
        )


@pytest.mark.asyncio
async def test_non_empty_summary_is_returned():
    summary = await _summarize_descriptions(
        "Entity",
        "Alpha Corp",
        ["first fact", "second fact"],
        _summary_config("<think>reasoning</think>Alpha Corp summary."),
        llm_response_cache=None,
    )

    assert summary == "Alpha Corp summary."


_D, _C = "<|#|>", "<|COMPLETE|>"


def _make_llm(summary_mode: dict):
    async def llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        if "Description List" in prompt:
            if summary_mode["empty"]:
                return "<think>reasoning, no answer</think>"
            return " ".join(re.findall(r'"Description": "(.*?)"', prompt))
        i = re.search(r"DOC-(\d+)X", prompt).group(1)
        return (
            f"entity{_D}Alpha Corp{_D}organization{_D}Alpha Corp fact number {i}.\n"
            f"entity{_D}Beta Inc{_D}organization{_D}Beta Inc fact number {i}.\n"
            f"relation{_D}Alpha Corp{_D}Beta Inc{_D}partner{_D}"
            f"Alpha-Beta relation fact number {i}.\n{_C}"
        )

    return llm


async def _embed(texts, **kwargs):
    rows = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        row = np.random.default_rng(seed).standard_normal(16)
        rows.append(row / np.linalg.norm(row))
    return np.array(rows, dtype=np.float32)


async def _graph_state(rag):
    node = await rag.chunk_entity_relation_graph.get_node("Alpha Corp")
    edge = await rag.chunk_entity_relation_graph.get_edge("Alpha Corp", "Beta Inc")
    docs = {
        doc_id: await rag.doc_status.get_by_id(doc_id)
        for doc_id in ("doc-0", "doc-1", "doc-2")
    }
    return node["description"], edge["description"], docs


@pytest.mark.asyncio
async def test_empty_summary_keeps_descriptions_until_a_retry(tmp_path):
    summary_mode = {"empty": True}
    rag = LightRAG(
        working_dir=str(tmp_path),
        llm_model_func=_make_llm(summary_mode),
        llm_model_name="fake",
        embedding_func=EmbeddingFunc(
            embedding_dim=16, max_token_size=8192, model_name="fake", func=_embed
        ),
        entity_extract_max_gleaning=0,
        force_llm_summary_on_merge=3,
    )
    await rag.initialize_storages()
    try:
        # Two fragments are joined without the LLM; the third needs a summary.
        for i in range(3):
            await rag.ainsert(
                f"DOC-{i}X text about Alpha Corp",
                ids=[f"doc-{i}"],
                file_paths=[f"d{i}.md"],
            )
        failed = await _graph_state(rag)

        # The empty answer was not cached: a manual retry asks the LLM again.
        summary_mode["empty"] = False
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.request_manual_retry(
            "retry-1",
            PipelineIngressMessage(
                kind="rescan", retry_failed=True, request_id="retry-1"
            ),
        )
        await rag.apipeline_process_enqueue_documents()
        retried = await _graph_state(rag)
    finally:
        await rag.finalize_storages()

    node_description, edge_description, docs = failed
    for i in (0, 1):
        assert f"Alpha Corp fact number {i}." in node_description
        assert f"Alpha-Beta relation fact number {i}." in edge_description
    assert "fact number 2" not in node_description
    assert [doc["status"] for doc in docs.values()] == [
        DocStatus.PROCESSED,
        DocStatus.PROCESSED,
        DocStatus.FAILED,
    ]
    assert "empty summary" in docs["doc-2"]["error_msg"]

    node_description, edge_description, docs = retried
    for i in (0, 1, 2):
        assert f"Alpha Corp fact number {i}." in node_description
        assert f"Alpha-Beta relation fact number {i}." in edge_description
    assert all(doc["status"] == DocStatus.PROCESSED for doc in docs.values())
