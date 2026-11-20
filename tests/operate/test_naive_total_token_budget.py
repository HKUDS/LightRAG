"""``max_total_tokens`` bounds the prompt a naive query sends, as it does for KG modes.

``naive_query`` sizes its chunks against ``max_total_tokens`` minus the empty
system prompt, the query and a fixed 200-token reserve, then renders the
context template and the reference list on top of them. The reserve does not
always cover that: a reference list with many or long paths, or a caller
template that renders ``{content_data}`` more than once, goes over the cap.
So the rendered prompt is checked and trailing chunks are dropped until it
fits. The KG modes do the same (``test_kg_total_token_budget.py``).
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag import operate
from lightrag.base import QueryParam
from lightrag.utils import TiktokenTokenizer

pytestmark = pytest.mark.offline

QUERY = "What changed in the third quarter?"


@pytest.fixture(scope="module")
def tokenizer():
    return TiktokenTokenizer()


def _chunks(count):
    """Chunks of about 180 tokens each, every one from its own source file."""
    return [
        {
            "id": f"chunk-{index}",
            "content": f"Quarterly evidence number {index}. " * 30,
            "file_path": (
                f"/srv/shared/finance/reports/2026/q3/"
                f"regional-summary-{index:02d}-final-reviewed.pdf"
            ),
        }
        for index in range(count)
    ]


async def _naive_query(tokenizer, chunks, max_total_tokens):
    answer_model = AsyncMock(return_value="answer")
    config = {"tokenizer": tokenizer, "role_llm_funcs": {"query": answer_model}}
    chunks_vdb = SimpleNamespace(
        cosine_better_than_threshold=0.2,
        query=AsyncMock(return_value=chunks),
    )
    param = QueryParam(
        mode="naive",
        max_total_tokens=max_total_tokens,
        chunk_top_k=len(chunks),
        enable_rerank=False,
    )
    result = await operate.naive_query(QUERY, chunks_vdb, param, config)
    return result, answer_model


async def test_naive_prompt_stays_within_max_total_tokens(tokenizer):
    chunk_count, max_total_tokens = 20, 4000
    chunks = _chunks(chunk_count)

    result, answer_model = await _naive_query(tokenizer, chunks, max_total_tokens)

    assert result.content == "answer"
    sent_system_prompt = answer_model.await_args.kwargs["system_prompt"]
    sent_tokens = len(tokenizer.encode(sent_system_prompt)) + len(
        tokenizer.encode(QUERY)
    )
    assert sent_tokens <= max_total_tokens
    # The reference list is rebuilt from the chunks that are kept.
    data = result.raw_data["data"]
    assert 0 < len(data["chunks"]) < chunk_count
    assert [chunk["reference_id"] for chunk in data["chunks"]] == [
        reference["reference_id"] for reference in data["references"]
    ]
    first_dropped = chunks[len(data["chunks"])]
    assert first_dropped["file_path"] not in sent_system_prompt


async def test_naive_returns_no_result_when_no_rendered_chunk_fits(
    tokenizer, lightrag_log_records
):
    """The chunk passes the chunk limit; its reference path is what does not fit."""
    chunk = {
        "id": "chunk-0",
        "content": "Short evidence.",
        "file_path": "/".join(f"archive-{level:03d}" for level in range(150)),
    }

    result, answer_model = await _naive_query(tokenizer, [chunk], 1000)

    assert result is None
    answer_model.assert_not_awaited()
    warnings = [
        record.getMessage()
        for record in lightrag_log_records
        if record.levelno == logging.WARNING
        and "Token budget left no room" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "reference list" in warnings[0]
