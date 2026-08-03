"""``process_chunks_unified``'s two-stage token truncation.

Counting a chunk list's tokens against the stored chunk dicts (the old,
single-stage approach) undercounts: what actually reaches the LLM is the
``{reference_id, content, content_headings?}`` projection built by
``generate_reference_list_from_chunks`` / ``render_chunks_context_text`` --
and ``reference_id`` itself is recomputed from each survivor's ``file_path``
frequency, which changes with the exact set of chunks kept. These tests
construct a case where truncation changes which ``file_path`` is most
frequent (and therefore its ``reference_id``), and pin that the truncation
returned by ``process_chunks_unified`` is verified against the REAL rendered
text, not an approximation -- plus that the whole two-stage flow is still one
executor submission.
"""

from __future__ import annotations

import json

import pytest

from lightrag.base import QueryParam
from lightrag.utils import (
    Tokenizer,
    TokenizerInterface,
    generate_reference_list_from_chunks,
    render_chunks_context_text,
    process_chunks_unified,
)

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    """One token per character -- makes the exact truncation boundary
    predictable so the test can construct a precise frequency-shift case."""

    def encode(self, content: str) -> list[int]:
        return [0] * len(content)

    def decode(self, tokens: list[int]) -> str:
        return "x" * len(tokens)


def _tok() -> Tokenizer:
    return Tokenizer("char", _CharTokenizer())


async def _run(chunks, chunk_token_limit, tokenizer):
    global_config = {"tokenizer": tokenizer}
    query_param = QueryParam(enable_rerank=False, chunk_top_k=None)
    return await process_chunks_unified(
        query="",
        unique_chunks=chunks,
        query_param=query_param,
        global_config=global_config,
        chunk_token_limit=chunk_token_limit,
    )


@pytest.mark.asyncio
async def test_truncation_result_matches_the_real_renderer_exactly():
    """The frequency-shift case: file_path "rare.txt" appears once, early;
    "common.txt" appears twice, but its second occurrence is the very last
    (largest) chunk -- the one truncation is likely to cut. If truncation is
    verified only against an approximation (not the real renderer with
    reference_id recomputed on the SURVIVING set), the returned text can
    silently drift from what really gets sent downstream.
    """
    chunks = [
        {"content": "A" * 10, "file_path": "common.txt", "chunk_id": "c1"},
        {"content": "B" * 10, "file_path": "rare.txt", "chunk_id": "c2"},
        # Large trailing chunk from common.txt again -- truncation should cut
        # this one under a tight budget, changing common.txt's frequency
        # count (2 -> 1) and therefore the reference_id assignment.
        {"content": "C" * 200, "file_path": "common.txt", "chunk_id": "c3"},
    ]
    tokenizer = _tok()

    # Budget fits chunk 1 + 2 comfortably but not chunk 3 as well.
    result = await _run(chunks, chunk_token_limit=150, tokenizer=tokenizer)

    kept_ids = {c["chunk_id"] for c in result}
    assert "c3" not in kept_ids  # the oversized trailing chunk was dropped
    assert {"c1", "c2"} <= kept_ids

    # The REAL renderer, run independently on the exact same survivor list,
    # must fit the budget -- this is what process_chunks_unified promised.
    _, rendered = generate_reference_list_from_chunks(
        [{k: v for k, v in c.items() if k != "id"} for c in result]
    )
    text = render_chunks_context_text(rendered)
    assert len(tokenizer.encode(text)) <= 150


@pytest.mark.asyncio
async def test_all_chunks_kept_when_they_fit():
    chunks = [
        {"content": "hello", "file_path": "a.txt", "chunk_id": "c1"},
        {"content": "world", "file_path": "b.txt", "chunk_id": "c2"},
    ]
    result = await _run(chunks, chunk_token_limit=1000, tokenizer=_tok())
    assert len(result) == 2


@pytest.mark.asyncio
async def test_empty_input_returns_empty():
    result = await _run([], chunk_token_limit=100, tokenizer=_tok())
    assert result == []


@pytest.mark.asyncio
async def test_the_two_stage_flow_is_a_single_executor_submission(monkeypatch):
    """Both truncation stages (approximate K, then real-renderer re-verify +
    K-decrement) must run inside ONE synchronous call submitted ONCE --
    otherwise the K-decrement loop's re-encodes would run back on the event
    loop, one of the exact failure modes this refactor exists to avoid."""
    import lightrag.utils as lr_utils

    submissions = 0
    real_executor = lr_utils.get_tokenizer_executor()

    class _CountingProxy:
        def submit(self, fn, /, *args, **kwargs):
            nonlocal submissions
            submissions += 1
            return real_executor.submit(fn, *args, **kwargs)

    monkeypatch.setattr(lr_utils, "get_tokenizer_executor", lambda: _CountingProxy())

    chunks = [
        {"content": "X" * 20, "file_path": f"f{i}.txt", "chunk_id": f"c{i}"}
        for i in range(30)
    ]
    await _run(chunks, chunk_token_limit=100, tokenizer=_tok())

    assert submissions == 1


@pytest.mark.asyncio
async def test_content_headings_are_preserved_in_the_final_render():
    chunks = [
        {
            "content": "hello",
            "content_headings": "Intro > Section 1",
            "file_path": "a.txt",
            "chunk_id": "c1",
        },
    ]
    result = await _run(chunks, chunk_token_limit=1000, tokenizer=_tok())
    _, rendered = generate_reference_list_from_chunks(result)
    text = render_chunks_context_text(rendered)
    parsed = [json.loads(line) for line in text.splitlines()]
    assert parsed[0]["content_headings"] == "Intro > Section 1"
