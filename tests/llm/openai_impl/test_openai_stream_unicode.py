"""Escaped Unicode split across Chat Completions stream chunks."""

from unittest.mock import patch

import pytest

from lightrag.llm.openai import openai_complete_if_cache
from tests.llm.openai_impl.test_openai_stream_generatorexit_cot_cleanup import (
    _FakeOpenAIStream,
    _make_chunk,
    _make_stream_client,
)

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_escape_split_across_chunks_is_decoded_per_text():
    """Reasoning and answer are decoded as two texts, each held across chunks.

    The reasoning's open escape is flushed before ``</think>``, the answer's at
    the end of the stream.
    """
    stream = _FakeOpenAIStream(
        [
            _make_chunk(reasoning_content=r"think \ud83d"),
            _make_chunk(reasoning_content=r"\ude00 \u4f"),
            _make_chunk(content=r"ans\u00"),
            _make_chunk(content=r"41 \u4f"),
        ]
    )

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=_make_stream_client(stream),
    ):
        gen = await openai_complete_if_cache.__wrapped__(
            model="deepseek-reasoner",
            prompt="hello",
            enable_cot=True,
            stream=True,
            api_key="test-key",
        )
        chunks = [chunk async for chunk in gen]

    assert chunks == [
        "<think>",
        "think ",
        "😀 ",
        r"\u4f",
        "</think>",
        "ans",
        "A ",
        r"\u4f",
    ]
