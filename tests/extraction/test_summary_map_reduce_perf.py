"""Regression tests for the map-reduce chunking path of
``_handle_entity_relation_summary``.

The map phase tokenizes every description to decide (a) whether the list
needs splitting and (b) where to cut the chunks. Historically the same
description was encoded twice per map-reduce iteration — once to sum the
total token count and again while building the chunks — which doubled the
tiktoken BPE cost on every entity/edge merge. The counts from the first
pass are now memoized and reused by the second.

These tests guard:

* the chunking output is unchanged (behavioral), and
* each description is encoded exactly once per map-reduce iteration
  (the perf property the fix introduced).
"""

from unittest.mock import AsyncMock

import pytest

from lightrag.utils import Tokenizer, TokenizerInterface


class _CountingTokenizer(TokenizerInterface):
    """1:1 character-to-token mapping that records every ``encode`` input."""

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, content: str):
        self.encoded.append(content)
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _make_chunking_config(mock_extract) -> tuple[dict, _CountingTokenizer]:
    """``global_config`` whose ``summary_context_size`` forces the map-reduce
    chunking path (4 five-token descriptions vs. a 10-token window)."""
    counter = _CountingTokenizer()
    tokenizer = Tokenizer("dummy", counter)
    global_config = {
        "role_llm_funcs": {"extract": mock_extract},
        "addon_params": {},
        "_resolved_summary_language": "English",
        "summary_length_recommended": 100,
        # 4 descriptions x 5 tokens = 20 tokens > 10-token window → chunking
        "summary_context_size": 10,
        "summary_max_tokens": 100_000,
        "force_llm_summary_on_merge": 10,
        "tokenizer": tokenizer,
    }
    return global_config, counter


@pytest.mark.offline
@pytest.mark.asyncio
async def test_map_reduce_chunking_is_preserved():
    """Forcing the chunking path must still split the descriptions into groups,
    summarize each via the LLM, and join the results — i.e. memoizing the
    token counts must not change the chunk boundaries or the final output."""
    from lightrag.operate import _handle_entity_relation_summary

    # Distinct summaries per reduce call so the joined result reflects the
    # number of chunks produced.
    mock_extract = AsyncMock(side_effect=["SUM_ONE", "SUM_TWO", "SUM_THREE"])
    global_config, _ = _make_chunking_config(mock_extract)

    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        "TEST_ENTITY",
        ["alpha", "beta", "gamma", "delta"],
        "<SEP>",
        global_config,
        llm_response_cache=None,
    )

    # Reduce phase ran (the LLM was used to summarize at least one chunk).
    assert llm_was_used is True
    # 4 five-token descriptions under a 10-token window → 2 chunks → 2 summaries
    # joined in the next iteration (len == 2 takes the no-LLM join exit).
    assert description == "SUM_ONE<SEP>SUM_TWO"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_map_phase_encodes_each_description_once():
    """The map phase must tokenize each description exactly once per
    map-reduce iteration. Pre-fix the chunk-building pass re-encoded every
    description, so each of the 4 inputs appeared 2x in the first iteration;
    post-fix it appears exactly once (the second pass reuses the counts)."""
    from lightrag.operate import _handle_entity_relation_summary

    mock_extract = AsyncMock(side_effect=["SUM_ONE", "SUM_TWO", "SUM_THREE"])
    global_config, counter = _make_chunking_config(mock_extract)
    inputs = ["alpha", "beta", "gamma", "delta"]

    await _handle_entity_relation_summary(
        "Entity",
        "TEST_ENTITY",
        inputs,
        "<SEP>",
        global_config,
        llm_response_cache=None,
    )

    # In the first (chunking) iteration each input description must be encoded
    # exactly once by the map phase. Pre-fix this was 2 per description.
    for desc in inputs:
        assert counter.encoded.count(desc) == 1, (
            f"description {desc!r} was encoded {counter.encoded.count(desc)} "
            f"time(s) by the map phase; expected exactly 1 (memoized counts)"
        )
