"""Regression tests for the map-reduce chunking path of
``_handle_entity_relation_summary``.

The map phase tokenizes every description to decide (a) whether the list
needs splitting and (b) where to cut the chunks. Historically the same
description was encoded twice per map-reduce iteration — once to sum the
total token count and again while building the chunks — which doubled the
tiktoken BPE cost on every entity/edge merge. The counts from the first
pass are now memoized and reused by the second.

These tests guard:

* the groups fit ``summary_context_size`` as the summary prompt renders them
  (one JSON line per description), so its trim drops nothing,
* each description is encoded exactly once per map-reduce iteration
  (the perf property the memoization introduced),
* the no-LLM join of a short list is decided on the bare texts it stores, and
* a description the prompt trim does drop is logged.
"""

import json
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
    chunking path: the map phase counts each description as its prompt line
    (``{"Description": "alpha"}`` plus a newline, 24-25 tokens), so two lines
    fit the 50-token window and four do not. A merge count of 3 sends the four
    inputs to the LLM and joins the two summaries without it."""
    counter = _CountingTokenizer()
    tokenizer = Tokenizer("dummy", counter)
    global_config = {
        "role_llm_funcs": {"extract": mock_extract},
        "addon_params": {},
        "_resolved_summary_language": "English",
        "summary_length_recommended": 100,
        # alpha/beta/gamma/delta lines: 25 + 24 + 25 + 25 = 99 > 50 → chunking
        "summary_context_size": 50,
        "summary_max_tokens": 100_000,
        "force_llm_summary_on_merge": 3,
        "tokenizer": tokenizer,
    }
    return global_config, counter


@pytest.mark.offline
@pytest.mark.asyncio
async def test_map_phase_groups_fit_the_prompt_and_encode_once():
    """Groups are sized by prompt lines, each encoded once per iteration.

    The four bare texts (19 tokens) fit the 50-token window, but their prompt
    lines (99 tokens) do not. Counting the bare text sent all four to one
    summary call, whose trim to the window kept alpha and beta and silently
    dropped gamma and delta. The chunk-building pass must also reuse the first
    pass's counts instead of re-encoding every line.
    """
    from lightrag.operate import _handle_entity_relation_summary

    # Distinct summaries per reduce call so the joined result reflects the
    # number of chunks produced.
    mock_extract = AsyncMock(side_effect=["SUM_ONE", "SUM_TWO", "SUM_THREE"])
    global_config, counter = _make_chunking_config(mock_extract)
    inputs = ["alpha", "beta", "gamma", "delta"]

    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        "TEST_ENTITY",
        inputs,
        "<SEP>",
        global_config,
        llm_response_cache=None,
    )

    # Lines of 25 + 24 | 25 + 25 tokens under a 50-token window → 2 chunks,
    # each prompt carrying both of its descriptions → 2 summaries joined in the
    # next iteration (2 is below the merge count: the no-LLM join exit).
    prompts = [call.args[0] for call in mock_extract.await_args_list]
    assert [[desc for desc in inputs if desc in prompt] for prompt in prompts] == [
        ["alpha", "beta"],
        ["gamma", "delta"],
    ]
    assert (description, llm_was_used) == ("SUM_ONE<SEP>SUM_TWO", True)
    # The map phase encodes each input's prompt line, plus its newline,
    # exactly once. The prompt trim encodes the joined lines, never this form.
    line_encodes = {
        desc: counter.encoded.count(
            json.dumps({"Description": desc}, ensure_ascii=False) + "\n"
        )
        for desc in inputs
    }
    assert line_encodes == dict.fromkeys(inputs, 1)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_short_list_whose_bare_texts_fit_is_joined_without_llm():
    """Below the merge count, the no-LLM join is decided on the bare texts.

    alpha, beta and gamma are 14 bare tokens, inside the 30-token window, but
    74 as prompt lines. Gating the join on the prompt lines sent them to
    map-reduce, whose forced alpha/beta pair overflowed the window and lost
    beta from the stored description.
    """
    from lightrag.operate import _handle_entity_relation_summary

    mock_extract = AsyncMock(side_effect=["SUM_ONE", "SUM_TWO"])
    global_config, _ = _make_chunking_config(mock_extract)
    global_config["summary_context_size"] = 30
    global_config["force_llm_summary_on_merge"] = 8

    result = await _handle_entity_relation_summary(
        "Entity",
        "TEST_ENTITY",
        ["alpha", "beta", "gamma"],
        "<SEP>",
        global_config,
        llm_response_cache=None,
    )

    assert result == ("alpha<SEP>beta<SEP>gamma", False)
    assert mock_extract.await_count == 0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_oversize_descriptions_left_out_of_the_prompt_are_logged(
    lightrag_log_records,
):
    """Two descriptions whose lines (25 + 24 tokens) overflow a 30-token window
    reach the final call together (a merge count of 2 rules out the plain
    join); its trim keeps only the first, and the loss must be logged rather
    than silent."""
    from lightrag.operate import _handle_entity_relation_summary

    mock_extract = AsyncMock(side_effect=["SUM_ONE"])
    global_config, _ = _make_chunking_config(mock_extract)
    global_config["summary_context_size"] = 30
    global_config["force_llm_summary_on_merge"] = 2

    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        "TEST_ENTITY",
        ["alpha", "beta"],
        "<SEP>",
        global_config,
        llm_response_cache=None,
    )

    (prompt,) = [call.args[0] for call in mock_extract.await_args_list]
    assert ("alpha" in prompt, "beta" in prompt) == (True, False)
    assert (description, llm_was_used) == ("SUM_ONE", True)
    warnings = [
        record.getMessage()
        for record in lightrag_log_records
        if "left out of the summary" in record.getMessage()
    ]
    assert warnings == [
        "Summarizing Entity:TEST_ENTITY: 1 of 2 descriptions exceed "
        "summary_context_size (30 tokens) and are left out of the summary"
    ]
