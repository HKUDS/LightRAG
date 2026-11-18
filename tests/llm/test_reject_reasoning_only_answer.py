"""A query answer that is a reasoning trace and nothing else is a failure.

``reject_reasoning_only_answer`` is the guard the query paths apply to a
non-streaming answer: the query role is called with ``enable_cot=True``, so a
binding that surfaces reasoning wraps it in ``<think>...</think>``, and a
reasoning-only response then looks like a non-empty answer.
"""

import logging

import pytest

from lightrag.exceptions import EmptyTruncatedResponseError, ReasoningOnlyResponseError
from lightrag.utils import TruncatedResponse, reject_reasoning_only_answer

pytestmark = pytest.mark.offline

REASONING = "SECRET-REASONING the manual says to hold the reset button"


@pytest.mark.parametrize(
    "response",
    [
        f"<think>{REASONING}</think>",
        f"  <think>{REASONING}</think>  \n",
        # Orphaned closing tag (a stream that began mid-trace).
        f"{REASONING}</think>",
        # Unclosed block: inline reasoning cut off before the answer began.
        f"<think>{REASONING}",
    ],
)
def test_reasoning_only_answer_is_rejected(response):
    with pytest.raises(ReasoningOnlyResponseError):
        reject_reasoning_only_answer(response)


def test_truncated_reasoning_only_answer_is_a_truncation_error():
    with pytest.raises(EmptyTruncatedResponseError):
        reject_reasoning_only_answer(TruncatedResponse(f"<think>{REASONING}</think>"))


@pytest.mark.parametrize(
    "response",
    [
        "Hold the reset button for ten seconds.",
        # The intended enable_cot output: a trace shown before the answer.
        f"<think>{REASONING}</think>Hold the reset button for ten seconds.",
        TruncatedResponse("Hold the reset button for"),
        # Empty and non-string responses are left to the existing handling.
        "",
        None,
    ],
)
def test_answers_and_non_answers_pass_through(response):
    reject_reasoning_only_answer(response)


def test_streaming_iterator_passes_through():
    async def _stream():
        yield "<think>"

    reject_reasoning_only_answer(_stream())


def test_log_and_error_carry_lengths_never_content(caplog):
    caplog.set_level(logging.DEBUG, logger="lightrag")
    response = f"<think>{REASONING}</think>"

    with pytest.raises(ReasoningOnlyResponseError) as caught:
        reject_reasoning_only_answer(response)

    assert "SECRET-REASONING" not in caplog.text
    assert "SECRET-REASONING" not in str(caught.value)
    assert f"response_len={len(response)}" in str(caught.value)
