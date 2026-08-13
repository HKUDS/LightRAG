"""Tests for the parser cancellation exception hierarchy and its
backward-compatible LLMBridge* aliases (see issue #3608).

The hierarchy moved from lightrag.parser.llm_bridge to
lightrag.parser.exceptions and was renamed ParseCancelled /
ParsePipelineCancelled / ParseShutdown to reflect its actual parser-wide
scope. The old LLMBridge* names remain importable from llm_bridge.py as
plain assignments (not subclasses), so isinstance/except behave identically
regardless of which name is used to catch.
"""

from __future__ import annotations

import pytest

from lightrag.parser.exceptions import (
    ParseCancelled,
    ParsePipelineCancelled,
    ParseShutdown,
)
from lightrag.parser.llm_bridge import (
    LLMBridgeCancelled,
    LLMBridgePipelineCancelled,
    LLMBridgeShutdown,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Alias identity: old names must be the SAME object as the new names, not
# subclasses — a subclass would break `except LLMBridgeCancelled` catching a
# raised ParseCancelled.
# ---------------------------------------------------------------------------


def test_llm_bridge_cancelled_is_parse_cancelled() -> None:
    assert LLMBridgeCancelled is ParseCancelled


def test_llm_bridge_pipeline_cancelled_is_parse_pipeline_cancelled() -> None:
    assert LLMBridgePipelineCancelled is ParsePipelineCancelled


def test_llm_bridge_shutdown_is_parse_shutdown() -> None:
    assert LLMBridgeShutdown is ParseShutdown


# ---------------------------------------------------------------------------
# Hierarchy shape, under the new names.
# ---------------------------------------------------------------------------


def test_parse_pipeline_cancelled_is_a_parse_cancelled() -> None:
    assert issubclass(ParsePipelineCancelled, ParseCancelled)


def test_parse_shutdown_is_a_parse_cancelled() -> None:
    assert issubclass(ParseShutdown, ParseCancelled)


def test_parse_cancelled_is_a_runtime_error() -> None:
    assert issubclass(ParseCancelled, RuntimeError)


# ---------------------------------------------------------------------------
# Cross-catch: raise via one name, catch via the other — both directions.
# ---------------------------------------------------------------------------


def test_raise_new_name_caught_by_old_name() -> None:
    with pytest.raises(LLMBridgeCancelled):
        raise ParseCancelled("boom")


def test_raise_old_name_caught_by_new_name() -> None:
    with pytest.raises(ParseCancelled):
        raise LLMBridgeCancelled("boom")


def test_raise_new_pipeline_cancelled_caught_by_old_name() -> None:
    with pytest.raises(LLMBridgePipelineCancelled):
        raise ParsePipelineCancelled("boom")


def test_raise_old_pipeline_cancelled_caught_by_new_name() -> None:
    with pytest.raises(ParsePipelineCancelled):
        raise LLMBridgePipelineCancelled("boom")


def test_raise_new_shutdown_caught_by_old_name() -> None:
    with pytest.raises(LLMBridgeShutdown):
        raise ParseShutdown("boom")


def test_raise_old_shutdown_caught_by_new_name() -> None:
    with pytest.raises(ParseShutdown):
        raise LLMBridgeShutdown("boom")


def test_raise_pipeline_cancelled_caught_by_base_class_either_name() -> None:
    # A subclass instance must still satisfy the broader base-class except,
    # under both the old and new spelling of the base.
    with pytest.raises(ParseCancelled):
        raise LLMBridgePipelineCancelled("boom")
    with pytest.raises(LLMBridgeCancelled):
        raise ParsePipelineCancelled("boom")
