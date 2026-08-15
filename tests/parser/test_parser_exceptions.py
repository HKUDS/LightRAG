"""Tests for the parser cancellation exception hierarchy (see issue #3608).

The hierarchy lives in lightrag.parser.exceptions as ParseCancelled /
ParsePipelineCancelled / ParseShutdown, reflecting its actual parser-wide
scope. The historical LLMBridge* names were internal-only and have been
removed outright — no compatibility aliases.
"""

from __future__ import annotations

import pytest

from lightrag.parser.exceptions import (
    ParseCancelled,
    ParsePipelineCancelled,
    ParseShutdown,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Hierarchy shape.
# ---------------------------------------------------------------------------


def test_parse_pipeline_cancelled_is_a_parse_cancelled() -> None:
    assert issubclass(ParsePipelineCancelled, ParseCancelled)


def test_parse_shutdown_is_a_parse_cancelled() -> None:
    assert issubclass(ParseShutdown, ParseCancelled)


def test_parse_cancelled_is_a_runtime_error() -> None:
    assert issubclass(ParseCancelled, RuntimeError)


# ---------------------------------------------------------------------------
# The internal-only LLMBridge* aliases are gone — the rename is complete, and
# nothing should quietly reintroduce the misleading old names.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old_name",
    ["LLMBridgeCancelled", "LLMBridgePipelineCancelled", "LLMBridgeShutdown"],
)
def test_old_llm_bridge_names_are_removed(old_name: str) -> None:
    import lightrag.parser.llm_bridge as llm_bridge

    assert not hasattr(llm_bridge, old_name)
