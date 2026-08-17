"""The cleanup tool's mode list must stay in sync with QueryParam.mode.

Cache keys are ``{mode}:{cache_type}:{hash}``, and the tool counts and deletes
per mode from an explicit list. A mode missing from that list is invisible to the
tool: its entries are neither reported nor removed, so an operator who runs the
tool believes the query cache is clean when it is not. ``naive`` was missing this
way. Deriving the expectation from the annotation rather than restating the list
means a newly added mode fails here instead of being silently skipped.
"""

import typing

import pytest

from lightrag.base import QueryParam
from lightrag.tools.clean_llm_query_cache import QUERY_MODES

pytestmark = pytest.mark.offline

# "bypass" answers straight from the LLM without touching the query cache, so it
# is the one mode that legitimately has no entries to clean.
_MODES_WITHOUT_CACHE = {"bypass"}


def _declared_query_modes() -> set[str]:
    hints = typing.get_type_hints(QueryParam)
    return set(typing.get_args(hints["mode"]))


def test_cleanup_tool_covers_every_cache_writing_query_mode():
    expected = _declared_query_modes() - _MODES_WITHOUT_CACHE
    missing = expected - set(QUERY_MODES)
    assert not missing, (
        f"QUERY_MODES misses {sorted(missing)}; their cache entries would be "
        "neither counted nor deleted by the cleanup tool"
    )


def test_cleanup_tool_declares_no_unknown_query_mode():
    unknown = set(QUERY_MODES) - _declared_query_modes()
    assert not unknown, f"QUERY_MODES lists non-existent modes: {sorted(unknown)}"


def test_cleanup_tool_does_not_scan_bypass_mode():
    """bypass writes no cache entries, so scanning for them would be dead work."""
    assert "bypass" not in QUERY_MODES
