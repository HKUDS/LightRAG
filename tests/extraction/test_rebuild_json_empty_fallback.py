"""A cached extraction result that is valid, empty JSON must not fall through
to the delimiter parser: that parser has no chance of finding a
"<|COMPLETE|>" marker in a JSON payload and would log a misleading warning
about it, even though the JSON was parsed successfully and simply carried no
entities or relationships.
"""

from __future__ import annotations

import logging

import pytest

from lightrag.operate import _rebuild_from_extraction_result

pytestmark = pytest.mark.offline


class DummyKV:
    async def get_by_id(self, key):
        return {"file_path": "doc.pdf"}


@pytest.fixture
def _propagate_lightrag_logger(monkeypatch):
    """``lightrag.utils.logger`` sets ``propagate = False``; restore
    propagation locally so ``caplog`` can capture WARNING records."""
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


@pytest.mark.asyncio
async def test_rebuild_accepts_valid_empty_json_without_delimiter_warning(
    caplog, _propagate_lightrag_logger
) -> None:
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        nodes, edges = await _rebuild_from_extraction_result(
            DummyKV(),
            '{"entities": [], "relationships": []}',
            chunk_id="chunk-1",
            timestamp=1,
        )
    assert nodes == {}
    assert edges == {}
    assert "Complete delimiter can not be found" not in caplog.text


@pytest.mark.asyncio
async def test_rebuild_still_returns_entities_from_nonempty_json() -> None:
    nodes, edges = await _rebuild_from_extraction_result(
        DummyKV(),
        '{"entities":[{"name":"Alice","type":"person",'
        '"description":"an engineer"}],"relationships":[]}',
        chunk_id="chunk-2",
        timestamp=1,
    )
    assert "Alice" in nodes


@pytest.mark.asyncio
async def test_rebuild_still_falls_back_on_unparseable_json_looking_text(
    caplog, _propagate_lightrag_logger
) -> None:
    """Text that starts with '{' but is not recoverable JSON, and also has no
    completion marker, is the pre-existing garbage-input case: falling
    through to the delimiter parser is correct there, unlike the
    valid-empty-JSON case above.

    The delimiter parser's finding is recorded at DEBUG rather than WARNING on
    this path: the JSON parser has already reported the failure at WARNING, and
    repeating it would make one bad chunk read as two problems. See
    ``test_rebuild_fallback_warning_dedup.py``.
    """
    with caplog.at_level(logging.DEBUG, logger="lightrag"):
        nodes, edges = await _rebuild_from_extraction_result(
            DummyKV(),
            "{not json at all, no delimiter either",
            chunk_id="chunk-3",
            timestamp=1,
        )
    assert nodes == {}
    assert edges == {}
    assert "Complete delimiter can not be found" in caplog.text
