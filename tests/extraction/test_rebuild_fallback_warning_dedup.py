"""One failure must be reported once.

When a cache-rebuild payload looks like JSON but cannot be parsed,
``_process_json_extraction_result`` logs the failure and
``_rebuild_from_extraction_result`` then tries the delimiter parser as a
speculative rescue. That rescue attempt must not announce the same failure a
second time with its own "Complete delimiter can not be found" WARNING: the
delimiter is missing because the payload was never delimiter-formatted, which
is a consequence of the already-reported failure rather than a new finding.

A genuinely delimiter-formatted payload that is missing its terminator is a
different matter -- nothing else reported it, so that WARNING stays. So is a
payload that parses cleanly but carries the wrong schema: the JSON parser
accepts it and says nothing, so demoting there would leave the chunk with no
parse-level warning at all.
"""

from __future__ import annotations

import logging

import pytest

from lightrag.operate import _rebuild_from_extraction_result

pytestmark = pytest.mark.offline

_DELIMITER_WARNING = "Complete delimiter can not be found"
_JSON_WARNING = "JSON extraction result is empty or unrecoverable"


class DummyKV:
    async def get_by_id(self, key):
        return {"file_path": "doc.pdf"}


@pytest.fixture
def _propagate_lightrag_logger(monkeypatch):
    """``lightrag.utils.logger`` sets ``propagate = False``; restore
    propagation locally so ``caplog`` can capture records."""
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


def _warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ]


@pytest.mark.asyncio
async def test_unparseable_json_payload_reports_the_failure_exactly_once(
    caplog, _propagate_lightrag_logger
) -> None:
    with caplog.at_level(logging.DEBUG, logger="lightrag"):
        nodes, edges = await _rebuild_from_extraction_result(
            DummyKV(),
            "{not json at all, no delimiter either",
            chunk_id="chunk-1",
            timestamp=1,
        )

    assert nodes == {}
    assert edges == {}

    warnings = _warnings(caplog)
    assert len(warnings) == 1, warnings
    assert _JSON_WARNING in warnings[0]

    # The rescue attempt's own finding is kept for diagnosis, just not at a
    # level that reads as a second, independent failure.
    assert _DELIMITER_WARNING in caplog.text


@pytest.mark.asyncio
async def test_delimiter_payload_missing_its_terminator_still_warns(
    caplog, _propagate_lightrag_logger
) -> None:
    """The payload never looked like JSON, so no one else reported anything and
    the delimiter parser's warning is the only signal there is."""
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        await _rebuild_from_extraction_result(
            DummyKV(),
            '("entity"<|#|>Alice<|#|>person<|#|>an engineer)',
            chunk_id="chunk-2",
            timestamp=1,
        )

    warnings = _warnings(caplog)
    assert any(_DELIMITER_WARNING in message for message in warnings), warnings


@pytest.mark.asyncio
async def test_json_payload_rescued_by_the_delimiter_parser_keeps_its_records(
    caplog, _propagate_lightrag_logger
) -> None:
    """Demoting the warning must not disturb the rescue itself: an empty JSON
    object with delimiter records appended still yields those records."""
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        nodes, _edges = await _rebuild_from_extraction_result(
            DummyKV(),
            '{}\n("entity"<|#|>Alice<|#|>person<|#|>an engineer)\n<|COMPLETE|>',
            chunk_id="chunk-3",
            timestamp=1,
        )

    assert "Alice" in nodes


@pytest.mark.asyncio
async def test_wrong_schema_json_payload_keeps_the_fallback_warning(
    caplog, _propagate_lightrag_logger
) -> None:
    """A payload that parses but has none of the extraction fields extracts
    nothing, and ``_process_json_extraction_result`` reports nothing about it.
    Demotion keys off the parse outcome rather than off the empty result, so
    this chunk keeps the one warning it has -- inferring "already reported"
    from emptiness would silence it entirely."""
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        nodes, edges = await _rebuild_from_extraction_result(
            DummyKV(),
            '{"answer": "no entities found"}',
            chunk_id="chunk-4",
            timestamp=1,
        )

    assert nodes == {}
    assert edges == {}

    warnings = _warnings(caplog)
    assert any(_DELIMITER_WARNING in message for message in warnings), warnings
    assert not any(_JSON_WARNING in message for message in warnings), warnings
