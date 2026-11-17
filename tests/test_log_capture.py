"""``lightrag_caplog`` / ``capture_lightrag_logs`` see each record exactly once.

They must hold on both sides of pytest 9.1, which started attaching caplog's
handler to non-propagating loggers itself (see ``tests/_log_capture.py``).
"""

from __future__ import annotations

import logging

import pytest

from tests._log_capture import capture_lightrag_logs

pytestmark = pytest.mark.offline

_logger = logging.getLogger("lightrag")


def _warnings_with(caplog, needle: str) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and needle in r.getMessage()
    ]


def test_fixture_captures_one_record_per_call(lightrag_caplog):
    with lightrag_caplog.at_level(logging.WARNING, logger="lightrag"):
        _logger.warning("fixture-probe")

    assert len(_warnings_with(lightrag_caplog, "fixture-probe")) == 1
    assert lightrag_caplog.text.count("fixture-probe") == 1


def test_context_manager_captures_one_record_per_call(caplog):
    with capture_lightrag_logs(caplog), caplog.at_level(logging.WARNING):
        _logger.warning("helper-probe")

    assert len(_warnings_with(caplog, "helper-probe")) == 1


def test_context_manager_leaves_pytests_own_attachment_in_place(caplog):
    attached_before = caplog.handler in _logger.handlers
    with capture_lightrag_logs(caplog):
        assert caplog.handler in _logger.handlers
    assert (caplog.handler in _logger.handlers) == attached_before


def test_records_survive_the_block(caplog):
    with capture_lightrag_logs(caplog), caplog.at_level(logging.WARNING):
        _logger.warning("after-probe")
    _logger.warning("outside-probe")  # may or may not be seen; not asserted

    assert len(_warnings_with(caplog, "after-probe")) == 1
