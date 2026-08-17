"""Native DOCX content-limit validation must *raise*, never ``sys.exit``.

The DOCX parser runs in-process inside the gunicorn/uvicorn worker (via
``LightRAG.parse_native`` → ``extract_docx_blocks``). A ``sys.exit`` there
raises ``SystemExit`` (a ``BaseException``), which slips past the pipeline's
per-document ``except Exception`` handler and tears down the whole worker —
gunicorn then reports ``WORKER TIMEOUT`` and kills it with SIGABRT (code 134),
interrupting the entire processing pipeline.

These tests lock in that the content-limit checks raise ``DocxContentError``
(an ``Exception`` subclass) so just the offending document fails while the
worker keeps running.
"""

from __future__ import annotations

import pytest

from lightrag.parser.docx.parse_document import (
    DocxContentError,
    MAX_HEADING_LENGTH,
    validate_heading_length,
)


def test_docx_content_error_is_an_exception_not_just_baseexception():
    # The pipeline parse worker catches ``except Exception``; SystemExit
    # (BaseException) would bypass it. Guard against a future regression that
    # reintroduces a BaseException-only error type.
    assert issubclass(DocxContentError, Exception)


def test_validate_heading_length_raises_instead_of_exiting():
    long_heading = "x" * (MAX_HEADING_LENGTH + 213)  # mirrors the 413-char report
    with pytest.raises(DocxContentError) as exc_info:
        validate_heading_length(long_heading, "p1")
    message = str(exc_info.value)
    assert f"Heading too long ({len(long_heading)} characters" in message
    assert "SOLUTION:" in message


def test_validate_heading_length_does_not_raise_within_limit():
    # Exactly at the limit and below must pass through silently.
    validate_heading_length("x" * MAX_HEADING_LENGTH, "p2")
    validate_heading_length("short heading", "p3")
