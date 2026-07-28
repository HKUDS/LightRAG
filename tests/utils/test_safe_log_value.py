"""``safe_log_value`` — the shared log-injection guard.

Untrusted input reaches the log in more than one place (rate-limit keys built
from a submitted username, operator-supplied identifiers in audited admin
actions), so the sanitizer lives in ``lightrag.utils`` and is tested here once.
Its call sites keep their own behavioural tests.
"""

import pytest

from lightrag.utils import safe_log_value

pytestmark = pytest.mark.offline


def test_control_characters_are_neutralized():
    # CR/LF would otherwise let a caller forge an extra log line.
    assert safe_log_value("a\r\nb") == "a??b"
    assert safe_log_value("admin\nCRITICAL forged") == "admin?CRITICAL forged"


def test_over_long_values_are_truncated():
    long = safe_log_value("x" * 500, max_length=100)
    assert long.startswith("x" * 100)
    assert "truncated" in long
    assert len(long) < 500


def test_printable_unicode_survives():
    # Sanitizing must not mangle legitimate non-ASCII identifiers.
    assert safe_log_value("报告.docx") == "报告.docx"
