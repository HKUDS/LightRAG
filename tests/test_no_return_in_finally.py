"""No ``return`` / ``break`` / ``continue`` may leave a ``finally`` block.

Such a jump silently discards the exception that is propagating -- including
``asyncio.CancelledError`` -- which breaks the "a failure must never be
swallowed" rule. Python 3.14 flags it with a ``SyntaxWarning`` (PEP 765);
ruff's B012 does not catch a jump nested in a ``try`` inside the ``finally``.
The check therefore uses the interpreter's own rule and runs on 3.14+ only.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import pytest

pytestmark = [
    pytest.mark.offline,
    pytest.mark.skipif(
        sys.version_info < (3, 14), reason="PEP 765 SyntaxWarning needs 3.14+"
    ),
]

_PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "lightrag"


def test_package_has_no_jump_out_of_finally():
    offenders = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        offenders += [
            f"{path.relative_to(_PACKAGE.parent)}:{w.lineno}: {w.message}"
            for w in caught
            if issubclass(w.category, SyntaxWarning) and "finally" in str(w.message)
        ]
    assert offenders == []
