"""Package-level environment isolation for the native markdown parser tests.

``tests/conftest.py`` runs ``load_dotenv(".env", override=False)`` at
``pytest_configure`` time, so a developer's real ``.env`` leaks every
``NATIVE_MD_IMAGE_*`` setting into these tests. The image-budget tests assert
on defaults, so a local ``.env`` that tunes any of them would make the suite
pass or fail depending on the machine.

This fixture has to live at package scope: the tests that depend on it are
spread across ``test_parser.py``, ``test_raw_cache.py``, ``test_e2e_sidecar.py``
and ``test_download_deadline.py``, and a module-scoped fixture in one of them
would not protect the others.

Ordering note: this runs BEFORE the module-level autouse fixtures in individual
test files (pytest resolves same-scope autouse fixtures outermost-first), so a
module is free to ``setenv`` whatever it needs afterwards.
"""

from __future__ import annotations

import pytest

_NATIVE_MD_ENV_VARS = (
    "NATIVE_MD_IMAGE_DOWNLOAD_ENABLED",
    "NATIVE_MD_IMAGE_DOWNLOAD_REQUIRED",
    "NATIVE_MD_IMAGE_DOWNLOAD_TIMEOUT",
    "NATIVE_MD_IMAGE_DOWNLOAD_TOTAL_TIMEOUT",
    "NATIVE_MD_IMAGE_MAX_BYTES",
    "NATIVE_MD_IMAGE_MAX_TOTAL_BYTES",
    "NATIVE_MD_IMAGE_MAX_REQUESTS",
    "NATIVE_MD_IMAGE_MAX_SVG_PIXELS",
    "NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS",
    "LIGHTRAG_FORCE_REPARSE_NATIVE",
)


@pytest.fixture(autouse=True)
def _isolate_native_md_env(monkeypatch):
    """Start every test in this package from the shipped defaults."""
    for name in _NATIVE_MD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
