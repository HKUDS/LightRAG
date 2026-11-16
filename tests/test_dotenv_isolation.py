"""The test suite must not load any ``.env`` (see ``tests/__init__.py``)."""

import os

import pytest
from dotenv import find_dotenv, load_dotenv

pytestmark = pytest.mark.offline

_KEY = "LIGHTRAG_TEST_DOTENV_ISOLATION_PROBE"


@pytest.fixture
def dotenv_dir(tmp_path, monkeypatch):
    # Skip only for an explicit opt-out. A missing variable means the switch in
    # tests/__init__.py is gone, and the tests below must then fail.
    switch = os.environ.get("PYTHON_DOTENV_DISABLED")
    if switch is not None and switch.casefold() not in {"1", "true", "t", "yes", "y"}:
        pytest.skip("PYTHON_DOTENV_DISABLED explicitly turned off for this run")
    (tmp_path / ".env").write_text(f"{_KEY}=leaked\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(_KEY, raising=False)
    return tmp_path


def test_load_dotenv_with_a_path_is_inert(dotenv_dir):
    """LightRAG's own pattern: ``load_dotenv(".env")`` at import time."""
    load_dotenv(dotenv_path=".env", override=False)
    assert _KEY not in os.environ


def test_a_searching_load_dotenv_is_inert(dotenv_dir):
    """A dependency's pattern: a bare search for the nearest ``.env``.

    pymilvus calls ``load_dotenv()`` on import, which walks up from its own
    install directory and found the checkout's ``.env`` whenever the venv sat
    inside it. An old python-dotenv ignores the switch; this test fails then.
    """
    load_dotenv(find_dotenv(usecwd=True), override=False)
    assert _KEY not in os.environ
