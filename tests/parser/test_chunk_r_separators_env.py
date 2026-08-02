"""Empty CHUNK_R_SEPARATORS must not crash default_chunker_config.

``default_chunker_config`` previously used bare ``json.loads(os.getenv(...))``,
which raises ``JSONDecodeError`` when the variable is present but empty.
Sibling ``load_chunk_separators`` already falls back to defaults.
"""

from __future__ import annotations

import json

import pytest

from lightrag.constants import DEFAULT_R_SEPARATORS
from lightrag.parser.routing import default_chunker_config


@pytest.mark.offline
@pytest.mark.parametrize("env_value", ["", "  ", "\t", "not-json", "[1,2]"])
def test_empty_or_invalid_chunk_r_separators_falls_back(
    env_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHUNK_R_SEPARATORS", env_value)
    separators = default_chunker_config()["recursive_character"]["separators"]
    assert separators == list(DEFAULT_R_SEPARATORS)


@pytest.mark.offline
def test_valid_chunk_r_separators_are_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_R_SEPARATORS", '["\\n\\n", "\\n", " "]')
    separators = default_chunker_config()["recursive_character"]["separators"]
    assert separators == ["\n\n", "\n", " "]


@pytest.mark.offline
def test_out_of_bounds_env_cascade_is_cached_and_warned_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A static bad environment value must not warn once per document/config read."""
    import lightrag.chunker.recursive_character as recursive_character

    separators = [f"env-cache-{index}" for index in range(70)]
    monkeypatch.setenv("CHUNK_R_SEPARATORS", json.dumps(separators))
    warnings: list[str] = []
    monkeypatch.setattr(recursive_character.logger, "warning", warnings.append)

    first = default_chunker_config()["recursive_character"]["separators"]
    second = default_chunker_config()["recursive_character"]["separators"]

    assert first == separators[:64]
    assert second == separators[:64]
    assert (
        sum("[CHUNK_R_SEPARATORS] separator cascade" in message for message in warnings)
        == 1
    )
