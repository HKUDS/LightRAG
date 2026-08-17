"""Empty CHUNK_V_BUFFER_SIZE must not crash default_chunker_config."""

from __future__ import annotations

import pytest

from lightrag.parser.routing import default_chunker_config


@pytest.mark.offline
@pytest.mark.parametrize("env_value", ["", "  ", "\t"])
def test_empty_chunk_v_buffer_size_falls_back(
    env_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHUNK_V_BUFFER_SIZE", env_value)
    assert default_chunker_config()["semantic_vector"]["buffer_size"] == 1


@pytest.mark.offline
def test_valid_chunk_v_buffer_size_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_V_BUFFER_SIZE", "3")
    assert default_chunker_config()["semantic_vector"]["buffer_size"] == 3
