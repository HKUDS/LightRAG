"""Empty env-backed QueryParam int defaults must not crash import.

``QueryParam`` field defaults previously used bare ``int(os.getenv(...))``,
which raises when the variable is present but empty (common in ``.env`` /
Compose). Sibling ``LightRAG`` knobs already go through ``get_env_value``.
"""

from __future__ import annotations

import pytest

from tests._env_import_probe import import_defaults_with_blank_env


_QUERYPARAM_INT_ENVS = [
    ("TOP_K", "40"),
    ("CHUNK_TOP_K", "20"),
    ("MAX_ENTITY_TOKENS", "6000"),
    ("MAX_RELATION_TOKENS", "8000"),
    ("MAX_TOTAL_TOKENS", "30000"),
]


# Every field reads its variable through the same get_env_value: each
# variable with "", and the other blank spellings on TOP_K.
@pytest.mark.offline
@pytest.mark.parametrize(
    "env_value,env_key,expected",
    [("", key, expected) for key, expected in _QUERYPARAM_INT_ENVS]
    + [(blank, *_QUERYPARAM_INT_ENVS[0]) for blank in ("  ", "\t")],
)
def test_empty_queryparam_int_env_falls_back_on_import(
    env_value: str, env_key: str, expected: str
) -> None:
    assert import_defaults_with_blank_env(env_value)[env_key] == expected
