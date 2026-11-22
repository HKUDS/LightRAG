"""Boolean env-backed dataclass defaults use the shared env parser."""

from __future__ import annotations

import pytest

from tests._env_import_probe import import_defaults_with_blank_env


@pytest.mark.offline
@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (
            "1",
            {
                "RERANK_BY_DEFAULT": "True",
                "ENTITY_EXTRACTION_USE_JSON": "True",
            },
        ),
        (
            "ture",
            {
                "RERANK_BY_DEFAULT": "True",
                "ENTITY_EXTRACTION_USE_JSON": "False",
            },
        ),
    ],
)
def test_bool_env_field_defaults_use_shared_parser(
    env_value: str, expected: dict[str, str]
) -> None:
    defaults = import_defaults_with_blank_env(env_value)
    assert {key: defaults[key] for key in expected} == expected
