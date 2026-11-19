"""Empty COSINE_THRESHOLD must not crash LightRAG class construction.

``cosine_better_than_threshold`` previously used bare ``float(os.getenv(...))``,
which raises ``ValueError`` when the variable is present but empty (common in
``.env`` / Docker Compose). Sibling knobs already go through ``get_env_value``.
"""

from __future__ import annotations

import pytest

from tests._env_import_probe import import_defaults_with_blank_env


@pytest.mark.offline
@pytest.mark.parametrize("env_value", ["", "  ", "\t"])
def test_empty_cosine_threshold_env_falls_back_on_import(env_value: str) -> None:
    assert import_defaults_with_blank_env(env_value)["COSINE_THRESHOLD"] == "0.2"
