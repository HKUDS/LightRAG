"""Empty env-backed int defaults must not crash LightRAG class construction.

Follow-up to the COSINE_THRESHOLD fix: several ``LightRAG`` field defaults still
used bare ``int(os.getenv(...))``, which raises when the variable is present but
empty (common in ``.env`` / Compose). ``env.example`` ships live
``EMBEDDING_BATCH_NUM=32``; clearing that value previously made import fail.
"""

from __future__ import annotations

import pytest

from tests._env_import_probe import import_defaults_with_blank_env


@pytest.mark.offline
@pytest.mark.parametrize("env_value", ["", "  ", "\t"])
def test_empty_embedding_batch_num_env_falls_back_on_import(env_value: str) -> None:
    assert import_defaults_with_blank_env(env_value)["EMBEDDING_BATCH_NUM"] == "10"


@pytest.mark.offline
def test_empty_llm_timeout_env_falls_back_on_import() -> None:
    assert import_defaults_with_blank_env("")["LLM_TIMEOUT"] == "240"
