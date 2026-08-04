"""Empty env-backed int defaults must not crash LightRAG class construction.

Follow-up to the COSINE_THRESHOLD fix: several ``LightRAG`` field defaults still
used bare ``int(os.getenv(...))``, which raises when the variable is present but
empty (common in ``.env`` / Compose). ``env.example`` ships live
``EMBEDDING_BATCH_NUM=32``; clearing that value previously made import fail.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_field_default(env_key: str, env_value: str, field_name: str) -> str:
    env = os.environ.copy()
    env[env_key] = env_value
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from lightrag.lightrag import LightRAG; "
            f"print(LightRAG.__dataclass_fields__[{field_name!r}].default)",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.offline
@pytest.mark.parametrize("env_value", ["", "  ", "\t"])
def test_empty_embedding_batch_num_env_falls_back_on_import(env_value: str) -> None:
    assert (
        _import_field_default("EMBEDDING_BATCH_NUM", env_value, "embedding_batch_num")
        == "10"
    )


@pytest.mark.offline
def test_empty_llm_timeout_env_falls_back_on_import() -> None:
    assert _import_field_default("LLM_TIMEOUT", "", "default_llm_timeout") == "240"
