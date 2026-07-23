"""Empty env-backed QueryParam int defaults must not crash import.

``QueryParam`` field defaults previously used bare ``int(os.getenv(...))``,
which raises when the variable is present but empty (common in ``.env`` /
Compose). Sibling ``LightRAG`` knobs already go through ``get_env_value``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_queryparam_field_default(
    env_key: str, env_value: str, field_name: str
) -> str:
    env = os.environ.copy()
    env[env_key] = env_value
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from lightrag.base import QueryParam; "
            f"print(QueryParam.__dataclass_fields__[{field_name!r}].default)",
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
@pytest.mark.parametrize(
    "env_key,field_name,expected",
    [
        ("TOP_K", "top_k", "40"),
        ("CHUNK_TOP_K", "chunk_top_k", "20"),
        ("MAX_ENTITY_TOKENS", "max_entity_tokens", "6000"),
        ("MAX_RELATION_TOKENS", "max_relation_tokens", "8000"),
        ("MAX_TOTAL_TOKENS", "max_total_tokens", "30000"),
    ],
)
def test_empty_queryparam_int_env_falls_back_on_import(
    env_value: str, env_key: str, field_name: str, expected: str
) -> None:
    assert _import_queryparam_field_default(env_key, env_value, field_name) == expected
