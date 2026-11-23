"""The import-time boolean env defaults use the shared boolean syntax."""

import json
import os
import subprocess
import sys

import pytest

from tests._env_import_probe import REPO_ROOT

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("env", "expected", "warn_keys"),
    # Each case pins one flag of each pair to an alias and the OTHER flag to a
    # typo, with the roles swapped between cases, so a wrong fallback default
    # or two flags reading the same variable fails one case.
    [
        pytest.param(
            {
                "RERANK_BY_DEFAULT": " ON ",
                "ENTITY_EXTRACTION_USE_JSON": "ture",
                "VERBOSE": "1",
                "LIGHTRAG_PERFORMANCE_TIMING_LOGS": "ture",
            },
            {
                "rerank": True,
                "json_extraction": False,
                "verbose": True,
                "timing": False,
            },
            ("ENTITY_EXTRACTION_USE_JSON", "LIGHTRAG_PERFORMANCE_TIMING_LOGS"),
            id="rerank-verbose-alias-json-timing-typo",
        ),
        pytest.param(
            {
                "RERANK_BY_DEFAULT": "ture",
                "ENTITY_EXTRACTION_USE_JSON": "yes",
                "VERBOSE": "ture",
                "LIGHTRAG_PERFORMANCE_TIMING_LOGS": " ON ",
            },
            {"rerank": True, "json_extraction": True, "verbose": False, "timing": True},
            ("RERANK_BY_DEFAULT", "VERBOSE"),
            id="json-timing-alias-rerank-verbose-typo",
        ),
    ],
)
def test_boolean_env_field_defaults(
    env: dict[str, str], expected: dict[str, bool], warn_keys: tuple[str, ...]
) -> None:
    # All four defaults are evaluated on import. Reading the LightRAG field
    # and the lightrag.utils flags exercises that evaluation without
    # constructing unrelated storages.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json

import lightrag.utils as utils
from lightrag import LightRAG, QueryParam

json_default = LightRAG.__dataclass_fields__["entity_extraction_use_json"].default
print(json.dumps({
    "rerank": QueryParam().enable_rerank,
    "json_extraction": json_default,
    "verbose": utils.VERBOSE_DEBUG,
    "timing": utils.PERFORMANCE_TIMING_LOGS,
}))
""",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHON_DOTENV_DISABLED": "1",
            **env,
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(result.stdout) == expected
    for key in warn_keys:
        assert f"WARNING: Environment variable {key}='ture'" in result.stderr
