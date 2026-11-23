"""The public dataclass defaults use the shared boolean environment syntax."""

import json
import os
import subprocess
import sys

import pytest

from tests._env_import_probe import REPO_ROOT

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("rerank", "json_extraction", "expected_json", "warn_key"),
    # Each case pins one field's alias parsing and the OTHER field's typo
    # fallback, so a wrong fallback default on either field fails one case.
    [
        pytest.param(
            " ON ",
            "ture",
            False,
            "ENTITY_EXTRACTION_USE_JSON",
            id="rerank-alias-json-typo-keeps-false",
        ),
        pytest.param(
            "ture",
            "yes",
            True,
            "RERANK_BY_DEFAULT",
            id="json-alias-rerank-typo-keeps-true",
        ),
    ],
)
def test_boolean_env_field_defaults(
    rerank: str, json_extraction: str, expected_json: bool, warn_key: str
) -> None:
    # Both defaults are evaluated on import. Reading the LightRAG field lets
    # us exercise that evaluation without constructing unrelated storages.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json

from lightrag import LightRAG, QueryParam

json_default = LightRAG.__dataclass_fields__["entity_extraction_use_json"].default
print(json.dumps({"rerank": QueryParam().enable_rerank, "json_extraction": json_default}))
""",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHON_DOTENV_DISABLED": "1",
            "RERANK_BY_DEFAULT": rerank,
            "ENTITY_EXTRACTION_USE_JSON": json_extraction,
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(result.stdout) == {
        "rerank": True,
        "json_extraction": expected_json,
    }
    assert f"WARNING: Environment variable {warn_key}='ture'" in result.stderr
