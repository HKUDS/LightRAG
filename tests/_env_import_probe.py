"""Shared subprocess probe for "empty env var must not crash import" tests.

Env-backed dataclass defaults are evaluated once, at import, so each check
needs a fresh interpreter. Spawning one per (variable, value) pair cost
~1.3s apiece and dominated the offline suite's slowest-tests list. Instead,
one interpreter per blank value sets EVERY probed variable to it and reports
every default at once; results are cached per value, so the whole family of
tests spawns one process per distinct blank value.

Setting all variables together keeps each assertion meaningful: every field
reads only its own variable, and a crash in any of them fails the import
with a traceback naming the field.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# env var -> (module, dataclass, field) whose default it feeds.
PROBED_FIELDS: dict[str, tuple[str, str, str]] = {
    "TOP_K": ("lightrag.base", "QueryParam", "top_k"),
    "CHUNK_TOP_K": ("lightrag.base", "QueryParam", "chunk_top_k"),
    "MAX_ENTITY_TOKENS": ("lightrag.base", "QueryParam", "max_entity_tokens"),
    "MAX_RELATION_TOKENS": ("lightrag.base", "QueryParam", "max_relation_tokens"),
    "MAX_TOTAL_TOKENS": ("lightrag.base", "QueryParam", "max_total_tokens"),
    "EMBEDDING_BATCH_NUM": ("lightrag.lightrag", "LightRAG", "embedding_batch_num"),
    "LLM_TIMEOUT": ("lightrag.lightrag", "LightRAG", "default_llm_timeout"),
    "COSINE_THRESHOLD": (
        "lightrag.lightrag",
        "LightRAG",
        "cosine_better_than_threshold",
    ),
}

_CHILD_SCRIPT = """
import importlib, json, sys
fields = json.loads(sys.argv[1])
out = {}
for key, (module, cls, field) in fields.items():
    owner = getattr(importlib.import_module(module), cls)
    out[key] = str(owner.__dataclass_fields__[field].default)
print(json.dumps(out))
"""


@functools.lru_cache(maxsize=None)
def import_defaults_with_blank_env(env_value: str) -> dict[str, str]:
    """Return ``{env var: str(field default)}`` with every probed var set to
    ``env_value`` in a fresh interpreter."""
    env = os.environ.copy()
    for key in PROBED_FIELDS:
        env[key] = env_value
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT, json.dumps(PROBED_FIELDS)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])
