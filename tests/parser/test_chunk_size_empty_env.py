"""Empty CHUNK_* size/overlap envs must not crash chunker defaults.

Follow-up to CHUNK_R_SEPARATORS / empty int field defaults: bare
``int(os.getenv(...))`` still crashed when size/overlap knobs were present
but empty (common in ``.env`` / Compose). Route through ``get_env_value``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from lightrag.constants import DEFAULT_CHUNK_P_SIZE
from lightrag.parser.routing import default_chunker_config

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.offline
@pytest.mark.parametrize("env_value", ["", "  ", "\t"])
@pytest.mark.parametrize(
    ("env_key", "strategy", "field", "expected"),
    [
        ("CHUNK_P_SIZE", "paragraph_semantic", "chunk_token_size", DEFAULT_CHUNK_P_SIZE),
        ("CHUNK_F_SIZE", "fixed_token", "chunk_token_size", None),
        ("CHUNK_R_SIZE", "recursive_character", "chunk_token_size", None),
        ("CHUNK_V_SIZE", "semantic_vector", "chunk_token_size", None),
        ("CHUNK_F_OVERLAP_SIZE", "fixed_token", "chunk_overlap_token_size", None),
        ("CHUNK_R_OVERLAP_SIZE", "recursive_character", "chunk_overlap_token_size", None),
        ("CHUNK_P_OVERLAP_SIZE", "paragraph_semantic", "chunk_overlap_token_size", None),
    ],
)
def test_empty_strategy_chunk_envs_fall_back(
    env_key: str,
    strategy: str,
    field: str,
    expected: int | None,
    env_value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env_key, env_value)
    cfg = default_chunker_config()[strategy]
    if expected is None:
        assert field not in cfg
    else:
        assert cfg[field] == expected


@pytest.mark.offline
def test_valid_chunk_f_size_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_F_SIZE", "900")
    assert default_chunker_config()["fixed_token"]["chunk_token_size"] == 900


@pytest.mark.offline
@pytest.mark.parametrize(
    "env_key",
    [
        "CHUNK_P_SIZE",
        "CHUNK_F_SIZE",
        "CHUNK_R_SIZE",
        "CHUNK_V_SIZE",
        "CHUNK_F_OVERLAP_SIZE",
        "CHUNK_R_OVERLAP_SIZE",
        "CHUNK_P_OVERLAP_SIZE",
    ],
)
@pytest.mark.parametrize("env_value", ["20OO", "1.5", "abc"])
def test_malformed_chunk_size_envs_raise(
    env_key: str, env_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env_key, env_value)
    with pytest.raises(ValueError, match=env_key):
        default_chunker_config()


@pytest.mark.offline
@pytest.mark.parametrize("env_value", ["", "  "])
def test_empty_chunk_size_env_does_not_crash_lightrag_init(
    env_value: str, tmp_path: Path
) -> None:
    env = os.environ.copy()
    env["CHUNK_SIZE"] = env_value
    env["CHUNK_OVERLAP_SIZE"] = env_value
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    script = f"""
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.lightrag import LightRAG

async def emb(texts):
    return np.zeros((len(texts), 8))

rag = LightRAG(
    working_dir={str(tmp_path / "wd")!r},
    embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=100, func=emb),
    llm_model_func=lambda *a, **k: "ok",
)
print(rag.chunk_token_size, rag.chunk_overlap_token_size)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1200 100"
