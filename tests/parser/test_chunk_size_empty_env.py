"""Empty CHUNK_* size/overlap envs must not crash chunker defaults.

Follow-up to CHUNK_R_SEPARATORS / empty int field defaults: bare
``int(os.getenv(...))`` still crashed when size/overlap knobs were present
but empty (common in ``.env`` / Compose). Route through ``get_env_value``.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

import pytest

from lightrag.constants import DEFAULT_CHUNK_P_SIZE
from lightrag.parser.routing import default_chunker_config

REPO_ROOT = Path(__file__).resolve().parents[2]


# Every variable is read by the shared ``_chunk_env_int``, so each variable
# is checked once (its own default and slot) and the blank and malformed
# spellings are swept on one of them: the spellings only differ inside it.
_CHUNK_ENVS = [
    ("CHUNK_P_SIZE", "paragraph_semantic", "chunk_token_size", DEFAULT_CHUNK_P_SIZE),
    ("CHUNK_F_SIZE", "fixed_token", "chunk_token_size", None),
    ("CHUNK_R_SIZE", "recursive_character", "chunk_token_size", None),
    ("CHUNK_V_SIZE", "semantic_vector", "chunk_token_size", None),
    ("CHUNK_F_OVERLAP_SIZE", "fixed_token", "chunk_overlap_token_size", None),
    ("CHUNK_R_OVERLAP_SIZE", "recursive_character", "chunk_overlap_token_size", None),
    ("CHUNK_P_OVERLAP_SIZE", "paragraph_semantic", "chunk_overlap_token_size", None),
]
_BLANKS = ["", "  ", "\t"]
_MALFORMED = ["20OO", "1.5", "abc"]


def _one_spelling_per_env(spellings):
    """Every env with the first spelling, plus the other spellings on the first env."""
    cases = [
        pytest.param(*env, spellings[0], id=f"{env[0]}-{spellings[0]!r}")
        for env in _CHUNK_ENVS
    ]
    cases += [
        pytest.param(*_CHUNK_ENVS[0], value, id=f"{_CHUNK_ENVS[0][0]}-{value!r}")
        for value in spellings[1:]
    ]
    return cases


@pytest.mark.offline
@pytest.mark.parametrize(
    ("env_key", "strategy", "field", "expected", "env_value"),
    _one_spelling_per_env(_BLANKS),
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
    ("env_key", "strategy", "field", "expected", "env_value"),
    _one_spelling_per_env(_MALFORMED),
)
def test_malformed_chunk_size_envs_raise(
    env_key: str,
    strategy: str,
    field: str,
    expected: int | None,
    env_value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del strategy, field, expected  # shared case table; only the key matters here
    monkeypatch.setenv(env_key, env_value)
    with pytest.raises(ValueError, match=env_key):
        default_chunker_config()


@pytest.mark.offline
@pytest.mark.parametrize(
    "env_value",
    [
        "",
    ],
)
def test_empty_chunk_size_env_does_not_crash_lightrag_init(
    env_value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHUNK_SIZE", env_value)
    monkeypatch.setenv("CHUNK_OVERLAP_SIZE", env_value)
    monkeypatch.setattr(
        "lightrag.lightrag.verify_storage_implementation", lambda *a: None
    )
    monkeypatch.setattr("lightrag.lightrag.check_storage_env_vars", lambda *a: None)

    class DummyStorage:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(
        "lightrag.lightrag.get_storage_class", lambda *a, **k: DummyStorage
    )

    async def emb(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 8))

    from lightrag.lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=100, func=emb),
        llm_model_func=lambda *a, **k: "ok",
    )
    assert rag.chunk_token_size == 1200
    assert rag.chunk_overlap_token_size == 100


@pytest.mark.offline
@pytest.mark.parametrize(
    "env_key",
    [
        "CHUNK_SIZE",
        "CHUNK_OVERLAP_SIZE",
        "CHUNK_P_SIZE",
        "CHUNK_F_SIZE",
        "CHUNK_R_SIZE",
        "CHUNK_V_SIZE",
    ],
)
def test_malformed_chunk_size_envs_raise_in_lightrag_init(
    env_key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed non-empty env values must raise ValueError on LightRAG init."""
    monkeypatch.setenv(env_key, "20OO")
    monkeypatch.setattr(
        "lightrag.lightrag.verify_storage_implementation", lambda *a: None
    )
    monkeypatch.setattr("lightrag.lightrag.check_storage_env_vars", lambda *a: None)

    class DummyStorage:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(
        "lightrag.lightrag.get_storage_class", lambda *a, **k: DummyStorage
    )

    async def emb(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 8))

    from lightrag.lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    with pytest.raises(ValueError, match=env_key):
        LightRAG(
            working_dir=str(tmp_path / "wd"),
            addon_params={"chunker": {"paragraph_semantic": {}}},
            embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=100, func=emb),
            llm_model_func=lambda *a, **k: "ok",
        )
