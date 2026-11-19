"""``QueryParam.enable_hybrid`` backward-compatibility contract (plan D4 / A3).

``QueryParam`` is a plain dataclass: field ORDER is public API because SDK
callers construct it positionally. ``enable_hybrid`` must be the LAST field,
every pre-existing field keeps its position, and the ``HYBRID_BY_DEFAULT``
default mirrors ``RERANK_BY_DEFAULT`` semantics -- only the literal "true"
(case-insensitive) enables; "1", "0", "yes", padded or empty values do not.

The env default is baked at class-definition time, so each env case runs in a
FRESH subprocess. That is deterministic (no import-order luck) and leaks no
reloaded class identity back into this process, unlike ``importlib.reload``.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lightrag.base import QueryParam

pytestmark = pytest.mark.offline

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The FULL field order: every pre-existing field unchanged, enable_hybrid last.
# Do NOT reorder to match a future refactor -- a change here is a change to the
# public positional constructor API.
_EXPECTED_FIELD_ORDER = (
    "mode",
    "only_need_context",
    "only_need_prompt",
    "response_type",
    "stream",
    "top_k",
    "chunk_top_k",
    "max_entity_tokens",
    "max_relation_tokens",
    "max_total_tokens",
    "hl_keywords",
    "ll_keywords",
    "conversation_history",
    "user_prompt",
    "enable_rerank",
    "include_references",
    "disable_user_prompt_prefix",
    "enable_hybrid",
)


def test_full_field_order_snapshot():
    names = tuple(f.name for f in dataclasses.fields(QueryParam))
    assert names == _EXPECTED_FIELD_ORDER


def test_enable_hybrid_is_the_last_field():
    names = [f.name for f in dataclasses.fields(QueryParam)]
    assert names[-1] == "enable_hybrid"
    assert names.index("disable_user_prompt_prefix") == len(names) - 2


def test_all_legacy_positional_args_still_bind():
    """Every old positional slot keeps its meaning; the new slot binds last."""
    param = QueryParam(
        "naive",
        True,
        True,
        "Bullet Points",
        True,
        7,
        5,
        111,
        222,
        333,
        ["hl"],
        ["ll"],
        [{"role": "user", "content": "hi"}],
        "be concise",
        False,
        True,
        True,
        True,
    )
    assert param.mode == "naive"
    assert param.only_need_context is True
    assert param.only_need_prompt is True
    assert param.response_type == "Bullet Points"
    assert param.stream is True
    assert param.top_k == 7
    assert param.chunk_top_k == 5
    assert param.max_entity_tokens == 111
    assert param.max_relation_tokens == 222
    assert param.max_total_tokens == 333
    assert param.hl_keywords == ["hl"]
    assert param.ll_keywords == ["ll"]
    assert param.conversation_history == [{"role": "user", "content": "hi"}]
    assert param.user_prompt == "be concise"
    assert param.enable_rerank is False
    assert param.include_references is True
    assert param.disable_user_prompt_prefix is True
    assert param.enable_hybrid is True


def test_legacy_only_positional_construction_leaves_hybrid_default():
    """17 legacy positional args must not spill into the new 18th slot."""
    param = QueryParam(
        "mix",
        False,
        False,
        "Multiple Paragraphs",
        False,
        40,
        20,
        6000,
        8000,
        30000,
        [],
        [],
        [],
        "be concise",
        False,
        True,
        True,
    )
    assert param.disable_user_prompt_prefix is True
    assert param.enable_hybrid == QueryParam().enable_hybrid


def _fresh_process_enable_hybrid(env_value: str | None) -> str:
    """Import QueryParam in a clean subprocess and print its baked default.

    A fresh import is the only deterministic way to observe a given
    HYBRID_BY_DEFAULT value: the default is evaluated once at class-definition
    time. ``None`` removes the variable entirely (the unset case).
    """
    env = {k: v for k, v in os.environ.items() if k != "HYBRID_BY_DEFAULT"}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    if env_value is not None:
        env["HYBRID_BY_DEFAULT"] = env_value
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from lightrag.base import QueryParam; print(QueryParam().enable_hybrid)",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        check=True,
        timeout=120,
    )
    return proc.stdout.strip()


def test_default_is_false_with_env_unset():
    assert _fresh_process_enable_hybrid(None) == "False"


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("true", "True"),
        ("TRUE", "True"),
        ("True", "True"),
        ("false", "False"),
        ("1", "False"),
        ("0", "False"),
        ("yes", "False"),
        ("", "False"),
        (" true", "False"),
        ("true ", "False"),
    ],
    ids=[
        "true",
        "TRUE",
        "True",
        "false",
        "one",
        "zero",
        "yes",
        "empty",
        "leading-space",
        "trailing-space",
    ],
)
def test_env_parsing_only_literal_true_enables(env_value, expected):
    """Pin the .lower() == "true" semantics: only literal true (any case) enables."""
    assert _fresh_process_enable_hybrid(env_value) == expected
