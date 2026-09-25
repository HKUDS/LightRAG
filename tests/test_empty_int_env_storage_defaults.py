"""Empty env-backed storage limits must not break the storage import.

Follow-up to the ``int(os.getenv(...))`` fallbacks in the ``LightRAG`` field
defaults: the Redis and Memgraph storage modules still converted their knobs
with bare ``int()``/``float()``, and those run when ``kg/factory.py`` imports
the module for a configured storage class. A documented limit that is present
but empty (``.env``/Compose) or mistyped therefore stopped the server from
starting, with a traceback naming ``int()`` rather than the offending key.
``env.example`` ships ``REDIS_MAX_CONNECTIONS=100`` and its three sibling keys
uncommented, and ``MAX_GRAPH_NODES`` as a commented example.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

REDIS_LIMITS = [
    ("MAX_CONNECTIONS", "REDIS_MAX_CONNECTIONS", "200", "17"),
    ("SOCKET_TIMEOUT", "REDIS_SOCKET_TIMEOUT", "30.0", "17.0"),
    ("SOCKET_CONNECT_TIMEOUT", "REDIS_CONNECT_TIMEOUT", "10.0", "17.0"),
    ("RETRY_ATTEMPTS", "REDIS_RETRY_ATTEMPTS", "3", "17"),
]


def _import_storage_constant(
    module_name: str, attr: str, env_key: str, env_value: str
) -> str:
    """Return ``module.attr`` as a fresh import of ``module_name`` sees it."""
    env = os.environ.copy()
    env[env_key] = env_value
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib; "
            f"module = importlib.import_module({module_name!r}); "
            f"print(module.{attr})",
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
@pytest.mark.parametrize(
    "attr,env_key,fallback",
    [(attr, env_key, fallback) for attr, env_key, fallback, _ in REDIS_LIMITS],
)
@pytest.mark.parametrize("env_value", ["", "off"])
def test_unusable_redis_limit_env_falls_back_on_import(
    attr: str, env_key: str, fallback: str, env_value: str
) -> None:
    module_attr = _import_storage_constant(
        "lightrag.kg.redis_impl", attr, env_key, env_value
    )
    assert module_attr == fallback


@pytest.mark.offline
def test_blank_padding_in_redis_limit_env_falls_back_on_import() -> None:
    assert (
        _import_storage_constant(
            "lightrag.kg.redis_impl", "MAX_CONNECTIONS", "REDIS_MAX_CONNECTIONS", "  "
        )
        == "200"
    )


@pytest.mark.offline
@pytest.mark.parametrize("env_value", ["", "  ", "off"])
def test_unusable_graph_node_limit_env_falls_back_on_import(env_value: str) -> None:
    assert (
        _import_storage_constant(
            "lightrag.kg.memgraph_impl", "MAX_GRAPH_NODES", "MAX_GRAPH_NODES", env_value
        )
        == "1000"
    )


@pytest.mark.offline
@pytest.mark.parametrize(
    "attr,env_key,populated",
    [(attr, env_key, populated) for attr, env_key, _, populated in REDIS_LIMITS],
)
def test_populated_redis_limit_env_still_wins(
    attr: str, env_key: str, populated: str
) -> None:
    assert (
        _import_storage_constant("lightrag.kg.redis_impl", attr, env_key, "17")
        == populated
    )


@pytest.mark.offline
def test_populated_graph_node_limit_env_still_wins() -> None:
    assert (
        _import_storage_constant(
            "lightrag.kg.memgraph_impl", "MAX_GRAPH_NODES", "MAX_GRAPH_NODES", "17"
        )
        == "17"
    )


@pytest.mark.offline
def test_storage_limits_keep_their_types_when_env_is_unset() -> None:
    """The fallbacks are the same ints/floats the module used before."""
    env = os.environ.copy()
    for env_key in (key for _, key, _, _ in REDIS_LIMITS):
        env.pop(env_key, None)
    env.pop("MAX_GRAPH_NODES", None)
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lightrag.kg.redis_impl as redis_impl; "
            "import lightrag.kg.memgraph_impl as memgraph_impl; "
            "print("
            "redis_impl.MAX_CONNECTIONS, redis_impl.SOCKET_TIMEOUT, "
            "redis_impl.SOCKET_CONNECT_TIMEOUT, redis_impl.RETRY_ATTEMPTS, "
            "type(redis_impl.MAX_CONNECTIONS).__name__, "
            "type(redis_impl.SOCKET_TIMEOUT).__name__, "
            "memgraph_impl.MAX_GRAPH_NODES"
            ")",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["200", "30.0", "10.0", "3", "int", "float", "1000"]
