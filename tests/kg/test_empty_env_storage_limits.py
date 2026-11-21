"""Unusable env-backed storage limits must not break the storage import.

The Redis and Memgraph storage modules read their limits at import time, and
``kg/factory.py`` imports the module for the configured storage class. A bare
``int()``/``float()`` there turned a limit left present but empty in
``.env``/Compose (``env.example`` ships the four Redis keys uncommented) into
a server that would not start. Each case imports both modules in a fresh
subprocess with every limit set to the same raw value.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_KEYS = (
    "REDIS_MAX_CONNECTIONS",
    "REDIS_SOCKET_TIMEOUT",
    "REDIS_CONNECT_TIMEOUT",
    "REDIS_RETRY_ATTEMPTS",
    "MAX_GRAPH_NODES",
)
# The parent's no-install guard (tests/conftest.py) does not reach a fresh
# interpreter, so the child refuses pipmaster.install itself: a missing SDK
# must fail the probe, never install one.
PROBE = (
    "import pipmaster; "
    "pipmaster.install = lambda *a, **k: (_ for _ in ()).throw("
    "RuntimeError(f'package install refused: {a}')); "
    "import lightrag.kg.redis_impl as r, lightrag.kg.memgraph_impl as m; "
    "print(repr((r.MAX_CONNECTIONS, r.SOCKET_TIMEOUT, r.SOCKET_CONNECT_TIMEOUT, "
    "r.RETRY_ATTEMPTS, m.MAX_GRAPH_NODES)))"
)
DEFAULTS = (200, 30.0, 10.0, 3, 1000)


@pytest.mark.parametrize(
    "raw,expected",
    [
        pytest.param(None, DEFAULTS, id="unset"),
        pytest.param("", DEFAULTS, id="empty"),
        pytest.param("  ", DEFAULTS, id="blank"),
        pytest.param("off", DEFAULTS, id="garbage"),
        pytest.param("17", (17, 17.0, 17.0, 17, 17), id="populated"),
    ],
)
def test_storage_limits_on_import(raw: str | None, expected: tuple) -> None:
    for sdk in ("redis", "neo4j"):
        if importlib.util.find_spec(sdk) is None:
            pytest.skip(f"{sdk} SDK not installed")
    env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    if raw is not None:
        env.update(dict.fromkeys(ENV_KEYS, raw))
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # repr keeps the int/float distinction: 200 vs 200.0 must not compare equal.
    assert result.stdout.strip() == repr(expected)
