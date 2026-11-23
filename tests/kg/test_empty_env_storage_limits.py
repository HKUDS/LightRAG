"""Unusable env-backed storage limits must not break the storage import.

The Redis and Memgraph storage modules read their limits at import time, and
``kg/factory.py`` imports the module for the configured storage class. A bare
``int()``/``float()`` there turned a limit left present but empty in
``.env``/Compose (``env.example`` ships the four Redis keys uncommented) into
a server that would not start.

Import-time values need a fresh import, not the parent's cached modules. One
child process serves every case: for each raw value it sets every limit to
it and executes both modules' source again under a name of their own, which
is what a fresh interpreter's import does, without paying for one per case.
"""

from __future__ import annotations

import functools
import importlib.util
import json
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
PROBE = """
import importlib.util, json, os, sys, traceback
import pipmaster
pipmaster.install = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError(f"package install refused: {a}"))
keys, cases = json.loads(sys.argv[1]), json.loads(sys.argv[2])

def fresh(module, case):
    origin = importlib.util.find_spec(module).origin
    name = f"{module}_probe_{case}"
    spec = importlib.util.spec_from_file_location(name, origin)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded  # dataclasses look the module up by name
    spec.loader.exec_module(loaded)
    return loaded

out = {}
for case, raw in cases.items():
    for key in keys:
        os.environ.pop(key, None)
    if raw is not None:
        os.environ.update(dict.fromkeys(keys, raw))
    try:
        r = fresh("lightrag.kg.redis_impl", case)
        m = fresh("lightrag.kg.memgraph_impl", case)
        out[case] = repr((r.MAX_CONNECTIONS, r.SOCKET_TIMEOUT,
                          r.SOCKET_CONNECT_TIMEOUT, r.RETRY_ATTEMPTS,
                          m.MAX_GRAPH_NODES))
    except Exception:
        out[case] = traceback.format_exc()
print(json.dumps(out))
"""
DEFAULTS = (200, 30.0, 10.0, 3, 1000)
CASES = {
    "unset": (None, DEFAULTS),
    "empty": ("", DEFAULTS),
    "blank": ("  ", DEFAULTS),
    "garbage": ("off", DEFAULTS),
    "populated": ("17", (17, 17.0, 17.0, 17, 17)),
}


@functools.lru_cache(maxsize=None)
def _import_results() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])
    )
    raws = {case: raw for case, (raw, _) in CASES.items()}
    result = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(ENV_KEYS), json.dumps(raws)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("case", list(CASES))
def test_storage_limits_on_import(case: str) -> None:
    for sdk in ("redis", "neo4j"):
        if importlib.util.find_spec(sdk) is None:
            pytest.skip(f"{sdk} SDK not installed")
    # repr keeps the int/float distinction: 200 vs 200.0 must not compare equal.
    assert _import_results()[case] == repr(CASES[case][1])
