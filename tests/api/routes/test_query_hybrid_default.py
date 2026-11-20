"""REST resolution of ``QueryRequest.enable_hybrid`` against ``HYBRID_BY_DEFAULT``.

``to_query_params`` forwards the request through
``model_dump(exclude_none=True)`` into ``QueryParam(**request_data)``, so the
field DEFAULT decides what an omitted field means: a concrete default is sent
on every request and overrides QueryParam's env-baked default, while ``None`` is
dropped so the env default applies. ``enable_rerank`` uses ``None`` for exactly
that reason; ``enable_hybrid`` must too, otherwise ``HYBRID_BY_DEFAULT=true``
is unreachable over REST.

The second surface is ``/query/data``: ``LightRAG.aquery_data`` copies the
incoming ``QueryParam`` into a ``data_param`` before dispatching to
``kg_query`` / ``naive_query``. Every retrieval-affecting field must ride that
copy -- a dropped ``enable_hybrid`` silently pins ``/query/data`` to the env
default, so an explicit client override (in either direction) never reaches
retrieval.

``QueryParam`` bakes its env default at class-definition (import) time, so each
case runs in a FRESH subprocess -- deterministic, with no import-order luck and
no reloaded class identity leaking back into this process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _rest_enable_hybrid(env_value: str | None, **request_fields) -> str:
    """Resolve ``enable_hybrid`` through the real QueryRequest -> QueryParam path.

    A fresh import is the only deterministic way to observe a given
    ``HYBRID_BY_DEFAULT`` value: the default is evaluated once, at class
    definition time. ``None`` removes the variable entirely (the unset case).
    """
    env = {k: v for k, v in os.environ.items() if k != "HYBRID_BY_DEFAULT"}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    if env_value is not None:
        env["HYBRID_BY_DEFAULT"] = env_value
    extra = "".join(f", {name}={value!r}" for name, value in request_fields.items())
    script = (
        "from lightrag.api.routers.query_routes import QueryRequest; "
        f"print(QueryRequest(query='what is this about'{extra})"
        ".to_query_params(is_stream=False).enable_hybrid)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        check=True,
        timeout=120,
    )
    return proc.stdout.strip()


@pytest.mark.parametrize(
    ("env_value", "request_fields", "expected"),
    [
        # An omitted field must fall through to QueryParam's baked env default;
        # only the literal "true" ever enables it.
        (None, {}, "False"),
        ("true", {}, "True"),
        ("1", {}, "False"),
        # An explicit client value always wins over the env default.
        ("true", {"enable_hybrid": False}, "False"),
        (None, {"enable_hybrid": True}, "True"),
        # Explicit null means "not sent": exclude_none drops it, so the env
        # default still applies.
        ("true", {"enable_hybrid": None}, "True"),
    ],
    ids=[
        "env_unset_omitted",
        "env_true_omitted",
        "env_one_omitted",
        "env_true_explicit_false",
        "env_unset_explicit_true",
        "env_true_explicit_none",
    ],
)
def test_enable_hybrid_resolution(env_value, request_fields, expected):
    assert _rest_enable_hybrid(env_value, **request_fields) == expected


def _aquery_data_forwarded_enable_hybrid(
    env_value: str | None, mode: str, **param_fields
) -> str:
    """Observe the ``enable_hybrid`` that ``aquery_data`` hands to retrieval.

    ``kg_query`` / ``naive_query`` are replaced by recorders that capture the
    ``QueryParam`` they receive, so the assertion reads the ``data_param``
    copy, not the caller's original. The storages are never touched: the
    recorder returns ``None`` and ``aquery_data`` reports the no-result shape.
    """
    env = {k: v for k, v in os.environ.items() if k != "HYBRID_BY_DEFAULT"}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    if env_value is not None:
        env["HYBRID_BY_DEFAULT"] = env_value
    extra = "".join(f", {name}={value!r}" for name, value in param_fields.items())
    script = (
        "import asyncio\n"
        "from types import SimpleNamespace\n"
        "import lightrag.lightrag as lr\n"
        "from lightrag.base import QueryParam\n"
        "recorded = []\n"
        "async def record(*args, **kwargs):\n"
        "    params = [a for a in args if isinstance(a, QueryParam)]\n"
        "    assert len(params) == 1, 'retrieval must receive exactly one QueryParam'\n"
        "    recorded.append(params[0].enable_hybrid)\n"
        "    return None\n"
        "lr.kg_query = record\n"
        "lr.naive_query = record\n"
        "async def noop():\n"
        "    pass\n"
        "ns = SimpleNamespace(\n"
        "    _build_global_config=lambda: {},\n"
        "    _query_done=noop,\n"
        "    chunk_entity_relation_graph=None,\n"
        "    entities_vdb=None,\n"
        "    relationships_vdb=None,\n"
        "    text_chunks=None,\n"
        "    chunks_vdb=None,\n"
        "    llm_response_cache=None,\n"
        ")\n"
        f"asyncio.run(lr.LightRAG.aquery_data(ns, 'what is this about', "
        f"param=QueryParam(mode={mode!r}{extra})))\n"
        "assert recorded, 'aquery_data never dispatched to retrieval'\n"
        "print(recorded[0])\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        check=True,
        timeout=120,
    )
    return proc.stdout.strip()


@pytest.mark.parametrize("mode", ["mix", "naive"])
@pytest.mark.parametrize(
    ("env_value", "param_fields", "expected"),
    [
        # An explicit client value must survive the data_param copy in BOTH
        # env-default directions; a dropped field falls back to the env
        # default and silently ignores the override.
        (None, {"enable_hybrid": True}, "True"),
        (None, {"enable_hybrid": False}, "False"),
        ("true", {"enable_hybrid": False}, "False"),
        ("true", {"enable_hybrid": True}, "True"),
    ],
    ids=[
        "env_unset_explicit_true",
        "env_unset_explicit_false",
        "env_true_explicit_false",
        "env_true_explicit_true",
    ],
)
def test_aquery_data_forwards_explicit_enable_hybrid(
    env_value, param_fields, expected, mode
):
    assert (
        _aquery_data_forwarded_enable_hybrid(env_value, mode, **param_fields)
        == expected
    )


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, "False"), ("true", "True")],
    ids=["env_unset_omitted", "env_true_omitted"],
)
def test_aquery_data_omitted_enable_hybrid_keeps_env_default(env_value, expected):
    assert _aquery_data_forwarded_enable_hybrid(env_value, "mix") == expected
