"""REST resolution of ``QueryRequest.enable_hybrid`` against ``HYBRID_BY_DEFAULT``.

``to_query_params`` forwards the request through
``model_dump(exclude_none=True)`` into ``QueryParam(**request_data)``, so the
field DEFAULT decides what an omitted field means: a concrete default is sent
on every request and overrides QueryParam's env-baked default, while ``None`` is
dropped so the env default applies. ``enable_rerank`` uses ``None`` for exactly
that reason; ``enable_hybrid`` must too, otherwise ``HYBRID_BY_DEFAULT=true``
is unreachable over REST.

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
    extra = "".join(
        f", {name}={value!r}" for name, value in request_fields.items()
    )
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