"""The import-time boolean env defaults use the shared boolean syntax."""

import importlib.util
import logging
import sys
from types import ModuleType

import pytest

import lightrag.base  # noqa: F401  (loaded so _fresh_copy can find its file)
import lightrag.lightrag  # noqa: F401
import lightrag.utils  # noqa: F401

pytestmark = pytest.mark.offline


def _fresh_copy(name: str) -> ModuleType:
    """Execute a throwaway copy of ``name`` to re-run its import-time defaults.

    ``importlib.reload`` would rebind every class in the shared module and
    break ``isinstance`` identity for later tests; the copy is never placed in
    ``sys.modules``. It costs milliseconds where a fresh interpreter costs ~1 s.
    """
    spec = importlib.util.spec_from_file_location(name, sys.modules[name].__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("env", "expected", "warn_keys"),
    # Each case pins one flag of each pair to an alias and the OTHER flag to a
    # typo, with the roles swapped between cases, so a wrong fallback default
    # or two flags reading the same variable fails one case. The renamed
    # LIGHTRAG_DOC_QUERY_TIMING_LOGS is set where timing must stay off, so a
    # restored fallback to it fails too.
    [
        pytest.param(
            {
                "RERANK_BY_DEFAULT": " ON ",
                "ENTITY_EXTRACTION_USE_JSON": "ture",
                "VERBOSE": "1",
                "LIGHTRAG_PERFORMANCE_TIMING_LOGS": "ture",
                "LIGHTRAG_DOC_QUERY_TIMING_LOGS": "true",
            },
            {
                "rerank": True,
                "json_extraction": False,
                "verbose": True,
                "timing": False,
            },
            {"ENTITY_EXTRACTION_USE_JSON", "LIGHTRAG_PERFORMANCE_TIMING_LOGS"},
            id="rerank-verbose-alias-json-timing-typo",
        ),
        pytest.param(
            {
                "RERANK_BY_DEFAULT": "ture",
                "ENTITY_EXTRACTION_USE_JSON": "yes",
                "VERBOSE": "ture",
                "LIGHTRAG_PERFORMANCE_TIMING_LOGS": " ON ",
            },
            {"rerank": True, "json_extraction": True, "verbose": False, "timing": True},
            {"RERANK_BY_DEFAULT", "VERBOSE"},
            id="json-timing-alias-rerank-verbose-typo",
        ),
    ],
)
def test_boolean_env_field_defaults(
    env: dict[str, str],
    expected: dict[str, bool],
    warn_keys: set[str],
    monkeypatch: pytest.MonkeyPatch,
    lightrag_log_records: list[logging.LogRecord],
) -> None:
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    logger = logging.getLogger("lightrag")
    level = logger.level
    try:
        utils = _fresh_copy("lightrag.utils")
        base = _fresh_copy("lightrag.base")
        core = _fresh_copy("lightrag.lightrag")
    finally:
        # The utils copy re-runs the shared logger's module-level setup.
        logger.setLevel(level)

    assert {
        "rerank": base.QueryParam().enable_rerank,
        "json_extraction": core.LightRAG.__dataclass_fields__[
            "entity_extraction_use_json"
        ].default,
        "verbose": utils.VERBOSE_DEBUG,
        "timing": utils.PERFORMANCE_TIMING_LOGS,
    } == expected
    warned = {
        record.args[0]
        for record in lightrag_log_records
        if "is not a recognized boolean" in record.msg
    }
    assert warned == warn_keys
