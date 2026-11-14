"""
Pytest configuration for LightRAG tests.

This file provides command-line options and fixtures for test configuration.
"""

import logging
import os
import sys

import pytest

from tests._log_capture import capture_lightrag_logs


@pytest.fixture
def lightrag_caplog(caplog):
    """``caplog``, seeing each ``lightrag`` record exactly once.

    Use it for any test that asserts on ``caplog.records`` / ``caplog.text``
    for records from the ``lightrag`` logger, and never set ``propagate`` on
    that logger alongside it: why, and how pytest 9.1 changed it, is in
    ``tests/_log_capture.py``. Inside a helper that receives ``caplog`` rather
    than a fixture, use ``capture_lightrag_logs`` from that module instead.
    ``caplog.at_level(..., logger="lightrag")`` still works as usual.
    """
    with capture_lightrag_logs(caplog):
        yield caplog


@pytest.fixture
def lightrag_log_records():
    """Records the ``lightrag`` logger emits during the test, and nothing else.

    A private collector on the logger itself, independent of caplog and of the
    pytest version (see ``tests/_log_capture.py``). Prefer it when the test
    only counts or inspects ``lightrag`` records; use ``lightrag_caplog`` when
    it needs caplog's API (``text``, ``at_level``, ``clear``).
    """
    logger = logging.getLogger("lightrag")
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector()
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


@pytest.fixture
def emulate_windows_newlines(monkeypatch):
    """Make ``Path.write_text`` translate newlines the way it does on Windows.

    Without ``newline=``, ``Path.write_text`` turns every ``\\n`` into
    ``os.linesep``: CRLF on Windows, a no-op on Linux and macOS. CI runs on
    Linux only, so a test that merely asserts "no CRLF in the file" passes
    there whether or not the code under test pins ``newline="\\n"``. Under
    this fixture a write that leaves ``newline`` unset (or ``None``) writes
    CRLF on every platform, so such a test goes red on Linux too. An explicit
    ``newline`` is passed through untouched.
    """
    import pathlib

    original = pathlib.Path.write_text

    def write_text(self, data, encoding=None, errors=None, newline=None):
        return original(
            self,
            data,
            encoding=encoding,
            errors=errors,
            newline="\r\n" if newline is None else newline,
        )

    monkeypatch.setattr(pathlib.Path, "write_text", write_text)


@pytest.fixture(autouse=True)
def _hermetic_mineru_env(monkeypatch):
    """Make every test start with parser-routing env vars in their unset state.

    ``lightrag/api/{auth,config}.py`` call ``load_dotenv(override=False)``
    at import time, leaking the developer's local ``.env`` into the test
    process. The MinerU test fixtures assume ``MINERU_API_MODE`` is unset
    (so it defaults to ``"local"`` per ``MinerURawClient.__init__`` /
    ``parser_engine_endpoint_requirement``):

    - A leaked ``MINERU_API_MODE=offical`` typo (or any invalid value)
      makes ``MinerURawClient()`` raise at construction.
    - A leaked ``MINERU_API_MODE=official`` flips
      ``parser_engine_endpoint_requirement`` to return
      ``"MINERU_API_TOKEN"`` instead of ``"MINERU_LOCAL_ENDPOINT"``,
      breaking the validation-error string match.

    ``LIGHTRAG_PARSER`` is cleared for the same reason: a routing rule
    like ``docx:mineru-iet`` in the developer's ``.env`` forces
    ``parser_routing.validate_parser_routing_config`` to require the
    corresponding endpoint (``MINERU_LOCAL_ENDPOINT`` /
    ``DOCLING_ENDPOINT``) at every ``create_app`` call, which then trips
    unrelated API/FastAPI tests (``test_bedrock_llm.py``,
    ``test_path_prefixes.py``).

    The ``MINERU_LOCAL_*`` parser options are stripped for the same reason:
    a developer ``.env`` that pins e.g. ``MINERU_LOCAL_PARSE_METHOD=ocr``
    leaks a non-default into tests that assume the built-in defaults
    (``test_client_local_mode_round_trip`` expects ``parse_method=auto``;
    ``test_invalid_when_local_parser_options_change`` toggles each option
    and expects the change to invalidate a bundle recorded with defaults).

    ``DOCLING_ADDITIONAL_SUFFIXES`` / ``MINERU_ADDITIONAL_SUFFIXES`` are cleared
    because they are live engine *capability* knobs
    (``ParserSpec.extra_suffixes_env``): a developer ``.env`` that opts in e.g.
    ``doc,ppt,xls`` widens ``suffix_capabilities()`` for that engine and the
    upload allowlist derived from it, so baseline suffix assertions would
    silently pass or fail depending on the developer's own deployment.

    ``DOCX_SMART_HEADING`` is pinned to ``"false"`` (not merely deleted):
    it is a live-env parser-routing knob (``routing.smart_heading_default_enabled``),
    so a developer ``.env`` that sets ``DOCX_SMART_HEADING=true`` seeds
    ``native(smart_heading=true)`` into the persisted ``parse_engine`` on
    every .docx enqueue, breaking baseline routing tests that expect a bare
    ``native`` — and it also makes ``create_app`` fail-fast on missing spaCy
    models (``validate_smart_heading_dependencies``), taking down otherwise
    unrelated API tests. ``delenv`` alone did not hold: the api-module
    import/reload path re-runs ``load_dotenv(".env", override=False)``, and
    because ``override=False`` only fills *unset* names, a deleted var is
    silently repopulated from ``.env``. An explicit ``"false"`` survives that
    repopulation. The developer's real opt-in is still honored for spaCy test
    selection — captured once in ``pytest_configure`` before this runs (see
    ``requires_spacy_models``), independent of this per-test neutralization.

    Strip these variables globally; tests that need a specific mode can
    still ``monkeypatch.setenv(...)`` themselves and monkeypatch will
    restore the inherited value at teardown.
    """
    monkeypatch.delenv("MINERU_API_MODE", raising=False)
    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)
    monkeypatch.delenv("MINERU_LOCAL_ENDPOINT", raising=False)
    monkeypatch.delenv("MINERU_OFFICIAL_ENDPOINT", raising=False)
    monkeypatch.delenv("MINERU_LOCAL_BACKEND", raising=False)
    monkeypatch.delenv("MINERU_LOCAL_PARSE_METHOD", raising=False)
    monkeypatch.delenv("MINERU_LOCAL_IMAGE_ANALYSIS", raising=False)
    monkeypatch.delenv("MINERU_LOCAL_START_PAGE_ID", raising=False)
    monkeypatch.delenv("MINERU_ADDITIONAL_SUFFIXES", raising=False)
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    monkeypatch.delenv("DOCLING_ADDITIONAL_SUFFIXES", raising=False)
    monkeypatch.setenv("DOCX_SMART_HEADING", "false")


@pytest.fixture(autouse=True)
def _reset_r_separator_caches():
    """Drop the process-wide ``CHUNK_R_SEPARATORS`` caches between tests.

    Both caches are keyed on the *raw* environment string and hold for the life
    of the process, including the one-time correction WARNING each of them
    emits. Without this reset, the second test in a session that happens to use
    the same ``CHUNK_R_SEPARATORS`` value sees zero warnings and fails an
    assertion that has nothing to do with what it is testing — or, worse, passes
    for the wrong reason. Tests must not have to invent globally-unique
    separator strings to stay independent.
    """
    from lightrag.multimodal_context import _cached_surrounding_chunk_separators
    from lightrag.parser.routing import _cached_env_r_separators

    _cached_env_r_separators.cache_clear()
    _cached_surrounding_chunk_separators.cache_clear()
    yield
    _cached_env_r_separators.cache_clear()
    _cached_surrounding_chunk_separators.cache_clear()


_API_MODULE_PREFIX = "lightrag.api"


@pytest.fixture(autouse=True)
def _restore_lightrag_api_module_identity():
    """Restore ``lightrag.api.*`` entries a test replaced or removed.

    ``lightrag.api.config`` / ``auth`` / ``utils_api`` compute module-level
    state at IMPORT time, and some of their functions re-read siblings through
    ``sys.modules`` at CALL time (``utils_api`` imports ``global_args`` inside
    the token-renewal check). A test that pops those modules and re-imports
    them under an isolated env therefore leaves a different object live than
    the one later tests imported at collection time: they patch a module
    nobody consults, and fail depending on which test ran first. This is
    session-wide rather than under ``tests/api/`` because such re-import
    helpers also live elsewhere (``tests/llm/``).

    REPLACEMENTS and REMOVALS are healed by identity comparison. When any
    entry had to be healed, the ``lightrag.api.*`` modules first imported
    during the test are evicted as well: they were bound to the replaced
    objects (``lightrag_server`` imported inside such a test holds the
    re-imported ``utils_api``), so keeping them would split one live module
    graph across two generations. The next import rebuilds them against the
    restored modules. A test that only imported new modules, without
    replacing any, is left alone. Parent-package attributes are kept in step
    in both directions, because ``from lightrag.api import x`` returns an
    existing attribute without consulting ``sys.modules``.
    """

    def _api_modules():
        return {
            name: module
            for name, module in sys.modules.items()
            if name == _API_MODULE_PREFIX or name.startswith(_API_MODULE_PREFIX + ".")
        }

    saved = _api_modules()
    yield
    healed = False
    for name, module in saved.items():
        if sys.modules.get(name) is module:
            continue
        healed = True
        sys.modules[name] = module
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and child and getattr(parent, child, None) is not module:
            setattr(parent, child, module)
    if not healed:
        return
    for name, module in _api_modules().items():
        if name in saved:
            continue
        del sys.modules[name]
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, child, None) is module:
            delattr(parent, child)


#: Populated in ``pytest_configure`` and read by ``requires_spacy_models`` /
#: ``pytest_terminal_summary``. Two independent facts:
#: - which pinned spaCy models are missing (empty tuple == all present);
#: - whether the developer opted into smart_heading (``DOCX_SMART_HEADING``),
#:   captured BEFORE the per-test ``_hermetic_mineru_env`` fixture pins the var
#:   to "false" — so opt-in drives spaCy test selection while routing/API tests
#:   still see a neutral value.
_SPACY_DOWNLOAD_HINT = "lightrag-download-cache --spacy-install"


def pytest_configure(config):
    """Register custom markers and capture the spaCy model / opt-in state."""
    config.addinivalue_line(
        "markers", "offline: marks tests as offline (no external dependencies)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring external services (skipped by default)",
    )
    config.addinivalue_line("markers", "requires_db: marks tests requiring database")
    config.addinivalue_line(
        "markers", "requires_api: marks tests requiring LightRAG API server"
    )
    config.addinivalue_line(
        "markers",
        "requires_spacy_models: needs the pinned spaCy language models "
        f"(install with `{_SPACY_DOWNLOAD_HINT}`)",
    )

    # The suite never loads .env (tests/__init__.py), but a developer's
    # DOCX_SMART_HEADING opt-in may live only there. Copy that one key, so the
    # rest of the file stays out of the process; _hermetic_mineru_env still
    # pins it to "false" inside each test.
    if "DOCX_SMART_HEADING" not in os.environ:
        from dotenv import dotenv_values

        opt_in = dotenv_values(".env").get("DOCX_SMART_HEADING")
        if opt_in is not None:
            os.environ["DOCX_SMART_HEADING"] = opt_in

    from lightrag.parser.docx.smart_heading import nlp

    config._missing_spacy_models = tuple(nlp.missing_spacy_models())

    try:
        from lightrag.parser.routing import smart_heading_default_enabled

        config._smart_heading_opted_in = smart_heading_default_enabled()
    except Exception:
        # A malformed DOCX_SMART_HEADING value surfaces at server startup, not
        # here — default to the gentle (skip, not fail) path for missing models.
        config._smart_heading_opted_in = False


def pytest_runtest_setup(item):
    """Gate ``@pytest.mark.requires_spacy_models`` tests on model availability.

    Single source of truth for every smart_heading test that needs the real
    pinned spaCy models, replacing per-file ad-hoc probes/skip messages:

    - models present -> run;
    - models missing + developer opted into smart_heading
      (``DOCX_SMART_HEADING=true``) -> FAIL loudly with the install command,
      because they have declared they use the feature;
    - models missing + not opted in -> skip, so contributors who never touch
      smart_heading are not forced to download the models.

    Either way ``pytest_terminal_summary`` restates the command at the end.

    It also fails every test that ``pytest_collection_modifyitems`` found
    without a gate marker (see ``GATE_MARKERS``).
    """
    if item.stash.get(_UNGATED_KEY, False):
        # Fail before any fixture runs; pytest_runtest_makereport makes the
        # failure stick even under xfail/skip markers.
        pytest.fail(_ungated_message(item), pytrace=False)
    if item.get_closest_marker("requires_spacy_models") is None:
        return
    missing = getattr(item.config, "_missing_spacy_models", ())
    if not missing:
        return
    detail = (
        f"pinned spaCy model(s) not installed ({', '.join(missing)}); "
        f"install with: {_SPACY_DOWNLOAD_HINT}"
    )
    if getattr(item.config, "_smart_heading_opted_in", False):
        pytest.fail(f"DOCX_SMART_HEADING is enabled but {detail}", pytrace=False)
    pytest.skip(detail)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Tell the developer how to get the spaCy models after the run finishes."""
    missing = getattr(config, "_missing_spacy_models", ())
    if not missing:
        return
    opted_in = getattr(config, "_smart_heading_opted_in", False)
    tr = terminalreporter
    tr.write_sep("=", "spaCy language models not installed", yellow=True, bold=True)
    tr.write_line(f"Missing model(s): {', '.join(missing)}")
    tr.write_line(
        "smart_heading (docx) tests that need them were "
        + ("FAILED (DOCX_SMART_HEADING is on)." if opted_in else "skipped.")
    )
    tr.write_line("To run them, install the pinned models:")
    tr.write_line(f"    {_SPACY_DOWNLOAD_HINT}")


def pytest_addoption(parser):
    """Add custom command-line options for LightRAG tests."""

    parser.addoption(
        "--keep-artifacts",
        action="store_true",
        default=False,
        help="Keep test artifacts (temporary directories and files) after test completion for inspection",
    )

    parser.addoption(
        "--stress-test",
        action="store_true",
        default=False,
        help="Enable stress test mode with more intensive workloads",
    )

    parser.addoption(
        "--test-workers",
        action="store",
        default=3,
        type=int,
        help="Number of parallel workers for stress tests (default: 3)",
    )

    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require external services (database, API server, etc.)",
    )


#: Every test must carry at least one of these. CI selects with ``-m offline``,
#: so a test with none of them is neither run in CI nor reported as skipped.
GATE_MARKERS = frozenset(
    {"offline", "integration", "requires_db", "requires_api", "pg_smoke"}
)


_UNGATED_KEY = pytest.StashKey[bool]()


def _ungated_message(item):
    return (
        f"{item.nodeid} carries none of the gate markers "
        f"{sorted(GATE_MARKERS)}, so CI's `-m offline` would deselect it "
        "and never run it. Add `pytestmark = pytest.mark.offline` to the "
        "module, or the matching integration/requires_* marker."
    )


# tryfirst on a wrapper makes it the outermost one, so it sees the setup
# report after the skipping plugin has rewritten it.
@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Report an ungated test's setup as failed, whatever its other markers.

    The ``pytest.fail()`` in ``pytest_runtest_setup`` is not enough on its
    own: an ``xfail`` marker turns it into XFAIL, and a ``skip``/``skipif``
    marker skips the test before that hook runs. Both would exit green and
    leave the test unrun, which is what the gate exists to prevent.
    """
    report = yield
    if call.when == "setup" and item.stash.get(_UNGATED_KEY, False):
        report.outcome = "failed"
        report.longrepr = _ungated_message(item)
        if hasattr(report, "wasxfail"):
            del report.wasxfail
    return report


# tryfirst: the gate check must see every collected item, before pytest's own
# ``-m`` handling deselects the ungated ones it is meant to catch.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Flag ungated tests, then skip integration tests by default.

    A test without a gate marker would be deselected silently by CI's
    ``-m offline`` and never run. Such a test is marked ``offline`` here, so
    that ``-m offline`` selects it, and flagged, so that
    ``pytest_runtest_setup`` and ``pytest_runtest_makereport`` fail it with
    the fix. This surfaces as an ordinary test failure in serial and xdist
    runs alike; raising from this hook would be an INTERNALERROR inside an
    xdist worker.

    Integration tests are skipped unless --run-integration flag is provided.
    This allows running offline tests quickly without needing external services.
    """
    for item in items:
        if not GATE_MARKERS.intersection(m.name for m in item.iter_markers()):
            item.add_marker(pytest.mark.offline)
            item.stash[_UNGATED_KEY] = True

    if config.getoption("--run-integration"):
        # If --run-integration is specified, run all tests
        return

    skip_integration = pytest.mark.skip(
        reason="Requires external services(DB/API), use --run-integration to run"
    )

    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def keep_test_artifacts(request):
    """
    Fixture to determine whether to keep test artifacts.

    Priority: CLI option > Environment variable > Default (False)
    """
    import os

    # Check CLI option first
    if request.config.getoption("--keep-artifacts"):
        return True

    # Fall back to environment variable
    return os.getenv("LIGHTRAG_KEEP_ARTIFACTS", "false").lower() == "true"


@pytest.fixture(scope="session")
def stress_test_mode(request):
    """
    Fixture to determine whether stress test mode is enabled.

    Priority: CLI option > Environment variable > Default (False)
    """
    import os

    # Check CLI option first
    if request.config.getoption("--stress-test"):
        return True

    # Fall back to environment variable
    return os.getenv("LIGHTRAG_STRESS_TEST", "false").lower() == "true"


@pytest.fixture(scope="session")
def parallel_workers(request):
    """
    Fixture to determine the number of parallel workers for stress tests.

    Priority: CLI option > Environment variable > Default (3)
    """
    import os

    # Check CLI option first
    cli_workers = request.config.getoption("--test-workers")
    if cli_workers != 3:  # Non-default value provided
        return cli_workers

    # Fall back to environment variable
    return int(os.getenv("LIGHTRAG_TEST_WORKERS", "3"))


@pytest.fixture(scope="session")
def run_integration_tests(request):
    """
    Fixture to determine whether to run integration tests.

    Priority: CLI option > Environment variable > Default (False)
    """
    import os

    # Check CLI option first
    if request.config.getoption("--run-integration"):
        return True

    # Fall back to environment variable
    return os.getenv("LIGHTRAG_RUN_INTEGRATION", "false").lower() == "true"


@pytest.fixture(autouse=True)
def _configured_embedding_model(monkeypatch):
    """Give every test that builds a server the EMBEDDING_MODEL it now requires.

    ``create_app`` refuses to start without one: the model's name is what each
    vector storage records beside its vectors, and it is the only thing that
    detects a later switch to a different model of the SAME dimension. A server
    that cannot name its model has no protected way through an embedding
    change, which LightRAG only supports as stop / rebuild / restart.

    This lives at the ROOT rather than under ``tests/api/`` because the
    requirement belongs to ``create_app``, not to a directory. Server-building
    tests are not confined to ``tests/api/`` -- ``tests/llm/bedrock_impl/`` and
    ``tests/llm/ollama_impl/`` call it too -- and a default scoped to one tree
    leaves every caller outside it failing on a condition it never meant to
    test. CI found exactly that.

    It sets the variable only when the environment does not already carry one,
    so a test that wants a specific model still wins, and a test that wants it
    UNSET still wins as well: a function-scoped fixture runs before the test
    body, so a later ``monkeypatch.delenv`` in the body takes effect.
    ``tests/api/test_embedding_model_required.py`` and
    ``tests/tools/test_rebuild_vdb.py`` both rely on that.

    The value is the default binding's OWN default model, not an invented
    name: a custom model obliges the operator to set ``EMBEDDING_DIM`` too
    (see ``create_optimized_embedding_function``), and making every such test
    carry a dimension it does not care about would be a second, unrelated
    change to what they configure.
    """
    import os

    if not (os.environ.get("EMBEDDING_MODEL") or "").strip():
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
