"""
Pytest configuration for LightRAG tests.

This file provides command-line options and fixtures for test configuration.
"""

import pytest


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
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    monkeypatch.setenv("DOCX_SMART_HEADING", "false")


#: Populated in ``pytest_configure`` and read by ``requires_spacy_models`` /
#: ``pytest_terminal_summary``. Two independent facts:
#: - which pinned spaCy models are missing (empty tuple == all present);
#: - whether the developer opted into smart_heading (``DOCX_SMART_HEADING``),
#:   captured BEFORE the per-test ``_hermetic_mineru_env`` fixture pins the var
#:   to "false" — so opt-in drives spaCy test selection while routing/API tests
#:   still see a neutral value.
_SPACY_DOWNLOAD_HINT = "lightrag-download-cache --spacy --spacy-install"


@pytest.fixture(autouse=True)
def _hermetic_paddleocr_vl_env(monkeypatch):
    """Make every test start with PaddleOCR-VL env vars in their unset state.

    Mirrors ``_hermetic_mineru_env``: ``load_dotenv(override=False)`` at API
    import time can leak a developer's local ``.env`` into the test process,
    and the PaddleOCR-VL option defaults (e.g. ``useChartRecognition=true``)
    feed both the live request payload and the cache options signature, so a
    leaked non-default would silently break option/signature tests. Strip all
    PaddleOCR-VL vars globally; tests that need a specific value still call
    ``monkeypatch.setenv(...)`` and monkeypatch restores the inherited value
    at teardown.
    """
    for name in (
        "PADDLEOCR_VL_API_MODE",
        "PADDLEOCR_VL_API_TOKEN",
        "PADDLEOCR_VL_ENDPOINT",
        "PADDLEOCR_VL_OFFICIAL_ENDPOINT",
        "PADDLEOCR_VL_LOCAL_ENDPOINT",
        "PADDLEOCR_VL_MODEL",
        "PADDLEOCR_VL_PAGE_RANGES",
        "PADDLEOCR_VL_BATCH_ID",
        "PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY",
        "PADDLEOCR_VL_USE_DOC_UNWARPING",
        "PADDLEOCR_VL_USE_LAYOUT_DETECTION",
        "PADDLEOCR_VL_USE_CHART_RECOGNITION",
        "PADDLEOCR_VL_USE_SEAL_RECOGNITION",
        "PADDLEOCR_VL_USE_OCR_FOR_IMAGE_BLOCK",
        "PADDLEOCR_VL_LAYOUT_THRESHOLD",
        "PADDLEOCR_VL_LAYOUT_NMS",
        "PADDLEOCR_VL_LAYOUT_UNCLIP_RATIO",
        "PADDLEOCR_VL_LAYOUT_MERGE_BBOXES_MODE",
        "PADDLEOCR_VL_LAYOUT_SHAPE_MODE",
        "PADDLEOCR_VL_PROMPT_LABEL",
        "PADDLEOCR_VL_FORMAT_BLOCK_CONTENT",
        "PADDLEOCR_VL_REPETITION_PENALTY",
        "PADDLEOCR_VL_TEMPERATURE",
        "PADDLEOCR_VL_TOP_P",
        "PADDLEOCR_VL_MIN_PIXELS",
        "PADDLEOCR_VL_MAX_PIXELS",
        "PADDLEOCR_VL_MAX_NEW_TOKENS",
        "PADDLEOCR_VL_MERGE_LAYOUT_BLOCKS",
        "PADDLEOCR_VL_MARKDOWN_IGNORE_LABELS",
        "PADDLEOCR_VL_VLM_EXTRA_ARGS",
        "PADDLEOCR_VL_SHOW_FORMULA_NUMBER",
        "PADDLEOCR_VL_RETURN_MARKDOWN_IMAGES",
        "PADDLEOCR_VL_RESTRUCTURE_PAGES",
        "PADDLEOCR_VL_MERGE_TABLES",
        "PADDLEOCR_VL_RELEVEL_TITLES",
        "PADDLEOCR_VL_PRETTIFY_MARKDOWN",
        "PADDLEOCR_VL_VISUALIZE",
        "PADDLEOCR_VL_ENGINE_VERSION",
        "PADDLEOCR_VL_TIMEOUT_SECONDS",
        "PADDLEOCR_VL_POLL_INTERVAL_SECONDS",
        "PADDLEOCR_VL_MAX_POLLS",
        "PADDLEOCR_VL_ALLOWED_ASSET_HOSTS",
    ):
        monkeypatch.delenv(name, raising=False)


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

    # Mirror the server's own .env load so DOCX_SMART_HEADING reflects the
    # developer's real configuration here, before _hermetic_mineru_env pins it.
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=False)

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
    """
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


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip integration tests by default.

    Integration tests are skipped unless --run-integration flag is provided.
    This allows running offline tests quickly without needing external services.
    """
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
