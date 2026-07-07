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
    monkeypatch.delenv("PADDLEOCR_VL_API_MODE", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_API_TOKEN", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_ENDPOINT", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_OFFICIAL_ENDPOINT", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_LOCAL_ENDPOINT", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_MODEL", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_PAGE_RANGES", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_BATCH_ID", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_USE_DOC_UNWARPING", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_USE_LAYOUT_DETECTION", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_USE_CHART_RECOGNITION", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_LAYOUT_THRESHOLD", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_LAYOUT_NMS", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_LAYOUT_UNCLIP_RATIO", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_LAYOUT_MERGE_BBOXES_MODE", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_LAYOUT_SHAPE_MODE", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_PROMPT_LABEL", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_REPETITION_PENALTY", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_TEMPERATURE", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_TOP_P", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_MIN_PIXELS", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_MAX_PIXELS", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_SHOW_FORMULA_NUMBER", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_RESTRUCTURE_PAGES", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_MERGE_TABLES", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_RELEVEL_TITLES", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_PRETTIFY_MARKDOWN", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_VISUALIZE", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_ENGINE_VERSION", raising=False)


def pytest_configure(config):
    """Register custom markers for LightRAG tests."""
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
