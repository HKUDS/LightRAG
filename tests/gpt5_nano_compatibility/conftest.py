"""
Pytest configuration for gpt5_nano_compatibility tests.

These tests require:
- OPENAI_API_KEY or LLM_BINDING_API_KEY environment variable
- Access to OpenAI API with gpt-5-nano model

Since these are integration tests that require external API access,
they are marked with the 'integration' marker and can be skipped
when running unit tests only.
"""

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require external services)"
    )


@pytest.fixture(scope="session")
def anyio_backend():
    """Specify the async backend for anyio tests."""
    return "asyncio"
