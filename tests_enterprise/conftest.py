from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring external services",
    )
    config.addinivalue_line(
        "markers",
        "requires_api: marks tests requiring LightRAG API server",
    )
    config.addinivalue_line(
        "markers",
        "requires_db: marks tests requiring database",
    )
