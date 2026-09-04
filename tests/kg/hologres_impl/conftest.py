import os
import uuid

import pytest

from lightrag.kg.hologres.capabilities import validate_test_schema_name
from lightrag.kg.hologres.client import HologresClient
from lightrag.kg.hologres.config import HologresConfig


_REQUIRED_LIVE_ENVIRONMENT = (
    "HOLOGRES_HOST",
    "HOLOGRES_PORT",
    "HOLOGRES_USER",
    "HOLOGRES_PASSWORD",
    "HOLOGRES_DATABASE",
)


@pytest.fixture
async def hologres_live_client():
    if any(not os.environ.get(name) for name in _REQUIRED_LIVE_ENVIRONMENT):
        pytest.skip("Hologres live credentials are unavailable")

    schema = validate_test_schema_name(f"lightrag_test_{uuid.uuid4().hex}")
    config = HologresConfig.from_env({**os.environ, "HOLOGRES_SCHEMA": schema})
    client = HologresClient(config)
    await client.open()
    try:
        yield client, schema
    finally:
        await client.close()
