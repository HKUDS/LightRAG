import pytest
from unittest.mock import MagicMock

pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def storage():
    """A pytest fixture that provides a mock storage object."""
    mock_storage = MagicMock()
    # You can configure the mock_storage here if needed
    yield mock_storage
