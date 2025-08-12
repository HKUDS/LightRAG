import pytest
import tempfile
from unittest.mock import MagicMock
import pytest_asyncio

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def storage():
    """A pytest fixture that provides a mock storage object."""
    mock_storage = MagicMock()
    # You can configure the mock_storage here if needed
    yield mock_storage


@pytest.fixture
def temp_working_dir():
    """Create a temporary working directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest_asyncio.fixture
async def initialize_shared_data():
    """Initialize shared data for storage tests."""
    from lightrag.kg.shared_storage import (
        initialize_share_data,
        initialize_pipeline_status,
    )

    # Initialize shared storage
    initialize_share_data(workers=1)
    await initialize_pipeline_status()
    return True


@pytest.fixture
def mock_embedding_func():
    """Mock embedding function."""

    async def mock_embed(text, **kwargs) -> list[float]:
        # Support both string and list input
        if isinstance(text, list):
            return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in text]
        else:
            return [0.1, 0.2, 0.3, 0.4, 0.5]

    mock_embed.embedding_dim = 5
    return mock_embed


@pytest.fixture
def mock_llm_func():
    """Mock LLM function."""

    async def mock_llm(prompt, **kwargs) -> str:
        return "Mock LLM response for: " + str(prompt)[:50]

    return mock_llm
