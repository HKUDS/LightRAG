"""
Pytest integration for the modular graph storage test suite
"""

import pytest
import os
from tests.test_graph_storage.core.storage_setup import (
    initialize_graph_test_storage,
    cleanup_kuzu_test_environment,
)
from tests.test_graph_storage.tests.basic import test_graph_basic
from tests.test_graph_storage.tests.advanced import test_graph_advanced
from tests.test_graph_storage.tests.batch import test_graph_batch_operations
from tests.test_graph_storage.tests.special_chars import test_graph_special_characters
from tests.test_graph_storage.tests.undirected import test_graph_undirected_property


@pytest.fixture
async def storage():
    """Fixture to provide a storage instance for tests"""
    storage_instance = await initialize_graph_test_storage()
    if storage_instance is None:
        pytest.skip("Failed to initialize storage")

    yield storage_instance

    # Cleanup
    if storage_instance and hasattr(storage_instance, "close"):
        await storage_instance.close()
    if storage_instance and hasattr(storage_instance, "_temp_dir"):
        cleanup_kuzu_test_environment(storage_instance._temp_dir)


@pytest.mark.asyncio
async def test_basic_graph_operations(storage):
    """Test basic graph operations"""
    os.environ["TEST_LANGUAGE"] = "english"
    result = await test_graph_basic(storage)
    assert result is True, "Basic graph operations test failed"


@pytest.mark.asyncio
async def test_advanced_graph_operations(storage):
    """Test advanced graph operations"""
    os.environ["TEST_LANGUAGE"] = "english"
    result = await test_graph_advanced(storage)
    assert result is True, "Advanced graph operations test failed"


@pytest.mark.asyncio
async def test_batch_graph_operations(storage):
    """Test batch graph operations"""
    os.environ["TEST_LANGUAGE"] = "english"
    result = await test_graph_batch_operations(storage)
    assert result is True, "Batch graph operations test failed"


@pytest.mark.asyncio
async def test_special_characters_handling(storage):
    """Test special characters handling"""
    os.environ["TEST_LANGUAGE"] = "english"
    result = await test_graph_special_characters(storage)
    assert result is True, "Special characters handling test failed"


@pytest.mark.asyncio
async def test_undirected_graph_properties(storage):
    """Test undirected graph properties"""
    os.environ["TEST_LANGUAGE"] = "english"
    result = await test_graph_undirected_property(storage)
    assert result is True, "Undirected graph properties test failed"


# Chinese language variants
@pytest.mark.asyncio
async def test_basic_graph_operations_chinese(storage):
    """Test basic graph operations with Chinese translations"""
    os.environ["TEST_LANGUAGE"] = "chinese"
    result = await test_graph_basic(storage)
    assert result is True, "Basic graph operations test (Chinese) failed"


@pytest.mark.asyncio
async def test_special_characters_handling_chinese(storage):
    """Test special characters handling with Chinese translations"""
    os.environ["TEST_LANGUAGE"] = "chinese"
    result = await test_graph_special_characters(storage)
    assert result is True, "Special characters handling test (Chinese) failed"
