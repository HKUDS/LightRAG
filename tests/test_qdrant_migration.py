import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from lightrag.utils import EmbeddingFunc
from lightrag.kg.qdrant_impl import QdrantVectorDBStorage


# Mock QdrantClient
@pytest.fixture
def mock_qdrant_client():
    with patch("lightrag.kg.qdrant_impl.QdrantClient") as mock_client_cls:
        client = mock_client_cls.return_value
        client.collection_exists.return_value = False
        client.count.return_value.count = 0
        # Mock payload schema and vector config for get_collection
        collection_info = MagicMock()
        collection_info.payload_schema = {}
        # Mock vector dimension to match mock_embedding_func (768d)
        collection_info.config.params.vectors.size = 768
        client.get_collection.return_value = collection_info
        yield client


# Mock get_data_init_lock to avoid async lock issues in tests
@pytest.fixture(autouse=True)
def mock_data_init_lock():
    with patch("lightrag.kg.qdrant_impl.get_data_init_lock") as mock_lock:
        mock_lock_ctx = AsyncMock()
        mock_lock.return_value = mock_lock_ctx
        yield mock_lock


# Mock Embedding function
@pytest.fixture
def mock_embedding_func():
    async def embed_func(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])

    func = EmbeddingFunc(embedding_dim=768, func=embed_func, model_name="test-model")
    return func


@pytest.mark.asyncio
async def test_qdrant_collection_naming(mock_qdrant_client, mock_embedding_func):
    """Test if collection name is correctly generated with model suffix"""
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )

    # Verify collection name contains model suffix
    expected_suffix = "test_model_768d"
    assert expected_suffix in storage.final_namespace
    assert storage.final_namespace == f"lightrag_vdb_chunks_{expected_suffix}"

    # Verify legacy namespace (should not include workspace, just the base collection name)
    assert storage.legacy_namespace == "lightrag_vdb_chunks"


@pytest.mark.asyncio
async def test_qdrant_migration_trigger(mock_qdrant_client, mock_embedding_func):
    """Test if migration logic is triggered correctly"""
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )

    # Setup mocks for migration scenario
    # 1. New collection does not exist
    mock_qdrant_client.collection_exists.side_effect = (
        lambda name: name == storage.legacy_namespace
    )

    # 2. Legacy collection exists and has data
    mock_qdrant_client.count.return_value.count = 100

    # 3. Mock scroll for data migration

    mock_point = MagicMock()
    mock_point.id = "old_id"
    mock_point.vector = [0.1] * 768
    mock_point.payload = {"content": "test"}

    # First call returns points, second call returns empty (end of scroll)
    mock_qdrant_client.scroll.side_effect = [([mock_point], "next_offset"), ([], None)]

    # Initialize storage (triggers migration)
    await storage.initialize()

    # Verify migration steps
    # 1. Legacy count checked
    mock_qdrant_client.count.assert_any_call(
        collection_name=storage.legacy_namespace, exact=True
    )

    # 2. New collection created
    mock_qdrant_client.create_collection.assert_called()

    # 3. Data scrolled from legacy
    assert mock_qdrant_client.scroll.call_count >= 1
    call_args = mock_qdrant_client.scroll.call_args_list[0]
    assert call_args.kwargs["collection_name"] == storage.legacy_namespace
    assert call_args.kwargs["limit"] == 500

    # 4. Data upserted to new
    mock_qdrant_client.upsert.assert_called()

    # 5. Payload index created
    mock_qdrant_client.create_payload_index.assert_called()


@pytest.mark.asyncio
async def test_qdrant_no_migration_needed(mock_qdrant_client, mock_embedding_func):
    """Test scenario where new collection already exists"""
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )

    # New collection exists and Legacy exists (warning case)
    # or New collection exists and Legacy does not exist (normal case)
    # Mocking case where both exist to test logic flow but without migration

    # Logic in code:
    # Case 1: Both exist -> Warning only
    # Case 2: Only new exists -> Ensure index

    # Let's test Case 2: Only new collection exists
    mock_qdrant_client.collection_exists.side_effect = (
        lambda name: name == storage.final_namespace
    )

    # Initialize
    await storage.initialize()

    # Should check index but NOT migrate
    # In Qdrant implementation, Case 2 calls get_collection
    mock_qdrant_client.get_collection.assert_called_with(storage.final_namespace)
    mock_qdrant_client.scroll.assert_not_called()


# ============================================================================
# Tests for scenarios described in design document (Lines 606-649)
# ============================================================================


@pytest.mark.asyncio
async def test_scenario_1_new_workspace_creation(
    mock_qdrant_client, mock_embedding_func
):
    """
    场景1：新建workspace
    预期：直接创建lightrag_vdb_chunks_text_embedding_3_large_3072d
    """
    # Use a large embedding model
    large_model_func = EmbeddingFunc(
        embedding_dim=3072,
        func=mock_embedding_func.func,
        model_name="text-embedding-3-large",
    )

    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=large_model_func,
        workspace="test_new",
    )

    # Case 3: Neither legacy nor new collection exists
    mock_qdrant_client.collection_exists.return_value = False

    # Initialize storage
    await storage.initialize()

    # Verify: Should create new collection with model suffix
    expected_collection = "lightrag_vdb_chunks_text_embedding_3_large_3072d"
    assert storage.final_namespace == expected_collection

    # Verify create_collection was called with correct name
    create_calls = [
        call for call in mock_qdrant_client.create_collection.call_args_list
    ]
    assert len(create_calls) > 0
    assert (
        create_calls[0][0][0] == expected_collection
        or create_calls[0].kwargs.get("collection_name") == expected_collection
    )

    # Verify no migration was attempted
    mock_qdrant_client.scroll.assert_not_called()

    print(
        f"✅ Scenario 1: New workspace created with collection '{expected_collection}'"
    )


@pytest.mark.asyncio
async def test_scenario_2_legacy_upgrade_migration(
    mock_qdrant_client, mock_embedding_func
):
    """
    场景2：从旧版本升级
    已存在lightrag_vdb_chunks（无后缀）
    预期：自动迁移数据到lightrag_vdb_chunks_text_embedding_ada_002_1536d
    """
    # Use ada-002 model
    ada_func = EmbeddingFunc(
        embedding_dim=1536,
        func=mock_embedding_func.func,
        model_name="text-embedding-ada-002",
    )

    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=ada_func,
        workspace="test_legacy",
    )

    legacy_collection = storage.legacy_namespace
    new_collection = storage.final_namespace

    # Case 4: Only legacy collection exists
    mock_qdrant_client.collection_exists.side_effect = (
        lambda name: name == legacy_collection
    )

    # Mock legacy collection info with 1536d vectors
    legacy_collection_info = MagicMock()
    legacy_collection_info.payload_schema = {}
    legacy_collection_info.config.params.vectors.size = 1536
    mock_qdrant_client.get_collection.return_value = legacy_collection_info

    # Mock legacy data
    mock_qdrant_client.count.return_value.count = 150

    # Mock scroll results (simulate migration in batches)

    mock_points = []
    for i in range(10):
        point = MagicMock()
        point.id = f"legacy-{i}"
        point.vector = [0.1] * 1536
        point.payload = {"content": f"Legacy document {i}", "id": f"doc-{i}"}
        mock_points.append(point)

    # First batch returns points, second batch returns empty
    mock_qdrant_client.scroll.side_effect = [(mock_points, "offset1"), ([], None)]

    # Initialize (triggers migration)
    await storage.initialize()

    # Verify: New collection should be created
    expected_new_collection = "lightrag_vdb_chunks_text_embedding_ada_002_1536d"
    assert storage.final_namespace == expected_new_collection

    # Verify migration steps
    # 1. Check legacy count
    mock_qdrant_client.count.assert_any_call(
        collection_name=legacy_collection, exact=True
    )

    # 2. Create new collection
    mock_qdrant_client.create_collection.assert_called()

    # 3. Scroll legacy data
    scroll_calls = [call for call in mock_qdrant_client.scroll.call_args_list]
    assert len(scroll_calls) >= 1
    assert scroll_calls[0].kwargs["collection_name"] == legacy_collection

    # 4. Upsert to new collection
    upsert_calls = [call for call in mock_qdrant_client.upsert.call_args_list]
    assert len(upsert_calls) >= 1
    assert upsert_calls[0].kwargs["collection_name"] == new_collection

    # 5. Verify legacy collection was automatically deleted after successful migration
    # This prevents Case 1 warnings on next startup
    delete_calls = [
        call for call in mock_qdrant_client.delete_collection.call_args_list
    ]
    assert (
        len(delete_calls) >= 1
    ), "Legacy collection should be deleted after successful migration"
    # Check if legacy_collection was passed to delete_collection
    deleted_collection = (
        delete_calls[0][0][0]
        if delete_calls[0][0]
        else delete_calls[0].kwargs.get("collection_name")
    )
    assert (
        deleted_collection == legacy_collection
    ), f"Expected to delete '{legacy_collection}', but deleted '{deleted_collection}'"

    print(
        f"✅ Scenario 2: Legacy data migrated from '{legacy_collection}' to '{expected_new_collection}' and legacy collection deleted"
    )


@pytest.mark.asyncio
async def test_scenario_3_multi_model_coexistence(mock_qdrant_client):
    """
    场景3：多模型并存
    预期：两个独立的collection，互不干扰
    """

    # Model A: bge-small with 768d
    async def embed_func_a(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])

    model_a_func = EmbeddingFunc(
        embedding_dim=768, func=embed_func_a, model_name="bge-small"
    )

    # Model B: bge-large with 1024d
    async def embed_func_b(texts, **kwargs):
        return np.array([[0.2] * 1024 for _ in texts])

    model_b_func = EmbeddingFunc(
        embedding_dim=1024, func=embed_func_b, model_name="bge-large"
    )

    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    # Create storage for workspace A with model A
    storage_a = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=model_a_func,
        workspace="workspace_a",
    )

    # Create storage for workspace B with model B
    storage_b = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=model_b_func,
        workspace="workspace_b",
    )

    # Verify: Collection names are different
    assert storage_a.final_namespace != storage_b.final_namespace

    # Verify: Model A collection
    expected_collection_a = "lightrag_vdb_chunks_bge_small_768d"
    assert storage_a.final_namespace == expected_collection_a

    # Verify: Model B collection
    expected_collection_b = "lightrag_vdb_chunks_bge_large_1024d"
    assert storage_b.final_namespace == expected_collection_b

    # Verify: Different embedding dimensions are preserved
    assert storage_a.embedding_func.embedding_dim == 768
    assert storage_b.embedding_func.embedding_dim == 1024

    print("✅ Scenario 3: Multi-model coexistence verified")
    print(f"   - Workspace A: {expected_collection_a} (768d)")
    print(f"   - Workspace B: {expected_collection_b} (1024d)")
    print("   - Collections are independent")


@pytest.mark.asyncio
async def test_case1_empty_legacy_auto_cleanup(mock_qdrant_client, mock_embedding_func):
    """
    Case 1a: 新旧collection都存在，且旧库为空
    预期：自动删除旧库
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )

    legacy_collection = storage.legacy_namespace
    new_collection = storage.final_namespace

    # Mock: Both collections exist
    mock_qdrant_client.collection_exists.side_effect = lambda name: name in [
        legacy_collection,
        new_collection,
    ]

    # Mock: Legacy collection is empty (0 records)
    def count_mock(collection_name, exact=True):
        mock_result = MagicMock()
        if collection_name == legacy_collection:
            mock_result.count = 0  # Empty legacy collection
        else:
            mock_result.count = 100  # New collection has data
        return mock_result

    mock_qdrant_client.count.side_effect = count_mock

    # Mock get_collection for Case 2 check
    collection_info = MagicMock()
    collection_info.payload_schema = {"workspace_id": True}
    mock_qdrant_client.get_collection.return_value = collection_info

    # Initialize storage
    await storage.initialize()

    # Verify: Empty legacy collection should be automatically cleaned up
    # Empty collections are safe to delete without data loss risk
    delete_calls = [
        call for call in mock_qdrant_client.delete_collection.call_args_list
    ]
    assert len(delete_calls) >= 1, "Empty legacy collection should be auto-deleted"
    deleted_collection = (
        delete_calls[0][0][0]
        if delete_calls[0][0]
        else delete_calls[0].kwargs.get("collection_name")
    )
    assert (
        deleted_collection == legacy_collection
    ), f"Expected to delete '{legacy_collection}', but deleted '{deleted_collection}'"

    print(
        f"✅ Case 1a: Empty legacy collection '{legacy_collection}' auto-deleted successfully"
    )


@pytest.mark.asyncio
async def test_case1_nonempty_legacy_warning(mock_qdrant_client, mock_embedding_func):
    """
    Case 1b: 新旧collection都存在，且旧库有数据
    预期：警告但不删除
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )

    legacy_collection = storage.legacy_namespace
    new_collection = storage.final_namespace

    # Mock: Both collections exist
    mock_qdrant_client.collection_exists.side_effect = lambda name: name in [
        legacy_collection,
        new_collection,
    ]

    # Mock: Legacy collection has data (50 records)
    def count_mock(collection_name, exact=True):
        mock_result = MagicMock()
        if collection_name == legacy_collection:
            mock_result.count = 50  # Legacy has data
        else:
            mock_result.count = 100  # New collection has data
        return mock_result

    mock_qdrant_client.count.side_effect = count_mock

    # Mock get_collection for Case 2 check
    collection_info = MagicMock()
    collection_info.payload_schema = {"workspace_id": True}
    mock_qdrant_client.get_collection.return_value = collection_info

    # Initialize storage
    await storage.initialize()

    # Verify: Legacy collection with data should be preserved
    # We never auto-delete collections that contain data to prevent accidental data loss
    delete_calls = [
        call for call in mock_qdrant_client.delete_collection.call_args_list
    ]
    # Check if legacy collection was deleted (it should not be)
    legacy_deleted = any(
        (call[0][0] if call[0] else call.kwargs.get("collection_name"))
        == legacy_collection
        for call in delete_calls
    )
    assert not legacy_deleted, "Legacy collection with data should NOT be auto-deleted"

    print(
        f"✅ Case 1b: Legacy collection '{legacy_collection}' with data preserved (warning only)"
    )
