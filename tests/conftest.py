import pytest
import os
import tempfile
import importlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)


# Mock embedding function for testing
def mock_embedding_func(text):
    """Mock embedding function that returns a fixed vector"""
    if isinstance(text, str):
        # Simple hash-based embedding for consistency
        hash_val = hash(text)
        return [float((hash_val >> i) & 1) for i in range(384)]
    elif isinstance(text, list):
        return [mock_embedding_func(item) for item in text]
    else:
        return [0.1] * 384


# Storage configuration mapping
STORAGES = {
    "NetworkXStorage": "lightrag.kg.networkx_impl",
    "Neo4JStorage": "lightrag.kg.neo4j_impl",
    "KuzuDBStorage": "lightrag.kg.kuzu_impl",
    "MongoDBStorage": "lightrag.kg.mongo_impl",
    "PGGraphStorage": "lightrag.kg.postgres_impl",
    "MemgraphStorage": "lightrag.kg.memgraph_impl",
}


def setup_kuzu_environment():
    """Set up temporary KuzuDB environment for testing"""
    temp_dir = tempfile.mkdtemp(prefix="kuzu_test_")
    kuzu_db_path = os.path.join(temp_dir, "test_graph.db")
    os.environ["KUZU_DB_PATH"] = kuzu_db_path
    return temp_dir, kuzu_db_path


def cleanup_kuzu_environment(temp_dir):
    """Clean up temporary KuzuDB environment"""
    import shutil

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def initialize_share_data():
    """Initialize shared data for NetworkXStorage"""
    try:
        from lightrag.kg.shared_storage import initialize_share_data

        initialize_share_data()
    except ImportError:
        # If shared_storage doesn't exist, skip initialization
        pass


async def create_storage_instance(storage_type="KuzuDBStorage"):
    """Create a storage instance for testing"""

    # KuzuDB special handling
    temp_dir = None
    if storage_type == "KuzuDBStorage":
        temp_dir, kuzu_db_path = setup_kuzu_environment()

    # Dynamic import of storage module
    module_path = STORAGES.get(storage_type)
    if not module_path:
        raise ValueError(f"Unknown storage type: {storage_type}")

    try:
        module = importlib.import_module(module_path)
        storage_class = getattr(module, storage_type)
    except (ImportError, AttributeError) as e:
        if temp_dir:
            cleanup_kuzu_environment(temp_dir)
        raise ImportError(f"Failed to import {storage_type}: {str(e)}")

    # Initialize storage instance
    global_config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        "working_dir": os.environ.get("WORKING_DIR", "./rag_storage"),
        "max_graph_nodes": 1000,
    }

    # Initialize shared data for NetworkXStorage
    if storage_type == "NetworkXStorage":
        initialize_share_data()

    try:
        # KuzuDB needs special initialization parameters
        if storage_type == "KuzuDBStorage":
            storage = storage_class(
                namespace="test_graph",
                global_config=global_config,
                embedding_func=mock_embedding_func,
                workspace="test_workspace",
            )
        else:
            storage = storage_class(
                namespace="test_graph",
                global_config=global_config,
                embedding_func=mock_embedding_func,
            )

        # Initialize connection
        await storage.initialize()

        # Store temp directory info for cleanup
        if temp_dir:
            storage._temp_dir = temp_dir

        return storage
    except Exception as e:
        if temp_dir:
            cleanup_kuzu_environment(temp_dir)
        raise RuntimeError(f"Failed to initialize {storage_type}: {str(e)}")


@pytest.fixture
async def storage():
    """Create a test storage instance"""
    # Default to KuzuDBStorage for testing
    storage_type = os.environ.get("LIGHTRAG_GRAPH_STORAGE", "KuzuDBStorage")

    storage_instance = await create_storage_instance(storage_type)

    yield storage_instance

    # Cleanup
    try:
        if hasattr(storage_instance, "_temp_dir"):
            cleanup_kuzu_environment(storage_instance._temp_dir)
        await storage_instance.finalize()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
async def kuzu_storage():
    """Create a KuzuDB storage instance for testing"""
    storage_instance = await create_storage_instance("KuzuDBStorage")

    yield storage_instance

    # Cleanup
    try:
        if hasattr(storage_instance, "_temp_dir"):
            cleanup_kuzu_environment(storage_instance._temp_dir)
        await storage_instance.finalize()
    except Exception:
        pass  # Ignore cleanup errors
