"""
Storage initialization and setup utilities
"""

import os
import sys
import importlib
import tempfile
import shutil
from typing import Optional, Tuple
from ascii_colors import ASCIIColors

# Add parent directory to path for imports
# sys.path.append(
#     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# )

from lightrag.kg import (
    STORAGE_IMPLEMENTATIONS,
    STORAGE_ENV_REQUIREMENTS,
    STORAGES,
    verify_storage_implementation,
)
from lightrag.kg.shared_storage import initialize_share_data
from .translation_engine import t


async def mock_embedding_func(texts):
    """Mock embedding function for testing"""
    import numpy as np

    return np.random.rand(len(texts), 10)  # Return 10-dimensional random vectors


def test_check_env_file() -> bool:
    """
    Check if .env file exists, issue warning if not
    Returns True to continue, False to exit
    """
    if not os.path.exists(".env"):
        warning_msg = t("warning_no_env")
        ASCIIColors.yellow(warning_msg)

        # Check if running in interactive terminal
        if sys.stdin.isatty():
            response = input(t("continue_execution"))
            if response.lower() != "yes":
                ASCIIColors.red(t("test_cancelled"))
                return False
    return True


def setup_kuzu_test_environment() -> Tuple[str, str]:
    """
    Setup KuzuDB test environment
    Returns tuple of (test_dir, kuzu_db_path)
    """
    # Create temporary directory for KuzuDB testing
    test_dir = tempfile.mkdtemp(prefix="kuzu_test_")
    kuzu_db_path = os.path.join(test_dir, "test_kuzu.db")

    # Set environment variables
    os.environ["KUZU_DB_PATH"] = kuzu_db_path
    os.environ["KUZU_WORKSPACE"] = "test_workspace"

    return test_dir, kuzu_db_path


def cleanup_kuzu_test_environment(test_dir: str) -> None:
    """
    Cleanup KuzuDB test environment
    """
    try:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
    except Exception as e:
        ASCIIColors.yellow(t("warning_cleanup_temp_dir_failed") % str(e))


async def initialize_graph_test_storage():
    """
    Initialize graph storage instance based on environment variables
    Returns initialized storage instance or None if failed
    """
    # Get graph storage type from environment
    graph_storage_type = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")

    # Verify storage type is valid
    try:
        verify_storage_implementation("GRAPH_STORAGE", graph_storage_type)
    except ValueError as e:
        ASCIIColors.red(t("error_general") % str(e))
        ASCIIColors.yellow(
            t("supported_graph_storage_types")
            % ", ".join(STORAGE_IMPLEMENTATIONS["GRAPH_STORAGE"]["implementations"])
        )
        return None

    # Check required environment variables
    required_env_vars = STORAGE_ENV_REQUIREMENTS.get(graph_storage_type, [])
    missing_env_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_env_vars:
        ASCIIColors.red(
            t("error_missing_env_vars")
            % (graph_storage_type, ", ".join(missing_env_vars))
        )
        return None

    # Special handling for KuzuDB: automatically setup test environment
    temp_dir = None
    if graph_storage_type == "KuzuDBStorage":
        temp_dir, kuzu_db_path = setup_kuzu_test_environment()
        ASCIIColors.cyan(t("kuzu_test_environment_setup") % kuzu_db_path)

    # Dynamically import the appropriate module
    module_path = STORAGES.get(graph_storage_type)
    if not module_path:
        ASCIIColors.red(t("error_module_path_not_found") % graph_storage_type)
        if temp_dir:
            cleanup_kuzu_test_environment(temp_dir)
        return None

    try:
        module = importlib.import_module(module_path, package="lightrag")
        storage_class = getattr(module, graph_storage_type)
    except (ImportError, AttributeError) as e:
        ASCIIColors.red(t("error_import_failed") % (graph_storage_type, str(e)))
        if temp_dir:
            cleanup_kuzu_test_environment(temp_dir)
        return None

    # Initialize storage instance
    global_config = {
        "embedding_batch_num": 10,  # Batch size
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.5  # Cosine similarity threshold
        },
        "working_dir": os.environ.get(
            "WORKING_DIR", "./rag_storage"
        ),  # Working directory
        "max_graph_nodes": 1000,  # Required for KuzuDB
    }

    # NetworkXStorage requires shared_storage initialization
    if graph_storage_type == "NetworkXStorage":
        initialize_share_data()  # Use single process mode

    try:
        storage = storage_class(
            namespace="test_graph",
            global_config=global_config,
            embedding_func=mock_embedding_func,
            workspace="test_workspace",
        )

        # Initialize connection
        await storage.initialize()

        # Store temporary directory info in storage object for later cleanup
        if temp_dir:
            storage._temp_dir = temp_dir

        return storage
    except Exception as e:
        ASCIIColors.red(t("error_initialization_failed") % (graph_storage_type, str(e)))
        if temp_dir:
            cleanup_kuzu_test_environment(temp_dir)
        return None
