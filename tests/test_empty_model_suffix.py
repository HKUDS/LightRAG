"""
Tests for handling empty model suffix in PostgreSQL and Qdrant storage.

This test module verifies that both storage backends gracefully handle
the case when _generate_collection_suffix() returns an empty string.
"""

from unittest.mock import Mock

from lightrag.base import BaseVectorStorage
from lightrag.utils import EmbeddingFunc


def dummy_embedding_func(*args, **kwargs):
    """Dummy embedding function for testing."""
    pass


class TestEmptyModelSuffix:
    """Test suite for handling empty model suffix scenarios."""

    def test_postgres_table_name_with_empty_suffix(self):
        """
        Test PostgreSQL table name generation when model_suffix is empty.

        Bug Fix Verification:
        - Before: table_name = "LIGHTRAG_VDB_CHUNKS_" (trailing underscore)
        - After: table_name = "LIGHTRAG_VDB_CHUNKS" (fallback to base name)
        """
        from lightrag.kg.postgres_impl import PostgresVectorDBStorage
        from lightrag.kg.shared_storage import namespace_to_table_name

        # Create a mock embedding function without get_model_identifier
        mock_embedding_func = Mock(spec=["embedding_dim"])
        mock_embedding_func.embedding_dim = 1536

        # Setup global_config without embedding_func
        global_config = {
            "embedding_batch_num": 100,
            "pgvector_precision": "hybrid",
            "pg_host": "localhost",
            "pg_port": 5432,
            "pg_user": "user",
            "pg_password": "password",
            "pg_database": "lightrag",
        }

        # Create PostgreSQL storage instance
        storage = PostgresVectorDBStorage(
            namespace="chunks",
            workspace="test",
            global_config=global_config,
            embedding_func=mock_embedding_func,
        )

        # Verify that:
        # 1. model_suffix is empty
        # 2. table_name doesn't have trailing underscore
        # 3. table_name equals the base table name
        assert storage.model_suffix == "", "model_suffix should be empty"
        assert not storage.table_name.endswith(
            "_"
        ), f"table_name should not have trailing underscore: {storage.table_name}"

        # Expected base table name
        expected_base = namespace_to_table_name("chunks")
        assert storage.table_name == expected_base, (
            f"table_name should fallback to base name when model_suffix is empty. "
            f"Expected: {expected_base}, Got: {storage.table_name}"
        )

    def test_qdrant_collection_name_with_empty_suffix(self):
        """
        Test Qdrant collection name generation when model_suffix is empty.

        Bug Fix Verification:
        - Before: final_namespace = "lightrag_vdb_chunks_" (trailing underscore)
        - After: final_namespace = "lightrag_vdb_chunks" (fallback to legacy name)
        """
        from lightrag.kg.qdrant_impl import QdrantVectorDBStorage

        # Create a mock embedding function without get_model_identifier
        mock_embedding_func = Mock(spec=["embedding_dim"])
        mock_embedding_func.embedding_dim = 1536

        # Setup global_config without embedding_func
        global_config = {
            "embedding_batch_num": 100,
            "qdrant_url": "http://localhost:6333",
        }

        # Create Qdrant storage instance
        storage = QdrantVectorDBStorage(
            namespace="chunks",
            workspace="test",
            global_config=global_config,
            embedding_func=mock_embedding_func,
        )

        # Verify that:
        # 1. model_suffix is empty
        # 2. final_namespace doesn't have trailing underscore
        # 3. final_namespace equals the legacy namespace
        assert (
            storage._generate_collection_suffix() == ""
        ), "model_suffix should be empty"
        assert not storage.final_namespace.endswith(
            "_"
        ), f"final_namespace should not have trailing underscore: {storage.final_namespace}"
        assert storage.final_namespace == storage.legacy_namespace, (
            f"final_namespace should fallback to legacy_namespace when model_suffix is empty. "
            f"Expected: {storage.legacy_namespace}, Got: {storage.final_namespace}"
        )

    def test_postgres_table_name_with_valid_suffix(self):
        """
        Test PostgreSQL table name generation with valid model suffix.

        Verification:
        - When embedding_func has get_model_identifier, use it
        - table_name has proper format: base_table_model_suffix
        """
        from lightrag.kg.postgres_impl import PostgresVectorDBStorage
        from lightrag.kg.shared_storage import namespace_to_table_name

        # Create a proper embedding function with model_name
        embedding_func = EmbeddingFunc(
            embedding_dim=1536,
            func=dummy_embedding_func,
            model_name="text-embedding-ada-002",
        )

        # Setup global_config
        global_config = {
            "embedding_batch_num": 100,
            "pgvector_precision": "hybrid",
            "pg_host": "localhost",
            "pg_port": 5432,
            "pg_user": "user",
            "pg_password": "password",
            "pg_database": "lightrag",
            "embedding_func": embedding_func,
        }

        # Create PostgreSQL storage instance
        storage = PostgresVectorDBStorage(
            namespace="chunks",
            workspace="test",
            global_config=global_config,
            embedding_func=embedding_func,
        )

        # Verify that:
        # 1. model_suffix is not empty
        # 2. table_name has correct format
        assert storage.model_suffix != "", "model_suffix should not be empty"
        assert (
            "_" in storage.table_name
        ), "table_name should contain underscore as separator"

        # Expected format: base_table_model_suffix
        expected_base = namespace_to_table_name("chunks")
        expected_model_id = embedding_func.get_model_identifier()
        expected_table_name = f"{expected_base}_{expected_model_id}"

        assert (
            storage.table_name == expected_table_name
        ), f"table_name format incorrect. Expected: {expected_table_name}, Got: {storage.table_name}"

    def test_qdrant_collection_name_with_valid_suffix(self):
        """
        Test Qdrant collection name generation with valid model suffix.

        Verification:
        - When embedding_func has get_model_identifier, use it
        - final_namespace has proper format: lightrag_vdb_namespace_model_suffix
        """
        from lightrag.kg.qdrant_impl import QdrantVectorDBStorage

        # Create a proper embedding function with model_name
        embedding_func = EmbeddingFunc(
            embedding_dim=1536,
            func=dummy_embedding_func,
            model_name="text-embedding-ada-002",
        )

        # Setup global_config
        global_config = {
            "embedding_batch_num": 100,
            "qdrant_url": "http://localhost:6333",
            "embedding_func": embedding_func,
        }

        # Create Qdrant storage instance
        storage = QdrantVectorDBStorage(
            namespace="chunks",
            workspace="test",
            global_config=global_config,
            embedding_func=embedding_func,
        )

        # Verify that:
        # 1. model_suffix is not empty
        # 2. final_namespace has correct format
        model_suffix = storage._generate_collection_suffix()
        assert model_suffix != "", "model_suffix should not be empty"
        assert (
            "_" in storage.final_namespace
        ), "final_namespace should contain underscore as separator"

        # Expected format: lightrag_vdb_namespace_model_suffix
        expected_model_id = embedding_func.get_model_identifier()
        expected_collection_name = f"lightrag_vdb_chunks_{expected_model_id}"

        assert (
            storage.final_namespace == expected_collection_name
        ), f"final_namespace format incorrect. Expected: {expected_collection_name}, Got: {storage.final_namespace}"

    def test_suffix_generation_fallback_chain(self):
        """
        Test the fallback chain in _generate_collection_suffix.

        Verification:
        1. Direct method: embedding_func.get_model_identifier()
        2. Global config fallback: global_config["embedding_func"].get_model_identifier()
        3. Final fallback: return empty string
        """

        # Create a concrete implementation for testing
        class TestStorage(BaseVectorStorage):
            async def query(self, *args, **kwargs):
                pass

            async def upsert(self, *args, **kwargs):
                pass

            async def delete_entity(self, *args, **kwargs):
                pass

            async def delete_entity_relation(self, *args, **kwargs):
                pass

            async def get_by_id(self, *args, **kwargs):
                pass

            async def get_by_ids(self, *args, **kwargs):
                pass

            async def delete(self, *args, **kwargs):
                pass

            async def get_vectors_by_ids(self, *args, **kwargs):
                pass

            async def index_done_callback(self):
                pass

            async def drop(self):
                pass

        # Case 1: Direct method available
        embedding_func = EmbeddingFunc(
            embedding_dim=1536, func=dummy_embedding_func, model_name="test-model"
        )
        storage = TestStorage(
            namespace="test",
            workspace="test",
            global_config={},
            embedding_func=embedding_func,
        )
        assert (
            storage._generate_collection_suffix() == "test_model_1536d"
        ), "Should use direct method when available"

        # Case 2: Global config fallback
        mock_embedding_func = Mock(spec=[])  # No get_model_identifier
        storage = TestStorage(
            namespace="test",
            workspace="test",
            global_config={"embedding_func": embedding_func},
            embedding_func=mock_embedding_func,
        )
        assert (
            storage._generate_collection_suffix() == "test_model_1536d"
        ), "Should fallback to global_config embedding_func"

        # Case 3: Final fallback (no embedding_func anywhere)
        storage = TestStorage(
            namespace="test",
            workspace="test",
            global_config={},
            embedding_func=mock_embedding_func,
        )
        assert (
            storage._generate_collection_suffix() == ""
        ), "Should return empty string when no model_identifier available"
