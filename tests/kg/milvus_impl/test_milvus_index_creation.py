"""
Tests for Milvus index creation behavior

This test suite validates:
1. P1: build_index_params uses compatibility helper
2. P2: Vector index creation failures are surfaced to callers
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch
from lightrag.kg.milvus_impl import (
    MILVUS_MAX_VARCHAR_BYTES,
    MilvusIndexConfig,
    MilvusVectorDBStorage,
)


def _make_storage(namespace="entities"):
    mock_embedding_func = MagicMock()
    mock_embedding_func.embedding_dim = 128
    return MilvusVectorDBStorage(
        namespace=namespace,
        workspace="test_workspace",
        global_config={
            "embedding_batch_num": 100,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.3,
            },
        },
        embedding_func=mock_embedding_func,
        meta_fields=set(),
    )


def _field_max_length(field):
    return int(field.params["max_length"])


def _collection_info(field_names):
    fields = [
        {"name": "id", "type": "VarChar", "is_primary": True},
        {"name": "vector", "type": "FloatVector", "params": {"dim": 128}},
        {"name": "created_at", "type": "Int64"},
    ]
    fields.extend(
        {
            "name": field_name,
            "type": "VarChar",
            "params": {"max_length": MILVUS_MAX_VARCHAR_BYTES},
        }
        for field_name in field_names
    )
    return {"fields": fields}


class _EmbeddingFunc:
    def __init__(self, dim=128, model_name="text-embedding-3-small"):
        self.embedding_dim = dim
        self.model_name = model_name


def _make_model_storage(namespace="entities", workspace="test_workspace", dim=128):
    return MilvusVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 100,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.3,
            },
        },
        embedding_func=_EmbeddingFunc(dim=dim),
        meta_fields=set(),
    )


def _wire_collection_state(storage, collections, describe_by_name=None):
    storage._client = MagicMock()
    describe_by_name = describe_by_name or {}

    def has_collection(collection_name):
        return collection_name in collections

    def create_collection(collection_name, schema):
        collections.add(collection_name)

    def drop_collection(collection_name):
        collections.discard(collection_name)

    def rename_collection(source, target):
        collections.discard(source)
        collections.add(target)

    def describe_collection(collection_name):
        return describe_by_name.get(collection_name, _collection_info([]))

    storage._client.has_collection.side_effect = has_collection
    storage._client.create_collection.side_effect = create_collection
    storage._client.drop_collection.side_effect = drop_collection
    storage._client.rename_collection.side_effect = rename_collection
    storage._client.describe_collection.side_effect = describe_collection
    return storage._client


@pytest.mark.offline
class TestMilvusIndexCreation:
    """Test index creation behavior and error handling"""

    @pytest.mark.parametrize(
        ("namespace", "expected_fields"),
        [
            ("entities", {"content", "source_id"}),
            ("relationships", {"content", "source_id"}),
            ("chunks", {"content"}),
        ],
    )
    def test_schema_promotes_core_metadata_fields(self, namespace, expected_fields):
        storage = _make_storage(namespace=namespace)

        fields_by_name = {
            field.name: field for field in storage._create_schema_for_namespace().fields
        }

        assert expected_fields.issubset(fields_by_name)
        for field_name in expected_fields:
            assert (
                _field_max_length(fields_by_name[field_name])
                == MILVUS_MAX_VARCHAR_BYTES
            )

    @pytest.mark.parametrize(
        ("namespace", "old_fields"),
        [
            ("entities", ["entity_name", "file_path"]),
            ("relationships", ["src_id", "tgt_id", "file_path"]),
            ("chunks", ["full_doc_id", "file_path"]),
        ],
    )
    def test_missing_core_metadata_fields_trigger_schema_migration(
        self, namespace, old_fields
    ):
        storage = _make_storage(namespace=namespace)

        with patch.object(storage, "_migrate_collection_schema") as migrate:
            storage._check_schema_compatibility(_collection_info(old_fields))

        migrate.assert_called_once_with()

    def test_migration_sanitizes_varchar_rows_before_insert(self):
        storage = _make_storage(namespace="entities")
        storage.final_namespace = "test_entities"
        storage._client = MagicMock()
        iterator = MagicMock()
        iterator.next.side_effect = [
            [
                {
                    "id": "ent-1",
                    "vector": [0.0] * 128,
                    "content": "x" * (MILVUS_MAX_VARCHAR_BYTES + 10),
                    "source_id": "源" * (MILVUS_MAX_VARCHAR_BYTES // 3 + 10),
                }
            ],
            [],
        ]
        storage._client.query_iterator.return_value = iterator

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema()

        inserted = storage._client.insert.call_args.kwargs["data"][0]
        assert len(inserted["content"].encode("utf-8")) <= MILVUS_MAX_VARCHAR_BYTES
        assert len(inserted["source_id"].encode("utf-8")) <= MILVUS_MAX_VARCHAR_BYTES
        inserted["source_id"].encode("utf-8").decode("utf-8")

    def test_migration_truncates_oversized_non_primary_identity_field(self):
        # Legacy $meta did not enforce the 512-byte entity_name limit, so an
        # oversized value must be truncated (not rejected) during migration so
        # one pathological row cannot abort the whole collection migration.
        storage = _make_storage(namespace="entities")
        normalized = storage._normalize_migration_row(
            {"id": "ent-1", "entity_name": "e" * 513, "content": "body"}
        )
        assert len(normalized["entity_name"].encode("utf-8")) == 512

    def test_migration_rejects_oversized_primary_key(self):
        # The primary key is never truncated, even during migration: collapsing
        # two ids would silently overwrite a row.
        storage = _make_storage(namespace="entities")
        with pytest.raises(ValueError, match="primary keys cannot be truncated"):
            storage._normalize_migration_row({"id": "i" * 65, "content": "body"})

    def test_legacy_without_vector_field_creates_fresh_suffixed_collection(self):
        # Old simple-schema collections have no vector field; their rows carry no
        # vectors, so migrating them into the required-vector schema would fail at
        # insert and block startup. They must be skipped and a fresh suffixed
        # collection created instead.
        storage = _make_model_storage()
        legacy_info = {
            "fields": [
                {"name": "id", "type": "VarChar", "is_primary": True},
                {
                    "name": "entity_name",
                    "type": "VarChar",
                    "params": {"max_length": 512},
                },
            ]
        }
        client = _wire_collection_state(
            storage,
            {storage.legacy_namespace},
            {storage.legacy_namespace: legacy_info},
        )

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_migrate_collection_schema") as migrate:
                storage._create_collection_if_not_exist()

        migrate.assert_not_called()
        client.query_iterator.assert_not_called()
        client.create_collection.assert_called_once()
        assert client.create_collection.call_args.kwargs["collection_name"] == (
            storage.final_namespace
        )
        client.load_collection.assert_called_with(storage.final_namespace)

    def test_model_suffix_collection_naming_with_workspace(self):
        storage = MilvusVectorDBStorage(
            namespace="chunks",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=_EmbeddingFunc(
                dim=3072, model_name="text-embedding-3-large"
            ),
            meta_fields=set(),
        )

        assert storage.legacy_namespace == "space1_chunks"
        assert storage.final_namespace == "space1_chunks_text_embedding_3_large_3072d"

    def test_model_suffix_collection_naming_without_workspace(self):
        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=_EmbeddingFunc(dim=2560, model_name=" qwen3-embedding:4b "),
            meta_fields=set(),
        )

        assert storage.legacy_namespace == "entities"
        assert storage.final_namespace == "entities_qwen3_embedding_4b_2560d"

    @pytest.mark.parametrize("model_name", ["", "   ", 123])
    def test_missing_or_invalid_model_name_keeps_legacy_collection_name(
        self, model_name
    ):
        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=_EmbeddingFunc(model_name=model_name),
            meta_fields=set(),
        )

        assert storage.model_suffix is None
        assert storage.legacy_namespace == "space1_entities"
        assert storage.final_namespace == "space1_entities"

    def test_creates_suffixed_collection_when_no_collection_exists(self):
        storage = _make_model_storage()
        client = _wire_collection_state(storage, set())

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._create_collection_if_not_exist()

        client.create_collection.assert_called_once()
        assert client.create_collection.call_args.kwargs["collection_name"] == (
            storage.final_namespace
        )
        client.query_iterator.assert_not_called()
        client.load_collection.assert_called_with(storage.final_namespace)

    def test_existing_suffixed_collection_is_validated_and_used(self):
        storage = _make_model_storage()
        client = _wire_collection_state(
            storage,
            {storage.final_namespace},
            {
                storage.final_namespace: _collection_info(
                    ["entity_name", "content", "source_id", "file_path"]
                )
            },
        )

        with patch.object(storage, "_migrate_collection_schema") as migrate:
            storage._create_collection_if_not_exist()

        migrate.assert_not_called()
        client.create_collection.assert_not_called()
        client.load_collection.assert_called_with(storage.final_namespace)

    def test_legacy_old_meta_schema_migrates_to_suffixed_collection(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(
            storage,
            collections,
            {
                storage.legacy_namespace: _collection_info(
                    ["entity_name", "file_path"]
                ),
                storage.final_namespace: _collection_info(
                    ["entity_name", "content", "source_id", "file_path"]
                ),
            },
        )

        def migrate_collection(**_kwargs):
            collections.add(storage.final_namespace)

        with patch.object(
            storage, "_migrate_collection_schema", side_effect=migrate_collection
        ) as migrate:
            storage._create_collection_if_not_exist()

        migrate.assert_called_once_with(
            source_collection_name=storage.legacy_namespace,
            target_collection_name=storage.final_namespace,
        )
        client.query_iterator.assert_not_called()
        client.insert.assert_not_called()
        client.create_collection.assert_not_called()
        client.load_collection.assert_called_with(storage.final_namespace)

    def test_legacy_new_schema_migrates_to_suffixed_collection(self):
        storage = _make_model_storage(namespace="chunks")
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(
            storage,
            collections,
            {
                storage.legacy_namespace: _collection_info(
                    ["full_doc_id", "content", "file_path"]
                ),
                storage.final_namespace: _collection_info(
                    ["full_doc_id", "content", "file_path"]
                ),
            },
        )

        def migrate_collection(**_kwargs):
            collections.add(storage.final_namespace)

        with patch.object(
            storage, "_migrate_collection_schema", side_effect=migrate_collection
        ) as migrate:
            storage._create_collection_if_not_exist()

        migrate.assert_called_once_with(
            source_collection_name=storage.legacy_namespace,
            target_collection_name=storage.final_namespace,
        )
        client.query_iterator.assert_not_called()
        client.insert.assert_not_called()
        client.rename_collection.assert_not_called()
        client.create_collection.assert_not_called()
        client.load_collection.assert_called_with(storage.final_namespace)

    def test_legacy_same_dimension_migrates_to_suffixed_collection(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(
            storage,
            collections,
            {
                storage.legacy_namespace: _collection_info(
                    ["entity_name", "content", "source_id"]
                )
            },
        )

        def migrate_collection(**_kwargs):
            collections.add(storage.final_namespace)

        with patch.object(
            storage, "_migrate_collection_schema", side_effect=migrate_collection
        ) as migrate:
            storage._create_collection_if_not_exist()

        migrate.assert_called_once_with(
            source_collection_name=storage.legacy_namespace,
            target_collection_name=storage.final_namespace,
        )
        client.describe_collection.assert_called_once_with(storage.legacy_namespace)
        client.query_iterator.assert_not_called()
        client.create_collection.assert_not_called()
        client.load_collection.assert_called_with(storage.final_namespace)

    def test_no_model_suffix_old_meta_schema_migrates_in_place(self):
        storage = _make_storage(namespace="entities")
        storage.model_suffix = None
        storage.legacy_namespace = storage.final_namespace
        client = _wire_collection_state(
            storage,
            {storage.final_namespace},
            {storage.final_namespace: _collection_info(["entity_name", "file_path"])},
        )

        with patch.object(storage, "_migrate_collection_schema") as migrate:
            storage._create_collection_if_not_exist()

        migrate.assert_called_once_with()
        client.create_collection.assert_not_called()
        client.load_collection.assert_called_with(storage.final_namespace)

    def test_legacy_dimension_mismatch_creates_suffixed_collection_without_migration(
        self,
    ):
        storage = _make_model_storage()
        legacy_info = _collection_info(["entity_name", "content", "source_id"])
        for field in legacy_info["fields"]:
            if field["name"] == "vector":
                field["params"]["dim"] = 256
        client = _wire_collection_state(
            storage,
            {storage.legacy_namespace},
            {storage.legacy_namespace: legacy_info},
        )

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_migrate_collection_schema") as migrate:
                storage._create_collection_if_not_exist()

        migrate.assert_not_called()
        client.query_iterator.assert_not_called()
        client.create_collection.assert_called_once()
        assert client.create_collection.call_args.kwargs["collection_name"] == (
            storage.final_namespace
        )

    def test_legacy_describe_failure_raises_without_creating_suffixed_collection(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        client.describe_collection.side_effect = RuntimeError("milvus unavailable")

        with pytest.raises(RuntimeError, match="milvus unavailable"):
            storage._create_collection_if_not_exist()

        client.describe_collection.assert_called_once_with(storage.legacy_namespace)
        client.query_iterator.assert_not_called()
        client.create_collection.assert_not_called()
        assert storage.legacy_namespace in collections
        assert storage.final_namespace not in collections

    def test_legacy_migration_failure_keeps_legacy_collection(self):
        storage = _make_model_storage()
        legacy_info = _collection_info(["entity_name", "content", "source_id"])
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(
            storage,
            collections,
            {storage.legacy_namespace: legacy_info},
        )

        with patch.object(
            storage,
            "_migrate_collection_schema",
            side_effect=RuntimeError("migration failed"),
        ) as migrate:
            with pytest.raises(RuntimeError, match="migration failed"):
                storage._create_collection_if_not_exist()

        migrate.assert_called_once_with(
            source_collection_name=storage.legacy_namespace,
            target_collection_name=storage.final_namespace,
        )
        client.query_iterator.assert_not_called()
        client.create_collection.assert_not_called()
        assert storage.legacy_namespace in collections
        assert storage.final_namespace not in collections

    def test_migration_insert_batches_use_build_upsert_batches(self):
        storage = _make_model_storage()
        storage._max_upsert_payload_bytes = 1024
        storage._max_upsert_records_per_batch = 2000
        client = _wire_collection_state(storage, {storage.legacy_namespace})
        iterator = MagicMock()
        iterator.next.side_effect = [
            [
                {
                    "id": f"ent-{i}",
                    "vector": [0.0] * 128,
                    "content": "x" * 300,
                }
                for i in range(2000)
            ],
            [],
        ]
        client.query_iterator.return_value = iterator

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(
                storage, "_build_upsert_batches", wraps=storage._build_upsert_batches
            ) as build_batches:
                with patch.object(storage, "_flush_pending_vector_ops") as flush:
                    storage._migrate_collection_schema(
                        source_collection_name=storage.legacy_namespace,
                        target_collection_name=storage.final_namespace,
                    )

        build_batches.assert_called()
        assert client.insert.call_count > 1
        flush.assert_not_called()

    def test_failed_legacy_migration_cleans_temp_and_keeps_legacy_collection(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        iterator = MagicMock()
        iterator.next.side_effect = [
            [{"id": "ent-1", "vector": [0.0] * 128, "content": "body"}],
        ]
        client.query_iterator.return_value = iterator
        client.insert.side_effect = RuntimeError("insert failed")

        with patch.object(storage, "_create_indexes_after_collection"):
            with pytest.raises(RuntimeError, match="Iterator-based migration failed"):
                storage._migrate_collection_schema(
                    source_collection_name=storage.legacy_namespace,
                    target_collection_name=storage.final_namespace,
                )

        assert storage.legacy_namespace in collections
        assert f"{storage.final_namespace}_temp" not in collections
        assert storage.final_namespace not in collections

    def test_vector_index_creation_failure_is_raised(self):
        """Test that vector index creation failures are raised to the caller (P2 fix)"""
        # Setup storage instance
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "HNSW",
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Mock the client and _get_index_params
        mock_client = MagicMock()
        mock_index_params = MagicMock()

        storage._client = mock_client
        storage.final_namespace = "test_entities"

        # Mock _get_index_params to return a valid IndexParams
        with patch.object(storage, "_get_index_params", return_value=mock_index_params):
            # Mock build_index_params to return the mock_index_params
            with patch.object(
                storage.index_config,
                "build_index_params",
                return_value=mock_index_params,
            ):
                # Mock create_index to raise an exception (simulating index creation failure)
                mock_client.create_index.side_effect = Exception(
                    "Index creation failed"
                )

                # Verify that the exception is raised (not caught and logged)
                with pytest.raises(Exception, match="Index creation failed"):
                    storage._create_indexes_after_collection()

    def test_scalar_index_creation_failure_is_logged_not_raised(self):
        """Test that scalar index creation failures are logged but not raised (existing behavior)"""
        # Setup storage instance
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "AUTOINDEX",  # No custom vector index
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Mock the client and _get_index_params
        mock_client = MagicMock()
        mock_index_params = MagicMock()

        storage._client = mock_client
        storage.final_namespace = "test_entities"

        # Mock _get_index_params to return a valid IndexParams for scalar indexes
        with patch.object(storage, "_get_index_params", return_value=mock_index_params):
            # Let vector AUTOINDEX creation succeed, then fail on scalar index creation
            mock_client.create_index.side_effect = [
                None,
                Exception("Scalar index creation failed"),
            ]

            # Verify that the function completes without raising (scalar index failures are logged)
            # This should not raise an exception
            storage._create_indexes_after_collection()

            # The function should complete successfully even though scalar index creation failed

    def test_build_index_params_uses_passed_index_params(self):
        """Test that build_index_params uses the passed index_params parameter (P1 fix)"""
        config = MilvusIndexConfig(
            index_type="HNSW",
            metric_type="COSINE",
            hnsw_m=32,
            hnsw_ef_construction=256,
        )

        mock_index_params = MagicMock()

        # Call build_index_params with the mock_index_params
        result = config.build_index_params(mock_index_params)

        # Verify that it used the passed index_params
        assert result == mock_index_params
        mock_index_params.add_index.assert_called_once()

    def test_build_index_params_raises_when_index_params_is_none_for_custom_type(self):
        """Test that build_index_params raises RuntimeError when index_params is None for custom types (P1 fix)"""
        config = MilvusIndexConfig(
            index_type="HNSW",
            metric_type="COSINE",
        )

        # Call with None (simulating compatibility helper returning None)
        # Should raise RuntimeError for non-AUTOINDEX types
        with pytest.raises(RuntimeError, match="IndexParams not available"):
            config.build_index_params(None)

    def test_build_index_params_returns_none_for_autoindex_when_index_params_is_none(
        self,
    ):
        """Test AUTOINDEX falls back to direct API parameters when IndexParams is unavailable."""
        config = MilvusIndexConfig(
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        # AUTOINDEX should still produce direct API parameters
        result = config.build_index_params(None)
        assert result == {
            "field_name": "vector",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
            "params": {},
        }

    def test_build_index_params_autoindex_uses_index_params_object(self):
        """Test AUTOINDEX still creates an explicit vector index when IndexParams is available."""
        config = MilvusIndexConfig(
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        mock_index_params = MagicMock()

        result = config.build_index_params(mock_index_params)

        assert result == mock_index_params
        mock_index_params.add_index.assert_called_once_with(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
            params={},
        )

    def test_create_indexes_uses_compatibility_helper(self):
        """Test that _create_indexes_after_collection uses _get_index_params (P1 fix)"""
        # Setup storage instance
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "HNSW",
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        # Mock the client
        mock_client = MagicMock()
        mock_index_params = MagicMock()

        storage._client = mock_client
        storage.final_namespace = "test_entities"

        # Spy on _get_index_params to verify it's called
        with patch.object(
            storage, "_get_index_params", return_value=mock_index_params
        ) as mock_get_index_params:
            # Call the method
            storage._create_indexes_after_collection()

            # Verify that _get_index_params was called at least once
            assert mock_get_index_params.call_count >= 1

    def test_version_probing_only_for_hnsw_sq(self):
        """Test that get_server_version is only called when index type requires it (P2 fix)"""
        from unittest.mock import AsyncMock

        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        # Test with HNSW (no version requirement) - should NOT call get_server_version
        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "HNSW",
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        mock_client = MagicMock()
        storage._client = mock_client

        # Mock the init lock as an async context manager
        mock_lock = AsyncMock()

        with patch(
            "lightrag.kg.milvus_impl.get_data_init_lock", return_value=mock_lock
        ):
            with patch.object(storage, "_create_collection_if_not_exist"):
                asyncio.run(storage.initialize())

        # get_server_version should NOT be called for HNSW
        mock_client.get_server_version.assert_not_called()

    def test_version_probing_called_for_hnsw_sq(self):
        """Test that get_server_version IS called when HNSW_SQ is configured (P2 fix)"""
        from unittest.mock import AsyncMock

        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "HNSW_SQ",
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        mock_client = MagicMock()
        mock_client.get_server_version.return_value = "2.6.9"
        storage._client = mock_client

        # Mock the init lock as an async context manager
        mock_lock = AsyncMock()

        with patch(
            "lightrag.kg.milvus_impl.get_data_init_lock", return_value=mock_lock
        ):
            with patch.object(storage, "_create_collection_if_not_exist"):
                asyncio.run(storage.initialize())

        # get_server_version SHOULD be called for HNSW_SQ
        mock_client.get_server_version.assert_called_once()

    def test_initialize_creates_missing_database_before_collection_setup(self):
        """Test that initialize bootstraps a missing configured Milvus database."""
        from unittest.mock import AsyncMock

        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        bootstrap_client = MagicMock()
        bootstrap_client.list_databases.return_value = ["default"]
        mock_lock = AsyncMock()

        with patch.dict(
            "os.environ",
            {
                "MILVUS_URI": "http://milvus:19530",
                "MILVUS_DB_NAME": "lightrag",
            },
            clear=False,
        ):
            with patch(
                "lightrag.kg.milvus_impl.MilvusClient", return_value=bootstrap_client
            ) as mock_client_cls:
                with patch(
                    "lightrag.kg.milvus_impl.get_data_init_lock",
                    return_value=mock_lock,
                ):
                    with patch.object(storage, "_create_collection_if_not_exist"):
                        asyncio.run(storage.initialize())

        mock_client_cls.assert_called_once_with(
            uri="http://milvus:19530",
            user=None,
            password=None,
            token=None,
        )
        bootstrap_client.list_databases.assert_called_once_with()
        bootstrap_client.create_database.assert_called_once_with("lightrag")
        bootstrap_client.use_database.assert_called_once_with("lightrag")

    def test_initialize_uses_existing_database_without_recreating_it(self):
        """Test that initialize switches to an existing configured Milvus database."""
        from unittest.mock import AsyncMock

        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        bootstrap_client = MagicMock()
        bootstrap_client.list_databases.return_value = ["default", "lightrag"]
        mock_lock = AsyncMock()

        with patch.dict(
            "os.environ",
            {
                "MILVUS_URI": "http://milvus:19530",
                "MILVUS_DB_NAME": "lightrag",
            },
            clear=False,
        ):
            with patch(
                "lightrag.kg.milvus_impl.MilvusClient", return_value=bootstrap_client
            ):
                with patch(
                    "lightrag.kg.milvus_impl.get_data_init_lock",
                    return_value=mock_lock,
                ):
                    with patch.object(storage, "_create_collection_if_not_exist"):
                        asyncio.run(storage.initialize())

        bootstrap_client.list_databases.assert_called_once_with()
        bootstrap_client.create_database.assert_not_called()
        bootstrap_client.use_database.assert_called_once_with("lightrag")

    def test_existing_collection_missing_vector_index_is_repaired(self):
        """Existing collections missing vector indexes should be repaired automatically."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )
        storage.final_namespace = "space1_entities"
        storage._client = MagicMock()
        storage._client.has_collection.return_value = True

        load_error = RuntimeError(
            "there is no vector index on field: [vector], please create index firstly"
        )

        with patch.object(storage._client, "describe_collection", return_value={}):
            with patch.object(storage, "_validate_collection_compatibility"):
                with patch.object(
                    storage,
                    "_ensure_collection_loaded",
                    side_effect=[load_error, None],
                ) as mock_load:
                    with patch.object(
                        storage, "_repair_missing_vector_index"
                    ) as mock_repair:
                        storage._create_collection_if_not_exist()

        assert mock_load.call_count == 2
        mock_repair.assert_called_once_with()

    def test_existing_collection_index_repair_failure_has_precise_error(self):
        """Index repair failures should not be reported as collection validation failures."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )
        storage.final_namespace = "space1_entities"
        storage._client = MagicMock()
        storage._client.has_collection.return_value = True

        load_error = RuntimeError(
            "there is no vector index on field: [vector], please create index firstly"
        )

        with patch.object(storage._client, "describe_collection", return_value={}):
            with patch.object(storage, "_validate_collection_compatibility"):
                with patch.object(
                    storage, "_ensure_collection_loaded", side_effect=load_error
                ):
                    with patch.object(
                        storage,
                        "_repair_missing_vector_index",
                        side_effect=RuntimeError("create index failed"),
                    ):
                        with pytest.raises(
                            RuntimeError,
                            match="Index repair failed for collection 'space1_entities'",
                        ):
                            storage._create_collection_if_not_exist()

    def test_existing_collection_non_index_validation_failure_still_raises(self):
        """Non-index validation failures should still stop initialization."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )
        storage.final_namespace = "space1_entities"
        storage._client = MagicMock()
        storage._client.has_collection.return_value = True

        with patch.object(storage._client, "describe_collection", return_value={}):
            with patch.object(
                storage,
                "_validate_collection_compatibility",
                side_effect=RuntimeError("dimension mismatch"),
            ):
                with pytest.raises(
                    RuntimeError,
                    match="Collection validation failed for 'space1_entities'",
                ):
                    storage._create_collection_if_not_exist()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
