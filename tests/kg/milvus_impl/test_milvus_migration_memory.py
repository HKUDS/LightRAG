"""
Tests for Milvus schema-migration memory back-pressure.

The bulk copy inserts the whole source collection into a temp collection with
no client-side throttle, so the Milvus data node accumulates insert-buffer
segments until its auto-flush catches up — a large migration can exhaust
server memory. These tests cover the periodic/final flush that bounds that
buffer and the optional inter-batch throttle.
"""

import pytest
from unittest.mock import MagicMock, patch
from pymilvus import MilvusException

from lightrag.kg.milvus_impl import MilvusVectorDBStorage


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


def _wire_collection_state(storage, collections):
    storage._client = MagicMock()

    def has_collection(collection_name):
        return collection_name in collections

    def create_collection(collection_name, schema):
        collections.add(collection_name)

    def drop_collection(collection_name):
        collections.discard(collection_name)

    def rename_collection(source, target):
        collections.discard(source)
        collections.add(target)

    storage._client.has_collection.side_effect = has_collection
    storage._client.create_collection.side_effect = create_collection
    storage._client.drop_collection.side_effect = drop_collection
    storage._client.rename_collection.side_effect = rename_collection
    return storage._client


def _wire_iterator(client, batches):
    """query_iterator yields the given list of row-batches then []."""

    def make_iterator(**kwargs):
        iterator = MagicMock()
        iterator.next.side_effect = list(batches) + [[]]
        return iterator

    client.query_iterator.side_effect = make_iterator


def _rows(n, start=0):
    return [
        {"id": f"ent-{i}", "vector": [0.0] * 128, "content": "body"}
        for i in range(start, start + n)
    ]


@pytest.mark.offline
class TestMigrationFlushBackpressure:
    def test_periodic_and_final_flush_by_row_interval(self):
        storage = _make_model_storage()
        storage._migration_flush_interval_rows = 3
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        temp = f"{storage.final_namespace}_temp"
        # Three iterator batches of 2 rows: cumulative 2, 4, 6.
        # Periodic flush trips once (at 4 >= 3), then a final flush at the end.
        _wire_iterator(client, [_rows(2), _rows(2, 2), _rows(2, 4)])

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        flush_targets = [call.args[0] for call in client.flush.call_args_list]
        assert flush_targets == [temp, temp]  # one periodic + one final
        assert storage.final_namespace in collections

    def test_flush_disabled_when_interval_zero(self):
        storage = _make_model_storage()
        storage._migration_flush_interval_rows = 0
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_iterator(client, [_rows(2), _rows(2, 2)])

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        client.flush.assert_not_called()
        assert storage.final_namespace in collections

    def test_no_final_flush_for_empty_source(self):
        storage = _make_model_storage()
        storage._migration_flush_interval_rows = 50000
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_iterator(client, [])  # empty source

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        client.flush.assert_not_called()

    def test_flush_connection_error_is_retryable(self):
        # A flush failing with a connection error must be retried like any other
        # connection-class migration failure (rebuild client, restart attempt).
        storage = _make_model_storage()
        storage._migration_flush_interval_rows = 1
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_iterator(client, [_rows(2)])

        flush_calls = {"n": 0}

        def flush(collection_name, **kwargs):
            flush_calls["n"] += 1
            if flush_calls["n"] == 1:
                raise MilvusException(
                    code=2, message="Fail connecting to server on host:19530"
                )

        client.flush.side_effect = flush

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client") as rebuild:
                with patch("lightrag.kg.milvus_impl.time.sleep"):
                    storage._migrate_collection_schema(
                        source_collection_name=storage.legacy_namespace,
                        target_collection_name=storage.final_namespace,
                    )

        rebuild.assert_called_once()
        assert storage.final_namespace in collections


@pytest.mark.offline
class TestMigrationBatchThrottle:
    def test_batch_sleep_throttles_between_batches(self):
        storage = _make_model_storage()
        storage._migration_flush_interval_rows = 0
        storage._migration_batch_sleep = 0.05
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_iterator(client, [_rows(2), _rows(2, 2)])

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch("lightrag.kg.milvus_impl.time.sleep") as sleep:
                storage._migrate_collection_schema(
                    source_collection_name=storage.legacy_namespace,
                    target_collection_name=storage.final_namespace,
                )

        # One sleep per non-empty iterator batch, none for the terminating [].
        assert [call.args[0] for call in sleep.call_args_list] == [0.05, 0.05]

    def test_no_sleep_when_disabled(self):
        storage = _make_model_storage()
        storage._migration_flush_interval_rows = 0
        storage._migration_batch_sleep = 0.0
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_iterator(client, [_rows(2)])

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch("lightrag.kg.milvus_impl.time.sleep") as sleep:
                storage._migrate_collection_schema(
                    source_collection_name=storage.legacy_namespace,
                    target_collection_name=storage.final_namespace,
                )

        sleep.assert_not_called()


@pytest.mark.offline
class TestMigrationBackpressureConfig:
    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("MILVUS_MIGRATION_FLUSH_INTERVAL_ROWS", "1234")
        monkeypatch.setenv("MILVUS_MIGRATION_BATCH_SLEEP", "0.25")
        storage = _make_model_storage()
        assert storage._migration_flush_interval_rows == 1234
        assert storage._migration_batch_sleep == 0.25

    def test_negative_values_clamped(self, monkeypatch):
        monkeypatch.setenv("MILVUS_MIGRATION_FLUSH_INTERVAL_ROWS", "-5")
        monkeypatch.setenv("MILVUS_MIGRATION_BATCH_SLEEP", "-1")
        storage = _make_model_storage()
        assert storage._migration_flush_interval_rows == 0
        assert storage._migration_batch_sleep == 0.0
