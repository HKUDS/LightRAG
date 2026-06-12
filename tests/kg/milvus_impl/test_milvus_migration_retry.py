"""
Tests for Milvus schema-migration resilience.

This suite validates the failure-handling hardening added after a production
outage where a transient Milvus connection failure mid-migration permanently
killed worker startup:

1. Connection-class failures retry the whole migration with a rebuilt client.
2. Non-connection failures keep failing fast (single attempt).
3. _is_retryable_connection_error classifies errors through cause chains.
4. The force-create fallback never fires on a connection error.
5. The temp collection is not loaded during the bulk copy.
6. Backup collections are released from memory after a successful migration.
7. A stale _old backup is dropped so the in-place rename can succeed.
8. An orphaned temp collection (crash between drop-source and rename-temp)
   is recovered instead of being shadowed by a fresh empty collection.
"""

import grpc
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


def _rows():
    return [{"id": "ent-1", "vector": [0.0] * 128, "content": "body"}]


def _wire_fresh_iterator_per_attempt(client, rows=None):
    """query_iterator returns a new exhausted-after-one-batch iterator per call."""

    def make_iterator(**kwargs):
        iterator = MagicMock()
        iterator.next.side_effect = [rows if rows is not None else _rows(), []]
        return iterator

    client.query_iterator.side_effect = make_iterator


class _FakeRpcError(grpc.RpcError):
    def __init__(self, status_code):
        self._status_code = status_code

    def code(self):
        return self._status_code


_CONNECT_FAILED = MilvusException(
    code=2, message="Fail connecting to server on host:19530"
)


@pytest.mark.offline
class TestMigrationRetry:
    def test_retryable_error_retries_with_rebuilt_client(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)

        insert_calls = {"n": 0}

        def insert(collection_name, data):
            insert_calls["n"] += 1
            if insert_calls["n"] == 1:
                raise MilvusException(
                    code=2, message="Fail connecting to server on host:19530"
                )

        client.insert.side_effect = insert

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client") as rebuild:
                with patch("lightrag.kg.milvus_impl.time.sleep") as sleep:
                    storage._migrate_collection_schema(
                        source_collection_name=storage.legacy_namespace,
                        target_collection_name=storage.final_namespace,
                    )

        rebuild.assert_called_once()
        sleep.assert_called_once()
        assert sleep.call_args.args[0] == pytest.approx(5.0)
        assert client.query_iterator.call_count == 2
        assert storage.final_namespace in collections
        assert f"{storage.final_namespace}_temp" not in collections
        # The legacy source stays in place as the backup.
        assert storage.legacy_namespace in collections

    def test_non_retryable_error_fails_fast(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        client.insert.side_effect = RuntimeError("insert failed")

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client") as rebuild:
                with patch("lightrag.kg.milvus_impl.time.sleep") as sleep:
                    with pytest.raises(
                        RuntimeError, match="Iterator-based migration failed"
                    ):
                        storage._migrate_collection_schema(
                            source_collection_name=storage.legacy_namespace,
                            target_collection_name=storage.final_namespace,
                        )

        rebuild.assert_not_called()
        sleep.assert_not_called()
        assert client.query_iterator.call_count == 1
        assert storage.legacy_namespace in collections
        assert storage.final_namespace not in collections

    def test_retries_exhausted_raises_with_backoff_sequence(self):
        storage = _make_model_storage()
        storage._migration_max_retries = 2
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        client.insert.side_effect = ValueError("Cannot invoke RPC on closed channel!")

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client"):
                with patch("lightrag.kg.milvus_impl.time.sleep") as sleep:
                    with pytest.raises(
                        RuntimeError, match="Iterator-based migration failed"
                    ):
                        storage._migrate_collection_schema(
                            source_collection_name=storage.legacy_namespace,
                            target_collection_name=storage.final_namespace,
                        )

        assert client.query_iterator.call_count == 3
        assert [call.args[0] for call in sleep.call_args_list] == [
            pytest.approx(5.0),
            pytest.approx(15.0),
        ]
        assert storage.legacy_namespace in collections

    def test_max_backoff_is_capped(self):
        storage = _make_model_storage()
        storage._migration_max_retries = 4
        storage._migration_retry_max_backoff = 20.0
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        client.insert.side_effect = _CONNECT_FAILED

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client"):
                with patch("lightrag.kg.milvus_impl.time.sleep") as sleep:
                    with pytest.raises(RuntimeError):
                        storage._migrate_collection_schema(
                            source_collection_name=storage.legacy_namespace,
                            target_collection_name=storage.final_namespace,
                        )

        assert [call.args[0] for call in sleep.call_args_list] == [
            pytest.approx(5.0),
            pytest.approx(15.0),
            pytest.approx(20.0),
            pytest.approx(20.0),
        ]


@pytest.mark.offline
class TestRetryableErrorClassification:
    @pytest.mark.parametrize(
        "error",
        [
            _FakeRpcError(grpc.StatusCode.UNAVAILABLE),
            _FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED),
            MilvusException(code=2, message="Fail connecting to server on host:19530"),
            MilvusException(code=1, message="server unavailable: ping timeout"),
            ValueError("Cannot invoke RPC on closed channel!"),
        ],
        ids=[
            "grpc-unavailable",
            "grpc-deadline-exceeded",
            "milvus-connect-failed",
            "milvus-connection-message",
            "closed-channel-valueerror",
        ],
    )
    def test_retryable_errors(self, error):
        assert MilvusVectorDBStorage._is_retryable_connection_error(error) is True

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("primary keys cannot be truncated"),
            MilvusException(code=1100, message="schema mismatch"),
            RuntimeError("Target collection already exists: foo"),
            KeyError("vector"),
        ],
        ids=[
            "value-error",
            "milvus-schema-error",
            "runtime-error",
            "key-error",
        ],
    )
    def test_non_retryable_errors(self, error):
        assert MilvusVectorDBStorage._is_retryable_connection_error(error) is False

    def test_cause_chain_is_walked(self):
        inner = ValueError("Cannot invoke RPC on closed channel!")
        outer = RuntimeError("Iterator-based migration failed for collection foo")
        outer.__cause__ = inner
        assert MilvusVectorDBStorage._is_retryable_connection_error(outer) is True

    def test_context_chain_is_walked(self):
        inner = MilvusException(code=2, message="Fail connecting to server")
        outer = RuntimeError("wrapper")
        outer.__context__ = inner
        assert MilvusVectorDBStorage._is_retryable_connection_error(outer) is True

    def test_self_referencing_chain_terminates(self):
        error = RuntimeError("loop")
        error.__cause__ = error
        assert MilvusVectorDBStorage._is_retryable_connection_error(error) is False


@pytest.mark.offline
class TestRebuildMilvusClient:
    def test_rebuild_replaces_client_and_closes_old(self):
        storage = _make_model_storage()
        old_client = MagicMock()
        new_client = MagicMock()
        storage._client = old_client

        with patch.object(
            storage, "_create_milvus_client", return_value=new_client
        ) as create:
            storage._rebuild_milvus_client()

        old_client.close.assert_called_once()
        create.assert_called_once()
        assert storage._client is new_client

    def test_rebuild_tolerates_close_failure_on_dead_client(self):
        storage = _make_model_storage()
        old_client = MagicMock()
        old_client.close.side_effect = ValueError(
            "Cannot invoke RPC on closed channel!"
        )
        new_client = MagicMock()
        storage._client = old_client

        with patch.object(storage, "_create_milvus_client", return_value=new_client):
            storage._rebuild_milvus_client()

        assert storage._client is new_client


@pytest.mark.offline
class TestForceCreateGuard:
    def test_connection_error_propagates_without_force_create(self):
        storage = _make_model_storage()
        storage._client = MagicMock()
        storage._client.has_collection.side_effect = MilvusException(
            code=2, message="Fail connecting to server on host:19530"
        )

        with pytest.raises(MilvusException):
            storage._create_collection_if_not_exist()

        storage._client.create_collection.assert_not_called()
        storage._client.drop_collection.assert_not_called()

    def test_non_connection_error_still_force_creates(self):
        storage = _make_model_storage()
        storage._client = MagicMock()
        storage._client.has_collection.side_effect = KeyError("boom")

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_ensure_collection_loaded"):
                storage._create_collection_if_not_exist()

        storage._client.create_collection.assert_called_once()


@pytest.mark.offline
class TestMigrationMemoryFootprint:
    def test_temp_collection_is_not_loaded_during_migration(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        client.load_collection.assert_not_called()
        assert storage.final_namespace in collections

    def test_suffix_migration_releases_legacy_backup(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        client.release_collection.assert_called_once_with(storage.legacy_namespace)
        assert storage.legacy_namespace in collections

    def test_in_place_migration_releases_old_backup(self):
        storage = _make_model_storage()
        collections = {storage.final_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema()

        backup_name = f"{storage.final_namespace}_old"
        client.release_collection.assert_called_once_with(backup_name)
        assert backup_name in collections
        assert storage.final_namespace in collections

    def test_release_failure_does_not_fail_migration(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        client.release_collection.side_effect = MilvusException(
            code=1, message="release failed"
        )

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        assert storage.final_namespace in collections


@pytest.mark.offline
class TestInPlaceMigrationSafety:
    def test_stale_old_backup_is_dropped_before_rename(self):
        storage = _make_model_storage()
        backup_name = f"{storage.final_namespace}_old"
        collections = {storage.final_namespace, backup_name}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema()

        # The stale backup was dropped so the rename could succeed; the source
        # now lives on as the fresh backup instead of being dropped outright.
        assert backup_name in collections
        assert storage.final_namespace in collections
        drop_calls = [call.args[0] for call in client.drop_collection.call_args_list]
        assert backup_name in drop_calls

    def test_orphaned_temp_collection_is_recovered_on_startup(self):
        storage = _make_model_storage()
        temp_name = f"{storage.final_namespace}_temp"
        collections = {temp_name}
        client = _wire_collection_state(storage, collections)

        with patch.object(storage, "_ensure_collection_loaded"):
            storage._create_collection_if_not_exist()

        assert storage.final_namespace in collections
        assert temp_name not in collections
        client.create_collection.assert_not_called()

    def test_startup_restores_old_backup_when_no_temp_survives(self):
        # Target gone, no temp, only the _old backup left: the source was
        # vacated but the migrated copy did not survive. Restore _old rather
        # than creating an empty collection over the last copy.
        storage = _make_model_storage()
        old_name = f"{storage.final_namespace}_old"
        collections = {old_name}
        client = _wire_collection_state(storage, collections)

        with patch.object(storage, "_ensure_collection_loaded"):
            storage._create_collection_if_not_exist()

        assert storage.final_namespace in collections
        assert old_name not in collections
        client.create_collection.assert_not_called()

    def test_inplace_recovery_precedes_legacy_migration(self):
        # A suffixed target was migrated in-place and interrupted after Step 3:
        # final renamed to _old, the completed copy sits in _temp, and the old
        # unsuffixed legacy backup still exists. Recovery MUST win over the
        # legacy migration, which would otherwise drop _temp and overwrite the
        # target with stale legacy data (losing every write since the split).
        storage = _make_model_storage()
        final = storage.final_namespace
        legacy = storage.legacy_namespace
        temp = f"{final}_temp"
        old = f"{final}_old"
        collections = {legacy, old, temp}
        client = _wire_collection_state(storage, collections)

        with patch.object(storage, "_ensure_collection_loaded"):
            with patch.object(storage, "_migrate_collection_schema") as migrate:
                storage._create_collection_if_not_exist()

        migrate.assert_not_called()
        assert final in collections  # _temp promoted to the target
        assert temp not in collections
        assert legacy in collections  # untouched
        client.create_collection.assert_not_called()
        drops = [call.args[0] for call in client.drop_collection.call_args_list]
        assert temp not in drops

    def test_partial_suffix_temp_with_legacy_is_remigrated_not_promoted(self):
        # An aborted suffix copy (legacy -> final) left a PARTIAL _temp while
        # legacy is intact and there is no _old. The partial temp must be
        # treated as scratch and re-migrated from legacy, never promoted.
        storage = _make_model_storage()
        final = storage.final_namespace
        legacy = storage.legacy_namespace
        temp = f"{final}_temp"
        collections = {legacy, temp}
        client = _wire_collection_state(storage, collections)

        with patch.object(storage, "_has_vector_field", return_value=True):
            with patch.object(storage, "_check_vector_dimension"):
                with patch.object(storage, "_migrate_collection_schema") as migrate:
                    with patch.object(storage, "_ensure_collection_loaded"):
                        storage._create_collection_if_not_exist()

        migrate.assert_called_once_with(
            source_collection_name=legacy,
            target_collection_name=final,
        )
        # Recovery (which promotes/restores via rename) must not have run.
        client.rename_collection.assert_not_called()


def _fail_specific_rename_once(client, collections, fail_source, fail_target, error):
    """Wrap the wired rename so one specific rename raises `error` on first call."""
    state = {"n": 0}

    def rename(source, target):
        if source == fail_source and target == fail_target:
            state["n"] += 1
            if state["n"] == 1:
                raise error
        collections.discard(source)
        collections.add(target)

    client.rename_collection.side_effect = rename


@pytest.mark.offline
class TestInPlaceCommitWindowRecovery:
    """The in-place commit window (source vacated, temp not yet promoted) must
    treat temp/_old as recoverable state, never as scratch to be dropped."""

    def test_step4_connection_failure_retry_promotes_temp(self):
        storage = _make_model_storage()
        final = storage.final_namespace
        temp = f"{final}_temp"
        old = f"{final}_old"
        collections = {final}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        # Step 4 (temp -> final) fails once with a connection error; the retry
        # must promote the surviving temp copy, not drop and re-copy.
        _fail_specific_rename_once(
            client,
            collections,
            temp,
            final,
            MilvusException(code=2, message="Fail connecting to server on host:19530"),
        )

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client") as rebuild:
                with patch("lightrag.kg.milvus_impl.time.sleep") as sleep:
                    storage._migrate_collection_schema()

        rebuild.assert_called_once()
        sleep.assert_called_once()
        assert final in collections
        assert temp not in collections
        assert old in collections  # backup preserved
        assert storage.final_namespace == final
        drops = [call.args[0] for call in client.drop_collection.call_args_list]
        assert temp not in drops  # never treated as scratch in the commit window

    def test_drop_source_fallback_then_step4_failure_recovers_temp(self):
        # rename(source -> _old) fails for a non-connection reason, so the
        # drop-source fallback runs (NO _old backup); Step 4 then fails with a
        # connection error. Without recovery this is total data loss; the retry
        # must promote the only surviving copy (temp).
        storage = _make_model_storage()
        final = storage.final_namespace
        temp = f"{final}_temp"
        old = f"{final}_old"
        collections = {final}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        state = {"temp_to_final": 0}

        def rename(source, target):
            if source == final and target == old:
                raise RuntimeError("rename to _old unsupported")  # forces drop-source
            if source == temp and target == final:
                state["temp_to_final"] += 1
                if state["temp_to_final"] == 1:
                    raise MilvusException(
                        code=2, message="Fail connecting to server on host:19530"
                    )
            collections.discard(source)
            collections.add(target)

        client.rename_collection.side_effect = rename

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client") as rebuild:
                with patch("lightrag.kg.milvus_impl.time.sleep"):
                    storage._migrate_collection_schema()

        rebuild.assert_called_once()
        assert final in collections
        assert temp not in collections
        assert old not in collections
        assert storage.final_namespace == final
        drops = [call.args[0] for call in client.drop_collection.call_args_list]
        assert temp not in drops

    def test_non_retryable_commit_failure_keeps_temp_for_startup_recovery(self):
        # A non-retryable Step 4 failure must still preserve temp (the source is
        # already vacated) so startup recovery can finish the commit.
        storage = _make_model_storage()
        final = storage.final_namespace
        temp = f"{final}_temp"
        old = f"{final}_old"
        collections = {final}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)

        def rename(source, target):
            if source == temp and target == final:
                raise RuntimeError("non-retryable rename failure")
            collections.discard(source)
            collections.add(target)

        client.rename_collection.side_effect = rename

        with patch.object(storage, "_create_indexes_after_collection"):
            with pytest.raises(RuntimeError, match="Iterator-based migration failed"):
                storage._migrate_collection_schema()

        assert temp in collections  # preserved as recovery state
        assert old in collections
        drops = [call.args[0] for call in client.drop_collection.call_args_list]
        assert temp not in drops
