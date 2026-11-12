"""Persistent local Qdrant reproduction of interrupted legacy migration."""

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient, models  # noqa: E402
from lightrag.kg.qdrant_impl import QdrantVectorDBStorage  # noqa: E402
from lightrag.exceptions import DataMigrationError  # noqa: E402

pytestmark = pytest.mark.offline


def test_partial_migration_restart_must_not_accept_incomplete_target(
    tmp_path, monkeypatch
):
    """A new client must recover or refuse, never accept only the first batch."""
    path = str(tmp_path / "qdrant")
    source = "lightrag_vdb_chunks"
    target = "lightrag_vdb_chunks_restart_test_3d"
    vectors = models.VectorParams(size=3, distance=models.Distance.COSINE)
    setup = dict(
        collection_name=target,
        namespace="chunks",
        workspace="restart-test",
        vectors_config=vectors,
        hnsw_config=models.HnswConfigDiff(),
        model_suffix="restart_test_3d",
    )
    client = QdrantClient(path=path)
    try:
        client.create_collection(source, vectors_config=vectors)
        client.upsert(
            source,
            points=[
                models.PointStruct(
                    id=index,
                    vector=[1.0, 0.0, 0.0],
                    payload={"id": f"chunk-{index}", "content": f"record {index}"},
                )
                for index in range(750)
            ],
            wait=True,
        )
        upsert = client.upsert
        writes = []

        def fail_second_batch(*args, **kwargs):
            if kwargs.get("collection_name") == target:
                writes.append(len(kwargs["points"]))
                if len(writes) == 2:
                    raise OSError("injected second migration batch failure")
            return upsert(*args, **kwargs)

        monkeypatch.setattr(client, "upsert", fail_second_batch)
        with pytest.raises(DataMigrationError):
            QdrantVectorDBStorage.setup_collection(client=client, **setup)
        assert writes == [500, 250]
        assert client.count(source, exact=True).count == 750
        assert client.count(target, exact=True).count == 500
    finally:
        client.close()

    # Reopen the persisted DB: no in-memory migration state survives.
    restarted = QdrantClient(path=path)
    try:
        assert restarted.count(target, exact=True).count == 500
        try:
            QdrantVectorDBStorage.setup_collection(client=restarted, **setup)
        except DataMigrationError:
            # Explicit refusal is safe; a recovery implementation can instead
            # return after restoring completeness. Neither may drop the source.
            assert restarted.count(source, exact=True).count == 750
            return
        assert restarted.count(source, exact=True).count == 750
        assert restarted.count(target, exact=True).count == 750, (
            "Restart accepted an incomplete migrated collection as ready: "
            "only 500 of 750 source points were published"
        )
    finally:
        restarted.close()


@pytest.fixture
def persistent_migration(tmp_path):
    from lightrag.kg.qdrant_impl import MIGRATION_PENDING_FIELD

    path = str(tmp_path / "persistent-qdrant")
    source = "lightrag_vdb_chunks"
    target = "lightrag_vdb_chunks_restart_test_3d"
    vectors = models.VectorParams(size=3, distance=models.Distance.COSINE)
    setup = dict(
        collection_name=target,
        namespace="chunks",
        workspace="owner",
        vectors_config=vectors,
        hnsw_config=models.HnswConfigDiff(),
        model_suffix="restart_test_3d",
    )
    client = QdrantClient(path=path)
    client.create_collection(source, vectors_config=vectors)
    client.upsert(
        source,
        points=[
            models.PointStruct(
                id=i,
                vector=[1.0, 0.0, 0.0],
                payload={"id": f"chunk-{i}", "workspace_id": "owner"},
            )
            for i in range(750)
        ],
        wait=True,
    )
    client.close()
    return path, source, target, setup, MIGRATION_PENDING_FIELD


def test_successful_migration_reopens_complete_without_pending(persistent_migration):
    path, source, target, setup, marker = persistent_migration
    client = QdrantClient(path=path)
    try:
        QdrantVectorDBStorage.setup_collection(client=client, **setup)
    finally:
        client.close()
    client = QdrantClient(path=path)
    try:
        QdrantVectorDBStorage.setup_collection(client=client, **setup)
        assert client.count(target, exact=True).count == 750
        points, _ = client.scroll(target, limit=1000, with_payload=True)
        assert len(points) == 750
        assert all(marker not in point.payload for point in points)
        assert client.count(source, exact=True).count == 750
    finally:
        client.close()


def test_pending_target_refused_without_legacy_source(persistent_migration):
    path, source, target, setup, marker = persistent_migration
    client = QdrantClient(path=path)
    try:
        client.create_collection(target, vectors_config=setup["vectors_config"])
        client.upsert(
            target,
            points=[
                models.PointStruct(
                    id=1,
                    vector=[1.0, 0.0, 0.0],
                    payload={"workspace_id": "owner", marker: True},
                )
            ],
            wait=True,
        )
        client.delete_collection(source)
    finally:
        client.close()
    client = QdrantClient(path=path)
    try:
        with pytest.raises(DataMigrationError, match="unfinished legacy migration"):
            QdrantVectorDBStorage.setup_collection(client=client, **setup)
        assert client.count(target, exact=True).count == 1
    finally:
        client.close()


def test_other_workspace_marker_does_not_block_or_get_cleared(persistent_migration):
    path, source, target, setup, marker = persistent_migration
    client = QdrantClient(path=path)
    try:
        client.create_collection(target, vectors_config=setup["vectors_config"])
        client.upsert(
            target,
            points=[
                models.PointStruct(
                    id=98765,
                    vector=[0.0, 1.0, 0.0],
                    payload={"workspace_id": "other", marker: True},
                )
            ],
            wait=True,
        )
        QdrantVectorDBStorage.setup_collection(client=client, **setup)
        assert client.count(target, exact=True).count == 751
        assert client.retrieve(target, ids=[98765])[0].payload[marker] is True
        with pytest.raises(DataMigrationError, match="unfinished legacy migration"):
            QdrantVectorDBStorage.setup_collection(
                client=client, **dict(setup, workspace="other")
            )
    finally:
        client.close()


@pytest.mark.parametrize("mode", ["before", "partial", "response_lost"])
def test_marker_cleanup_failure_is_durable_and_safe(
    persistent_migration, monkeypatch, mode
):
    path, source, target, setup, marker = persistent_migration
    client = QdrantClient(path=path)
    try:
        delete = client.delete_payload

        def fail_cleanup(*args, **kwargs):
            if mode == "partial":
                points, _ = client.scroll(target, limit=100, with_payload=True)
                delete(
                    collection_name=target,
                    keys=[marker],
                    points=[point.id for point in points],
                    wait=True,
                )
            elif mode == "response_lost":
                delete(*args, **kwargs)
            raise OSError("injected cleanup response failure")

        monkeypatch.setattr(client, "delete_payload", fail_cleanup)
        with pytest.raises(DataMigrationError):
            QdrantVectorDBStorage.setup_collection(client=client, **setup)
        assert client.count(target, exact=True).count == 750
    finally:
        client.close()
    client = QdrantClient(path=path)
    try:
        if mode == "response_lost":
            QdrantVectorDBStorage.setup_collection(client=client, **setup)
        else:
            with pytest.raises(DataMigrationError, match="unfinished legacy migration"):
                QdrantVectorDBStorage.setup_collection(client=client, **setup)
        points, _ = client.scroll(target, limit=1000, with_payload=True)
        marked = sum(point.payload.get(marker) is True for point in points)
        assert marked == {"before": 750, "partial": 650, "response_lost": 0}[mode]
        assert client.count(source, exact=True).count == 750
    finally:
        client.close()


def test_pending_marker_count_outage_does_not_admit_existing_target(
    persistent_migration, monkeypatch
):
    path, source, target, setup, marker = persistent_migration
    client = QdrantClient(path=path)
    try:
        QdrantVectorDBStorage.setup_collection(client=client, **setup)
        count = client.count

        def fail_pending_count(*args, **kwargs):
            condition = kwargs.get("count_filter")
            if condition and marker in condition.model_dump_json():
                raise OSError("pending marker count unavailable")
            return count(*args, **kwargs)

        monkeypatch.setattr(client, "count", fail_pending_count)
        with pytest.raises(OSError, match="pending marker count unavailable"):
            QdrantVectorDBStorage.setup_collection(client=client, **setup)
        assert count(target, exact=True).count == 750
    finally:
        client.close()


def test_operator_scoped_recovery_preserves_legacy_and_peer(
    persistent_migration, monkeypatch
):
    """Offline maintenance only: no writers; never call adapter.drop()."""
    path, source, target, setup, marker = persistent_migration
    owner_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="workspace_id", match=models.MatchValue(value="owner")
            )
        ]
    )
    client = QdrantClient(path=path)
    try:
        client.create_collection(target, vectors_config=setup["vectors_config"])
        client.upsert(
            target,
            points=[
                models.PointStruct(
                    id=98765,
                    vector=[0.0, 1.0, 0.0],
                    payload={"workspace_id": "peer", "content": "must survive"},
                )
            ],
            wait=True,
        )
        upsert = client.upsert
        writes = 0

        def fail_second(*args, **kwargs):
            nonlocal writes
            if kwargs.get("collection_name") == target:
                writes += 1
                if writes == 2:
                    raise OSError("second batch failure")
            return upsert(*args, **kwargs)

        monkeypatch.setattr(client, "upsert", fail_second)
        with pytest.raises(DataMigrationError):
            QdrantVectorDBStorage.setup_collection(client=client, **setup)
    finally:
        client.close()
    client = QdrantClient(path=path)
    try:
        with pytest.raises(DataMigrationError, match="unfinished legacy migration"):
            QdrantVectorDBStorage.setup_collection(client=client, **setup)
        rows, offset = client.scroll(
            target, scroll_filter=owner_filter, limit=1000, with_payload=True
        )
        assert offset is None
        assert len(rows) == 500
        assert all(row.payload.get(marker) is True for row in rows)
        # This proof is required before manual deletion. A mixed marked /
        # unmarked target needs investigation, not blanket cleanup.
        client.delete(
            target,
            points_selector=models.FilterSelector(filter=owner_filter),
            wait=True,
        )
        assert client.count(source, exact=True).count == 750
        assert client.count(target, exact=True).count == 1
        QdrantVectorDBStorage.setup_collection(client=client, **setup)
        assert client.count(target, count_filter=owner_filter, exact=True).count == 750
        assert client.count(source, exact=True).count == 750
        peer = client.retrieve(target, ids=[98765])[0]
        assert peer.payload == {"workspace_id": "peer", "content": "must survive"}
    finally:
        client.close()


@pytest.mark.parametrize("existing_target", [False, True])
def test_pending_counts_require_completed_boolean_index(
    persistent_migration, monkeypatch, existing_target
):
    """Strict-mode spy checks ordering; local engine cannot enforce indexes."""
    path, source, target, setup, marker = persistent_migration
    client = QdrantClient(path=path)
    try:
        if existing_target:
            client.create_collection(target, vectors_config=setup["vectors_config"])
        indexed = set()
        pending_reads = []
        original_index = client.create_payload_index
        original_count = client.count

        def create_index(*args, **kwargs):
            result = original_index(*args, **kwargs)
            if kwargs.get("field_name") == marker:
                assert kwargs["field_schema"] == models.PayloadSchemaType.BOOL
                assert kwargs.get("wait") is True
                indexed.add(kwargs["collection_name"])
            return result

        def strict_count(*args, **kwargs):
            condition = kwargs.get("count_filter")
            if condition and marker in condition.model_dump_json():
                collection = kwargs["collection_name"]
                if collection not in indexed:
                    raise RuntimeError("Strict mode: unindexed pending filter")
                pending_reads.append(collection)
            return original_count(*args, **kwargs)

        monkeypatch.setattr(client, "create_payload_index", create_index)
        monkeypatch.setattr(client, "count", strict_count)
        QdrantVectorDBStorage.setup_collection(client=client, **setup)
        # New target checks after publication; existing target additionally
        # checks before legacy discovery. Both must have a completed BOOL index.
        assert pending_reads == [target] * (2 if existing_target else 1)
        assert original_count(target, exact=True).count == 750
    finally:
        client.close()
