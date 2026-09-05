"""Regression tests for workspace/index collisions on the OpenSearch backend.

``_sanitize_index_name`` lowercases the workspace and folds every character
outside ``[a-z0-9_-]`` to ``_``, so distinct workspaces (``TeamA`` / ``teama``,
``v1.0`` / ``v1_0``) resolve to the same physical index. Sharing an index means
sharing documents, overwriting each other's rows and -- because ``drop()``
deletes the whole index on this backend -- destroying each other's data.

Each storage class must therefore record its *original* workspace namespace in
the index mapping's ``_meta`` and refuse to attach to an index claimed by a
different one. See issue #3827.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from opensearchpy.exceptions import OpenSearchException  # type: ignore  # noqa: E402

from lightrag.kg.opensearch_impl import (  # noqa: E402
    ClientManager,
    OpenSearchDocStatusStorage,
    OpenSearchGraphStorage,
    OpenSearchKVStorage,
    OpenSearchVectorDBStorage,
    WorkspaceIndexCollisionError,
    _EDGE_ID_CANONICAL_META_FLAG,
    _FINAL_NAMESPACE_META_KEY,
    _WORKSPACE_META_KEY,
    _claim_index_for_workspace,
    _sanitize_index_name,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _mock_lock():
    yield


def _mock_lock_factory(*args, **kwargs):
    return _mock_lock()


@pytest.fixture(autouse=True)
def patch_locks():
    """Run initialize() without the shared-storage machinery."""
    cache: dict[tuple, asyncio.Lock] = {}

    def namespace_lock(namespace, workspace=None, enable_logging=False):
        return cache.setdefault((namespace, workspace), asyncio.Lock())

    with (
        patch(
            "lightrag.kg.opensearch_impl.get_data_init_lock",
            side_effect=_mock_lock_factory,
        ),
        patch(
            "lightrag.kg.opensearch_impl.get_namespace_lock",
            side_effect=namespace_lock,
        ),
        patch("lightrag.kg.opensearch_impl._shard_doc_supported", True),
    ):
        yield


class FakeCluster:
    """Minimal stateful stand-in for one shared OpenSearch cluster.

    Index mappings are kept in a dict, so two storage instances built against
    the same ``FakeCluster`` really do collide on the same index name -- which
    is the whole point of these tests. A per-call ``AsyncMock`` cannot express
    that: it has no memory of what the previous instance created.
    """

    def __init__(self):
        self.mappings: dict[str, dict] = {}

    # -- indices sub-client -------------------------------------------------
    async def exists(self, index):
        return index in self.mappings

    async def create(self, index, body=None):
        # Mirrors OpenSearch: creating an existing index is an error. Tests
        # that need the lost-race path drive it explicitly.
        assert index not in self.mappings, f"index {index} already exists"
        self.mappings[index] = dict((body or {}).get("mappings", {}))

    async def get_mapping(self, index):
        if index not in self.mappings:
            return {}
        return {index: {"mappings": self.mappings[index]}}

    async def put_mapping(self, index, body):
        mappings = self.mappings[index]
        if "_meta" in body:
            mappings["_meta"] = body["_meta"]
        if "properties" in body:
            mappings.setdefault("properties", {}).update(body["properties"])

    async def delete(self, index, **kwargs):
        self.mappings.pop(index, None)

    async def refresh(self, index=None, **kwargs):
        return None

    # -- client -------------------------------------------------------------
    def client(self):
        from opensearchpy import AsyncOpenSearch

        client = AsyncMock(spec=AsyncOpenSearch)
        client.indices = AsyncMock()
        client.indices.exists = AsyncMock(side_effect=self.exists)
        client.indices.create = AsyncMock(side_effect=self.create)
        client.indices.get_mapping = AsyncMock(side_effect=self.get_mapping)
        client.indices.put_mapping = AsyncMock(side_effect=self.put_mapping)
        client.indices.delete = AsyncMock(side_effect=self.delete)
        client.indices.refresh = AsyncMock(side_effect=self.refresh)
        client.transport = AsyncMock()
        client.transport.perform_request = AsyncMock(
            side_effect=Exception("PPL not available")
        )
        # Empty index: no scheduling backfill, no edge migration work.
        client.count = AsyncMock(return_value={"count": 0})
        client.search = AsyncMock(
            return_value={"hits": {"hits": [], "total": {"value": 0}}}
        )
        return client


class _Embed:
    embedding_dim = 8
    max_token_size = 100

    async def __call__(self, texts):
        return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)


@pytest.fixture
def cluster():
    return FakeCluster()


@pytest.fixture
def global_config():
    return {
        "embedding_batch_num": 10,
        "max_graph_nodes": 1000,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


STORAGE_CASES = [
    pytest.param(OpenSearchKVStorage, "text_chunks", id="kv"),
    pytest.param(OpenSearchDocStatusStorage, "doc_status", id="doc_status"),
    pytest.param(OpenSearchGraphStorage, "chunk_entity_relation", id="graph"),
    pytest.param(OpenSearchVectorDBStorage, "entities", id="vector"),
]


async def _initialize(cls, namespace, workspace, global_config, cluster):
    storage = cls(
        namespace=namespace,
        global_config=global_config,
        embedding_func=_Embed(),
        workspace=workspace,
    )
    with patch.object(ClientManager, "get_client", return_value=cluster.client()):
        await storage.initialize()
    return storage


# ---------------------------------------------------------------------------
# The collision itself
# ---------------------------------------------------------------------------


class TestCaseVariantWorkspacesCollide:
    def test_index_names_really_collide(self):
        """The precondition: the name mapping is not injective."""
        assert _sanitize_index_name("TeamA_text_chunks") == "teama_text_chunks"
        assert _sanitize_index_name("teama_text_chunks") == "teama_text_chunks"
        assert _sanitize_index_name("v1.0_text_chunks") == "v1_0_text_chunks"
        assert _sanitize_index_name("v1_0_text_chunks") == "v1_0_text_chunks"

    @pytest.mark.parametrize("cls,namespace", STORAGE_CASES)
    @pytest.mark.asyncio
    async def test_second_workspace_is_refused(
        self, cls, namespace, global_config, cluster
    ):
        """``teama`` must not attach to the index ``TeamA`` created."""
        await _initialize(cls, namespace, "TeamA", global_config, cluster)

        with pytest.raises(WorkspaceIndexCollisionError) as excinfo:
            await _initialize(cls, namespace, "teama", global_config, cluster)

        message = str(excinfo.value)
        assert f"TeamA_{namespace}" in message
        assert f"teama_{namespace}" in message
        # The way out is renaming a workspace. Dropping the index would
        # destroy the other workspace's data, so every mention of dropping
        # must stay negated.
        assert "Rename one of the" in message
        assert message.count("drop") == message.count("Do NOT drop")

    @pytest.mark.parametrize("cls,namespace", STORAGE_CASES)
    @pytest.mark.asyncio
    async def test_punctuation_variant_is_refused(
        self, cls, namespace, global_config, cluster
    ):
        """Folding of non-alphanumeric characters collides the same way."""
        await _initialize(cls, namespace, "v1.0", global_config, cluster)

        with pytest.raises(WorkspaceIndexCollisionError):
            await _initialize(cls, namespace, "v1_0", global_config, cluster)

    @pytest.mark.parametrize("cls,namespace", STORAGE_CASES)
    @pytest.mark.asyncio
    async def test_same_workspace_reattaches(
        self, cls, namespace, global_config, cluster
    ):
        """The ordinary multi-worker case must stay silent."""
        await _initialize(cls, namespace, "TeamA", global_config, cluster)
        await _initialize(cls, namespace, "TeamA", global_config, cluster)
        await _initialize(cls, namespace, "TeamA", global_config, cluster)

    @pytest.mark.parametrize("cls,namespace", STORAGE_CASES)
    @pytest.mark.asyncio
    async def test_distinct_workspaces_are_untouched(
        self, cls, namespace, global_config, cluster
    ):
        """Workspaces that do not fold together keep working independently."""
        await _initialize(cls, namespace, "TeamA", global_config, cluster)
        await _initialize(cls, namespace, "TeamB", global_config, cluster)


class TestAmbiguousJoinedNamespace:
    """``{workspace}_{namespace}`` is itself a lossy join.

    Workspace ``foo`` with namespace ``text_chunks`` and workspace ``foo_text``
    with namespace ``chunks`` -- both real LightRAG namespaces -- produce the
    same joined name and therefore the same index. Comparing only the joined
    name would wave the second deployment through.
    """

    @pytest.mark.asyncio
    async def test_joined_names_really_collide(self, global_config, cluster):
        kv = OpenSearchKVStorage(
            namespace="text_chunks",
            global_config=global_config,
            embedding_func=_Embed(),
            workspace="foo",
        )
        vector = OpenSearchVectorDBStorage(
            namespace="chunks",
            global_config=global_config,
            embedding_func=_Embed(),
            workspace="foo_text",
        )
        assert kv._index_name == vector._index_name == "foo_text_chunks"
        assert kv.final_namespace == vector.final_namespace == "foo_text_chunks"

    @pytest.mark.asyncio
    async def test_different_workspace_same_joined_name_is_refused(
        self, global_config, cluster
    ):
        await _initialize(
            OpenSearchKVStorage, "text_chunks", "foo", global_config, cluster
        )
        with pytest.raises(WorkspaceIndexCollisionError) as excinfo:
            await _initialize(
                OpenSearchVectorDBStorage, "chunks", "foo_text", global_config, cluster
            )
        message = str(excinfo.value)
        assert "'foo'" in message
        assert "'foo_text'" in message

    @pytest.mark.asyncio
    async def test_partial_marker_is_not_adopted(self, cluster):
        """An identity we cannot fully match is not ours to overwrite."""
        cluster.mappings["teama_text_chunks"] = {
            "_meta": {_WORKSPACE_META_KEY: "TeamA"},
            "properties": {},
        }
        client = cluster.client()
        with pytest.raises(WorkspaceIndexCollisionError):
            await _claim_index_for_workspace(
                client, "teama_text_chunks", "TeamA", "TeamA_text_chunks"
            )
        client.indices.put_mapping.assert_not_awaited()


class TestMarkerIsRecorded:
    @pytest.mark.asyncio
    async def test_every_created_index_carries_the_marker(self, global_config, cluster):
        await _initialize(
            OpenSearchGraphStorage,
            "chunk_entity_relation",
            "TeamA",
            global_config,
            cluster,
        )
        # The graph storage owns two indices; both must be marked, or the
        # unmarked one silently stays shareable.
        assert set(cluster.mappings) == {
            "teama_chunk_entity_relation-nodes",
            "teama_chunk_entity_relation-edges",
        }
        for mappings in cluster.mappings.values():
            meta = mappings["_meta"]
            assert meta[_WORKSPACE_META_KEY] == "TeamA"
            assert meta[_FINAL_NAMESPACE_META_KEY] == "TeamA_chunk_entity_relation"
        # The edge-migration flag also lives in ``_meta`` and is written after
        # the marker; neither write may clobber the other.
        edges_meta = cluster.mappings["teama_chunk_entity_relation-edges"]["_meta"]
        assert edges_meta[_EDGE_ID_CANONICAL_META_FLAG] is True

    @pytest.mark.asyncio
    async def test_drop_and_recreate_remarks_the_index(self, global_config, cluster):
        storage = await _initialize(
            OpenSearchKVStorage, "text_chunks", "TeamA", global_config, cluster
        )
        with patch.object(ClientManager, "get_client", return_value=cluster.client()):
            await storage.drop()
            await storage._ensure_index_ready()
        assert cluster.mappings["teama_text_chunks"]["_meta"] == {
            _WORKSPACE_META_KEY: "TeamA",
            _FINAL_NAMESPACE_META_KEY: "TeamA_text_chunks",
        }


class TestLegacyIndexAdoption:
    """Indices created before the marker existed carry no ``_meta``."""

    @pytest.mark.asyncio
    async def test_unmarked_index_is_adopted(self, global_config, cluster):
        cluster.mappings["teama_text_chunks"] = {
            "properties": {"__mirrored_id": {"type": "keyword"}}
        }
        await _initialize(
            OpenSearchKVStorage, "text_chunks", "TeamA", global_config, cluster
        )
        assert cluster.mappings["teama_text_chunks"]["_meta"] == {
            _WORKSPACE_META_KEY: "TeamA",
            _FINAL_NAMESPACE_META_KEY: "TeamA_text_chunks",
        }

    @pytest.mark.asyncio
    async def test_adoption_is_first_come_first_served(self, global_config, cluster):
        cluster.mappings["teama_text_chunks"] = {"properties": {}}
        await _initialize(
            OpenSearchKVStorage, "text_chunks", "TeamA", global_config, cluster
        )
        with pytest.raises(WorkspaceIndexCollisionError):
            await _initialize(
                OpenSearchKVStorage, "text_chunks", "teama", global_config, cluster
            )

    @pytest.mark.asyncio
    async def test_adoption_preserves_existing_meta(self, cluster):
        """``put_mapping`` replaces ``_meta`` wholesale, so it must be merged.

        Dropping ``_EDGE_ID_CANONICAL_META_FLAG`` here would silently re-run
        the one-time canonical edge-id reindex on every later startup.
        """
        cluster.mappings["teama_graph-edges"] = {
            "_meta": {_EDGE_ID_CANONICAL_META_FLAG: True},
            "properties": {},
        }
        client = cluster.client()
        await _claim_index_for_workspace(
            client, "teama_graph-edges", "TeamA", "TeamA_graph"
        )
        assert cluster.mappings["teama_graph-edges"]["_meta"] == {
            _EDGE_ID_CANONICAL_META_FLAG: True,
            _WORKSPACE_META_KEY: "TeamA",
            _FINAL_NAMESPACE_META_KEY: "TeamA_graph",
        }


class TestConcurrentAdoption:
    @pytest.mark.asyncio
    async def test_same_workspace_racing_adoption_is_idempotent(self, cluster):
        """Two workers of one deployment writing the same marker must not raise."""
        cluster.mappings["teama_text_chunks"] = {"properties": {}}
        client = cluster.client()
        await asyncio.gather(
            *[
                _claim_index_for_workspace(
                    client, "teama_text_chunks", "TeamA", "TeamA_text_chunks"
                )
                for _ in range(4)
            ]
        )
        assert cluster.mappings["teama_text_chunks"]["_meta"] == {
            _WORKSPACE_META_KEY: "TeamA",
            _FINAL_NAMESPACE_META_KEY: "TeamA_text_chunks",
        }

    @pytest.mark.asyncio
    async def test_losing_an_adoption_race_is_detected(self, cluster):
        """A different workspace claiming between our write and re-read raises."""
        cluster.mappings["teama_text_chunks"] = {"properties": {}}
        client = cluster.client()
        original_put = cluster.put_mapping

        async def steal_after_write(index, body):
            await original_put(index, body)
            cluster.mappings[index]["_meta"] = {
                _WORKSPACE_META_KEY: "teama",
                _FINAL_NAMESPACE_META_KEY: "teama_text_chunks",
            }

        client.indices.put_mapping = AsyncMock(side_effect=steal_after_write)
        with pytest.raises(WorkspaceIndexCollisionError):
            await _claim_index_for_workspace(
                client, "teama_text_chunks", "TeamA", "TeamA_text_chunks"
            )

    @pytest.mark.asyncio
    async def test_invisible_write_is_not_a_collision(self, cluster):
        """A marker that is not yet readable back must not fail startup."""
        cluster.mappings["teama_text_chunks"] = {"properties": {}}
        client = cluster.client()

        async def swallow_write(index, body):
            return None

        client.indices.put_mapping = AsyncMock(side_effect=swallow_write)
        await _claim_index_for_workspace(
            client, "teama_text_chunks", "TeamA", "TeamA_text_chunks"
        )


class TestUnwritableMapping:
    """A read-only account must not be turned away by the marker write."""

    @pytest.mark.asyncio
    async def test_startup_survives_a_rejected_marker_write(
        self, global_config, cluster, caplog
    ):
        cluster.mappings["teama_text_chunks"] = {
            "properties": {"__mirrored_id": {"type": "keyword"}}
        }
        client = cluster.client()
        client.indices.put_mapping = AsyncMock(
            side_effect=OpenSearchException("security_exception: no write privilege")
        )
        storage = OpenSearchKVStorage(
            namespace="text_chunks",
            global_config=global_config,
            embedding_func=_Embed(),
            workspace="TeamA",
        )
        # lightrag's logger does not propagate, so caplog sees nothing unless
        # propagation is enabled for the duration of the assertion.
        from lightrag.utils import logger as lightrag_logger

        previous = lightrag_logger.propagate
        lightrag_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=lightrag_logger.name):
                with patch.object(ClientManager, "get_client", return_value=client):
                    await storage.initialize()
        finally:
            lightrag_logger.propagate = previous

        assert "teama_text_chunks" in caplog.text
        assert "stays unmarked" in caplog.text
        assert "_meta" not in cluster.mappings["teama_text_chunks"]

    @pytest.mark.asyncio
    async def test_collision_still_raises_without_write_access(self, cluster):
        """Detecting a foreign marker needs no write, so it must still fail."""
        cluster.mappings["teama_text_chunks"] = {
            "_meta": {
                _WORKSPACE_META_KEY: "teama",
                _FINAL_NAMESPACE_META_KEY: "teama_text_chunks",
            }
        }
        client = cluster.client()
        client.indices.put_mapping = AsyncMock(
            side_effect=OpenSearchException("security_exception: no write privilege")
        )
        with pytest.raises(WorkspaceIndexCollisionError):
            await _claim_index_for_workspace(
                client, "teama_text_chunks", "TeamA", "TeamA_text_chunks"
            )
        client.indices.put_mapping.assert_not_awaited()
