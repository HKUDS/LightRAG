"""Embedding-space provenance on the OpenSearch vector index.

The index name on this backend carries no model information, so an operator who
swaps to a *different model of the same dimension* silently keeps querying the
previous model's vectors — confidently wrong neighbours, detected by nothing.
The fix is a marker in the index mapping's ``_meta`` and one choke point,
``_assert_index_is_usable``, that every attach path runs.

Three of those paths exist, and on ``main`` they checked different things; the
loser of an ``indices.create`` race checked neither dimension nor model, so
which facts got verified depended on how the instance happened to arrive. That
is a bug independent of the model swap: a deployment whose workspace folds onto
the same index could mark itself ready against vectors of another dimension.

Rules pinned here (see ``docs/design/VectorSpaceProvenance.md``):

- every created index records the model and dimension it was built for;
- all three attach paths refuse a foreign embedding space, with a TYPED
  exception, because ``lightrag-rebuild-vdb`` answers it by dropping the index;
- absent evidence never refuses — an index predating the marker stays servable;
- a refused instance stays drop-capable, which is what makes the refusal
  recoverable instead of a wedge.

The cluster fake is stateful on purpose: two storage instances built against
one ``FakeCluster`` really do meet on the same index, which a per-call
``AsyncMock`` cannot express.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from opensearchpy.exceptions import RequestError  # type: ignore  # noqa: E402

from lightrag.exceptions import VectorSpaceMismatchError  # noqa: E402
from lightrag.kg.opensearch_impl import (  # noqa: E402
    ClientManager,
    OpenSearchVectorDBStorage,
    _FINAL_NAMESPACE_META_KEY,
    _WORKSPACE_META_KEY,
)
from lightrag.kg.vector_space import (  # noqa: E402
    VECTOR_SPACE_DIM_KEY,
    VECTOR_SPACE_MODEL_KEY,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fakes
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
    ):
        yield


class FakeCluster:
    """Stateful stand-in for one shared OpenSearch cluster."""

    def __init__(self):
        self.mappings: dict[str, dict] = {}

    async def exists(self, index):
        return index in self.mappings

    async def create(self, index, body=None):
        if index in self.mappings:
            raise RequestError(
                400,
                "resource_already_exists_exception",
                {"error": "resource_already_exists_exception"},
            )
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
        client.count = AsyncMock(return_value={"count": 0})
        client.search = AsyncMock(
            return_value={"hits": {"hits": [], "total": {"value": 0}}}
        )
        client.mget = AsyncMock(return_value={"docs": []})
        return client

    # -- helpers used by the tests ------------------------------------------

    def meta_of(self, index):
        return self.mappings[index].get("_meta", {})

    def seed_index(self, index, *, model, dim, workspace="ws", namespace="entities"):
        """Place an index built by another process/configuration."""
        meta = {
            _WORKSPACE_META_KEY: workspace,
            _FINAL_NAMESPACE_META_KEY: f"{workspace}_{namespace}",
        }
        if model is not None:
            meta[VECTOR_SPACE_MODEL_KEY] = model
        if dim is not None:
            meta[VECTOR_SPACE_DIM_KEY] = dim
        self.mappings[index] = {
            "properties": {"vector": {"type": "knn_vector", "dimension": dim}},
            "_meta": meta,
        }


class _Embed:
    max_token_size = 100

    def __init__(self, model_name="model-a", embedding_dim=8):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    async def __call__(self, texts, **kwargs):
        return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)


@pytest.fixture
def cluster():
    return FakeCluster()


@pytest.fixture
def global_config():
    return {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


def _storage(global_config, embed, workspace="ws", namespace="entities"):
    return OpenSearchVectorDBStorage(
        namespace=namespace,
        global_config=global_config,
        embedding_func=embed,
        workspace=workspace,
    )


async def _initialize(storage, cluster):
    with patch.object(ClientManager, "get_client", return_value=cluster.client()):
        await storage.initialize()
    return storage


INDEX = "ws_entities"


# ---------------------------------------------------------------------------
# The marker is recorded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_created_index_records_its_embedding_space(global_config, cluster):
    await _initialize(_storage(global_config, _Embed("bge-m3", 8)), cluster)

    meta = cluster.meta_of(INDEX)
    assert meta[VECTOR_SPACE_MODEL_KEY] == "bge-m3"
    assert meta[VECTOR_SPACE_DIM_KEY] == 8
    # The ownership identity the marker rides alongside must survive.
    assert meta[_WORKSPACE_META_KEY] == "ws"
    assert meta[_FINAL_NAMESPACE_META_KEY] == "ws_entities"


@pytest.mark.asyncio
async def test_an_unknown_model_records_no_model_key(global_config, cluster):
    # A recorded None would be indistinguishable from "written before the
    # marker existed", making the never-refuse rule permanent for this index.
    await _initialize(_storage(global_config, _Embed(None, 8)), cluster)

    meta = cluster.meta_of(INDEX)
    assert VECTOR_SPACE_MODEL_KEY not in meta
    assert meta[VECTOR_SPACE_DIM_KEY] == 8


# ---------------------------------------------------------------------------
# Attach path 1: the index already exists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attaching_refuses_an_index_built_by_another_model(
    global_config, cluster
):
    """The defect: same dimension, different model, nothing notices."""
    cluster.seed_index(INDEX, model="bge-m3", dim=8)

    with pytest.raises(VectorSpaceMismatchError) as excinfo:
        await _initialize(_storage(global_config, _Embed("e5-large", 8)), cluster)

    message = str(excinfo.value)
    assert "'bge-m3' -> 'e5-large'" in message
    assert "lightrag-rebuild-vdb" in message
    # Never a raw repair command: put_mapping replaces _meta wholesale, so a
    # hand-run PUT carrying only the new keys would strip the ownership
    # identity and hand the index to a folding-equivalent deployment.
    assert "_mapping" not in message


@pytest.mark.asyncio
async def test_attaching_accepts_the_same_model(global_config, cluster):
    cluster.seed_index(INDEX, model="bge-m3", dim=8)
    storage = await _initialize(_storage(global_config, _Embed("bge-m3", 8)), cluster)
    assert storage._index_ready is True


@pytest.mark.asyncio
async def test_an_index_predating_the_marker_is_still_servable(global_config, cluster):
    """Absent evidence never refuses, or every pre-upgrade index is refused."""
    cluster.seed_index(INDEX, model=None, dim=8)
    storage = await _initialize(_storage(global_config, _Embed("bge-m3", 8)), cluster)
    assert storage._index_ready is True


@pytest.mark.asyncio
async def test_attaching_is_not_a_backfill(global_config, cluster):
    """An unmarked index is served, NOT stamped with this process's model.

    Blind adoption would record the new model's name over the old model's
    vectors for an operator who upgrades and swaps models in one step — a lie
    recorded permanently, after which the gate can never fire. Backfill needs
    evidence that the vectors really came from this model, and that evidence is
    gathered one layer up.
    """
    cluster.seed_index(INDEX, model=None, dim=8)
    await _initialize(_storage(global_config, _Embed("bge-m3", 8)), cluster)
    assert VECTOR_SPACE_MODEL_KEY not in cluster.meta_of(INDEX)


@pytest.mark.asyncio
async def test_attaching_refuses_a_foreign_dimension(global_config, cluster):
    cluster.seed_index(INDEX, model="bge-m3", dim=16)

    with pytest.raises(VectorSpaceMismatchError, match="16 -> 8"):
        await _initialize(_storage(global_config, _Embed("bge-m3", 8)), cluster)


# ---------------------------------------------------------------------------
# Attach path 2: losing the indices.create race
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_losing_the_create_race_still_validates_compatibility(
    global_config, cluster
):
    """The bug this closes, independent of any model swap.

    Two workers start together; the loser of indices.create attaches to an
    index it did not build and, on main, validated OWNERSHIP only. A workspace
    that folds onto the same index could then mark itself ready against vectors
    of another dimension entirely.
    """
    storage = _storage(global_config, _Embed("e5-large", 8))
    client = cluster.client()

    async def _exists_then_lose(index):
        # exists() says no, and a peer creates the index before our create().
        cluster.seed_index(index, model="bge-m3", dim=8)
        return False

    client.indices.exists = AsyncMock(side_effect=_exists_then_lose)

    with patch.object(ClientManager, "get_client", return_value=client):
        with pytest.raises(VectorSpaceMismatchError, match="'bge-m3' -> 'e5-large'"):
            await storage.initialize()

    assert storage._index_ready is False


@pytest.mark.asyncio
async def test_losing_the_create_race_accepts_a_compatible_index(
    global_config, cluster
):
    storage = _storage(global_config, _Embed("bge-m3", 8))
    client = cluster.client()

    async def _exists_then_lose(index):
        cluster.seed_index(index, model="bge-m3", dim=8)
        return False

    client.indices.exists = AsyncMock(side_effect=_exists_then_lose)

    with patch.object(ClientManager, "get_client", return_value=client):
        await storage.initialize()

    assert storage._index_ready is True


@pytest.mark.asyncio
async def test_an_unreadable_mapping_after_the_create_race_fails_closed(
    global_config, cluster
):
    """ "I could not look" is not evidence that the index is usable.

    A mapping that PARSES but records nothing is absent evidence and is served
    (see above). A get_mapping that FAILS is a different thing entirely, and it
    must propagate rather than let this instance mark itself ready one line
    later.
    """
    storage = _storage(global_config, _Embed("bge-m3", 8))
    client = cluster.client()
    calls = {"n": 0}

    async def _exists_then_lose(index):
        cluster.seed_index(index, model="bge-m3", dim=8)
        return False

    async def _get_mapping(index):
        calls["n"] += 1
        # The claim check reads first; fail the compatibility read after it.
        if calls["n"] <= 1:
            return {index: {"mappings": cluster.mappings[index]}}
        raise ConnectionError("cluster unreachable")

    client.indices.exists = AsyncMock(side_effect=_exists_then_lose)
    client.indices.get_mapping = AsyncMock(side_effect=_get_mapping)

    with patch.object(ClientManager, "get_client", return_value=client):
        with pytest.raises(ConnectionError):
            await storage.initialize()

    assert storage._index_ready is False


# ---------------------------------------------------------------------------
# Attach path 3: the read-path readiness probe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_presence_recheck_refuses_a_foreign_model(global_config, cluster):
    """A peer can rebuild the index under another model while we are marked.

    The probe is the only thing standing between that and a read path serving
    another embedding space's neighbours as ours.
    """
    storage = await _initialize(_storage(global_config, _Embed("bge-m3", 8)), cluster)
    storage._mark_index_missing()
    # The index came back — rebuilt by a different embedding configuration.
    cluster.seed_index(INDEX, model="e5-large", dim=8)

    with pytest.raises(VectorSpaceMismatchError, match="'e5-large' -> 'bge-m3'"):
        await storage.query("anything", top_k=5)

    assert storage._index_ready is False


@pytest.mark.asyncio
async def test_presence_recheck_accepts_the_same_model(global_config, cluster):
    storage = await _initialize(_storage(global_config, _Embed("bge-m3", 8)), cluster)
    storage._mark_index_missing()

    assert await storage.query("anything", top_k=5) == []
    assert storage._index_ready is True


# ---------------------------------------------------------------------------
# A refusal must stay recoverable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refused_instance_can_still_drop(global_config, cluster):
    """The wedge this removes.

    initialize() used to assign _flush_lock AFTER the compatibility gate, so a
    refused instance carried None and drop() died on ``async with None``. The
    operator then had to delete the index through the OpenSearch API by hand —
    exactly the out-of-band step lightrag-rebuild-vdb exists to avoid.
    """
    cluster.seed_index(INDEX, model="bge-m3", dim=8)
    storage = _storage(global_config, _Embed("e5-large", 8))

    with patch.object(ClientManager, "get_client", return_value=cluster.client()):
        with pytest.raises(VectorSpaceMismatchError):
            await storage.initialize()

        assert storage._flush_lock is not None
        assert storage.client is not None

        result = await storage.drop()

    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_drop_reprovisions_the_index_in_the_current_space(global_config, cluster):
    """drop() + initialize() is the tool's whole recovery; it must converge."""
    cluster.seed_index(INDEX, model="bge-m3", dim=8)
    storage = _storage(global_config, _Embed("e5-large", 16))
    client = cluster.client()

    with patch.object(ClientManager, "get_client", return_value=client):
        with pytest.raises(VectorSpaceMismatchError):
            await storage.initialize()

        assert (await storage.drop())["status"] == "success"

        meta = cluster.meta_of(INDEX)
        assert meta[VECTOR_SPACE_MODEL_KEY] == "e5-large"
        assert meta[VECTOR_SPACE_DIM_KEY] == 16
        assert cluster.mappings[INDEX]["properties"]["vector"]["dimension"] == 16

        # The second attach is the ordinary path and no longer refuses.
        await storage.initialize()

    assert storage._index_ready is True
