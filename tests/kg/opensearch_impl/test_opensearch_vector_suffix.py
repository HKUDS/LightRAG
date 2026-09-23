"""Model-suffix isolation for OpenSearch vector indexes.

Every index created before the suffix is legacy. Startup copies it when
this workspace owns it and the recorded dimension and model do not disagree.
A destination with fewer documents than that legacy index is copied again;
one with at least as many is left in place. A mismatch is skipped, not
raised. ``drop`` deletes a compatible owned legacy index so the next
startup does not copy the cleared corpus back, and leaves an incompatible
one in place. A legacy mapping that cannot be read fails the drop before
the serving index is deleted.

See ``docs/design/VectorSpaceProvenance.md``.
"""

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from opensearchpy.exceptions import NotFoundError, OpenSearchException, RequestError

from lightrag.exceptions import DataMigrationError, VectorSpaceMismatchError
from lightrag.kg.opensearch_impl import (
    ClientManager,
    OpenSearchVectorDBStorage,
    _FINAL_NAMESPACE_META_KEY,
    _WORKSPACE_META_KEY,
)
from lightrag.kg.vector_space import VECTOR_SPACE_DIM_KEY, VECTOR_SPACE_MODEL_KEY

pytestmark = pytest.mark.offline


@asynccontextmanager
async def _mock_lock():
    yield


def _mock_lock_factory(*args, **kwargs):
    return _mock_lock()


@pytest.fixture(autouse=True)
def patch_locks():
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


class _Embed:
    max_token_size = 100

    def __init__(self, model_name="bge-m3", embedding_dim=8):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    async def __call__(self, texts, **kwargs):
        return [[0.0] * self.embedding_dim for _ in texts]


class Cluster:
    """One OpenSearch cluster, just enough for index create / reindex / drop."""

    def __init__(self):
        self.indices: dict[str, dict] = {}
        self.reindex_calls: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.fail_reindex = False
        # None copies every source document. An int copies that many, then
        # reports success, so a short destination can stay short.
        self.reindex_copy_limit: int | None = None
        self.fail_delete = False
        # Fail get_mapping on the Nth call for an index (1-based), counted
        # only while the entry is set. Ownership reads the mapping once;
        # the compatibility read is the next call.
        self.fail_get_mapping_on: dict[str, int] = {}
        self.get_mapping_calls: dict[str, int] = {}

    def client(self):
        from opensearchpy import AsyncOpenSearch

        client = AsyncMock(spec=AsyncOpenSearch)
        client.indices = AsyncMock()
        client.indices.exists = AsyncMock(side_effect=self.exists)
        client.indices.create = AsyncMock(side_effect=self.create)
        client.indices.get_mapping = AsyncMock(side_effect=self.get_mapping)
        client.indices.put_mapping = AsyncMock(side_effect=self.put_mapping)
        client.indices.delete = AsyncMock(side_effect=self.delete)
        client.count = AsyncMock(side_effect=self.count)
        client.reindex = AsyncMock(side_effect=self.reindex)
        return client

    async def exists(self, index):
        return index in self.indices

    async def create(self, index, body=None):
        if index in self.indices:
            raise RequestError(
                400,
                "resource_already_exists_exception",
                {"error": "resource_already_exists_exception"},
            )
        self.indices[index] = {
            "mappings": dict((body or {}).get("mappings") or {}),
            "docs": {},
        }

    async def get_mapping(self, index):
        if index not in self.indices:
            raise NotFoundError(404, "index_not_found_exception", "no such index")
        nth = self.get_mapping_calls.get(index, 0) + 1
        self.get_mapping_calls[index] = nth
        if self.fail_get_mapping_on.get(index) == nth:
            raise OpenSearchException("mapping unavailable")
        return {index: {"mappings": self.indices[index]["mappings"]}}

    async def put_mapping(self, index, body):
        mappings = self.indices[index]["mappings"]
        if "_meta" in body:
            mappings["_meta"] = body["_meta"]

    async def delete(self, index, **kwargs):
        self.deleted.append(index)
        if self.fail_delete:
            raise OpenSearchException("delete failed")
        if index not in self.indices:
            raise NotFoundError(404, "index_not_found_exception", "no such index")
        del self.indices[index]

    async def count(self, index=None, **kwargs):
        return {"count": len(self.indices[index]["docs"])}

    async def reindex(
        self, *, body, params=None, refresh=None, wait_for_completion=None, **kwargs
    ):
        # OpenSearch reads conflicts from the reindex body. A query param is
        # rejected as unrecognized before any document is copied.
        assert body["conflicts"] == "proceed"
        assert "conflicts" not in (params or {})
        source = body["source"]["index"]
        dest = body["dest"]["index"]
        self.reindex_calls.append((source, dest))
        if self.fail_reindex:
            self.indices[dest]["docs"]["partial"] = {"content": "partial"}
            raise OpenSearchException("reindex failed")
        created = 0
        for doc_id, source_doc in self.indices[source]["docs"].items():
            if (
                self.reindex_copy_limit is not None
                and created >= self.reindex_copy_limit
            ):
                break
            self.indices[dest]["docs"][doc_id] = dict(source_doc)
            created += 1
        return {
            "total": created,
            "created": created,
            "updated": 0,
            "failures": [],
            "timed_out": False,
        }

    def seed(
        self,
        index: str,
        *,
        workspace: str = "ws",
        final_namespace: str = "ws_entities",
        model: str | None = None,
        dim: int | None = 8,
        docs: dict | None = None,
        include_meta: bool = True,
    ):
        meta = {
            _WORKSPACE_META_KEY: workspace,
            _FINAL_NAMESPACE_META_KEY: final_namespace,
        }
        if model is not None:
            meta[VECTOR_SPACE_MODEL_KEY] = model
        if dim is not None:
            meta[VECTOR_SPACE_DIM_KEY] = dim
        mappings: dict = {"properties": {}}
        if dim is not None:
            mappings["properties"]["vector"] = {
                "type": "knn_vector",
                "dimension": dim,
            }
        if include_meta:
            mappings["_meta"] = meta
        self.indices[index] = {"mappings": mappings, "docs": dict(docs or {})}

    def meta(self, index: str) -> dict:
        return self.indices[index]["mappings"].get("_meta") or {}

    def doc_ids(self, index: str) -> set[str]:
        return set(self.indices[index]["docs"])


@pytest.fixture
def cluster():
    return Cluster()


@pytest.fixture
def global_config():
    return {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


@contextmanager
def _warnings(caplog):
    """lightrag's logger does not propagate, so caplog needs it turned on."""
    from lightrag.utils import logger as lightrag_logger

    previous = lightrag_logger.propagate
    lightrag_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger=lightrag_logger.name):
            yield
    finally:
        lightrag_logger.propagate = previous


def _storage(global_config, embed=None, workspace="ws", namespace="entities"):
    return OpenSearchVectorDBStorage(
        namespace=namespace,
        global_config=global_config,
        embedding_func=embed or _Embed(),
        workspace=workspace,
    )


async def _init(storage, cluster):
    with patch.object(ClientManager, "get_client", return_value=cluster.client()):
        await storage.initialize()
    return storage


def test_suffix_is_applied_when_model_name_is_present(global_config):
    storage = _storage(global_config, _Embed("bge-m3", 8))
    assert storage.model_suffix == "bge_m3_8d"
    assert storage._legacy_index_name == "ws_entities"
    assert storage.final_namespace == "ws_entities"
    assert storage._index_name == "ws_entities_bge_m3_8d"


def test_suffix_is_absent_without_model_name(global_config):
    storage = _storage(global_config, _Embed(None, 8))
    assert storage.model_suffix is None
    assert storage._index_name == "ws_entities"
    assert storage._legacy_index_name == "ws_entities"


def test_distinct_models_do_not_share_an_index(global_config):
    first = _storage(global_config, _Embed("bge-m3", 8))
    second = _storage(global_config, _Embed("e5-large", 8))
    assert first._index_name != second._index_name


def test_index_name_over_255_bytes_is_rejected(global_config):
    with pytest.raises(ValueError, match="255"):
        _storage(global_config, _Embed("m" * 300, 8))


@pytest.mark.asyncio
async def test_fresh_index_records_model_and_dimension(global_config, cluster):
    storage = await _init(_storage(global_config), cluster)

    meta = cluster.meta(storage._index_name)
    assert meta[VECTOR_SPACE_MODEL_KEY] == "bge-m3"
    assert meta[VECTOR_SPACE_DIM_KEY] == 8
    assert meta[_WORKSPACE_META_KEY] == "ws"
    assert meta[_FINAL_NAMESPACE_META_KEY] == "ws_entities"
    assert cluster.reindex_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_model", [None, "bge-m3"])
async def test_owned_legacy_index_is_copied_once(global_config, cluster, legacy_model):
    storage = _storage(global_config)
    cluster.seed(
        storage._legacy_index_name,
        model=legacy_model,
        dim=8,
        docs={"a": {"content": "a"}, "b": {"content": "b"}},
        include_meta=legacy_model is not None,
    )

    await _init(storage, cluster)

    assert cluster.doc_ids(storage._index_name) == {"a", "b"}
    assert cluster.doc_ids(storage._legacy_index_name) == {"a", "b"}
    assert cluster.reindex_calls == [(storage._legacy_index_name, storage._index_name)]
    assert cluster.meta(storage._index_name)[VECTOR_SPACE_MODEL_KEY] == "bge-m3"
    # The legacy index keeps its own provenance. Adoption may record the
    # workspace, and it must not stamp a model onto an index that did not
    # already name one.
    legacy_meta = cluster.meta(storage._legacy_index_name)
    if legacy_model is None:
        assert VECTOR_SPACE_MODEL_KEY not in legacy_meta
        assert legacy_meta[_WORKSPACE_META_KEY] == "ws"
        assert legacy_meta[_FINAL_NAMESPACE_META_KEY] == "ws_entities"
    else:
        assert legacy_meta[VECTOR_SPACE_MODEL_KEY] == "bge-m3"

    restarted = _storage(global_config)
    await _init(restarted, cluster)
    assert cluster.reindex_calls == [(storage._legacy_index_name, storage._index_name)]
    assert cluster.doc_ids(storage._index_name) == {"a", "b"}


@pytest.mark.asyncio
async def test_legacy_model_mismatch_skips_without_raising(global_config, cluster):
    storage = _storage(global_config, _Embed("bge-m3", 8))
    cluster.seed(
        storage._legacy_index_name,
        model="e5-large",
        dim=8,
        docs={"a": {"content": "a"}},
    )

    await _init(storage, cluster)

    assert cluster.doc_ids(storage._index_name) == set()
    assert cluster.doc_ids(storage._legacy_index_name) == {"a"}
    assert cluster.reindex_calls == []
    assert (
        cluster.meta(storage._legacy_index_name)[VECTOR_SPACE_MODEL_KEY] == "e5-large"
    )


@pytest.mark.asyncio
async def test_legacy_dimension_mismatch_skips_without_raising(global_config, cluster):
    storage = _storage(global_config, _Embed("bge-m3", 8))
    cluster.seed(
        storage._legacy_index_name,
        model="bge-m3",
        dim=16,
        docs={"a": {"content": "a"}},
    )

    await _init(storage, cluster)

    assert cluster.doc_ids(storage._index_name) == set()
    assert cluster.doc_ids(storage._legacy_index_name) == {"a"}
    assert cluster.reindex_calls == []


@pytest.mark.asyncio
async def test_legacy_ownership_mismatch_skips_and_does_not_rewrite_meta(
    global_config, cluster
):
    storage = _storage(global_config)
    cluster.seed(
        storage._legacy_index_name,
        workspace="other",
        final_namespace="other_entities",
        model="bge-m3",
        dim=8,
        docs={"a": {"content": "a"}},
    )
    before = dict(cluster.meta(storage._legacy_index_name))

    await _init(storage, cluster)

    assert cluster.meta(storage._legacy_index_name) == before
    assert cluster.doc_ids(storage._legacy_index_name) == {"a"}
    assert cluster.doc_ids(storage._index_name) == set()
    assert cluster.reindex_calls == []


@pytest.mark.asyncio
async def test_empty_legacy_index_is_deleted(global_config, cluster):
    storage = _storage(global_config)
    cluster.seed(storage._legacy_index_name, model=None, dim=8, docs={})

    await _init(storage, cluster)

    assert storage._legacy_index_name not in cluster.indices
    assert cluster.doc_ids(storage._index_name) == set()
    assert cluster.reindex_calls == []


@pytest.mark.asyncio
async def test_empty_suffixed_index_still_receives_the_legacy_copy(
    global_config, cluster
):
    """An empty suffixed index is not 'already migrated'.

    ``drop`` recreates that empty index. If it left the legacy index behind,
    the next startup would copy the cleared corpus back.
    """
    storage = _storage(global_config)
    cluster.seed(storage._index_name, model="bge-m3", dim=8, docs={})
    cluster.seed(
        storage._legacy_index_name,
        model="bge-m3",
        dim=8,
        docs={"a": {"content": "a"}},
    )

    await _init(storage, cluster)

    assert cluster.doc_ids(storage._index_name) == {"a"}
    assert cluster.reindex_calls == [(storage._legacy_index_name, storage._index_name)]


@pytest.mark.asyncio
async def test_nonempty_suffixed_index_is_not_copied_into_again(
    global_config, cluster, caplog
):
    """Equal counts are already covered, so the legacy index is left alone.

    The operator is told to delete that legacy index only in this case.
    """
    storage = _storage(global_config)
    cluster.seed(
        storage._index_name,
        model="bge-m3",
        dim=8,
        docs={"kept": {"content": "kept"}},
    )
    cluster.seed(
        storage._legacy_index_name,
        model="bge-m3",
        dim=8,
        docs={"legacy": {"content": "legacy"}},
    )

    with _warnings(caplog):
        await _init(storage, cluster)

    assert cluster.doc_ids(storage._index_name) == {"kept"}
    assert cluster.doc_ids(storage._legacy_index_name) == {"legacy"}
    assert cluster.reindex_calls == []
    assert "Not copying" in caplog.text
    assert storage._legacy_index_name in caplog.text


@pytest.mark.asyncio
async def test_larger_destination_is_not_copied_into_again(
    global_config, cluster, caplog
):
    """More destination rows than the legacy index is also already covered."""
    storage = _storage(global_config)
    cluster.seed(
        storage._index_name,
        model="bge-m3",
        dim=8,
        docs={"kept": {"content": "kept"}, "also": {"content": "also"}},
    )
    cluster.seed(
        storage._legacy_index_name,
        model="bge-m3",
        dim=8,
        docs={"legacy": {"content": "legacy"}},
    )

    with _warnings(caplog):
        await _init(storage, cluster)

    assert cluster.doc_ids(storage._index_name) == {"kept", "also"}
    assert cluster.doc_ids(storage._legacy_index_name) == {"legacy"}
    assert cluster.reindex_calls == []
    assert storage._index_name not in cluster.deleted
    assert "Not copying" in caplog.text


@pytest.mark.asyncio
async def test_short_destination_is_reindexed_without_dropping_extra_rows(
    global_config, cluster, caplog
):
    """A killed reindex leaves a short index. The next start copies again.

    Document ids are idempotent, so rows that exist only in the destination
    stay. The legacy index is the complete copy and must not be offered for
    deletion while the destination is still short.
    """
    storage = _storage(global_config)
    cluster.seed(
        storage._index_name,
        model="bge-m3",
        dim=8,
        docs={"a": {"content": "partial"}, "extra": {"content": "only-here"}},
    )
    cluster.seed(
        storage._legacy_index_name,
        model="bge-m3",
        dim=8,
        docs={
            "a": {"content": "a"},
            "b": {"content": "b"},
            "c": {"content": "c"},
        },
    )

    with _warnings(caplog):
        await _init(storage, cluster)

    assert storage._index_ready is True
    assert cluster.doc_ids(storage._index_name) == {"a", "b", "c", "extra"}
    assert cluster.indices[storage._index_name]["docs"]["a"]["content"] == "a"
    assert cluster.indices[storage._index_name]["docs"]["extra"]["content"] == (
        "only-here"
    )
    assert cluster.doc_ids(storage._legacy_index_name) == {"a", "b", "c"}
    assert cluster.reindex_calls == [
        (storage._legacy_index_name, storage._index_name)
    ]
    assert storage._index_name not in cluster.deleted
    assert storage._legacy_index_name not in cluster.deleted
    assert "Not copying" not in caplog.text


@pytest.mark.asyncio
async def test_undeleatable_short_destination_is_refused_until_it_covers(
    global_config, cluster, caplog
):
    """Startup stays failed while a short index cannot be deleted or filled.

    ``_discard_partial_migration`` logs and returns when delete fails. The
    next start must still refuse that short index instead of attaching, and
    it must retry the reindex without deleting destination-only rows first.
    """
    storage = _storage(global_config)
    cluster.seed(
        storage._index_name,
        model="bge-m3",
        dim=8,
        docs={"a": {"content": "partial"}, "extra": {"content": "only-here"}},
    )
    cluster.seed(
        storage._legacy_index_name,
        model="bge-m3",
        dim=8,
        docs={
            "a": {"content": "a"},
            "b": {"content": "b"},
            "c": {"content": "c"},
        },
    )
    cluster.reindex_copy_limit = 0
    cluster.fail_delete = True

    with _warnings(caplog):
        with pytest.raises(DataMigrationError, match="verification failed"):
            await _init(storage, cluster)

    assert storage._index_ready is False
    assert cluster.doc_ids(storage._index_name) == {"a", "extra"}
    assert storage._index_name in cluster.indices
    assert "Not copying" not in caplog.text

    restarted = _storage(global_config)
    with _warnings(caplog):
        with pytest.raises(DataMigrationError, match="verification failed"):
            await _init(restarted, cluster)

    assert restarted._index_ready is False
    assert cluster.doc_ids(storage._index_name) == {"a", "extra"}
    assert cluster.doc_ids(storage._legacy_index_name) == {"a", "b", "c"}
    assert cluster.reindex_calls == [
        (storage._legacy_index_name, storage._index_name),
        (storage._legacy_index_name, storage._index_name),
    ]
    assert "Not copying" not in caplog.text

    cluster.reindex_copy_limit = None
    cluster.fail_delete = False
    recovered = _storage(global_config)
    with _warnings(caplog):
        await _init(recovered, cluster)

    assert recovered._index_ready is True
    assert cluster.doc_ids(storage._index_name) == {"a", "b", "c", "extra"}
    assert cluster.indices[storage._index_name]["docs"]["extra"]["content"] == (
        "only-here"
    )
    assert cluster.doc_ids(storage._legacy_index_name) == {"a", "b", "c"}
    assert "Not copying" not in caplog.text


@pytest.mark.asyncio
async def test_failed_copy_raises_and_removes_the_partial_destination(
    global_config, cluster
):
    storage = _storage(global_config)
    cluster.seed(
        storage._legacy_index_name,
        model="bge-m3",
        dim=8,
        docs={"a": {"content": "a"}},
    )
    cluster.fail_reindex = True

    with pytest.raises(DataMigrationError, match=storage._index_name):
        await _init(storage, cluster)

    assert storage._index_name not in cluster.indices
    assert cluster.doc_ids(storage._legacy_index_name) == {"a"}


@pytest.mark.asyncio
async def test_drop_clears_legacy_so_the_next_start_does_not_resurrect(
    global_config, cluster
):
    storage = _storage(global_config)
    cluster.seed(
        storage._legacy_index_name,
        model="bge-m3",
        dim=8,
        docs={"a": {"content": "a"}, "b": {"content": "b"}},
    )
    await _init(storage, cluster)
    assert cluster.doc_ids(storage._index_name) == {"a", "b"}

    assert (await storage.drop())["status"] == "success"
    assert storage._legacy_index_name not in cluster.indices
    assert cluster.doc_ids(storage._index_name) == set()

    restarted = await _init(_storage(global_config), cluster)
    assert cluster.doc_ids(restarted._index_name) == set()
    assert cluster.reindex_calls == [(storage._legacy_index_name, storage._index_name)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("legacy_model", "legacy_dim", "other_index"),
    [
        ("e5-large", 8, "ws_entities_e5_large_8d"),
        ("bge-m3", 16, "ws_entities_bge_m3_16d"),
    ],
)
async def test_drop_leaves_an_incompatible_owned_legacy_index(
    global_config, cluster, legacy_model, legacy_dim, other_index
):
    """Clear removes the current index, not a previous model's pre-suffix rows.

    Startup already refuses to copy a model or dimension mismatch, so
    deleting that unsuffixed index is not what stops it coming back. The
    other space's suffixed index is a different name and stays as well.
    """
    storage = _storage(global_config)
    assert storage._index_name == "ws_entities_bge_m3_8d"
    assert storage._legacy_index_name == "ws_entities"
    assert other_index not in (storage._index_name, storage._legacy_index_name)
    cluster.seed(
        other_index,
        model=legacy_model,
        dim=legacy_dim,
        docs={"other": {"content": "other"}},
    )
    cluster.seed(
        storage._legacy_index_name,
        model=legacy_model,
        dim=legacy_dim,
        docs={"a": {"content": "a"}},
    )

    await _init(storage, cluster)
    assert cluster.doc_ids(storage._index_name) == set()
    assert cluster.reindex_calls == []

    assert (await storage.drop())["status"] == "success"

    assert storage._legacy_index_name in cluster.indices
    assert cluster.doc_ids(storage._legacy_index_name) == {"a"}
    assert (
        cluster.meta(storage._legacy_index_name)[VECTOR_SPACE_MODEL_KEY]
        == legacy_model
    )
    assert cluster.doc_ids(other_index) == {"other"}
    assert cluster.doc_ids(storage._index_name) == set()

    await _init(_storage(global_config), cluster)
    assert cluster.doc_ids(storage._legacy_index_name) == {"a"}
    assert cluster.doc_ids(storage._index_name) == set()
    assert cluster.doc_ids(other_index) == {"other"}
    assert cluster.reindex_calls == []


@pytest.mark.asyncio
async def test_drop_mapping_failure_does_not_delete_the_serving_index(
    global_config, cluster
):
    """A legacy mapping that cannot be read aborts before either delete.

    Ownership consumes the first mapping read. The compatibility read is
    the next one; failing it must not wipe the serving index or the legacy
    index and report success.
    """
    storage = _storage(global_config)
    cluster.seed(
        storage._index_name,
        model="bge-m3",
        dim=8,
        docs={"kept": {"content": "kept"}},
    )
    cluster.seed(
        storage._legacy_index_name,
        model="bge-m3",
        dim=8,
        docs={"a": {"content": "a"}},
    )
    await _init(storage, cluster)
    assert cluster.doc_ids(storage._index_name) == {"kept"}
    assert cluster.doc_ids(storage._legacy_index_name) == {"a"}

    cluster.get_mapping_calls.clear()
    cluster.fail_get_mapping_on[storage._legacy_index_name] = 2

    result = await storage.drop()

    assert result["status"] == "error"
    assert "mapping unavailable" in result["message"]
    assert cluster.doc_ids(storage._index_name) == {"kept"}
    assert cluster.doc_ids(storage._legacy_index_name) == {"a"}
    assert storage._index_name not in cluster.deleted
    assert storage._legacy_index_name not in cluster.deleted


@pytest.mark.asyncio
async def test_drop_leaves_a_legacy_index_owned_by_another_workspace(
    global_config, cluster
):
    storage = _storage(global_config)
    cluster.seed(
        storage._legacy_index_name,
        workspace="other",
        final_namespace="other_entities",
        model="bge-m3",
        dim=8,
        docs={"a": {"content": "a"}},
    )
    await _init(storage, cluster)

    assert (await storage.drop())["status"] == "success"

    assert cluster.doc_ids(storage._legacy_index_name) == {"a"}
    assert cluster.meta(storage._legacy_index_name)[_WORKSPACE_META_KEY] == "other"
    assert cluster.doc_ids(storage._index_name) == set()

    await _init(_storage(global_config), cluster)
    assert cluster.doc_ids(storage._legacy_index_name) == {"a"}
    assert cluster.doc_ids(storage._index_name) == set()
    assert cluster.reindex_calls == []


@pytest.mark.asyncio
async def test_dimension_guard_still_refuses_without_model_name(global_config, cluster):
    """No model_name means no suffix, so the guard is the only backstop."""
    storage = _storage(global_config, _Embed(None, 8))
    assert storage._index_name == "ws_entities"
    cluster.seed(storage._index_name, model="bge-m3", dim=16, docs={"a": {}})

    with pytest.raises(VectorSpaceMismatchError, match="16 -> 8"):
        await _init(storage, cluster)

    assert storage._flush_lock is not None
    assert (await storage.drop())["status"] == "success"
    assert cluster.meta(storage._index_name)[VECTOR_SPACE_DIM_KEY] == 8
    assert VECTOR_SPACE_MODEL_KEY not in cluster.meta(storage._index_name)
