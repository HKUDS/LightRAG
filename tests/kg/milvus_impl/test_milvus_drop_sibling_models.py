"""Milvus workspace clear: every copy of the workspace's vectors goes, and only
collections it can prove it owns.

Two things are pinned here. ``drop()`` removes the legacy collection and the
``_old`` / ``_temp`` backups before it rebuilds the active collection, because
each is a source ``initialize()`` refills a free collection name from.
``drop_sibling_models()`` drops the other models' collections on evidence of
ownership, never on a name match alone.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("pymilvus", reason="pymilvus is required for Milvus storage tests")

from lightrag.kg.milvus_impl import MilvusVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import initialize_share_data  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

pytestmark = pytest.mark.offline

OURS = "[lightrag:workspace=a;namespace=chunks]"
VECTOR_SCHEMA = [{"name": "id"}, {"name": "vector"}]


@pytest.fixture(autouse=True)
def _shared_storage():
    initialize_share_data(workers=1)


def fake_milvus(collections=None, fields=None):
    """A client that remembers which collections exist.

    ``collections`` maps a name to its description; ``fields`` overrides the
    schema ``describe_collection`` reports for a name. Every drop is recorded
    in order on ``client.dropped``, and a rename moves the entry -- enough for
    ``initialize()``'s recovery and migration branches to act on what a clear
    left behind.
    """
    client = MagicMock()
    client.collections = dict(collections or {})
    client.dropped = []
    schemas = dict(fields or {})

    def drop(name, *args, **kwargs):
        del client.collections[name]
        client.dropped.append(name)

    def create(collection_name=None, schema=None, **kwargs):
        client.collections[collection_name] = getattr(schema, "description", "")

    def rename(old_name, new_name, *args, **kwargs):
        client.collections[new_name] = client.collections.pop(old_name)

    client.has_collection.side_effect = lambda name, *a, **k: name in client.collections
    client.list_collections.side_effect = lambda *a, **k: list(client.collections)
    client.describe_collection.side_effect = lambda name, *a, **k: {
        "description": client.collections[name],
        "fields": schemas.get(name, VECTOR_SCHEMA),
    }
    client.drop_collection.side_effect = drop
    client.create_collection.side_effect = create
    client.rename_collection.side_effect = rename
    return client


def _storage(client, *, model="model-b", dim=768, workspace="a", namespace="chunks"):
    async def embed(texts, **kwargs):
        return np.array([[0.1] * dim for _ in texts])

    storage = MilvusVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(embedding_dim=dim, func=embed, model_name=model),
        meta_fields={"content"},
    )
    storage._client = client
    return storage


# ---------------------------------------------------------------------------
# The ownership marker
# ---------------------------------------------------------------------------


def test_new_collections_carry_the_ownership_marker():
    storage = _storage(fake_milvus())
    schema = storage._create_schema_for_namespace()
    assert storage._ownership_marker() == OURS
    assert OURS in schema.description


def test_model_container_name_is_the_name_that_model_would_attach_to():
    """The rebuild tool vouches for a container by this name, so it must be
    the exact name an instance configured for that model computes."""
    other = _storage(fake_milvus(), model="Model-A", dim=1536)
    assert (
        _storage(fake_milvus()).model_container_name("Model-A", 1536)
        == other.final_namespace
    )


# ---------------------------------------------------------------------------
# drop_sibling_models
# ---------------------------------------------------------------------------


async def test_drops_marked_siblings_only():
    client = fake_milvus(
        {
            "a_chunks_model_b_768d": f"LightRAG chunks vector storage {OURS}",
            "a_chunks_model_a_1536d": f"LightRAG chunks vector storage {OURS}",
            # Another workspace whose prefix collides with ours.
            "a_chunks_m_entities_model_a_1536d": (
                "LightRAG entities vector storage "
                "[lightrag:workspace=a_chunks_m;namespace=entities]"
            ),
        }
    )
    result = await _storage(client).drop_sibling_models()

    assert result["status"] == "success"
    assert client.dropped == ["a_chunks_model_a_1536d"]
    assert sorted(client.collections) == [
        "a_chunks_m_entities_model_a_1536d",
        "a_chunks_model_b_768d",
    ]


async def test_a_siblings_backups_go_before_the_sibling_itself():
    """A crash between the drops must never leave the name free beside a
    backup: ``initialize()`` would restore the backup into it."""
    client = fake_milvus(
        {
            "a_chunks_model_a_1536d": OURS,
            "a_chunks_model_a_1536d_old": OURS,
            "a_chunks_model_a_1536d_temp": OURS,
        }
    )
    result = await _storage(client).drop_sibling_models()

    assert result["status"] == "success"
    assert client.dropped == [
        "a_chunks_model_a_1536d_temp",
        "a_chunks_model_a_1536d_old",
        "a_chunks_model_a_1536d",
    ]


async def test_a_backup_shares_the_verdict_of_the_collection_it_belongs_to():
    """``_old`` keeps the description of the collection it was renamed from,
    which may predate the marker its rebuilt successor carries."""
    client = fake_milvus(
        {
            "a_chunks_model_a_1536d": OURS,
            "a_chunks_model_a_1536d_old": "LightRAG chunks vector storage",
        }
    )
    result = await _storage(client).drop_sibling_models()

    assert result["status"] == "success"
    assert client.collections == {}


async def test_the_active_models_own_backups_are_not_siblings():
    client = fake_milvus(
        {
            "a_chunks_model_b_768d": OURS,
            "a_chunks_model_b_768d_old": OURS,
            "a_chunks_model_b_768d_temp": OURS,
        }
    )
    result = await _storage(client).drop_sibling_models()

    assert result["status"] == "success"
    assert client.dropped == []


async def test_unmarked_candidates_are_kept_and_reported():
    client = fake_milvus({"a_chunks_model_a_1536d": "LightRAG chunks vector storage"})
    storage = _storage(client)

    result = await storage.drop_sibling_models()

    assert result["status"] == "error"
    assert "a_chunks_model_a_1536d" in result["message"]
    assert client.dropped == []
    assert await storage.unvouched_sibling_containers() == ["a_chunks_model_a_1536d"]


async def test_a_vouched_unmarked_sibling_is_dropped_with_its_backups():
    client = fake_milvus(
        {
            "a_chunks_model_a_1536d": "LightRAG chunks vector storage",
            "a_chunks_model_a_1536d_old": "LightRAG chunks vector storage",
            "a_chunks_model_c_8d": "LightRAG chunks vector storage",
        }
    )
    result = await _storage(client).drop_sibling_models(
        vouched=["a_chunks_model_a_1536d"]
    )

    # Still an error: the one nobody vouched for is left and named.
    assert result["status"] == "error"
    assert "a_chunks_model_c_8d" in result["message"]
    assert "a_chunks_model_a_1536d" not in result["message"]
    assert client.dropped == ["a_chunks_model_a_1536d_old", "a_chunks_model_a_1536d"]


async def test_a_voucher_never_overrides_another_owners_marker():
    foreign = "[lightrag:workspace=a_chunks_m;namespace=entities]"
    client = fake_milvus({"a_chunks_m_entities_model_a_1536d": foreign})
    storage = _storage(client)

    result = await storage.drop_sibling_models(
        vouched=["a_chunks_m_entities_model_a_1536d"]
    )

    assert result["status"] == "success"
    assert client.dropped == []
    assert await storage.unvouched_sibling_containers() == []


@pytest.mark.parametrize(
    "family",
    [
        pytest.param(
            {
                "a_chunks_m_entities_x_8d": "FOREIGN",
                "a_chunks_m_entities_x_8d_old": OURS,
            },
            id="our-marker-on-the-backup",
        ),
        pytest.param(
            {
                "a_chunks_m_entities_x_8d": OURS,
                "a_chunks_m_entities_x_8d_old": "FOREIGN",
            },
            id="our-marker-on-the-collection",
        ),
    ],
)
@pytest.mark.parametrize("vouched", [(), ("a_chunks_m_entities_x_8d",)])
async def test_a_family_marked_for_two_owners_is_refused_whole(family, vouched):
    """Workspace ``a`` + ``chunks`` + model ``m_entities_x`` and workspace
    ``a_chunks_m`` + ``entities`` + model ``x`` fold onto one collection name.
    A schema migration by either leaves an ``_old`` carrying the other's
    marker beside its own live collection, and a backup must never authorize
    dropping a collection that names another owner."""
    foreign = "[lightrag:workspace=a_chunks_m;namespace=entities]"
    client = fake_milvus(
        {name: foreign if d == "FOREIGN" else d for name, d in family.items()}
    )
    storage = _storage(client)

    result = await storage.drop_sibling_models(vouched=vouched)

    assert result["status"] == "error"
    assert "another owner" in result["message"]
    assert "a_chunks_m_entities_x_8d" in result["message"]
    assert client.dropped == []
    # Nothing a voucher could settle, so it is not offered for one.
    assert await storage.unvouched_sibling_containers() == []


async def test_listing_unvouched_siblings_drops_nothing():
    client = fake_milvus(
        {
            "a_chunks_model_a_1536d": "",
            "a_chunks_model_c_8d": OURS,
            "a_chunks_model_b_768d": "",  # the active collection
            "a_chunks": "",  # the legacy collection
        }
    )
    assert await _storage(client).unvouched_sibling_containers() == [
        "a_chunks_model_a_1536d"
    ]
    assert client.dropped == []


async def test_a_backend_error_is_an_error_result():
    client = fake_milvus()
    client.list_collections.side_effect = ConnectionError("milvus down")

    result = await _storage(client).drop_sibling_models()

    assert result == {"status": "error", "message": "milvus down"}


# ---------------------------------------------------------------------------
# drop(): the active model's backups and the legacy collection
# ---------------------------------------------------------------------------


async def test_drop_removes_every_backup_before_it_rebuilds_the_collection():
    """Backups first: a crash after them leaves the active collection still
    populated with nothing to refill it from; the other order leaves a free
    name beside a backup."""
    client = fake_milvus(
        {
            "a_chunks_model_b_768d": OURS,
            "a_chunks_model_b_768d_old": OURS,
            "a_chunks_model_b_768d_temp": OURS,
            "a_chunks": "LightRAG chunks vector storage",
            "a_chunks_old": "LightRAG chunks vector storage",
            "a_chunks_temp": "LightRAG chunks vector storage",
        }
    )
    result = await _storage(client).drop()

    assert result["status"] == "success"
    assert client.dropped[-1] == "a_chunks_model_b_768d"
    assert sorted(client.dropped[:-1]) == [
        "a_chunks",
        "a_chunks_model_b_768d_old",
        "a_chunks_model_b_768d_temp",
        "a_chunks_old",
        "a_chunks_temp",
    ]
    # Rebuilt empty, and nothing else is left.
    assert list(client.collections) == ["a_chunks_model_b_768d"]
    assert OURS in client.collections["a_chunks_model_b_768d"]


async def test_drop_leaves_a_legacy_collection_nothing_would_migrate(
    lightrag_log_records,
):
    """No ``vector`` field: ``initialize()`` skips it, so it cannot come back
    -- and it may not be LightRAG's collection at all."""
    client = fake_milvus(
        {"a_chunks_model_b_768d": OURS, "a_chunks": ""},
        fields={"a_chunks": [{"name": "id"}, {"name": "text"}]},
    )
    result = await _storage(client).drop()

    assert result["status"] == "success"
    assert client.dropped == ["a_chunks_model_b_768d"]
    assert "a_chunks" in client.collections
    assert any(
        "'a_chunks'" in r.getMessage() and "no vector field" in r.getMessage()
        for r in lightrag_log_records
    )


async def test_drop_leaves_a_backup_marked_as_another_owners(lightrag_log_records):
    foreign = "[lightrag:workspace=a_chunks_model;namespace=entities]"
    client = fake_milvus(
        {"a_chunks_model_b_768d": OURS, "a_chunks_model_b_768d_old": foreign}
    )
    result = await _storage(client).drop()

    assert result["status"] == "success"
    assert client.dropped == ["a_chunks_model_b_768d"]
    assert client.collections["a_chunks_model_b_768d_old"] == foreign
    assert any("another owner's" in r.getMessage() for r in lightrag_log_records)


async def test_drop_without_a_model_suffix_treats_no_collection_as_legacy():
    """With no model name the active collection IS the un-suffixed one: it is
    rebuilt once, not also dropped as its own legacy backup."""
    client = fake_milvus({"a_chunks": "", "a_chunks_old": ""})
    storage = _storage(client, model=None)
    assert storage.final_namespace == storage.legacy_namespace == "a_chunks"

    result = await storage.drop()

    assert result["status"] == "success"
    assert client.dropped == ["a_chunks_old", "a_chunks"]
    assert list(client.collections) == ["a_chunks"]


# ---------------------------------------------------------------------------
# The regression: nothing comes back when the operator switches models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leftover",
    [
        pytest.param({"a_chunks": "LightRAG chunks vector storage"}, id="legacy"),
        pytest.param({"a_chunks_model_a_768d_old": OURS}, id="sibling-old-backup"),
        pytest.param({"a_chunks_model_a_768d_temp": OURS}, id="sibling-temp"),
    ],
)
async def test_a_cleared_sibling_is_created_empty_when_its_model_returns(leftover):
    """A workspace cleared under model B, then started under model A. Each
    leftover is a source ``initialize()`` would have refilled A's free
    collection name from -- the legacy migration, or the interrupted-migration
    recovery -- putting the cleared rows back."""
    client = fake_milvus(
        {
            "a_chunks_model_b_768d": OURS,
            "a_chunks_model_a_768d": OURS,
            **leftover,
        }
    )
    result = await _storage(client).drop_all_models()
    assert result["status"] == "success"
    assert list(client.collections) == ["a_chunks_model_b_768d"]

    returning = _storage(client, model="model-a")
    returning._create_collection_if_not_exist()

    assert "a_chunks_model_a_768d" in client.collections
    client.rename_collection.assert_not_called()
    client.query_iterator.assert_not_called()
    client.insert.assert_not_called()
    assert client.create_collection.call_args.kwargs["collection_name"] == (
        "a_chunks_model_a_768d"
    )
