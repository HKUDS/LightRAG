"""``supports_hybrid_query`` capability protocol (plan D3) and its graceful fallback.

P1 pins the protocol BEFORE any consumer exists. The class-level flag is False
on every registered vector backend except ``PGVectorStorage``, it is a
``ClassVar`` (never a dataclass ``__init__`` field, so no backend constructor
signature moves -- the ``requires_single_writer`` precedent), and an
``enable_hybrid=True`` request through the real retrieval entry point
``_get_vector_context`` leaves an unsupported backend on its plain vector path.

The graceful cases are the real gate now that P3 wires ``hybrid_query``: a
consumer must check the flag and fall back to vector-only, never crash or
change results on a backend that does not support hybrid retrieval, and
never reach a ``hybrid_query`` method on such a backend.
``NoopVectorDBStorage``'s vector-only path IS its designed
``StorageCapabilityError`` refusal, so "graceful" there means the flag adds no
new failure mode on top of that refusal.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lightrag.base import BaseVectorStorage, QueryParam
from lightrag.exceptions import StorageCapabilityError
from lightrag.kg import STORAGE_IMPLEMENTATIONS
from lightrag.kg.factory import get_storage_class
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.kg.noop_vector_db_impl import NoopVectorDBStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.operate import _get_vector_context
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline

_VECTOR_BACKENDS = STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]

DIM = 8


@pytest.mark.parametrize("backend_name", _VECTOR_BACKENDS)
def test_capability_flag_matches_the_protocol(backend_name):
    """Only PGVectorStorage advertises hybrid support; every other backend False."""
    cls = get_storage_class(backend_name)
    assert issubclass(cls, BaseVectorStorage)
    assert cls.supports_hybrid_query is (backend_name == "PGVectorStorage")


@pytest.mark.parametrize("backend_name", _VECTOR_BACKENDS)
def test_capability_flag_is_not_a_dataclass_field(backend_name):
    """ClassVar, not an __init__ field: the constructor signature must not move."""
    cls = get_storage_class(backend_name)
    field_names = {f.name for f in dataclasses.fields(cls)}
    assert "supports_hybrid_query" not in field_names


def test_base_default_is_false_and_classvar():
    assert BaseVectorStorage.supports_hybrid_query is False
    assert "supports_hybrid_query" not in {
        f.name for f in dataclasses.fields(BaseVectorStorage)
    }


async def _embed(texts, **kwargs):
    return np.array([[0.1] * DIM for _ in texts], dtype=np.float32)


@pytest.fixture
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def test_unsupported_backend_hybrid_request_stays_vector_only(tmp_path, _shared_data):
    """A real registered unsupported backend (Nano) returns the same vector-only
    hits with enable_hybrid=True as with it False -- no exception, no new leg."""
    storage = NanoVectorDBStorage(
        namespace="test_hybrid_chunks",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=DIM, max_token_size=512, func=_embed
        ),
        meta_fields={"content"},
    )
    await storage.initialize()
    await storage.upsert({"chunk-1": {"content": "hello hybrid world"}})
    await storage.index_done_callback()

    assert storage.supports_hybrid_query is False

    off = await _get_vector_context(
        "hello", storage, QueryParam(enable_hybrid=False), query_embedding=[0.1] * DIM
    )
    on = await _get_vector_context(
        "hello", storage, QueryParam(enable_hybrid=True), query_embedding=[0.1] * DIM
    )

    assert len(on) == 1
    assert on[0]["content"] == "hello hybrid world"
    assert on[0]["source_type"] == "vector"
    assert on[0]["chunk_id"] == "chunk-1"
    assert on == off


async def test_noop_backend_hybrid_request_keeps_its_capability_refusal():
    """Noop refuses vector retrieval by design; enable_hybrid=True must not turn
    that designed refusal into a different (hybrid-specific) crash."""
    storage = NoopVectorDBStorage(
        namespace="test_hybrid_chunks",
        workspace="ws",
        global_config={},
        embedding_func=None,
    )
    assert storage.supports_hybrid_query is False
    with pytest.raises(StorageCapabilityError, match="Vector retrieval is disabled"):
        await _get_vector_context(
            "hello",
            storage,
            QueryParam(enable_hybrid=True),
            query_embedding=[0.1] * DIM,
        )


_UNSUPPORTED_BACKENDS = [b for b in _VECTOR_BACKENDS if b != "PGVectorStorage"]


@pytest.mark.parametrize("backend_name", _UNSUPPORTED_BACKENDS)
async def test_dispatch_guard_never_reaches_hybrid_query(backend_name):
    """With a consumer wired (P3), enable_hybrid=True on ANY unsupported
    backend must never reach a hybrid_query method, even one attached to the
    instance: the dispatch gate checks the capability flag first. Service
    backends are exercised through ``__new__`` (no connection needed at this
    seam); Noop keeps its real designed refusal. The recording fake raises
    AND records, so a swallowed AssertionError cannot hide a dispatch leak.
    """
    cls = get_storage_class(backend_name)
    storage = cls.__new__(cls)
    storage.cosine_better_than_threshold = 0.2
    hybrid_calls: list[tuple] = []

    async def recording_hybrid(query, top_k, query_embedding=None):
        hybrid_calls.append((query, top_k))
        raise AssertionError("hybrid_query reached on an unsupported backend")

    storage.hybrid_query = recording_hybrid

    if backend_name == "NoopVectorDBStorage":
        with pytest.raises(
            StorageCapabilityError, match="Vector retrieval is disabled"
        ):
            await _get_vector_context(
                "hello",
                storage,
                QueryParam(enable_hybrid=True),
                query_embedding=[0.1] * DIM,
            )
    else:
        query_calls: list[tuple] = []

        async def fake_query(query, top_k, query_embedding=None):
            query_calls.append((query, top_k))
            return [{"content": "vector hit", "id": "c-1", "file_path": "a.txt"}]

        storage.query = fake_query
        chunks = await _get_vector_context(
            "hello",
            storage,
            QueryParam(enable_hybrid=True),
            query_embedding=[0.1] * DIM,
        )
        assert [c["chunk_id"] for c in chunks] == ["c-1"]
        assert query_calls == [("hello", 20)]

    assert hybrid_calls == []
