"""Lone-surrogate records must not wedge a ``NanoVectorDBStorage`` namespace.

The namespace file is written by the external ``NanoVectorDB.save`` with
``json.dump(..., ensure_ascii=False)`` into a UTF-8 file, so a lone surrogate
in any record string made every commit on that namespace raise
``UnicodeEncodeError`` — and the poisoned row stayed buffered, so later
commits failed identically until the row was deleted. ``upsert`` strips
such characters from record values, so a buffered record can never wedge
the namespace.
"""

import numpy as np
import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")  # noqa: F841

from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402

DIM = 8


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def _embed(texts, **kwargs):
    return np.array(
        [[0.1] * DIM for _ in texts],
        dtype=np.float32,
    )


def _make_storage(tmp_path, meta_fields=None) -> NanoVectorDBStorage:
    return NanoVectorDBStorage(
        namespace="test_vectors",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=DIM, max_token_size=512, func=_embed
        ),
        meta_fields=meta_fields or {"content", "tags", "extra"},
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_upsert_strips_lone_surrogate_and_commit_succeeds(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    await storage.upsert({"bad": {"content": "hello \ud800 world"}})

    # Pre-fix this raised UnicodeEncodeError: the surrogate cannot be
    # UTF-8-encoded by the namespace file's json.dump(ensure_ascii=False).
    await storage.index_done_callback()

    row = await storage.get_by_id("bad")
    assert row["content"] == "hello  world"

    # The file on disk must be a valid snapshot: a fresh instance over the
    # same directory reads the row back.
    reloaded = _make_storage(tmp_path)
    await reloaded.initialize()
    assert (await reloaded.get_by_id("bad"))["content"] == "hello  world"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_poisoned_row_no_longer_wedges_namespace(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    await storage.upsert({"bad": {"content": "hello \ud800 world"}})
    await storage.index_done_callback()

    # Pre-fix this second commit raised the same UnicodeEncodeError: the
    # poisoned row stayed buffered and wedged the whole namespace.
    await storage.upsert({"good": {"content": "clean text"}})
    await storage.index_done_callback()

    assert (await storage.get_by_id("bad"))["content"] == "hello  world"
    assert (await storage.get_by_id("good"))["content"] == "clean text"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_clean_content_and_nested_values_unchanged_except_surrogates(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    await storage.upsert(
        {
            "id1": {
                "content": "plain text with emoji \U0001f600 and entity &amp;",
                "tags": ["keep", "drop\udc00me"],
                "extra": {"note": "x\udfffy", "count": 3, "nc": "a\ufffeb\uffffc"},
            }
        }
    )
    await storage.index_done_callback()

    # Clean strings pass through byte-identical; only the lone surrogates go.
    # U+FFFE / U+FFFF are UTF-8-encodable and were always persisted, so they
    # must survive the commit and a reload unchanged.
    expected = {
        "content": "plain text with emoji \U0001f600 and entity &amp;",
        "tags": ["keep", "dropme"],
        "extra": {"note": "xy", "count": 3, "nc": "a\ufffeb\uffffc"},
    }
    row = await storage.get_by_id("id1")
    assert {k: row[k] for k in expected} == expected

    reloaded = _make_storage(tmp_path)
    await reloaded.initialize()
    row = await reloaded.get_by_id("id1")
    assert {k: row[k] for k in expected} == expected


@pytest.mark.offline
@pytest.mark.asyncio
async def test_encodable_identity_field_is_not_rewritten(tmp_path):
    """An entity name holding U+FFFE must keep matching its graph key."""
    storage = _make_storage(tmp_path, meta_fields={"entity_name", "content"})
    await storage.initialize()

    await storage.upsert(
        {"ent-x": {"entity_name": "A\ufffeB", "content": "A\ufffeB\ndesc"}}
    )
    await storage.index_done_callback()

    reloaded = _make_storage(tmp_path, meta_fields={"entity_name", "content"})
    await reloaded.initialize()
    assert (await reloaded.get_by_id("ent-x"))["entity_name"] == "A\ufffeB"
