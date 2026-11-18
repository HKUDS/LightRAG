"""``PGKVStorage.delete`` / ``PGDocStatusStorage.delete`` must raise on failure.

Both used to log the database error and return normally, so a caller could not
tell a delete that failed from one that landed. The document-deletion pipeline
relies on the raise to keep the document's retry record; see
``tests/pipeline/test_pg_delete_failures.py`` for that end-to-end behavior.
"""

import pytest

pytest.importorskip("asyncpg")

from lightrag.kg.postgres_impl import (  # noqa: E402
    PGDocStatusStorage,
    PGKVStorage,
)
from lightrag.namespace import NameSpace  # noqa: E402


pytestmark = pytest.mark.offline


class _DeleteUnavailable(Exception):
    pass


class _FailingDB:
    async def _run_with_retry(self, operation):
        raise _DeleteUnavailable("delete unavailable")


def _make_storage(cls, namespace):
    """Build a storage instance exercising only the attributes ``delete`` reads."""
    storage = object.__new__(cls)
    storage.namespace = namespace
    storage.workspace = "test_ws"
    storage._max_delete_records_per_batch = 0
    storage.db = _FailingDB()
    return storage


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cls", "namespace"),
    [
        (PGKVStorage, NameSpace.KV_STORE_TEXT_CHUNKS),
        (PGKVStorage, NameSpace.KV_STORE_FULL_DOCS),
        (PGDocStatusStorage, NameSpace.DOC_STATUS),
    ],
)
async def test_delete_propagates_database_failure(cls, namespace):
    storage = _make_storage(cls, namespace)

    with pytest.raises(_DeleteUnavailable, match="delete unavailable"):
        await storage.delete(["id-1", "id-2"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cls", "namespace"),
    [
        (PGKVStorage, NameSpace.KV_STORE_TEXT_CHUNKS),
        (PGDocStatusStorage, NameSpace.DOC_STATUS),
    ],
)
async def test_empty_delete_does_not_touch_the_database(cls, namespace):
    storage = _make_storage(cls, namespace)

    await storage.delete([])
