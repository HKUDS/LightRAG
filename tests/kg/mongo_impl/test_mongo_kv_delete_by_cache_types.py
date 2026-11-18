"""``MongoKVStorage.delete_by_cache_types``: one ``delete_many`` whose regex
must select exactly what ``parse_cache_key`` would, durable on return."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.kg.mongo_impl import MongoKVStorage
from lightrag.utils import parse_cache_key

pytestmark = pytest.mark.offline


def _storage(deleted_count):
    storage = MongoKVStorage.__new__(MongoKVStorage)
    storage.namespace = "llm_response_cache"
    storage.workspace = "ws"
    storage.global_config = {}
    collection = MagicMock()
    collection.delete_many = AsyncMock(
        return_value=SimpleNamespace(deleted_count=deleted_count)
    )
    storage._data = collection
    return storage, collection


_KEYS = [
    "mix:query:q1",
    "naive:keywords:k1",
    "default:extract:e1",
    "default:summary:s1",
    "default:smartheading:h1",
    "mix:query",  # two parts: does not parse, must be kept
    "mix:queryx:q",  # a type that merely starts with "query"
    "a:b:query:c",  # "query" in the hash, not the type
]


@pytest.mark.parametrize("cache_types", [{"query", "keywords"}, {"query"}])
async def test_the_filter_matches_exactly_what_parse_cache_key_selects(cache_types):
    storage, collection = _storage(deleted_count=7)

    deleted = await storage.delete_by_cache_types(cache_types)

    assert deleted == 7
    (filter_,), _ = collection.delete_many.call_args
    pattern = re.compile(filter_["_id"]["$regex"])
    selected = {k for k in _KEYS if pattern.search(k)}
    expected = {
        k
        for k in _KEYS
        if (p := parse_cache_key(k)) is not None and p[1] in cache_types
    }
    assert selected == expected


async def test_no_types_issues_no_delete():
    """``^[^:]*:(?:):`` would match every key whose type is empty."""
    storage, collection = _storage(deleted_count=99)

    assert await storage.delete_by_cache_types(set()) == 0
    collection.delete_many.assert_not_called()
