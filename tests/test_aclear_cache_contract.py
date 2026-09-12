"""``LightRAG.aclear_cache`` must not report a drop that did not happen.

``BaseKVStorage.drop`` reports a non-raising failure as
``{"status": "error"}``. That dict is truthy, so the old truthiness check
logged "Cleared all cache" on exactly the failures it was meant to catch, and
the broad ``except`` swallowed the raising ones. Both are silent failure: the
caller (now ``DELETE /documents?clear_llm_cache=true``) decides what a
half-cleared cache means for its operation, and it can only do that if the
failure reaches it.
"""

import pytest

from lightrag.lightrag import LightRAG

pytestmark = pytest.mark.offline


class _CacheStorage:
    def __init__(self, drop_result=None, drop_error: Exception | None = None):
        self.drop_result = drop_result or {"status": "success", "message": "dropped"}
        self.drop_error = drop_error
        self.drop_calls = 0
        self.commit_calls = 0

    async def drop(self):
        self.drop_calls += 1
        if self.drop_error is not None:
            raise self.drop_error
        return self.drop_result

    async def index_done_callback(self):
        self.commit_calls += 1


class _FakeRag:
    """Only the attributes ``aclear_cache`` touches; building a real LightRAG
    would drag in storages and an event-loop owner this contract ignores."""

    aclear_cache = LightRAG.aclear_cache

    def __init__(self, cache):
        self.llm_response_cache = cache
        self.text_chunks = None


async def test_aclear_cache_commits_on_a_successful_drop():
    cache = _CacheStorage()
    await _FakeRag(cache).aclear_cache()

    assert cache.drop_calls == 1
    assert cache.commit_calls == 1


async def test_aclear_cache_raises_when_drop_reports_an_error():
    cache = _CacheStorage(drop_result={"status": "error", "message": "backend down"})

    with pytest.raises(RuntimeError, match="backend down"):
        await _FakeRag(cache).aclear_cache()

    # The commit is skipped: there is nothing to publish, and reporting the
    # flush as done would compound the misreport.
    assert cache.commit_calls == 0


async def test_aclear_cache_propagates_a_raising_drop():
    cache = _CacheStorage(drop_error=ValueError("connection reset"))

    with pytest.raises(ValueError, match="connection reset"):
        await _FakeRag(cache).aclear_cache()


async def test_aclear_cache_is_a_noop_without_a_cache_storage():
    rag = _FakeRag(None)

    await rag.aclear_cache()  # must not raise
