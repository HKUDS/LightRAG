"""``BaseVectorStorage.drop_all_models`` is the workspace-clear contract."""

from unittest.mock import AsyncMock

import pytest

from lightrag.base import BaseVectorStorage

pytestmark = pytest.mark.offline


class _Stub(BaseVectorStorage):
    """BaseVectorStorage with its abstract surface waived: only the concrete
    ``drop_all_models`` / ``drop_sibling_models`` under test are exercised."""


_Stub.__abstractmethods__ = frozenset()


def _storage(drop_result, sibling_result):
    storage = object.__new__(_Stub)
    storage.drop = AsyncMock(return_value=drop_result)
    storage.drop_sibling_models = AsyncMock(return_value=sibling_result)
    return storage


OK = {"status": "success", "message": "ok"}


async def test_clears_the_active_model_then_its_siblings():
    storage = _storage(OK, OK)
    assert (await storage.drop_all_models())["status"] == "success"
    storage.drop.assert_awaited_once()
    storage.drop_sibling_models.assert_awaited_once()


async def test_siblings_are_untouched_when_the_active_drop_fails():
    failed = {"status": "error", "message": "nope"}
    storage = _storage(failed, OK)
    assert await storage.drop_all_models() == failed
    storage.drop_sibling_models.assert_not_awaited()


async def test_a_sibling_failure_is_reported_not_swallowed():
    storage = _storage(OK, {"status": "error", "message": "stuck"})
    result = await storage.drop_all_models()
    assert result["status"] == "error"
    assert "stuck" in result["message"]


async def test_default_has_no_siblings():
    s = object.__new__(_Stub)
    assert (await s.drop_sibling_models())["status"] == "success"
