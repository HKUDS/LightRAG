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


async def test_vouched_names_reach_the_sibling_sweep_and_not_drop():
    storage = _storage(OK, OK)
    await storage.drop_all_models(vouched=["a_chunks_m_8d"])
    storage.drop.assert_awaited_once_with()
    storage.drop_sibling_models.assert_awaited_once_with(vouched=["a_chunks_m_8d"])


async def test_default_names_no_container_and_lists_nothing_unvouched():
    s = object.__new__(_Stub)
    assert s.model_container_name("m", 8) is None
    assert await s.unvouched_sibling_containers() == []


@pytest.mark.parametrize(
    "model_name, expected",
    [
        ("Text-Embedding-3.Large", "text_embedding_3_large_8d"),
        ("  m  ", "m_8d"),
        ("   ", None),
        (None, None),
    ],
)
def test_the_suffix_fold_is_one_definition(model_name, expected):
    """``model_container_name`` and the active container share this fold; a
    second copy that drifted would vouch for a name nothing attaches to."""
    assert _Stub._collection_suffix_for(model_name, 8) == expected
