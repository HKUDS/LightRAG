"""Registry and interface contract for the KEYWORD_STORAGE role."""

import inspect

from lightrag.kg import STORAGE_IMPLEMENTATIONS, STORAGE_ENV_REQUIREMENTS, STORAGES


def test_keyword_storage_role_registered():
    role = STORAGE_IMPLEMENTATIONS["KEYWORD_STORAGE"]
    assert "Bm25KeywordStorage" in role["implementations"]
    assert "index_entities" in role["required_methods"]
    assert "search" in role["required_methods"]


def test_bm25_keyword_storage_env_requirements_empty():
    assert STORAGE_ENV_REQUIREMENTS["Bm25KeywordStorage"] == []


def test_bm25_keyword_storage_module_mapped():
    assert STORAGES["Bm25KeywordStorage"] == ".kg.bm25s_keyword_impl"


def test_base_keyword_storage_interface():
    from lightrag.base import BaseKeywordStorage

    assert inspect.isabstract(BaseKeywordStorage)
    sig = inspect.signature(BaseKeywordStorage.search)
    assert list(sig.parameters) == ["self", "query", "top_k"]
    sig = inspect.signature(BaseKeywordStorage.index_entities)
    assert list(sig.parameters) == ["self", "names"]


def test_factory_resolves_bm25_keyword_storage():
    pytest_bm25s = __import__("pytest").importorskip("bm25s")  # noqa: F841
    from lightrag.kg.factory import get_storage_class
    from lightrag.base import BaseKeywordStorage

    cls = get_storage_class("Bm25KeywordStorage")
    assert issubclass(cls, BaseKeywordStorage)
