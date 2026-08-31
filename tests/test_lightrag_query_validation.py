from types import SimpleNamespace

import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.query_validation import RAGQueryTooShortError


pytestmark = pytest.mark.offline


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["aquery_llm", "aquery_data"])
async def test_core_rag_query_entry_points_reject_short_queries_before_storage_access(
    method_name,
):
    rag = SimpleNamespace()

    with pytest.raises(RAGQueryTooShortError):
        await getattr(LightRAG, method_name)(
            rag,
            "中",
            param=QueryParam(mode="mix"),
        )
