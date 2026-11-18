"""``DELETE /documents/query_cache``: clear the query-answer cache only.

The route is the REST face of ``LightRAG.aclear_query_cache`` (see
``tests/test_aclear_query_cache.py`` for the storage-level contract). These
tests pin what is the route's own: the selector and its default, that it is
accepted while the pipeline is busy -- it takes no reservation, by design --
and that a commit that did not land reaches the client as an error rather than
as a success.

See HKUDS/LightRAG#4082.
"""

import importlib
import sys
from uuid import uuid4

import numpy as np
import pytest
from fastapi import HTTPException

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag import LightRAG  # noqa: E402
from lightrag.kg.shared_storage import get_namespace_data  # noqa: E402
from lightrag.utils import EmbeddingFunc, Tokenizer, load_json  # noqa: E402

DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


_ROWS = {
    "mix:query:q1": "query",
    "local:query:q2": "query",
    "hybrid:keywords:k1": "keywords",
    "default:extract:e1": "extract",
    "default:summary:s1": "summary",
    "default:smartheading:h1": "smartheading",
}
_NEVER_DELETED = {"default:extract:e1", "default:summary:s1", "default:smartheading:h1"}


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"querycacheroute-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
    )
    await rag.initialize_storages()
    await rag.llm_response_cache.upsert(
        {key: {"return": "cached", "cache_type": t} for key, t in _ROWS.items()}
    )
    await rag.llm_response_cache.index_done_callback()
    return rag


def _endpoint(rag, input_dir):
    router = create_document_routes(rag, DocumentManager(str(input_dir)))
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "clear_query_cache"
    ][-1]


def _on_disk(rag) -> set[str]:
    return set(load_json(rag.llm_response_cache._file_name) or {})


def test_the_route_is_registered_as_a_delete():
    router = create_document_routes(object(), DocumentManager("."))
    routes = [r for r in router.routes if getattr(r, "name", "") == "clear_query_cache"]
    assert [(r.path, r.methods) for r in routes] == [
        ("/documents/query_cache", {"DELETE"})
    ]


@pytest.mark.parametrize(
    ("cache_type", "expected_types", "expected_kept"),
    [
        (None, ["keywords", "query"], set()),
        (["query"], ["query"], {"hybrid:keywords:k1"}),
        (["keywords"], ["keywords"], {"mix:query:q1", "local:query:q2"}),
        (["query", "keywords", "query"], ["keywords", "query"], set()),
    ],
    ids=["default-both", "query-only", "keywords-only", "duplicates"],
)
async def test_the_selector_and_its_default(
    tmp_path, cache_type, expected_types, expected_kept
):
    rag = await _build_rag(tmp_path)
    try:
        response = await _endpoint(rag, tmp_path)(cache_type=cache_type)

        assert response.status == "success"
        assert response.cache_types == expected_types
        assert response.deleted == 3 - len(expected_kept)
        assert _on_disk(rag) == _NEVER_DELETED | expected_kept
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize("flag", ["busy", "destructive_busy"])
async def test_the_route_succeeds_while_the_pipeline_is_busy(tmp_path, flag):
    """The opposite of ``DELETE /documents``: refusing here would block the
    call exactly when an operator most needs it, for no integrity gain."""
    rag = await _build_rag(tmp_path)
    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status[flag] = True
    try:
        response = await _endpoint(rag, tmp_path)(cache_type=None)

        assert response.status == "success"
        assert response.deleted == 3
        assert _on_disk(rag) == _NEVER_DELETED
        assert pipeline_status[flag] is True, "the route must not touch the flag"
    finally:
        pipeline_status[flag] = False
        await rag.finalize_storages()


async def test_a_failed_chunk_commit_is_reported_as_an_error(tmp_path):
    rag = await _build_rag(tmp_path)
    try:

        async def _chunk_commit_fails():
            raise RuntimeError("chunk store is down")

        rag.text_chunks.index_done_callback = _chunk_commit_fails

        with pytest.raises(HTTPException) as exc_info:
            await _endpoint(rag, tmp_path)(cache_type=None)

        assert exc_info.value.status_code == 500
        # Never reported as cleared: the file still holds every row.
        assert _on_disk(rag) == set(_ROWS)
    finally:
        await rag.finalize_storages()
