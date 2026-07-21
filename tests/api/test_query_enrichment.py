from types import SimpleNamespace
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from lightrag.api.routers.query_routes import create_query_routes  # noqa: E402
sys.argv = _original_argv


class _Rag:
    async def aquery_llm(self, *_args, **_kwargs):
        return {
            "llm_response": {"content": "The answer", "is_streaming": False},
            "data": {
                "references": [{"reference_id": "r1", "file_path": "report.pdf"}],
                "chunks": [
                    {
                        "reference_id": "r1",
                        "file_path": "report.pdf",
                        "full_doc_id": "doc-1",
                        "content": "retrieved context",
                    }
                ],
                "entities": [{"entity_name": "ClinicalModernBERT"}],
                "relationships": [
                    {"src_id": "ClinicalModernBERT", "tgt_id": "FlashAttention"}
                ],
            },
        }


def test_query_enrichment_is_opt_in(tmp_path):
    manager = SimpleNamespace(input_dir=tmp_path, workspace="workspace-key")
    app = FastAPI()
    app.include_router(create_query_routes(_Rag(), document_manager=manager))
    client = TestClient(app)

    plain = client.post("/query", json={"query": "hello world"})
    assert plain.status_code == 200
    assert plain.json()["content_blocks"] is None

    enriched = client.post(
        "/query",
        json={
            "query": "hello world",
            "include_content_blocks": True,
            "include_graph_context": True,
        },
    )
    payload = enriched.json()
    assert enriched.status_code == 200
    assert payload["content_blocks"] == [{"type": "markdown", "markdown": "The answer"}]
    assert payload["graph_context"]["nodes"] == ["ClinicalModernBERT"]
    assert payload["references"][0]["document_id"] == "doc-1"


def test_stream_query_emits_optional_context_line(tmp_path):
    manager = SimpleNamespace(input_dir=tmp_path, workspace="workspace-key")
    app = FastAPI()
    app.include_router(create_query_routes(_Rag(), document_manager=manager))
    response = TestClient(app).post(
        "/query/stream",
        json={
            "query": "hello world",
            "stream": False,
            "include_content_blocks": True,
            "include_graph_context": True,
        },
    )
    assert response.status_code == 200
    assert '"content_blocks"' in response.text
    assert '"graph_context"' in response.text
