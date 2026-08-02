import json
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


def test_video_scene_is_returned_as_timecoded_evidence(tmp_path):
    parsed_dir = tmp_path / "__parsed__" / "demo.mp4.parsed"
    asset_dir = parsed_dir / "demo.blocks.assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "scene-0001.jpg").write_bytes(b"jpeg")
    block_id = "block-1"
    (parsed_dir / "demo.blocks.jsonl").write_text(
        json.dumps({"type": "meta", "doc_id": "doc-1"})
        + "\n"
        + json.dumps({"type": "content", "blockid": block_id, "content": '<drawing id="im-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-0001" />'})
        + "\n"
    )
    (parsed_dir / "demo.drawings.json").write_text(json.dumps({"drawings": {
        "im-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-0001": {
            "blockid": block_id,
            "path": "demo.blocks.assets/scene-0001.jpg",
            "caption": "scene-0001: 00:00:00.000 – 00:00:04.000",
            "extras": {"media_type": "video_contact_sheet", "scene_id": "scene-0001", "start_ms": 0, "end_ms": 4000},
        }
    }}))
    manager = SimpleNamespace(input_dir=tmp_path, workspace="workspace-key")
    from lightrag.api.query_enrichment import QueryEnricher

    blocks, _ = QueryEnricher(manager).content_blocks("answer", [{"content": '<drawing id="im-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-0001" />'}])
    assert blocks[1]["type"] == "video_scene"
    assert blocks[1]["start_ms"] == 0
    assert blocks[1]["end_ms"] == 4000
