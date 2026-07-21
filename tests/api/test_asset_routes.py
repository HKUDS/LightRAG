import json
import sys
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from lightrag.api.assets import WorkspaceAssetCatalog  # noqa: E402
from lightrag.api.routers.asset_routes import create_asset_routes  # noqa: E402
sys.argv = _original_argv


def _asset_app(tmp_path):
    parsed = tmp_path / "__parsed__" / "report.parsed"
    assets = parsed / "report.blocks.assets"
    assets.mkdir(parents=True)
    (assets / "figure.png").write_bytes(b"PNG bytes")
    doc_id = "doc-" + "1" * 32
    (parsed / "report.blocks.jsonl").write_text(
        json.dumps({"type": "meta", "doc_id": doc_id}) + "\n"
    )
    drawing_id = "im-" + "1" * 32 + "-0001"
    (parsed / "report.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    drawing_id: {
                        "path": "report.blocks.assets/figure.png",
                        "caption": "Figure",
                    }
                }
            }
        )
    )
    manager = SimpleNamespace(input_dir=tmp_path, workspace="workspace-key")
    app = FastAPI()
    app.include_router(create_asset_routes(manager))
    return app, manager


def test_asset_is_binary_and_has_cache_headers(tmp_path):
    app, manager = _asset_app(tmp_path)
    asset = WorkspaceAssetCatalog(manager).records()[0]
    response = TestClient(app).get(f"/assets/{asset.asset_id}")
    assert response.status_code == 200
    assert response.content == b"PNG bytes"
    assert response.headers["content-type"] == "image/png"
    assert response.headers["content-disposition"].startswith("inline")
    assert response.headers["etag"].startswith('"sha256-')


def test_unknown_asset_is_not_disclosed(tmp_path):
    app, _manager = _asset_app(tmp_path)
    response = TestClient(app).get("/assets/ast_does-not-exist")
    assert response.status_code == 404
