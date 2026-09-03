import os
import sys

# Patch sys.argv before any lightrag import so parse_args doesn't consume pytest flags
_original_argv = sys.argv.copy()
sys.argv = ["lightrag"]

try:
    from lightrag.api.config import initialize_config

    initialize_config(force=True)
    from lightrag.api.routers.config_routes import update_env_file, create_config_routes
finally:
    sys.argv = _original_argv

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_update_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=old_val\n")

    update_env_file(
        {"EXISTING_KEY": "new_val", "LANGFUSE_PUBLIC_KEY": "pk-test"},
        env_path=str(env_file),
    )

    content = env_file.read_text()
    assert "EXISTING_KEY=new_val" in content
    assert "LANGFUSE_PUBLIC_KEY=pk-test" in content


def test_langfuse_config_routes(monkeypatch, tmp_path):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-123")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-456")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    app = FastAPI()
    app.include_router(create_config_routes(api_key=None))
    client = TestClient(app)

    # Test GET
    res = client.get("/api/config/langfuse")
    assert res.status_code == 200
    data = res.json()
    assert data["public_key"] == "pk-123"
    assert data["secret_key_set"] is True
    assert data["host"] == "https://cloud.langfuse.com"
    assert data["enabled"] is True

    # Test POST
    monkeypatch.chdir(tmp_path)

    post_res = client.post(
        "/api/config/langfuse",
        json={"public_key": "pk-updated", "host": "https://custom.langfuse.com"},
    )
    assert post_res.status_code == 200
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-updated"
    assert os.environ["LANGFUSE_HOST"] == "https://custom.langfuse.com"
